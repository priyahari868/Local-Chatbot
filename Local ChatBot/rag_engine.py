import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import PyPDF2
import io
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, persist_directory: str = "./vector_store"):
        """Initialize the RAG engine with ChromaDB and sentence transformer."""
        self.persist_directory = persist_directory
        self.embedding_model_name = "all-MiniLM-L6-v2"
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection_name = "documents"
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        # Ollama settings
        self.ollama_url = "http://localhost:11434"
        self.model_name = "mistral"

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""

    def split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If we're not at the end, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end != -1 and word_end > start + chunk_size // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks

    def add_document(self, text: str, filename: str) -> bool:
        """Add a document to the vector database."""
        try:
            # Split text into chunks
            chunks = self.split_text_into_chunks(text)
            
            if not chunks:
                logger.warning("No chunks created from document")
                return False
            
            logger.info(f"Processing {len(chunks)} chunks from {filename}")
            
            # Generate embeddings for chunks
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Create unique IDs for chunks
            ids = [f"{filename}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
            
            # Create metadata
            metadatas = [{"filename": filename, "chunk_id": i} for i in range(len(chunks))]
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False

    def query_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query the vector database for relevant documents."""
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return []

    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def check_model_availability(self) -> bool:
        """Check if the specified model is available in Ollama."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'].split(':')[0] for model in models]
                return self.model_name in available_models
            return False
        except:
            return False

    def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama local LLM."""
        try:
            # Check Ollama connection
            if not self.check_ollama_connection():
                return "❌ Ollama is not running. Please start Ollama service first."
            
            # Check model availability
            if not self.check_model_availability():
                return f"❌ Model '{self.model_name}' is not available. Please pull it first using: ollama pull {self.model_name}"
            
            # Make request to Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"❌ Error from Ollama: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return "❌ Request timed out. The model might be processing a complex query."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"❌ Error generating response: {str(e)}"

    def rag_query(self, user_query: str) -> Dict[str, Any]:
        """Perform RAG query: retrieve relevant documents and generate response."""
        try:
            # Retrieve relevant documents
            relevant_docs = self.query_documents(user_query, n_results=3)
            
            if not relevant_docs:
                return {
                    'answer': "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    'sources': [],
                    'context_used': ""
                }
            
            # Prepare context from retrieved documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(relevant_docs, 1):
                context_parts.append(f"[Context {i}]:\n{doc['content']}")
                sources.append({
                    'filename': doc['metadata'].get('filename', 'Unknown'),
                    'chunk_id': doc['metadata'].get('chunk_id', 0),
                    'similarity': 1 - doc['distance']  # Convert distance to similarity
                })
            
            context = "\n\n".join(context_parts)
            
            # Create prompt for LLM
            prompt = f"""Based on the context below, answer the question. If the context doesn't contain relevant information to answer the question, say so clearly.

Context:
{context}

Question: {user_query}

Answer:"""
            
            # Generate response
            answer = self.generate_response(prompt)
            
            return {
                'answer': answer,
                'sources': sources,
                'context_used': context
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                'answer': f"❌ Error processing query: {str(e)}",
                'sources': [],
                'context_used': ""
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'model_name': self.embedding_model_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_chunks': 0, 'collection_name': self.collection_name, 'model_name': self.embedding_model_name}

    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False