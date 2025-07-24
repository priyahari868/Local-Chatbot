import streamlit as st
import os
import chromadb
from chromadb.config import Settings
from rag_engine import RAGEngine
import time

# Page configuration
st.set_page_config(
    page_title="Local Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
        background-color: #f8f9fa;
    }
    .context-box {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .source-info {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []

def initialize_rag_engine():
    """Initialize the RAG engine."""
    if st.session_state.rag_engine is None:
        with st.spinner("🔄 Initializing RAG Engine..."):
            try:
                st.session_state.rag_engine = RAGEngine()
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG Engine: {str(e)}")
                return False
    return True

def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 Local Chatbot</h1>', unsafe_allow_html=True)
    
    # Initialize RAG engine
    if not initialize_rag_engine():
        st.stop()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Display collection stats
        if st.session_state.rag_engine:
            stats = st.session_state.rag_engine.get_collection_stats()
            st.info(f"📊 **Documents in DB**: {stats['total_chunks']} chunks")
            st.info(f"🧠 **Embedding Model**: {stats['model_name']}")
        
        st.markdown("---")
        
        # File upload section
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['txt', 'pdf'],
            help="Upload .txt or .pdf files to add to the knowledge base"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                process_documents(uploaded_files)
        
        st.markdown("---")
        
        # System status
        st.subheader("🔧 System Status")
        check_system_status()
        
        st.markdown("---")
        
        # Clear database
        if st.button("🗑️ Clear Database", help="Remove all documents from the database"):
            if st.session_state.rag_engine.clear_collection():
                st.success("✅ Database cleared!")
                st.session_state.uploaded_docs = []
                st.rerun()
            else:
                st.error("❌ Failed to clear database")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display uploaded documents
        if st.session_state.uploaded_docs:
            st.subheader("📋 Uploaded Documents")
            for doc in st.session_state.uploaded_docs:
                st.markdown(f"📄 **{doc['name']}** - {doc['chunks']} chunks")
            st.markdown("---")
        
        # Chat input
        user_query = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know?",
            help="Enter your question and press Enter"
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            ask_button = st.button("🔍 Ask", type="primary")
        with col_btn2:
            if st.button("🧹 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process query
        if (ask_button or user_query) and user_query.strip():
            process_query(user_query.strip())
        
        # Display chat history
        display_chat_history()
    
    with col2:
        st.header("📖 Instructions")
        st.markdown("""
        **Getting Started:**
        1. **Install Ollama** on your system
        2. **Pull a model**: `ollama pull mistral`
        3. **Start Ollama**: `ollama serve`
        4. **Upload documents** using the sidebar
        5. **Ask questions** about your documents
        
        **Supported Formats:**
        - Text files (.txt)
        - PDF files (.pdf)
        
        **Features:**
        - 🔒 Completely offline and private
        - 🚀 Fast semantic search
        - 🧠 Local LLM integration
        - 💾 Persistent vector storage
        """)
        
        st.markdown("---")
        st.header("🎯 Tips")
        st.markdown("""
        - **Be specific** in your questions
        - **Ask follow-up questions** for clarification
        - **Upload multiple documents** for comprehensive answers
        - **Clear the database** to start fresh
        """)

def process_documents(uploaded_files):
    """Process uploaded documents and add them to the vector database."""
    success_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i) / len(uploaded_files))
        
        try:
            # Extract text based on file type
            if uploaded_file.name.endswith('.pdf'):
                text = st.session_state.rag_engine.extract_text_from_pdf(uploaded_file)
            else:  # .txt file
                text = str(uploaded_file.read(), "utf-8")
            
            if not text.strip():
                st.error(f"❌ No text extracted from {uploaded_file.name}")
                continue
            
            # Add document to vector database
            if st.session_state.rag_engine.add_document(text, uploaded_file.name):
                success_count += 1
                # Track uploaded documents
                chunks = len(st.session_state.rag_engine.split_text_into_chunks(text))
                st.session_state.uploaded_docs.append({
                    'name': uploaded_file.name,
                    'chunks': chunks
                })
                st.success(f"✅ Successfully processed {uploaded_file.name}")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    progress_bar.progress(1.0)
    status_text.text(f"Completed! Processed {success_count}/{len(uploaded_files)} documents.")
    
    if success_count > 0:
        time.sleep(1)
        st.rerun()

def process_query(user_query):
    """Process user query and generate response."""
    if not st.session_state.uploaded_docs:
        st.warning("⚠️ Please upload some documents first!")
        return
    
    with st.spinner("🔍 Searching documents and generating response..."):
        # Perform RAG query
        result = st.session_state.rag_engine.rag_query(user_query)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'query': user_query,
            'answer': result['answer'],
            'sources': result['sources'],
            'context': result['context_used']
        })
        
    st.rerun()

def display_chat_history():
    """Display the chat history."""
    if not st.session_state.chat_history:
        st.info("👋 Upload some documents and ask a question to get started!")
        return
    
    st.subheader("💭 Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            # User query
            st.markdown(f"**🤔 You asked:** {chat['query']}")
            
            # Assistant response
            st.markdown("**🤖 Assistant:**")
            st.markdown(f'<div class="chat-message">{chat["answer"]}</div>', unsafe_allow_html=True)
            
            # Sources information
            if chat['sources']:
                st.markdown("**📚 Sources:**")
                for source in chat['sources']:
                    similarity_percent = source['similarity'] * 100
                    st.markdown(
                        f'<div class="source-info">📄 {source["filename"]} (chunk {source["chunk_id"]}) - Similarity: {similarity_percent:.1f}%</div>',
                        unsafe_allow_html=True
                    )
            
            # Show context (expandable)
            if chat['context']:
                with st.expander("👁️ View Retrieved Context"):
                    st.markdown(f'<div class="context-box">{chat["context"]}</div>', unsafe_allow_html=True)
            
            st.markdown("---")

def check_system_status():
    """Check and display system status."""
    if st.session_state.rag_engine:
        # Check Ollama connection
        ollama_status = st.session_state.rag_engine.check_ollama_connection()
        if ollama_status:
            st.success("✅ Ollama connected")
        else:
            st.error("❌ Ollama not running")
            st.markdown("Start Ollama: `ollama serve`")
        
        # Check model availability
        model_status = st.session_state.rag_engine.check_model_availability()
        if model_status:
            st.success(f"✅ Model '{st.session_state.rag_engine.model_name}' ready")
        else:
            st.error(f"❌ Model '{st.session_state.rag_engine.model_name}' not available")
            st.markdown(f"Pull model: `ollama pull {st.session_state.rag_engine.model_name}`")

if __name__ == "__main__":
    main()