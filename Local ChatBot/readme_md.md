# 🤖 Local LLM + RAG Chatbot

A complete end-to-end Retrieval-Augmented Generation (RAG) chatbot that runs entirely offline using free and open-source tools.

## 🎯 Features

- ✅ **100% Offline & Private** - No external API calls
- ✅ **Local LLM Integration** - Uses Ollama with Mistral/Llama2
- ✅ **Document Upload** - Support for .txt and .pdf files
- ✅ **Vector Database** - ChromaDB for persistent storage
- ✅ **Semantic Search** - Find relevant context using embeddings
- ✅ **User-friendly UI** - Clean Streamlit web interface
- ✅ **Docker Support** - Easy containerized deployment

## 🏗️ Architecture

```
User (Streamlit UI)
↓
Document Upload (.txt/.pdf)
↓
Split + Embed Chunks (sentence-transformers)
↓
Store in ChromaDB (local vector DB)
↓
User Query
↓
Query vector DB for relevant chunks
↓
Assemble prompt (query + context)
↓
Send to local LLM (Ollama)
↓
Display LLM response
```

## 📋 Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.com](https://ollama.com)
3. **Git** (optional, for cloning)

## 🚀 Quick Start

### Method 1: Local Installation

1. **Clone or create the project directory:**
```bash
mkdir rag_chatbot
cd rag_chatbot
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install and setup Ollama:**
```bash
# Install Ollama (follow instructions at ollama.com)
# Then pull a model
ollama pull mistral
```

4. **Start Ollama service:**
```bash
ollama serve
```

5. **Run the application:**
```bash
streamlit run app.py
```

6. **Open your browser:**
Navigate to `http://localhost:8501`

### Method 2: Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t rag-chatbot .
```

2. **Run the container:**
```bash
docker run -p 8501:8501 -p 11434:11434 -v $(pwd)/vector_store:/app/vector_store rag-chatbot
```

3. **Access the application:**
Open `http://localhost:8501` in your browser

## 📖 Usage Guide

### 1. Upload Documents
- Click the sidebar "Upload Documents" section
- Select .txt or .pdf files
- Click "🚀 Process Documents"
- Wait for processing to complete

### 2. Ask Questions
- Type your question in the text input
- Click "🔍 Ask" or press Enter
- View the response with sources and context

### 3. System Status
- Check the sidebar for system status
- Ensure Ollama is running and model is available
- Monitor document count in the database

## 🔧 Configuration

### Changing the LLM Model

Edit `rag_engine.py`:
```python
self.model_name = "llama2"  # Change from "mistral" to "llama2" or other models
```

Then pull the new model:
```bash
ollama pull llama2
```

### Adjusting Chunk Size

Modify the chunk size in `rag_engine.py`:
```python
def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100):
```

### Embedding Model

Change the embedding model:
```python
self.embedding_model_name = "all-mpnet-base-v2"  # More accurate but slower
```

## 📁 Project Structure

```
rag_chatbot/
├── app.py                 # Streamlit UI application
├── rag_engine.py          # RAG logic and LLM integration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md             # This file
└── vector_store/         # ChromaDB persistent directory (created automatically)
```

## 🛠️ Troubleshooting

### Ollama Issues
- **"Ollama not running"**: Start Ollama with `ollama serve`
- **"Model not available"**: Pull the model with `ollama pull mistral`
- **Connection refused**: Check if Ollama is running on port 11434

### Document Processing
- **No text extracted**: Check if PDF is text-based (not scanned images)
- **Processing fails**: Try smaller documents first
- **Empty responses**: Ensure documents contain relevant text

### Performance Issues
- **Slow responses**: Use smaller models like `tinyllama`
- **