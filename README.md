# RAG Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) chatbot system that allows you to chat with your PDF documents using state-of-the-art AI technology.

## 🌟 Features

- **PDF Processing**: Extract and process text from PDF documents with multiple extraction methods
- **Intelligent Chunking**: Split documents into optimal chunks for better retrieval
- **Vector Embeddings**: Generate high-quality embeddings using sentence-transformers or OpenAI
- **Vector Database**: Store and index embeddings using ChromaDB or FAISS
- **Advanced Retrieval**: Multi-stage retrieval with re-ranking and context optimization
- **LLM Integration**: Support for OpenAI GPT models, Anthropic Claude, and local models
- **Conversation Memory**: Maintain context across chat sessions
- **Multiple Interfaces**: Streamlit, Gradio, FastAPI, and CLI interfaces
- **Source Citations**: Get references to original documents and page numbers

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-chatbot

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file based on the example:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required: Set at least one API key
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Customize other settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### 3. Add Your Documents

Place your PDF files in the `./data/pdfs/` directory:

```bash
mkdir -p data/pdfs
# Copy your PDF files to data/pdfs/
```

### 4. Run the Chatbot

Choose your preferred interface:

#### Streamlit Web Interface (Recommended)
```bash
python main.py streamlit
```

#### CLI Interface
```bash
python main.py cli
```

#### Gradio Web Interface
```bash
python main.py gradio
```

#### REST API
```bash
python main.py api
```

## 📖 Usage Examples

### Basic Usage

```python
from src.chatbot import RAGChatbot

# Configure the chatbot
config = {
    'openai_api_key': 'your-api-key',
    'llm_provider': 'openai',
    'llm_model': 'gpt-4o-mini',
    'embedding_model': 'sentence-transformers/all-mpnet-base-v2'
}

# Initialize and load documents
chatbot = RAGChatbot(config)
result = chatbot.load_documents('./data/pdfs')

# Chat with your documents
response = chatbot.chat("What are the main topics in the documents?")
print(response['response'])
```

### Advanced Usage

See `example_usage.py` for comprehensive examples including:
- Document processing pipeline
- Session management
- Performance testing
- Advanced configuration

## 🏗️ Architecture

The RAG chatbot consists of several key components:

```
📄 PDF Documents
    ↓
📝 PDF Extractor (PyPDF2, pdfplumber, PyMuPDF)
    ↓
✂️ Document Processor (LangChain text splitters)
    ↓
🔢 Embedding Generator (sentence-transformers, OpenAI)
    ↓
🗃️ Vector Store (ChromaDB, FAISS)
    ↓
🔍 Document Retriever (similarity search + re-ranking)
    ↓
🧠 LLM Provider (OpenAI, Anthropic, local models)
    ↓
💬 Chat Interface (Streamlit, Gradio, FastAPI, CLI)
```

### Core Components

- **PDF Processor**: Handles text extraction from PDFs with fallback methods
- **Document Processor**: Chunks documents intelligently for optimal retrieval
- **Embedding Generator**: Creates vector representations of text
- **Vector Store**: Indexes and stores embeddings for fast similarity search
- **Document Retriever**: Finds relevant context with advanced ranking
- **LLM Provider**: Generates responses using various AI models
- **Conversation Manager**: Maintains chat history and context
- **Chat Interfaces**: Multiple ways to interact with the system

## 🛠️ Configuration

### Core Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `LLM_PROVIDER` | AI provider (`openai`, `anthropic`, `local`) | `openai` |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `sentence-transformers/all-mpnet-base-v2` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `MAX_RETRIEVAL_DOCS` | Max documents to retrieve | `5` |
| `TEMPERATURE` | LLM temperature | `0.7` |

### Vector Database Options

- **ChromaDB** (default): Persistent vector database with metadata filtering
- **FAISS**: High-performance similarity search library

### Embedding Options

- **Sentence Transformers**: Local embedding models (recommended)
- **OpenAI Embeddings**: Cloud-based embeddings via API

### LLM Provider Options

- **OpenAI**: GPT-4, GPT-3.5-turbo models
- **Anthropic**: Claude models
- **Local**: Support for local models (Ollama, etc.)

## 📁 Project Structure

```
rag-chatbot/
├── src/                          # Source code
│   ├── pdf_processor/           # PDF text extraction
│   ├── embeddings/              # Document processing & embeddings
│   ├── retrieval/               # Vector storage & retrieval
│   ├── llm/                     # LLM providers & prompts
│   ├── chatbot/                 # Main chatbot & interfaces
│   └── utils/                   # Utilities
├── config/                      # Configuration
├── data/                        # Data directory
│   ├── pdfs/                   # Input PDF files
│   ├── processed/              # Processed documents
│   └── vector_db/              # Vector database
├── tests/                       # Test suite
├── main.py                      # Main entry point
├── example_usage.py             # Usage examples
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_pdf_processor.py -v
python -m pytest tests/test_embeddings.py -v
python -m pytest tests/test_chatbot.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 🔧 Advanced Features

### Custom PDF Processing

```python
from src.pdf_processor import PDFExtractor

extractor = PDFExtractor()
result = extractor.extract_text("document.pdf", method="auto")
```

### Custom Document Chunking

```python
from src.embeddings import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=500,
    chunk_overlap=100,
    splitter_type="recursive"
)
```

### Custom Retrieval

```python
from src.retrieval import DocumentRetriever

retriever = DocumentRetriever(
    vector_store=vector_store,
    embedding_generator=embedding_gen,
    default_top_k=10,
    enable_reranking=True
)
```

## 🌐 API Reference

When running with the FastAPI interface, the following endpoints are available:

- `POST /chat` - Send a chat message
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /sessions` - List active sessions
- `GET /sessions/{session_id}` - Get session history
- `DELETE /sessions/{session_id}` - Delete session
- `POST /upload` - Upload and process documents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**No API Key Error**
- Make sure you've set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your `.env` file

**Import Errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're using the correct Python environment

**PDF Processing Errors**
- Some PDFs may require different extraction methods
- Try different PDF files if one fails to process

**Memory Issues**
- Reduce `CHUNK_SIZE` and `MAX_RETRIEVAL_DOCS` for large documents
- Use FAISS instead of ChromaDB for better memory efficiency

**Slow Performance**
- Use local embeddings instead of OpenAI embeddings
- Reduce the number of chunks per document
- Consider using a faster LLM model

### Getting Help

- Check the `example_usage.py` file for comprehensive examples
- Review the test files for API usage examples
- Open an issue on GitHub for bugs or feature requests

## 🔮 Future Enhancements

- [ ] Support for more document formats (Word, PowerPoint, etc.)
- [ ] Multi-language support
- [ ] Advanced query understanding
- [ ] Document summarization
- [ ] Integration with more vector databases
- [ ] Improved web interface with chat history
- [ ] Mobile-responsive design
- [ ] Batch document processing
- [ ] Advanced analytics and insights