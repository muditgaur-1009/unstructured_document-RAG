# Unstructured_Document-RAG 📚

A Streamlit-based application that implements a Retrieval-Augmented Generation (RAG) system for intelligent document question-answering. The system processes PDF documents, creates semantic embeddings, and answers questions using context-aware retrieval and language model inference.

## 🌟 Features

- **PDF Processing**: Upload and process multiple PDF documents
- **Smart Text Chunking**: Implements recursive character splitting with optimal chunk sizes
- **Advanced Retrieval**: Uses FAISS for efficient similarity search with MMR reranking
- **Contextual Compression**: Implements LLM-based context compression for better answer relevance
- **Interactive UI**: Clean and intuitive Streamlit interface
- **Source Attribution**: Displays source documents and page numbers for transparency

## 🛠️ Prerequisites

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)
- 8GB RAM minimum (16GB recommended)

### System Dependencies
For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install poppler-utils python3-magic
```

For macOS:
```bash
brew install poppler magic
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/muditgaur-1009/unstructured_document-RAG.git
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the application in your web browser (typically at `http://localhost:8501`)

3. Use the interface to:
   - Upload PDF documents
   - Ask questions about the documents
   - View answers with source attributions
   - Reset the system when needed

## 💡 How It Works

1. **Document Processing**:
   - PDF documents are processed using `unstructured`
   - Text is split into chunks using `RecursiveCharacterTextSplitter`
   - Each chunk maintains metadata about its source and page number

2. **Embedding and Indexing**:
   - Text chunks are embedded using BAAI/bge-large-en-v1.5
   - FAISS creates a searchable vector index
   - MMR reranking ensures diverse context retrieval

3. **Question Answering**:
   - Questions are processed against the vector index
   - Relevant contexts are retrieved and compressed
   - LLaMA model generates answers based on retrieved context

## 🔧 Configuration

Key parameters that can be adjusted in the code:

```python
# Text splitting parameters
chunk_size=500
chunk_overlap=50

# Retrieval parameters
search_kwargs={
    "k": 6,
    "fetch_k": 10
}

# LLM parameters
temperature=0.7
top_p=0.9
repeat_penalty=1.1
num_ctx=4096
```

## ⚠️ Known Limitations

- Large PDF files may require significant processing time
- GPU acceleration recommended for optimal performance
- Limited by local LLM capabilities and context window

## 🔄 Memory Management

The application includes automatic cleanup of temporary files:
- Temporary upload directory is created at startup
- Files are automatically cleaned up on application exit
- Manual reset available through the UI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit
- Uses LangChain for RAG implementation
- Powered by Ollama and FAISS
- Embedding model from BAAI
