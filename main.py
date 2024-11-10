import streamlit as st
import os
import torch
import tempfile
from pathlib import Path
import shutil
import atexit
from typing import List

# Import text processing components
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Set page configuration
st.set_page_config(
    page_title="Enhanced RAG Document QA System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'document_count' not in st.session_state:
    st.session_state.document_count = 0

# Create temporary directories
temp_upload_dir = tempfile.mkdtemp()

@st.cache_resource
def get_embeddings_model():
    """Initialize the embedding model """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",  # Better embedding model
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    )

def create_text_splitter():
    """Create an improved text splitter with better chunking strategy"""
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )

def process_document(file_path: str, file_name: str) -> List[Document]:
    """Process a single document with improved chunking"""
    try:
        # Extract text from PDF
        elements = partition_pdf(
            filename=file_path,
            strategy="fast",
            include_metadata=True
        )
        
        # Combine text while preserving structure
        text_chunks = []
        current_page = None
        current_text = ""
        
        for element in elements:
            if hasattr(element, 'text') and element.text.strip():
                page_num = getattr(element, 'page_number', None)
                if page_num != current_page and current_text:
                    text_chunks.append((current_text, current_page))
                    current_text = ""
                current_page = page_num
                current_text += element.text + "\n"
        
        if current_text:
            text_chunks.append((current_text, current_page))
        
        # Create text splitter and split documents
        text_splitter = create_text_splitter()
        documents = []
        
        for text, page_num in text_chunks:
            splits = text_splitter.create_documents([text])
            for split in splits:
                documents.append(
                    Document(
                        page_content=split.page_content,
                        metadata={
                            'source': file_name,
                            'page_number': page_num,
                            'chunk_size': len(split.page_content)
                        }
                    )
                )
        
        return documents
    except Exception as e:
        st.error(f"Error processing file {file_name}: {str(e)}")
        return []

def process_uploaded_files(uploaded_files):
    """Process uploaded files with improved chunking and error handling"""
    documents = []
    for uploaded_file in uploaded_files:
        with st.status(f"Processing {uploaded_file.name}...") as status:
            # Save uploaded file temporarily
            file_path = os.path.join(temp_upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the document
            doc_chunks = process_document(file_path, uploaded_file.name)
            documents.extend(doc_chunks)
            status.update(label=f"Processed {uploaded_file.name}: {len(doc_chunks)} chunks created")
    
    return documents

def initialize_qa_system(_documents):
    """Initialize an enhanced QA system with better retrieval and reasoning"""
    try:
        # Initialize embeddings and vector store
        embeddings = get_embeddings_model()
        vectorstore = FAISS.from_documents(_documents, embeddings)
        
        # Create base retriever
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",  # Using MMR for diversity (Maximal Marginal Relevance (MMR))
            search_kwargs={
                "k": 6,
                "fetch_k": 10
            }
        )
        
        # Initialize Ollama LLM
        llm = OllamaLLM(
            model="llama3.2:1b",
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096  # Increased context window
        )
        
        # Create compressor for better context selection
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )

        # Enhanced prompt template
        template = """You are a helpful AI assistant tasked with answering questions based on the provided documents.
        
        Context information is below:
        ---------------------
        {context}
        ---------------------
        
        Given the context information and no prior knowledge, answer the question step by step:
        Question: {question}
        
        Let's approach this systematically:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Create enhanced QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            },
            return_source_documents=True
        )

        return qa_chain

    except Exception as e:
        st.error(f"Error initializing QA system: {str(e)}")
        return None

# UI Layout
st.title('📚Document Question-Answering System')


# File upload section
with st.container():
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your PDF files for analysis"
    )

    if uploaded_files and not st.session_state.processed_files:
        with st.spinner('Processing documents...'):
            documents = process_uploaded_files(uploaded_files)
            
            if documents:
                st.session_state.document_count = len(documents)
                st.session_state.qa_chain = initialize_qa_system(documents)
                st.session_state.processed_files = True
                st.success(f'Successfully processed {len(uploaded_files)} documents into {len(documents)} chunks!')

# Question and answer section
if st.session_state.processed_files and st.session_state.qa_chain:
    with st.container():
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know about the documents?"
        )
        
        if question:
            with st.spinner('Analyzing documents and generating answer...'):
                try:
                    result = st.session_state.qa_chain({"query": question})
                    
                    st.markdown("### Answer")
                    st.write(result["result"])
                    
                    with st.expander("View source documents"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Source {i+1}** (from {doc.metadata['source']}, page {doc.metadata['page_number']})")
                            st.markdown(f"```\n{doc.page_content}\n```")
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

# Reset button
if st.button('Reset Application'):
    st.session_state.processed_files = False
    st.session_state.qa_chain = None
    st.session_state.document_count = 0
    shutil.rmtree(temp_upload_dir, ignore_errors=True)
    st.experimental_rerun()

# Cleanup function
def cleanup():
    """Clean up temporary directories"""
    shutil.rmtree(temp_upload_dir, ignore_errors=True)

# Register cleanup function
atexit.register(cleanup)