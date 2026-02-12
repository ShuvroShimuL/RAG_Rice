import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle

# PDF processing
import PyPDF2
from pypdf import PdfReader

# LangChain document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handle PDF document processing and text chunking
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 processed_dir: str = "./data/processed"):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            processed_dir: Directory to save processed documents
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_dir = processed_dir
        
        # Create directories
        os.makedirs(processed_dir, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF document using LangChain PyPDFLoader
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        try:
            if isinstance(file_path, str):
                # File path provided
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"PDF file not found: {file_path}")
                
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(file_path)
                    doc.metadata['document_type'] = 'agricultural_guide'
                
                logger.info(f"Loaded {len(documents)} pages from {file_path}")
                return documents
                
            else:
                # Streamlit uploaded file
                return self._load_uploaded_pdf(file_path)
                
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []
    
    def _load_uploaded_pdf(self, uploaded_file) -> List[Document]:
        """
        Load PDF from Streamlit uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects
        """
        try:
            documents = []
            
            # Read PDF using PyPDF2
            pdf_reader = PdfReader(uploaded_file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source_file': uploaded_file.name,
                            'page': page_num + 1,
                            'document_type': 'agricultural_guide'
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} pages from uploaded file: {uploaded_file.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing uploaded PDF: {e}")
            return []
    
    def load_multiple_pdfs(self, pdf_directory: str) -> List[Document]:
        """
        Load multiple PDF files from directory
        
        Args:
            pdf_directory: Directory containing PDF files
            
        Returns:
            List of all documents from all PDFs
        """
        if not os.path.exists(pdf_directory):
            logger.error(f"PDF directory not found: {pdf_directory}")
            return []
        
        documents = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return []
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            try:
                file_docs = self.load_pdf(pdf_path)
                documents.extend(file_docs)
                logger.info(f"Processed: {pdf_file} ({len(file_docs)} pages)")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        if not documents:
            return []
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []
    
    def save_processed_documents(self, documents: List[Document], filename: str = "processed_docs.pkl"):
        """
        Save processed documents to disk
        
        Args:
            documents: List of documents to save
            filename: Name of the file to save
        """
        try:
            filepath = os.path.join(self.processed_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(documents, f)
            
            logger.info(f"Saved {len(documents)} processed documents to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving processed documents: {e}")
    
    def load_processed_documents(self, filename: str = "processed_docs.pkl") -> List[Document]:
        """
        Load processed documents from disk
        
        Args:
            filename: Name of the file to load
            
        Returns:
            List of loaded documents
        """
        try:
            filepath = os.path.join(self.processed_dir, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"Processed documents file not found: {filepath}")
                return []
            
            with open(filepath, 'rb') as f:
                documents = pickle.load(f)
            
            logger.info(f"Loaded {len(documents)} processed documents from {filepath}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading processed documents: {e}")
            return []
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about processed documents
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {'total_docs': 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        # Count unique source files
        source_files = set(doc.metadata.get('source_file', 'unknown') for doc in documents)
        
        return {
            'total_docs': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_chars_per_doc': total_chars // len(documents),
            'average_words_per_doc': total_words // len(documents),
            'unique_source_files': len(source_files),
            'source_files': list(source_files)
        }

# Utility functions
def load_pdf(file_path_or_uploaded_file):
    """
    Utility function to load PDF
    Compatible with your existing polished_chatbot.py
    """
    processor = DocumentProcessor()
    return processor.load_pdf(file_path_or_uploaded_file)

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Utility function to split documents
    Compatible with your existing polished_chatbot.py
    """
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return processor.split_documents(documents)
