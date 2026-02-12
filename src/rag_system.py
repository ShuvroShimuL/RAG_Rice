import os
import yaml
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables
load_dotenv()

# Groq and LangChain
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Local imports
from .config import config
from .document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiceAdvisoryRAG:
    """
    Complete RAG system for Rice Advisory with Groq integration
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the RAG system"""
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.ml_models = {}
        self.document_processor = DocumentProcessor(
            chunk_size=self.config['document_processing']['chunk_size'],
            chunk_overlap=self.config['document_processing']['chunk_overlap']
        )
        
        # Setup components
        self._setup_embeddings()
        self._load_ml_models()
        self._setup_vector_store()
        
    def _setup_embeddings(self):
        """Initialize embedding model"""
        try:
            embedding_model = self.config['models']['embedding_model']
            self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
            logger.info(f"Embeddings initialized with model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
     # Initialize Groq LLM
        try:
            self.llm = ChatGroq(
                api_key=SecretStr(os.getenv('GROQ_API_KEY') or ""),
                model=self.config['models']['llm_model'],
                temperature=self.config['groq']['temperature'],
                max_tokens=self.config['groq']['max_tokens']
            )
            logger.info(f"Groq LLM initialized with model: {self.config['models']['llm_model']}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            raise
    
    def _setup_lm(self):
        """Initialize the Language Model"""
        # Add any additional setup for the LLM here
        pass

    def _load_ml_models(self):
        """Load existing ML models for yield prediction"""
        try:
            models_config = self.config.get('ml_models', {})
            
            for model_name, model_path in models_config.items():
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.ml_models[model_name] = pickle.load(f)
                    logger.info(f"Loaded ML model: {model_name}")
                else:
                    logger.warning(f"ML model not found: {model_path}")
                    
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
    
    def _setup_vector_store(self):
        """Initialize or load existing vector store"""
        try:
            persist_dir = self.config['vector_db']['persist_directory']
            collection_name = self.config['vector_db']['collection_name']
            
            # Create directory if it doesn't exist
            os.makedirs(persist_dir, exist_ok=True)
            
            # Try to load existing vector store
            if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
                self.vector_store = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                logger.info("Loaded existing vector store")
                self._setup_qa_chain()
            else:
                logger.info("Vector store not found. Call process_documents() to create one.")
                
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
    
    def _setup_qa_chain(self):
        """Setup the QA chain"""
        if self.vector_store and self.llm:
            try:
                # Create custom prompt template
                template = """You are an expert rice cultivation advisor in Bangladesh. Use the following context to provide comprehensive, practical advice for rice farmers.

Context: {context}

Question: {question}

Instructions:
- Provide specific, actionable advice for rice farmers in Bangladesh
- Include practical tips for cultivation, pest management, and yield optimization
- Use simple, clear language that farmers can understand
- Focus on local agricultural practices and conditions
- If the context doesn't contain enough information, mention what additional information might be needed

Answer:"""

                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
                
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
                logger.info("QA chain initialized successfully")
                
            except Exception as e:
                logger.error(f"Error setting up QA chain: {e}")
    
    def process_documents(self, force_rebuild: bool = False) -> bool:
        """
        Process PDF documents and create/update vector store
        
        Args:
            force_rebuild: If True, rebuild the entire vector store
            
        Returns:
            bool: Success status
        """
        try:
            pdf_dir = self.config['data']['pdfs_directory']
            
            if not os.path.exists(pdf_dir):
                logger.error(f"PDF directory not found: {pdf_dir}")
                return False
            
            # Load documents
            documents = self.document_processor.load_multiple_pdfs(pdf_dir)
            
            if not documents:
                logger.error("No documents were successfully processed")
                return False
            
            # Split documents into chunks
            chunks = self.document_processor.split_documents(documents)
            logger.info(f"Created {len(chunks)} document chunks")
            
            # Save processed documents
            self.document_processor.save_processed_documents(chunks)
            
            # Create or update vector store
            persist_dir = self.config['vector_db']['persist_directory']
            collection_name = self.config['vector_db']['collection_name']
            
            if force_rebuild or not self.vector_store:
                # Create new vector store
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=persist_dir,
                    collection_name=collection_name
                )
                logger.info("Created new vector store")
            else:
                # Add to existing vector store
                self.vector_store.add_documents(chunks)
                logger.info("Updated existing vector store")
            
            # Setup QA chain
            self._setup_qa_chain()
            
            logger.info("Document processing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False
    
    def predict_yield(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict rice yield using ML models
        
        Args:
            features: Dictionary with rice variety, region, weather data, etc.
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if 'yield_prediction' not in self.ml_models:
                return {"error": "Yield prediction model not loaded"}
            
            # Prepare features for prediction
            model = self.ml_models['yield_prediction']
            region_encoder = self.ml_models.get('region_encoder')
            
            # Create feature vector (adjust based on your model's requirements)
            feature_vector = []
            
            # Add numerical features
            for feature in ['temperature', 'rainfall', 'humidity', 'cultivation_area']:
                feature_vector.append(features.get(feature, 0))
            
            # Encode categorical features
            if region_encoder and 'region' in features:
                try:
                    encoded_region = region_encoder.transform([features['region']])[0]
                    feature_vector.append(encoded_region)
                except:
                    feature_vector.append(0)  # Default encoding
            
            # Make prediction
            prediction = model.predict([feature_vector])[0]
            
            # Generate recommendations based on prediction
            recommendations = self._generate_recommendations(features, prediction)
            
            return {
                'predicted_yield': round(prediction, 2),
                'unit': 'tons/acre',
                'confidence': 0.85,
                'recommendations': recommendations,
                'input_features': features
            }
            
        except Exception as e:
            logger.error(f"Error in yield prediction: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, features: Dict[str, Any], predicted_yield: float) -> List[str]:
        """Generate agricultural recommendations based on prediction"""
        recommendations = []
        
        # Yield-based recommendations
        if predicted_yield < 3.0:
            recommendations.extend([
                "Consider using high-yielding variety (HYV) seeds",
                "Improve soil fertility with organic fertilizers",
                "Ensure proper water management and irrigation",
                "Apply recommended doses of NPK fertilizers"
            ])
        elif predicted_yield > 5.0:
            recommendations.extend([
                "Maintain current agricultural practices",
                "Monitor for pest and disease management",
                "Ensure timely harvesting for optimal quality"
            ])
        
        # Weather-based recommendations
        rainfall = features.get('rainfall', 0)
        humidity = features.get('humidity', 0)
        temperature = features.get('temperature', 25)
        
        if rainfall < 1000:
            recommendations.append("Implement water-saving irrigation techniques")
        elif rainfall > 2000:
            recommendations.append("Ensure proper drainage to prevent waterlogging")
        
        if humidity > 80:
            recommendations.append("Monitor for fungal diseases due to high humidity")
        
        if temperature > 35:
            recommendations.append("Provide shade or cooling measures during extreme heat")
        elif temperature < 20:
            recommendations.append("Consider cold-tolerant rice varieties")
        
        return recommendations
    
    def query(self, question: str, include_ml_prediction: bool = False, 
              ml_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with optional ML prediction
        
        Args:
            question: User question
            include_ml_prediction: Whether to include yield prediction
            ml_features: Features for ML prediction
            
        Returns:
            Dictionary with response and optional prediction
        """
        try:
            if not self.qa_chain:
                return {
                    "error": "QA chain not initialized. Please process documents first.",
                    "response": "I need to process agricultural documents before I can answer questions. Please run process_documents() first."
                }
            
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Prepare response
            response_data = {
                "response": result['result'],
                "sources": []
            }
            
            # Add source information
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    response_data["sources"].append({
                        'source_file': doc.metadata.get('source_file', 'Unknown'),
                        'page': doc.metadata.get('page', 'Unknown'),
                        'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
            
            # Add ML prediction if requested
            if include_ml_prediction and ml_features:
                prediction_result = self.predict_yield(ml_features)
                response_data["yield_prediction"] = prediction_result
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                "error": str(e),
                "response": "An error occurred while processing your question."
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "vector_store_initialized": self.vector_store is not None,
            "qa_chain_initialized": self.qa_chain is not None,
            "embeddings_loaded": self.embeddings is not None,
            "llm_loaded": self.llm is not None,
            "ml_models_loaded": list(self.ml_models.keys()),
            "groq_model": self.config['models']['llm_model'],
            "embedding_model": self.config['models']['embedding_model']
        }

# Utility function for compatibility with existing code
def create_vector_store(chunks):
    """
    Create vector store from chunks
    Compatible with your existing polished_chatbot.py
    """
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./data/vector_store",
            collection_name="documents"
        )
        
        return vector_store, embeddings
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

# Utility function for easy initialization
def create_rice_advisory_rag(config_path: str = "config/settings.yaml") -> RiceAdvisoryRAG:
    """Create and return a RiceAdvisoryRAG instance"""
    return RiceAdvisoryRAG(config_path)
