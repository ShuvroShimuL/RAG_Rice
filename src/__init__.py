"""
Rice Advisory RAG System
A comprehensive RAG-based chatbot for rice cultivation advisory.
"""

__version__ = "1.0.0"
__author__ = "Shamimul Shimul"

from .rag_system import RiceAdvisoryRAG, create_rice_advisory_rag
from .evaluation_system import RAGEvaluationSystem, create_evaluation_system
from .advanced_features import AdvancedRAGFeatures, create_advanced_features
from .config import Config
__all__ = [
    'RiceAdvisoryRAG',
    'create_rice_advisory_rag',
    'RAGEvaluationSystem', 
    'create_evaluation_system',
    'AdvancedRAGFeatures',
    'create_advanced_features'
]
