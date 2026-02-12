import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """Configuration management for Rice Advisory RAG System"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/settings.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    return yaml.safe_load(file) or {}
            else:
                return self._default_config()
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'models': {
                'llm_model': 'llama-3.1-8b-instant',  # Groq model
                'embedding_model': 'all-MiniLM-L6-v2'
            },
            'vector_db': {
                'collection_name': 'rice_advisory',
                'persist_directory': './data/vector_store'
            },
            'document_processing': {
                'chunk_size': 1000,
                'chunk_overlap': 200
            },
            'data': {
                'pdfs_directory': './data/pdfs',
                'processed_directory': './data/processed',
                'ml_dataset': './data/merged_dataset_final.csv'
            },
            'ml_models': {
                'yield_prediction': './models/rice_yield_model.pkl',
                'region_encoder': './models/region_encoder.pkl',
                'rice_types': './models/rice_types.pkl'
            },
            'groq': {
                'api_key': os.getenv('GROQ_API_KEY'),
                'temperature': 0.3,
                'max_tokens': 1000
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

# Global config instance
config = Config()
