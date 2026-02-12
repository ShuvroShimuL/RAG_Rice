import pickle
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Any, Optional

class YieldPredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.region_encoder = None
        self.rice_types = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        with open(self.config['ml_models']['yield_prediction'], 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.config['ml_models']['region_encoder'], 'rb') as f:
            self.region_encoder = pickle.load(f)
        
        with open(self.config['ml_models']['rice_types'], 'rb') as f:
            self.rice_types = pickle.load(f)
    
    def predict_yield(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict rice yield based on input features
        
        Args:
            features: Dictionary containing:
                - region: Region name
                - rice_type: Type of rice
                - temperature: Temperature
                - rainfall: Rainfall
                - humidity: Humidity
                - cultivation_area: Cultivation area
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Prepare input data
            input_data = self.prepare_input(features)
            
            # Ensure model is loaded
            if self.model is None:
                raise ValueError("Yield prediction model is not loaded.")
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            # Calculate confidence intervals (if available)
            confidence = self.calculate_confidence(input_data)
            
            return {
                'predicted_yield': round(prediction, 2),
                'confidence': confidence,
                'recommendations': self.generate_recommendations(features, prediction)
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'predicted_yield': None
            }
    
    def prepare_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare input data for model prediction"""
        # This should match your model's expected input format
        # Based on your merged_dataset_final.csv structure
        
        input_df = pd.DataFrame([features])
        
        # Handle categorical encoding
        if 'region' in features:
            if self.region_encoder is None:
                raise ValueError("Region encoder is not loaded.")
            input_df['region_encoded'] = self.region_encoder.transform([features['region']])[0]
        
        # Select features in the same order as training
        feature_columns = ['temperature', 'rainfall', 'humidity', 'cultivation_area', 'region_encoded']
        
        return input_df[feature_columns].values
    
    def calculate_confidence(self, input_data: np.ndarray) -> float:
        """Calculate prediction confidence"""
        # Implement confidence calculation based on your model
        # This is a placeholder - adjust based on your model type
        return 0.85  # Example confidence score
    
    def generate_recommendations(self, features: Dict[str, Any], predicted_yield: float) -> List[str]:
        """Generate recommendations based on prediction"""
        recommendations = []
        
        # Add recommendations based on predicted yield and input features
        if predicted_yield < 3.0:  # Low yield threshold
            recommendations.append("Consider improving soil fertility with organic fertilizers")
            recommendations.append("Optimize irrigation schedule based on weather conditions")
        
        if features.get('rainfall', 0) < 1000:  # Low rainfall
            recommendations.append("Implement drip irrigation to conserve water")
        
        if features.get('humidity', 0) > 80:  # High humidity
            recommendations.append("Monitor for fungal diseases and apply preventive measures")
        
        return recommendations

# Make sure to import RiceAdvisoryRAG from its module
from rag_system import RiceAdvisoryRAG

class HybridRiceAdvisorySystem:
    """Combines RAG and ML for comprehensive rice advisory"""
    
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.rag_system = RiceAdvisoryRAG(config_path)
        self.yield_predictor = YieldPredictor(self.config)
    
    def get_comprehensive_advice(self, query: str, ml_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Provide comprehensive advice combining RAG and ML predictions
        """
        result = {
            'rag_response': self.rag_system.query(query),
            'yield_prediction': None,
            'combined_recommendations': []
        }
        
        if ml_features:
            yield_result = self.yield_predictor.predict_yield(ml_features)
            result['yield_prediction'] = yield_result
            
            # Combine recommendations
            if 'recommendations' in yield_result:
                result['combined_recommendations'] = yield_result['recommendations']
        
        return result
