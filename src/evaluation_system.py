import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
import asyncio

# Evaluation metrics
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ollama

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationSample:
    """Single evaluation sample"""
    question: str
    contexts: List[str]
    answer: str
    ground_truth: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EvaluationResults:
    """Results from RAG evaluation"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    detailed_scores: Dict[str, Any]
    timestamp: str

class RAGEvaluationSystem:
    """
    Comprehensive RAG evaluation system using multiple metrics
    Implements RAGAS-style evaluation without requiring external dependencies
    """
    
    def __init__(self, 
                 llm_model: str = "llama3.2:3b",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize evaluation system
        
        Args:
            llm_model: Ollama model for LLM-based evaluations
            embedding_model: SentenceTransformer model for embeddings
        """
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts"""
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded")
        
        return self.embedding_model.encode(texts)
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        try:
            embeddings = self._get_embeddings([text1, text2])
            similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _ollama_query(self, prompt: str, max_retries: int = 3) -> str:
        """Query Ollama with retry mechanism"""
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response['message']['content'].strip()
            
            except Exception as e:
                logger.warning(f"Ollama query attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All Ollama query attempts failed")
                    return ""
        
        return ""
    
    def evaluate_faithfulness(self, sample: EvaluationSample) -> float:
        """
        Evaluate faithfulness: how factually consistent is the answer with the context
        
        Args:
            sample: Evaluation sample containing question, contexts, and answer
            
        Returns:
            Faithfulness score between 0 and 1
        """
        try:
            context = "\n".join(sample.contexts)
            
            # Extract claims from the answer
            claims_prompt = f"""
Extract all factual claims from the following answer. List each claim on a separate line.

Answer: {sample.answer}

Claims:
"""
            claims_text = self._ollama_query(claims_prompt)
            
            if not claims_text:
                return 0.0
            
            claims = [claim.strip() for claim in claims_text.split('\n') if claim.strip()]
            
            if not claims:
                return 0.0
            
            # Check each claim against the context
            supported_claims = 0
            
            for claim in claims:
                verification_prompt = f"""
Context: {context}

Claim: {claim}

Question: Can this claim be inferred from or supported by the given context?
Answer only 'YES' or 'NO'.

Answer:
"""
                
                verification = self._ollama_query(verification_prompt)
                
                if verification.upper().startswith('YES'):
                    supported_claims += 1
            
            faithfulness_score = supported_claims / len(claims) if claims else 0.0
            
            logger.info(f"Faithfulness: {supported_claims}/{len(claims)} claims supported = {faithfulness_score:.3f}")
            
            return faithfulness_score
            
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {e}")
            return 0.0
    
    def evaluate_answer_relevancy(self, sample: EvaluationSample) -> float:
        """
        Evaluate answer relevancy: how relevant is the answer to the question
        
        Args:
            sample: Evaluation sample
            
        Returns:
            Answer relevancy score between 0 and 1
        """
        try:
            # Generate questions from the answer (reverse engineering)
            generation_prompt = f"""
Given the following answer, generate 3 different questions that this answer would appropriately address.
Make the questions diverse but all should be answerable by the given answer.

Answer: {sample.answer}

Generate exactly 3 questions, one per line:
"""
            
            generated_questions_text = self._ollama_query(generation_prompt)
            
            if not generated_questions_text:
                return 0.0
            
            generated_questions = [
                q.strip().lstrip('123456789.-) ') 
                for q in generated_questions_text.split('\n') 
                if q.strip()
            ][:3]  # Take only first 3
            
            if not generated_questions:
                return 0.0
            
            # Calculate cosine similarity between original question and generated questions
            similarities = []
            for gen_question in generated_questions:
                if gen_question:
                    similarity = self._cosine_similarity(sample.question, gen_question)
                    similarities.append(similarity)
            
            if not similarities:
                return 0.0
            
            answer_relevancy = np.mean(similarities)
            
            logger.info(f"Answer Relevancy: {len(similarities)} questions, mean similarity = {answer_relevancy:.3f}")
            
            return float(answer_relevancy)
            
        except Exception as e:
            logger.error(f"Error evaluating answer relevancy: {e}")
            return 0.0
    
    def evaluate_context_precision(self, sample: EvaluationSample) -> float:
        """
        Evaluate context precision: how relevant are the retrieved contexts
        
        Args:
            sample: Evaluation sample
            
        Returns:
            Context precision score between 0 and 1
        """
        try:
            if not sample.contexts:
                return 0.0
            
            relevant_contexts = 0
            
            for i, context in enumerate(sample.contexts):
                relevance_prompt = f"""
Question: {sample.question}
Context: {context}

Question: Is this context relevant for answering the given question?
Answer only 'YES' or 'NO'.

Answer:
"""
                
                relevance = self._ollama_query(relevance_prompt)
                
                if relevance.upper().startswith('YES'):
                    relevant_contexts += 1
            
            precision = relevant_contexts / len(sample.contexts)
            
            logger.info(f"Context Precision: {relevant_contexts}/{len(sample.contexts)} contexts relevant = {precision:.3f}")
            
            return precision
            
        except Exception as e:
            logger.error(f"Error evaluating context precision: {e}")
            return 0.0
    
    def evaluate_context_recall(self, sample: EvaluationSample) -> float:
        """
        Evaluate context recall: how much of the ground truth is covered by contexts
        
        Args:
            sample: Evaluation sample (requires ground_truth)
            
        Returns:
            Context recall score between 0 and 1
        """
        try:
            if not sample.ground_truth or not sample.contexts:
                return 0.0
            
            context = "\n".join(sample.contexts)
            
            # Extract key information from ground truth
            info_extraction_prompt = f"""
Extract the key factual information from the following ground truth answer.
List each piece of information on a separate line.

Ground Truth: {sample.ground_truth}

Key Information:
"""
            
            key_info_text = self._ollama_query(info_extraction_prompt)
            
            if not key_info_text:
                return 0.0
            
            key_info_points = [info.strip() for info in key_info_text.split('\n') if info.strip()]
            
            if not key_info_points:
                return 0.0
            
            # Check how many key information points are covered by the contexts
            covered_points = 0
            
            for info_point in key_info_points:
                coverage_prompt = f"""
Context: {context}
Information: {info_point}

Question: Is this information covered or supported by the given context?
Answer only 'YES' or 'NO'.

Answer:
"""
                
                coverage = self._ollama_query(coverage_prompt)
                
                if coverage.upper().startswith('YES'):
                    covered_points += 1
            
            recall = covered_points / len(key_info_points)
            
            logger.info(f"Context Recall: {covered_points}/{len(key_info_points)} info points covered = {recall:.3f}")
            
            return recall
            
        except Exception as e:
            logger.error(f"Error evaluating context recall: {e}")
            return 0.0
    
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResults:
        """
        Evaluate a single sample across all metrics
        
        Args:
            sample: Evaluation sample
            
        Returns:
            EvaluationResults with all metric scores
        """
        logger.info(f"Evaluating sample: {sample.question[:50]}...")
        
        # Calculate individual metrics
        faithfulness = self.evaluate_faithfulness(sample)
        answer_relevancy = self.evaluate_answer_relevancy(sample)
        context_precision = self.evaluate_context_precision(sample)
        
        # Context recall only if ground truth is available
        context_recall = 0.0
        if sample.ground_truth:
            context_recall = self.evaluate_context_recall(sample)
        
        # Calculate overall score (RAGAS score)
        scores = [faithfulness, answer_relevancy, context_precision]
        if context_recall > 0:
            scores.append(context_recall)
        
        overall_score = np.mean(scores) if scores else 0.0
        
        # Prepare detailed scores
        detailed_scores = {
            'individual_metrics': {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_precision': context_precision,
                'context_recall': context_recall
            },
            'sample_metadata': sample.metadata or {},
            'evaluation_settings': {
                'llm_model': self.llm_model,
                'embedding_model': self.embedding_model_name
            }
        }
        
        return EvaluationResults(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            overall_score=float(overall_score),
            detailed_scores=detailed_scores,
            timestamp=datetime.now().isoformat()
        )
    
    def evaluate_dataset(self, samples: List[EvaluationSample]) -> Dict[str, Any]:
        """
        Evaluate multiple samples and return aggregated results
        
        Args:
            samples: List of evaluation samples
            
        Returns:
            Dictionary with aggregated evaluation results
        """
        logger.info(f"Evaluating dataset with {len(samples)} samples...")
        
        if not samples:
            return {"error": "No samples provided for evaluation"}
        
        all_results = []
        
        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i+1}/{len(samples)}")
            result = self.evaluate_sample(sample)
            all_results.append(result)
        
        # Aggregate results
        faithfulness_scores = [r.faithfulness for r in all_results]
        answer_relevancy_scores = [r.answer_relevancy for r in all_results]
        context_precision_scores = [r.context_precision for r in all_results]
        context_recall_scores = [r.context_recall for r in all_results if r.context_recall > 0]
        overall_scores = [r.overall_score for r in all_results]
        
        aggregated_results = {
            'dataset_summary': {
                'total_samples': len(samples),
                'samples_with_ground_truth': len(context_recall_scores),
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'mean_scores': {
                'faithfulness': np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
                'answer_relevancy': np.mean(answer_relevancy_scores) if answer_relevancy_scores else 0.0,
                'context_precision': np.mean(context_precision_scores) if context_precision_scores else 0.0,
                'context_recall': np.mean(context_recall_scores) if context_recall_scores else 0.0,
                'overall_score': np.mean(overall_scores) if overall_scores else 0.0
            },
            'std_scores': {
                'faithfulness': np.std(faithfulness_scores) if faithfulness_scores else 0.0,
                'answer_relevancy': np.std(answer_relevancy_scores) if answer_relevancy_scores else 0.0,
                'context_precision': np.std(context_precision_scores) if context_precision_scores else 0.0,
                'context_recall': np.std(context_recall_scores) if context_recall_scores else 0.0,
                'overall_score': np.std(overall_scores) if overall_scores else 0.0
            },
            'individual_results': [
                {
                    'sample_index': i,
                    'question': samples[i].question,
                    'faithfulness': result.faithfulness,
                    'answer_relevancy': result.answer_relevancy,
                    'context_precision': result.context_precision,
                    'context_recall': result.context_recall,
                    'overall_score': result.overall_score
                }
                for i, result in enumerate(all_results)
            ]
        }
        
        return aggregated_results
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to JSON file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def create_test_samples(self) -> List[EvaluationSample]:
        """Create sample test data for rice advisory evaluation"""
        return [
            EvaluationSample(
                question="What are the best practices for Boro rice cultivation in Bangladesh?",
                contexts=[
                    "Boro rice is cultivated during the dry season (December to May) in Bangladesh. It requires proper water management and irrigation.",
                    "High-yielding varieties (HYV) of Boro rice should be planted with recommended spacing of 20cm x 15cm.",
                    "Fertilizer application should include urea, TSP, and MOP in proper ratios for optimal yield."
                ],
                answer="For Boro rice cultivation in Bangladesh, follow these best practices: 1) Plant during December-January using HYV seeds, 2) Maintain proper irrigation throughout the dry season, 3) Apply balanced fertilizers including urea, TSP, and MOP, 4) Use recommended plant spacing of 20cm x 15cm.",
                ground_truth="Boro rice cultivation requires dry season planting, proper irrigation, balanced fertilization, and appropriate spacing for optimal results.",
                metadata={"crop_type": "rice", "season": "boro", "region": "bangladesh"}
            ),
            EvaluationSample(
                question="How can I control pests in rice fields?",
                contexts=[
                    "Integrated Pest Management (IPM) is the most effective approach for rice pest control.",
                    "Common rice pests include stem borer, brown planthopper, and rice hispa.",
                    "Biological control using natural predators and parasites is environmentally friendly."
                ],
                answer="To control pests in rice fields, implement Integrated Pest Management (IPM) strategies including biological control with natural predators, targeted pesticide application when necessary, and regular field monitoring to detect pest infestations early.",
                ground_truth="Effective rice pest control involves IPM strategies, biological control methods, and timely interventions based on pest monitoring.",
                metadata={"topic": "pest_control", "crop": "rice"}
            )
        ]

# Utility functions
def create_evaluation_system(llm_model: str = "llama3.2:3b", 
                           embedding_model: str = "all-MiniLM-L6-v2") -> RAGEvaluationSystem:
    """Create and return a RAGEvaluationSystem instance"""
    return RAGEvaluationSystem(llm_model, embedding_model)

if __name__ == "__main__":
    # Example usage
    evaluator = create_evaluation_system()
    
    # Create test samples
    test_samples = evaluator.create_test_samples()
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(test_samples)
    
    # Print results
    print("\n=== RAG Evaluation Results ===")
    print(f"Overall Score: {results['mean_scores']['overall_score']:.3f}")
    print(f"Faithfulness: {results['mean_scores']['faithfulness']:.3f}")
    print(f"Answer Relevancy: {results['mean_scores']['answer_relevancy']:.3f}")
    print(f"Context Precision: {results['mean_scores']['context_precision']:.3f}")
    print(f"Context Recall: {results['mean_scores']['context_recall']:.3f}")
    
    # Save results
    evaluator.save_evaluation_results(results, "evaluation_results/rag_evaluation.json")
