import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict
import asyncio
from dataclasses import dataclass, asdict

# Advanced RAG features
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryAnalysis:
    """Analysis results for a user query"""
    original_query: str
    query_type: str
    expanded_queries: List[str]
    key_entities: List[str]
    intent: str
    complexity_score: float
    suggested_followups: List[str]

@dataclass
class ResponseEnhancement:
    """Enhanced response with additional features"""
    original_response: str
    enhanced_response: str
    confidence_score: float
    reasoning_chain: List[str]
    evidence_quality: float
    alternative_perspectives: List[str]

class AdvancedRAGFeatures:
    """
    Advanced features for RAG system including:
    - Query expansion and reformulation
    - Multi-step reasoning
    - Response enhancement
    - Confidence scoring
    - Context reranking
    - Query intent analysis
    """
    
    def __init__(self,
                 llm_model: str = "llama3.2:3b",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize advanced RAG features
        
        Args:
            llm_model: Ollama model for LLM operations
            embedding_model: SentenceTransformer model for embeddings
        """
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model
        
        # Initialize models
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # Initialize query cache and analytics
        self.query_cache = {}

        # Use a regular dict for query_analytics to allow mixed value types
        self.query_analytics = {
            'total_queries': 0,
            'rice_queries': 0,
            'pest_queries': 0,
            'fertilizer_queries': 0,
            'other_queries': 0,
            'quality_scores': []
        }
        self.feedback_data = []
    
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
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze user query to understand intent and complexity
        
        Args:
            query: User input query
            
        Returns:
            QueryAnalysis with detailed analysis results
        """
        try:
            # Analyze query type and intent
            analysis_prompt = f"""
Analyze the following agricultural query and provide structured information:

Query: "{query}"

Please analyze and respond in the following format:
QUERY_TYPE: [factual/procedural/diagnostic/comparative/predictive]
INTENT: [brief description of what user wants to know]
KEY_ENTITIES: [comma-separated list of key agricultural terms, crops, practices]
COMPLEXITY: [simple/medium/complex]

Analysis:
"""
            
            analysis_text = self._ollama_query(analysis_prompt)
            
            # Parse analysis results
            query_type = "factual"
            intent = "General agricultural information"
            key_entities = []
            complexity = "medium"
            
            if analysis_text:
                lines = analysis_text.split('\n')
                for line in lines:
                    if line.startswith('QUERY_TYPE:'):
                        query_type = line.split(':', 1)[1].strip()
                    elif line.startswith('INTENT:'):
                        intent = line.split(':', 1)[1].strip()
                    elif line.startswith('KEY_ENTITIES:'):
                        entities_text = line.split(':', 1)[1].strip()
                        key_entities = [e.strip() for e in entities_text.split(',') if e.strip()]
                    elif line.startswith('COMPLEXITY:'):
                        complexity = line.split(':', 1)[1].strip()
            
            # Generate expanded queries
            expanded_queries = self.expand_query(query, key_entities)
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(query, key_entities)
            
            # Generate suggested follow-ups
            suggested_followups = self.generate_followup_questions(query, intent)
            
            return QueryAnalysis(
                original_query=query,
                query_type=query_type,
                expanded_queries=expanded_queries,
                key_entities=key_entities,
                intent=intent,
                complexity_score=complexity_score,
                suggested_followups=suggested_followups
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return QueryAnalysis(
                original_query=query,
                query_type="unknown",
                expanded_queries=[query],
                key_entities=[],
                intent="Unknown intent",
                complexity_score=0.5,
                suggested_followups=[]
            )
    
    def expand_query(self, query: str, entities: Optional[List[str]] = None) -> List[str]:
        """
        Generate multiple variations of the query for better retrieval
        
        Args:
            query: Original query
            entities: Key entities extracted from query
            
        Returns:
            List of expanded query variations
        """
        try:
            expansion_prompt = f"""
Generate 3-4 alternative ways to ask the following agricultural question. 
Make them more specific and include relevant technical terms.

Original Question: {query}

Key Entities: {', '.join(entities) if entities else 'Not specified'}

Generate alternative questions (one per line):
"""
            
            expansion_text = self._ollama_query(expansion_prompt)
            
            if not expansion_text:
                return [query]
            
            # Parse expanded queries
            expanded = [query]  # Include original
            lines = expansion_text.split('\n')
            
            for line in lines:
                cleaned_line = line.strip().lstrip('1234567890.-) ')
                if cleaned_line and cleaned_line not in expanded and len(cleaned_line) > 10:
                    expanded.append(cleaned_line)
            
            return expanded[:5]  # Limit to 5 total queries
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]
    
    def _calculate_complexity_score(self, query: str, entities: List[str]) -> float:
        """Calculate query complexity score based on various factors"""
        try:
            score = 0.0
            
            # Length factor
            word_count = len(query.split())
            if word_count > 15:
                score += 0.3
            elif word_count > 8:
                score += 0.2
            else:
                score += 0.1
            
            # Entity count factor
            entity_count = len(entities)
            if entity_count > 5:
                score += 0.3
            elif entity_count > 2:
                score += 0.2
            else:
                score += 0.1
            
            # Question type factor
            if any(word in query.lower() for word in ['how', 'why', 'compare', 'analyze']):
                score += 0.2
            
            # Technical terms factor
            technical_terms = ['cultivation', 'fertilizer', 'pesticide', 'irrigation', 'yield', 'variety']
            if any(term in query.lower() for term in technical_terms):
                score += 0.2
            
            return min(score, 1.0)
            
        except:
            return 0.5
    
    def generate_followup_questions(self, query: str, intent: str) -> List[str]:
        """Generate relevant follow-up questions"""
        try:
            followup_prompt = f"""
Based on this agricultural query and intent, suggest 3 relevant follow-up questions that a farmer might ask next.

Original Query: {query}
Intent: {intent}

Generate follow-up questions (one per line):
"""
            
            followup_text = self._ollama_query(followup_prompt)
            
            if not followup_text:
                return []
            
            followups = []
            lines = followup_text.split('\n')
            
            for line in lines:
                cleaned_line = line.strip().lstrip('1234567890.-) ')
                if cleaned_line and len(cleaned_line) > 10:
                    followups.append(cleaned_line)
            
            return followups[:3]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []
    
    def rerank_contexts(self, query: str, contexts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Rerank retrieved contexts based on relevance and quality
        
        Args:
            query: User query
            contexts: List of retrieved contexts
            top_k: Number of top contexts to return
            
        Returns:
            List of (context, relevance_score) tuples, sorted by relevance
        """
        try:
            if not contexts or not self.embedding_model:
                return [(ctx, 0.0) for ctx in contexts[:top_k]]
            
            # Get embeddings
            query_embedding = self.embedding_model.encode([query])
            context_embeddings = self.embedding_model.encode(contexts)
            
            # Calculate semantic similarity
            similarities = cosine_similarity(query_embedding, context_embeddings)[0]
            
            # Calculate quality scores
            quality_scores = []
            for context in contexts:
                quality_score = self._calculate_context_quality(context)
                quality_scores.append(quality_score)
            
            # Combine similarity and quality (weighted)
            combined_scores = []
            for i, (sim_score, qual_score) in enumerate(zip(similarities, quality_scores)):
                combined_score = 0.7 * sim_score + 0.3 * qual_score
                combined_scores.append((contexts[i], combined_score))
            
            # Sort by combined score
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            
            return combined_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error reranking contexts: {e}")
            return [(ctx, 0.0) for ctx in contexts[:top_k]]
    
    def _calculate_context_quality(self, context: str) -> float:
        """Calculate quality score for a context"""
        try:
            score = 0.0
            
            # Length factor (not too short, not too long)
            length = len(context.split())
            if 50 <= length <= 200:
                score += 0.3
            elif 20 <= length <= 300:
                score += 0.2
            else:
                score += 0.1
            
            # Information density (keywords)
            agricultural_keywords = [
                'rice', 'cultivation', 'farming', 'crop', 'yield', 'fertilizer',
                'irrigation', 'pest', 'disease', 'variety', 'harvest', 'planting'
            ]
            keyword_count = sum(1 for keyword in agricultural_keywords if keyword.lower() in context.lower())
            score += min(keyword_count * 0.1, 0.4)
            
            # Structure quality (has punctuation, proper sentences)
            if '.' in context and context[0].isupper():
                score += 0.2
            
            # Numbers and specific data
            if any(char.isdigit() for char in context):
                score += 0.1
            
            return min(score, 1.0)
            
        except:
            return 0.5
    
    def enhance_response(self, original_response: str, context: str, query: str) -> ResponseEnhancement:
        """
        Enhance the original response with additional features
        
        Args:
            original_response: Original RAG response
            context: Retrieved context
            query: User query
            
        Returns:
            ResponseEnhancement with improved response and metadata
        """
        try:
            # Generate reasoning chain
            reasoning_chain = self._generate_reasoning_chain(query, context, original_response)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(original_response, context)
            
            # Assess evidence quality
            evidence_quality = self._assess_evidence_quality(context)
            
            # Generate alternative perspectives
            alternative_perspectives = self._generate_alternatives(query, original_response)
            
            # Create enhanced response
            enhancement_prompt = f"""
Improve the following agricultural advice response by making it more comprehensive, practical, and farmer-friendly.

Original Query: {query}
Original Response: {original_response}
Context: {context}

Enhanced Response Guidelines:
1. Make it more actionable and specific
2. Add practical tips where relevant
3. Include potential challenges and solutions
4. Maintain accuracy and clarity
5. Use farmer-friendly language

Enhanced Response:
"""
            
            enhanced_text = self._ollama_query(enhancement_prompt)
            
            if not enhanced_text:
                enhanced_text = original_response
            
            return ResponseEnhancement(
                original_response=original_response,
                enhanced_response=enhanced_text,
                confidence_score=confidence_score,
                reasoning_chain=reasoning_chain,
                evidence_quality=evidence_quality,
                alternative_perspectives=alternative_perspectives
            )
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return ResponseEnhancement(
                original_response=original_response,
                enhanced_response=original_response,
                confidence_score=0.5,
                reasoning_chain=[],
                evidence_quality=0.5,
                alternative_perspectives=[]
            )
    
    def _generate_reasoning_chain(self, query: str, context: str, response: str) -> List[str]:
        """Generate step-by-step reasoning chain"""
        try:
            reasoning_prompt = f"""
Explain the step-by-step reasoning process for how the following response was derived from the context.

Query: {query}
Context: {context}
Response: {response}

Break down the reasoning into 3-4 clear steps:
"""
            
            reasoning_text = self._ollama_query(reasoning_prompt)
            
            if not reasoning_text:
                return ["Analysis based on provided agricultural context"]
            
            steps = []
            lines = reasoning_text.split('\n')
            
            for line in lines:
                cleaned_line = line.strip().lstrip('1234567890.-) ')
                if cleaned_line and len(cleaned_line) > 10:
                    steps.append(cleaned_line)
            
            return steps[:4]
            
        except Exception as e:
            logger.error(f"Error generating reasoning chain: {e}")
            return ["Reasoning based on agricultural knowledge"]
    
    def _calculate_confidence_score(self, response: str, context: str) -> float:
        """Calculate confidence score for the response"""
        try:
            if not self.embedding_model:
                return 0.5
            
            # Semantic alignment between response and context
            response_embedding = self.embedding_model.encode([response])
            context_embedding = self.embedding_model.encode([context])
            
            alignment_score = cosine_similarity(response_embedding, context_embedding)[0][0]
            
            # Response completeness (length and structure)
            completeness_score = 0.0
            word_count = len(response.split())
            
            if word_count >= 30:
                completeness_score = 0.3
            elif word_count >= 15:
                completeness_score = 0.2
            else:
                completeness_score = 0.1
            
            # Specificity (numbers, specific terms)
            specificity_score = 0.0
            if any(char.isdigit() for char in response):
                specificity_score += 0.1
            
            agricultural_terms = ['variety', 'fertilizer', 'irrigation', 'harvest', 'planting', 'cultivation']
            term_count = sum(1 for term in agricultural_terms if term.lower() in response.lower())
            specificity_score += min(term_count * 0.05, 0.2)
            
            # Combine scores
            confidence = 0.5 * alignment_score + 0.3 * completeness_score + 0.2 * specificity_score
            
            return float(min(confidence, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _assess_evidence_quality(self, context: str) -> float:
        """Assess the quality of evidence in the context"""
        try:
            quality_score = 0.0
            
            # Source indicators
            source_indicators = ['research', 'study', 'experiment', 'data', 'analysis', 'findings']
            if any(indicator in context.lower() for indicator in source_indicators):
                quality_score += 0.3
            
            # Specific data
            if any(char.isdigit() for char in context):
                quality_score += 0.2
            
            # Technical accuracy indicators
            technical_terms = ['scientific', 'method', 'procedure', 'technique', 'protocol']
            if any(term in context.lower() for term in technical_terms):
                quality_score += 0.2
            
            # Completeness
            if len(context.split()) > 50:
                quality_score += 0.2
            
            # Authority indicators
            authority_terms = ['expert', 'specialist', 'institute', 'university', 'department']
            if any(term in context.lower() for term in authority_terms):
                quality_score += 0.1
            
            return min(quality_score, 1.0)
            
        except:
            return 0.5
    
    def _generate_alternatives(self, query: str, response: str) -> List[str]:
        """Generate alternative perspectives or approaches"""
        try:
            alternatives_prompt = f"""
For the following agricultural query and response, suggest 2-3 alternative approaches or perspectives that a farmer might consider.

Query: {query}
Response: {response}

Alternative approaches (one per line):
"""
            
            alternatives_text = self._ollama_query(alternatives_prompt)
            
            if not alternatives_text:
                return []
            
            alternatives = []
            lines = alternatives_text.split('\n')
            
            for line in lines:
                cleaned_line = line.strip().lstrip('1234567890.-) ')
                if cleaned_line and len(cleaned_line) > 10:
                    alternatives.append(cleaned_line)
            
            return alternatives[:3]
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            return []
    
    def multi_step_reasoning(self, query: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Perform multi-step reasoning for complex queries
        
        Args:
            query: Complex user query
            contexts: Retrieved contexts
            
        Returns:
            Dictionary with reasoning steps and final answer
        """
        try:
            reasoning_prompt = f"""
Break down this complex agricultural question into steps and provide a comprehensive answer.

Question: {query}

Available Information:
{chr(10).join([f"- {ctx}" for ctx in contexts])}

Please structure your response as follows:
STEP 1: [First step of analysis]
STEP 2: [Second step of analysis]
STEP 3: [Third step if needed]
CONCLUSION: [Final comprehensive answer]

Analysis:
"""
            
            reasoning_text = self._ollama_query(reasoning_prompt)
            
            if not reasoning_text:
                return {
                    "steps": ["Unable to perform multi-step analysis"],
                    "conclusion": "Please try rephrasing your question",
                    "confidence": 0.1
                }
            
            # Parse reasoning steps
            steps = []
            conclusion = ""
            
            lines = reasoning_text.split('\n')
            current_step = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('STEP'):
                    if current_step:
                        steps.append(current_step.strip())
                    current_step = line
                elif line.startswith('CONCLUSION:'):
                    if current_step:
                        steps.append(current_step.strip())
                    conclusion = line.replace('CONCLUSION:', '').strip()
                    break
                elif current_step:
                    current_step += " " + line
            
            if current_step and not conclusion:
                steps.append(current_step.strip())
            
            return {
                "steps": steps,
                "conclusion": conclusion or "Analysis completed based on available information",
                "confidence": 0.8,
                "contexts_used": len(contexts)
            }
            
        except Exception as e:
            logger.error(f"Error in multi-step reasoning: {e}")
            return {
                "steps": ["Error in analysis"],
                "conclusion": "Unable to complete multi-step analysis",
                "confidence": 0.1
            }
    
    def track_query_analytics(self, query: str, response_quality: Optional[float] = None):
        """Track query analytics for system improvement"""
        try:
            # Update query analytics
            self.query_analytics['total_queries'] += 1
            
            # Categorize query
            query_lower = query.lower()
            if any(word in query_lower for word in ['rice', 'paddy', 'cultivation']):
                self.query_analytics['rice_queries'] += 1
            elif any(word in query_lower for word in ['pest', 'insect', 'disease']):
                self.query_analytics['pest_queries'] += 1
            elif any(word in query_lower for word in ['fertilizer', 'nutrient', 'soil']):
                self.query_analytics['fertilizer_queries'] += 1
            else:
                self.query_analytics['other_queries'] += 1
            
            # Track quality if provided
            if response_quality is not None:
                if 'quality_scores' not in self.query_analytics:
                    self.query_analytics['quality_scores'] = []
                self.query_analytics['quality_scores'].append(response_quality)
            
            # Track timestamp
            self.query_analytics['last_query'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error tracking analytics: {e}")
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Get analytics report"""
        try:
            total_queries = self.query_analytics.get('total_queries', 0)
            
            if total_queries == 0:
                return {"message": "No queries tracked yet"}
            
            # Calculate averages
            quality_scores = self.query_analytics.get('quality_scores', [])
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            
            return {
                "total_queries": total_queries,
                "query_categories": {
                    "rice_cultivation": self.query_analytics.get('rice_queries', 0),
                    "pest_management": self.query_analytics.get('pest_queries', 0),
                    "fertilizer_nutrition": self.query_analytics.get('fertilizer_queries', 0),
                    "other": self.query_analytics.get('other_queries', 0)
                },
                "average_response_quality": round(avg_quality, 3),
                "total_quality_ratings": len(quality_scores),
                "last_query_time": self.query_analytics.get('last_query', 'Never'),
                "system_performance": {
                    "embedding_model": self.embedding_model_name,
                    "llm_model": self.llm_model
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            return {"error": str(e)}
    
    def save_analytics(self, filepath: str):
        """Save analytics data"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            analytics_data = {
                "query_analytics": dict(self.query_analytics),
                "feedback_data": self.feedback_data,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analytics_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analytics saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving analytics: {e}")

# Utility functions
def create_advanced_features(llm_model: str = "llama3.2:3b",
                           embedding_model: str = "all-MiniLM-L6-v2") -> AdvancedRAGFeatures:
    """Create and return an AdvancedRAGFeatures instance"""
    return AdvancedRAGFeatures(llm_model, embedding_model)

if __name__ == "__main__":
    # Example usage
    advanced_rag = create_advanced_features()
    
    # Test query analysis
    query = "How can I improve my Boro rice yield in Bangladesh during drought conditions?"
    analysis = advanced_rag.analyze_query(query)
    
    print("=== Query Analysis ===")
    print(f"Query Type: {analysis.query_type}")
    print(f"Intent: {analysis.intent}")
    print(f"Key Entities: {analysis.key_entities}")
    print(f"Complexity Score: {analysis.complexity_score:.3f}")
    print(f"Expanded Queries: {analysis.expanded_queries}")
    print(f"Follow-up Questions: {analysis.suggested_followups}")
    
    # Test context reranking
    sample_contexts = [
        "Rice cultivation requires proper water management and fertilization.",
        "Boro rice is grown during the dry season in Bangladesh from December to May.",
        "Drought-resistant rice varieties can help maintain yield during water stress.",
        "Weather conditions significantly impact rice production in South Asia."
    ]
    
    reranked = advanced_rag.rerank_contexts(query, sample_contexts)
    print("\n=== Context Reranking ===")
    for i, (context, score) in enumerate(reranked):
        print(f"{i+1}. Score: {score:.3f} - {context[:60]}...")
    
    # Test multi-step reasoning
    reasoning_result = advanced_rag.multi_step_reasoning(query, sample_contexts)
    print("\n=== Multi-Step Reasoning ===")
    for i, step in enumerate(reasoning_result['steps']):
        print(f"Step {i+1}: {step}")
    print(f"Conclusion: {reasoning_result['conclusion']}")
    
    # Track analytics
    advanced_rag.track_query_analytics(query, 0.85)
    analytics = advanced_rag.get_analytics_report()
    print("\n=== Analytics Report ===")
    print(f"Total Queries: {analytics['total_queries']}")
    print(f"Query Categories: {analytics['query_categories']}")
