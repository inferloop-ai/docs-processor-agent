"""
Embedding Agent
Generates vector embeddings for document chunks using various embedding models
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import time
import hashlib

# LangChain and embedding imports
from langchain.schema import Document
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings
)
from langchain_anthropic import AnthropicEmbeddings
from sentence_transformers import SentenceTransformer
import openai

from .base_agent import BaseAgent, AgentResult


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embedding: List[float]
    model_name: str
    embedding_dimension: int
    processing_time: float
    token_count: Optional[int] = None


class EmbeddingAgent(BaseAgent):
    """
    Embedding Agent
    
    Capabilities:
    - Multi-provider embedding generation (OpenAI, HuggingFace, SentenceTransformers)
    - Batch processing for efficiency
    - Embedding caching and deduplication
    - Multiple embedding strategies
    - Quality validation and normalization
    - Performance optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Load embedding-specific configuration
        self.embedder_config = self.config.get('agents', {}).get('embedder', {})
        self.embedding_config = self.config.get('embeddings', {})
        
        # Embedding settings
        self.batch_size = self.embedder_config.get('batch_size', 50)
        self.max_retries = self.embedder_config.get('max_retries', 3)
        self.normalize_embeddings = self.embedder_config.get('normalize_embeddings', True)
        
        # Model configuration
        self.embedding_provider = self.embedding_config.get('embedding_provider', 'openai')
        self.embedding_model = self.embedding_config.get('embedding_model', 'text-embedding-3-small')
        self.embedding_dimension = self.embedding_config.get('embedding_dimensions', 1536)
        
        # Initialize embedding models
        self.embedding_models = {}
        self._initialize_embedding_models()
        
        # Caching
        self.embedding_cache = {}
        self.cache_enabled = True
        
        self.log_info(f"EmbeddingAgent initialized with provider: {self.embedding_provider}")
    
    def _initialize_embedding_models(self):
        """Initialize embedding models for different providers"""
        try:
            # OpenAI embeddings
            if self.embedding_provider == 'openai' or 'openai' in self.embedding_provider:
                try:
                    self.embedding_models['openai'] = OpenAIEmbeddings(
                        model=self.embedding_model,
                        chunk_size=self.batch_size
                    )
                    self.log_info("OpenAI embeddings initialized")
                except Exception as e:
                    self.log_warning(f"Failed to initialize OpenAI embeddings: {e}")
            
            # Sentence Transformers
            if self.embedding_provider == 'sentence-transformers' or 'sentence-transformers' in self.embedding_provider:
                try:
                    model_name = self.embedding_model if self.embedding_model != 'text-embedding-3-small' else 'all-MiniLM-L6-v2'
                    
                    self.embedding_models['sentence-transformers'] = SentenceTransformerEmbeddings(
                        model_name=model_name
                    )
                    
                    # Also initialize direct sentence transformer for batch processing
                    self.sentence_transformer = SentenceTransformer(model_name)
                    self.log_info(f"Sentence Transformers initialized with model: {model_name}")
                except Exception as e:
                    self.log_warning(f"Failed to initialize Sentence Transformers: {e}")
            
            # HuggingFace embeddings
            if self.embedding_provider == 'huggingface':
                try:
                    self.embedding_models['huggingface'] = HuggingFaceEmbeddings(
                        model_name=self.embedding_model
                    )
                    self.log_info("HuggingFace embeddings initialized")
                except Exception as e:
                    self.log_warning(f"Failed to initialize HuggingFace embeddings: {e}")
            
            # Anthropic embeddings (if available)
            if self.embedding_provider == 'anthropic':
                try:
                    self.embedding_models['anthropic'] = AnthropicEmbeddings()
                    self.log_info("Anthropic embeddings initialized")
                except Exception as e:
                    self.log_warning(f"Failed to initialize Anthropic embeddings: {e}")
                    
        except Exception as e:
            self.log_error(f"Error initializing embedding models: {e}")
    
    async def process(self, input_data: Union[List[Document], Dict[str, Any]], **kwargs) -> AgentResult:
        """
        Generate embeddings for documents
        
        Args:
            input_data: List of Document objects OR dict with 'documents' key
            **kwargs: Additional parameters
                - provider: Override embedding provider
                - batch_size: Override batch size
                - normalize: Override normalization setting
        
        Returns:
            AgentResult with embedded documents and metadata
        """
        try:
            # Parse input
            if isinstance(input_data, list):
                documents = input_data
                document_id = kwargs.get('document_id', 'unknown')
            elif isinstance(input_data, dict):
                documents = input_data.get('documents', [])
                document_id = input_data.get('document_id', 'unknown')
            else:
                return AgentResult(
                    success=False,
                    error="Input must be list of Documents or dict with 'documents' key"
                )
            
            if not documents:
                return AgentResult(
                    success=False,
                    error="No documents provided for embedding"
                )
            
            # Get embedding settings
            provider = kwargs.get('provider', self.embedding_provider)
            batch_size = kwargs.get('batch_size', self.batch_size)
            normalize = kwargs.get('normalize', self.normalize_embeddings)
            
            # Generate embeddings
            embedding_results = await self._generate_embeddings_batch(
                documents, provider, batch_size, normalize
            )
            
            # Create embedded documents
            embedded_documents = []
            embedding_metadata = []
            
            for i, (doc, embedding_result) in enumerate(zip(documents, embedding_results)):
                # Create new document with embedding in metadata
                embedded_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        'embedding': embedding_result.embedding,
                        'embedding_model': embedding_result.model_name,
                        'embedding_dimension': embedding_result.embedding_dimension,
                        'embedding_processing_time': embedding_result.processing_time
                    }
                )
                
                embedded_documents.append(embedded_doc)
                embedding_metadata.append({
                    'chunk_index': i,
                    'embedding_dimension': embedding_result.embedding_dimension,
                    'model_name': embedding_result.model_name,
                    'processing_time': embedding_result.processing_time,
                    'token_count': embedding_result.token_count
                })
            
            # Calculate statistics
            total_processing_time = sum(r.processing_time for r in embedding_results)
            average_processing_time = total_processing_time / len(embedding_results)
            total_tokens = sum(r.token_count or 0 for r in embedding_results)
            
            result_data = {
                "embedded_documents": embedded_documents,
                "embedding_metadata": embedding_metadata,
                "statistics": {
                    "total_documents": len(documents),
                    "total_processing_time": total_processing_time,
                    "average_processing_time": average_processing_time,
                    "total_tokens": total_tokens,
                    "embedding_dimension": embedding_results[0].embedding_dimension if embedding_results else 0,
                    "model_used": embedding_results[0].model_name if embedding_results else "",
                    "provider": provider
                }
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                confidence=0.95,
                metadata={
                    "document_id": document_id,
                    "embedding_provider": provider,
                    "batch_processing": True
                }
            )
            
        except Exception as e:
            self.log_error(f"Error in embedding generation: {str(e)}")
            return AgentResult(
                success=False,
                error=f"Embedding generation failed: {str(e)}"
            )
    
    async def _generate_embeddings_batch(self, documents: List[Document], 
                                       provider: str, batch_size: int, 
                                       normalize: bool) -> List[EmbeddingResult]:
        """Generate embeddings for a batch of documents"""
        results = []
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = await self._process_embedding_batch(batch, provider, normalize)
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(documents):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _process_embedding_batch(self, batch: List[Document], 
                                     provider: str, normalize: bool) -> List[EmbeddingResult]:
        """Process a single batch of documents"""
        batch_texts = [doc.page_content for doc in batch]
        
        # Check cache first
        cached_results = []
        uncached_texts = []
        uncached_indices = []
        
        if self.cache_enabled:
            for i, text in enumerate(batch_texts):
                cache_key = self._get_cache_key(text, provider)
                if cache_key in self.embedding_cache:
                    cached_results.append((i, self.embedding_cache[cache_key]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = batch_texts
            uncached_indices = list(range(len(batch_texts)))
        
        # Generate embeddings for uncached texts
        uncached_embeddings = []
        if uncached_texts:
            if provider == 'openai':
                uncached_embeddings = await self._generate_openai_embeddings(uncached_texts)
            elif provider == 'sentence-transformers':
                uncached_embeddings = await self._generate_sentence_transformer_embeddings(uncached_texts)
            elif provider == 'huggingface':
                uncached_embeddings = await self._generate_huggingface_embeddings(uncached_texts)
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        
        # Combine cached and new results
        all_results = [None] * len(batch_texts)
        
        # Add cached results
        for index, result in cached_results:
            all_results[index] = result
        
        # Add new results and cache them
        for i, (index, embedding_result) in enumerate(zip(uncached_indices, uncached_embeddings)):
            all_results[index] = embedding_result
            
            # Cache the result
            if self.cache_enabled:
                cache_key = self._get_cache_key(batch_texts[index], provider)
                self.embedding_cache[cache_key] = embedding_result
        
        # Apply normalization if requested
        if normalize:
            for result in all_results:
                if result and result.embedding:
                    result.embedding = self._normalize_embedding(result.embedding)
        
        return all_results
    
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings using OpenAI"""
        start_time = time.time()
        
        try:
            embeddings_model = self.embedding_models.get('openai')
            if not embeddings_model:
                raise ValueError("OpenAI embeddings not initialized")
            
            # Generate embeddings
            embeddings = await embeddings_model.aembed_documents(texts)
            
            processing_time = time.time() - start_time
            avg_time_per_text = processing_time / len(texts)
            
            # Estimate token count (rough approximation)
            total_chars = sum(len(text) for text in texts)
            estimated_tokens = total_chars // 4  # Rough approximation
            
            results = []
            for embedding in embeddings:
                results.append(EmbeddingResult(
                    embedding=embedding,
                    model_name=self.embedding_model,
                    embedding_dimension=len(embedding),
                    processing_time=avg_time_per_text,
                    token_count=estimated_tokens // len(embeddings)
                ))
            
            return results
            
        except Exception as e:
            self.log_error(f"OpenAI embedding generation failed: {e}")
            raise
    
    async def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings using Sentence Transformers"""
        start_time = time.time()
        
        try:
            if hasattr(self, 'sentence_transformer'):
                # Use direct sentence transformer for better batch processing
                embeddings = self.sentence_transformer.encode(texts, convert_to_numpy=True)
            else:
                embeddings_model = self.embedding_models.get('sentence-transformers')
                if not embeddings_model:
                    raise ValueError("Sentence Transformers not initialized")
                embeddings = await embeddings_model.aembed_documents(texts)
            
            processing_time = time.time() - start_time
            avg_time_per_text = processing_time / len(texts)
            
            results = []
            for embedding in embeddings:
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                results.append(EmbeddingResult(
                    embedding=embedding,
                    model_name=getattr(self, 'sentence_transformer', self.embedding_models.get('sentence-transformers')).get_sentence_embedding_dimension() if hasattr(self, 'sentence_transformer') else len(embedding),
                    embedding_dimension=len(embedding),
                    processing_time=avg_time_per_text,
                    token_count=None  # Sentence transformers don't provide token count
                ))
            
            return results
            
        except Exception as e:
            self.log_error(f"Sentence Transformers embedding generation failed: {e}")
            raise
    
    async def _generate_huggingface_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings using HuggingFace"""
        start_time = time.time()
        
        try:
            embeddings_model = self.embedding_models.get('huggingface')
            if not embeddings_model:
                raise ValueError("HuggingFace embeddings not initialized")
            
            embeddings = await embeddings_model.aembed_documents(texts)
            
            processing_time = time.time() - start_time
            avg_time_per_text = processing_time / len(texts)
            
            results = []
            for embedding in embeddings:
                results.append(EmbeddingResult(
                    embedding=embedding,
                    model_name=self.embedding_model,
                    embedding_dimension=len(embedding),
                    processing_time=avg_time_per_text,
                    token_count=None
                ))
            
            return results
            
        except Exception as e:
            self.log_error(f"HuggingFace embedding generation failed: {e}")
            raise
    
    def _get_cache_key(self, text: str, provider: str) -> str:
        """Generate cache key for text and provider"""
        content_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{provider}:{self.embedding_model}:{content_hash}"
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to unit vector"""
        try:
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            
            if norm == 0:
                return embedding  # Return original if zero vector
            
            normalized = embedding_array / norm
            return normalized.tolist()
            
        except Exception as e:
            self.log_warning(f"Error normalizing embedding: {e}")
            return embedding
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.log_error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_documents(self, query_embedding: List[float], 
                             document_embeddings: List[Tuple[str, List[float]]], 
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar documents to a query embedding"""
        try:
            similarities = []
            
            for doc_id, doc_embedding in document_embeddings:
                similarity = self.calculate_similarity(query_embedding, doc_embedding)
                similarities.append((doc_id, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            self.log_error(f"Error finding similar documents: {e}")
            return []
    
    async def embed_query(self, query: str, provider: Optional[str] = None) -> Optional[List[float]]:
        """Embed a single query string"""
        try:
            provider = provider or self.embedding_provider
            
            # Create a temporary document
            temp_doc = Document(page_content=query, metadata={})
            
            # Generate embedding
            results = await self._process_embedding_batch([temp_doc], provider, self.normalize_embeddings)
            
            if results and results[0]:
                return results[0].embedding
            
            return None
            
        except Exception as e:
            self.log_error(f"Error embedding query: {e}")
            return None
    
    def get_embedding_dimension(self, provider: Optional[str] = None) -> int:
        """Get the embedding dimension for a provider"""
        provider = provider or self.embedding_provider
        
        dimension_map = {
            'openai': {
                'text-embedding-3-small': 1536,
                'text-embedding-3-large': 3072,
                'text-embedding-ada-002': 1536
            },
            'sentence-transformers': {
                'all-MiniLM-L6-v2': 384,
                'all-mpnet-base-v2': 768,
                'all-distilroberta-v1': 768
            }
        }
        
        if provider in dimension_map and self.embedding_model in dimension_map[provider]:
            return dimension_map[provider][self.embedding_model]
        
        return self.embedding_dimension
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        self.log_info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.embedding_cache),
            "cache_enabled": self.cache_enabled,
            "memory_usage_mb": len(str(self.embedding_cache)) / (1024 * 1024)
        }
    
    def set_batch_size(self, batch_size: int):
        """Set batch size for processing"""
        self.batch_size = max(1, min(batch_size, 1000))  # Clamp between 1 and 1000
        self.log_info(f"Batch size set to {self.batch_size}")
    
    def enable_cache(self, enabled: bool = True):
        """Enable or disable caching"""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()
        self.log_info(f"Caching {'enabled' if enabled else 'disabled'}")
    
    async def validate_embeddings(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Validate embedding quality and consistency"""
        try:
            if not embeddings:
                return {"valid": False, "error": "No embeddings provided"}
            
            # Check dimensions consistency
            dimensions = [len(emb) for emb in embeddings]
            if len(set(dimensions)) > 1:
                return {
                    "valid": False, 
                    "error": f"Inconsistent embedding dimensions: {set(dimensions)}"
                }
            
            # Check for zero vectors
            zero_vectors = sum(1 for emb in embeddings if all(x == 0 for x in emb))
            
            # Calculate basic statistics
            avg_dimension = sum(dimensions) / len(dimensions)
            avg_magnitude = np.mean([np.linalg.norm(emb) for emb in embeddings])
            
            return {
                "valid": True,
                "statistics": {
                    "total_embeddings": len(embeddings),
                    "embedding_dimension": int(avg_dimension),
                    "zero_vectors": zero_vectors,
                    "average_magnitude": float(avg_magnitude),
                    "dimension_consistency": len(set(dimensions)) == 1
                }
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported embedding providers"""
        return list(self.embedding_models.keys())
    
    def switch_provider(self, provider: str, model: Optional[str] = None):
        """Switch to a different embedding provider"""
        if provider not in self.embedding_models:
            raise ValueError(f"Provider {provider} not available. Available: {list(self.embedding_models.keys())}")
        
        self.embedding_provider = provider
        if model:
            self.embedding_model = model
        
        # Clear cache when switching providers
        self.clear_cache()
        
        self.log_info(f"Switched to provider: {provider}, model: {self.embedding_model}")
    
    async def benchmark_providers(self, test_texts: List[str]) -> Dict[str, Any]:
        """Benchmark different embedding providers"""
        results = {}
        
        for provider in self.embedding_models.keys():
            try:
                start_time = time.time()
                
                # Create temporary documents
                temp_docs = [Document(page_content=text) for text in test_texts]
                
                # Generate embeddings
                embedding_results = await self._process_embedding_batch(
                    temp_docs, provider, False
                )
                
                processing_time = time.time() - start_time
                
                results[provider] = {
                    "success": True,
                    "processing_time": processing_time,
                    "time_per_text": processing_time / len(test_texts),
                    "embedding_dimension": embedding_results[0].embedding_dimension if embedding_results else 0,
                    "texts_processed": len(embedding_results)
                }
                
            except Exception as e:
                results[provider] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results