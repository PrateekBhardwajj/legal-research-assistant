"""
Embedding manager for generating and managing document embeddings.
Supports OpenAI and HuggingFace embedding models.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import time

try:
    import openai
    # Check if we're using the new OpenAI client
    OPENAI_V1 = hasattr(openai, 'OpenAI')
    if OPENAI_V1:
        openai_client = None  # Will be initialized later
except ImportError:
    openai = None
    OPENAI_V1 = False
    openai_client = None
from sentence_transformers import SentenceTransformer
import tiktoken
from loguru import logger

from .document_processor import DocumentChunk


@dataclass
class EmbeddingResult:
    """Represents an embedding result for a document chunk."""
    chunk_id: str
    embedding: List[float]
    model_name: str
    timestamp: float
    token_count: int = None


class EmbeddingManager:
    """Manages document embeddings using various embedding models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_api_key = config.get('api_key')
        self.embedding_model = config.get('embedding_model', 'text-embedding-ada-002')
        self.fallback_model = config.get('fallback_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize OpenAI client
        self.openai_client = None
        if self.openai_api_key:
            if OPENAI_V1:
                # New OpenAI client (v1.0+)
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            else:
                # Legacy OpenAI (v0.x)
                openai.api_key = self.openai_api_key
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
        
        # Initialize local embedding model as fallback
        self._init_local_model()
    
    def _init_local_model(self):
        """Initialize local sentence transformer model."""
        models_to_try = [
            self.fallback_model,
            'all-MiniLM-L6-v2',
            'paraphrase-MiniLM-L6-v2',
            'distilbert-base-nli-stsb-mean-tokens'
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying to initialize model: {model_name}")
                self.local_model = SentenceTransformer(model_name, device='cpu')
                logger.info(f"Successfully initialized local embedding model: {model_name}")
                self.fallback_model = model_name  # Update to working model
                return
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_name}: {e}")
                continue
        
        logger.error("Failed to initialize any local embedding model")
        self.local_model = None
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[EmbeddingResult]:
        """Generate embeddings for a list of document chunks."""
        embeddings = []
        
        for chunk in chunks:
            try:
                embedding_result = self._generate_single_embedding(chunk)
                if embedding_result:
                    embeddings.append(embedding_result)
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {chunk.chunk_id}: {e}")
                continue
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def _generate_single_embedding(self, chunk: DocumentChunk) -> Optional[EmbeddingResult]:
        """Generate embedding for a single document chunk."""
        text = self._preprocess_text(chunk.content)
        
        # Try OpenAI first
        if self.openai_api_key:
            try:
                return self._generate_openai_embedding(chunk.chunk_id, text)
            except Exception as e:
                logger.warning(f"OpenAI embedding failed for {chunk.chunk_id}: {e}")
        
        # Fallback to local model
        if self.local_model:
            try:
                return self._generate_local_embedding(chunk.chunk_id, text)
            except Exception as e:
                logger.error(f"Local embedding failed for {chunk.chunk_id}: {e}")
        
        return None
    
    def _generate_openai_embedding(self, chunk_id: str, text: str) -> EmbeddingResult:
        """Generate embedding using OpenAI API."""
        # Count tokens
        token_count = self._count_tokens(text) if self.tokenizer else None
        
        # Check token limit (8191 for text-embedding-ada-002)
        max_tokens = 8191
        if token_count and token_count > max_tokens:
            text = self._truncate_text(text, max_tokens)
            token_count = max_tokens
        
        # Generate embedding
        if OPENAI_V1 and self.openai_client:
            # New OpenAI client (v1.0+)
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
        else:
            # Legacy OpenAI (v0.x)
            response = openai.Embedding.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response['data'][0]['embedding']
        
        return EmbeddingResult(
            chunk_id=chunk_id,
            embedding=embedding,
            model_name=self.embedding_model,
            timestamp=time.time(),
            token_count=token_count
        )
    
    def _generate_local_embedding(self, chunk_id: str, text: str) -> EmbeddingResult:
        """Generate embedding using local SentenceTransformer model."""
        embedding = self.local_model.encode(text, convert_to_tensor=False)
        
        return EmbeddingResult(
            chunk_id=chunk_id,
            embedding=embedding.tolist(),
            model_name=self.fallback_model,
            timestamp=time.time()
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Handle very long texts by truncating at sentence boundaries
        if len(text) > 8000:  # Conservative limit
            sentences = text.split('. ')
            truncated_text = ""
            for sentence in sentences:
                if len(truncated_text + sentence) > 7500:
                    break
                truncated_text += sentence + ". "
            text = truncated_text.strip()
        
        return text
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if not self.tokenizer:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return len(text) // 4
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if not self.tokenizer:
            # Rough truncation based on character count
            max_chars = max_tokens * 4
            return text[:max_chars]
        
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
        except Exception as e:
            logger.warning(f"Text truncation failed: {e}")
            max_chars = max_tokens * 4
            return text[:max_chars]
    
    def batch_generate_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 100) -> List[EmbeddingResult]:
        """Generate embeddings in batches for efficiency."""
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            batch_embeddings = self.generate_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Add delay to respect API rate limits
            if self.openai_api_key and i + batch_size < len(chunks):
                time.sleep(1)  # 1 second delay between batches
        
        return all_embeddings
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def find_similar_chunks(self, query_embedding: List[float], chunk_embeddings: List[EmbeddingResult], top_k: int = 5) -> List[tuple]:
        """Find most similar chunks to a query embedding."""
        similarities = []
        
        for chunk_embedding in chunk_embeddings:
            similarity = self.compute_similarity(query_embedding, chunk_embedding.embedding)
            similarities.append((chunk_embedding.chunk_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        preprocessed_query = self._preprocess_text(query)
        
        # Try OpenAI first
        if self.openai_api_key:
            try:
                if OPENAI_V1 and self.openai_client:
                    # New OpenAI client (v1.0+)
                    response = self.openai_client.embeddings.create(
                        input=preprocessed_query,
                        model=self.embedding_model
                    )
                    return response.data[0].embedding
                else:
                    # Legacy OpenAI (v0.x)
                    response = openai.Embedding.create(
                        input=preprocessed_query,
                        model=self.embedding_model
                    )
                    return response['data'][0]['embedding']
            except Exception as e:
                logger.warning(f"OpenAI query embedding failed: {e}")
        
        # Fallback to local model
        if self.local_model:
            try:
                embedding = self.local_model.encode(preprocessed_query, convert_to_tensor=False)
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Local query embedding failed: {e}")
        
        raise Exception("Failed to generate query embedding")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the current model."""
        if self.embedding_model == "text-embedding-ada-002":
            return 1536
        elif self.local_model:
            # Get dimension from local model
            return self.local_model.get_sentence_embedding_dimension()
        else:
            return 384  # Default dimension for all-MiniLM-L6-v2
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding is properly formatted."""
        if not isinstance(embedding, list):
            return False
        
        if len(embedding) == 0:
            return False
        
        # Check if all elements are numbers
        try:
            for val in embedding:
                float(val)
        except (TypeError, ValueError):
            return False
        
        # Check dimension
        expected_dim = self.get_embedding_dimension()
        if len(embedding) != expected_dim:
            logger.warning(f"Embedding dimension mismatch: {len(embedding)} != {expected_dim}")
        
        return True
    
    def save_embeddings(self, embeddings: List[EmbeddingResult], file_path: str):
        """Save embeddings to file (for caching purposes)."""
        import pickle
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved {len(embeddings)} embeddings to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def load_embeddings(self, file_path: str) -> List[EmbeddingResult]:
        """Load embeddings from file."""
        import pickle
        
        try:
            with open(file_path, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"Loaded {len(embeddings)} embeddings from {file_path}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return []
