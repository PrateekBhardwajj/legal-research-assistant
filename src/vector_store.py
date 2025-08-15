"""
Vector store implementation for legal document retrieval.
Supports ChromaDB and Pinecone backends.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import uuid
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from loguru import logger

from .document_processor import DocumentChunk
from .embedding_manager import EmbeddingResult


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores."""
    
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingResult]) -> bool:
        """Add documents and their embeddings to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get total number of documents in the store."""
        pass


class ChromaVectorStore(VectorStoreInterface):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, config: Dict[str, Any]):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        self.config = config
        self.persist_directory = config.get('persist_directory', './chromadb')
        self.collection_name = config.get('collection_name', 'legal_documents')
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Create or get collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Initialized ChromaDB vector store: {self.collection_name}")
    
    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Legal documents for RAG system"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingResult]) -> bool:
        """Add documents and their embeddings to ChromaDB."""
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
            
            # Prepare data for insertion
            ids = []
            documents = []
            metadatas = []
            embeddings_list = []
            
            for chunk, embedding in zip(chunks, embeddings):
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)
                
                # Prepare metadata (ChromaDB requires JSON-serializable metadata)
                metadata = {
                    'document_id': chunk.document_id,
                    'document_type': chunk.document_type,
                    'filename': chunk.metadata.get('filename', ''),
                    'file_path': chunk.metadata.get('file_path', ''),
                    'section': chunk.section or '',
                    'page_number': chunk.page_number or 0,
                    'chunk_id': chunk.chunk_id,
                    'embedding_model': embedding.model_name,
                    'timestamp': embedding.timestamp
                }
                metadatas.append(metadata)
                embeddings_list.append(embedding.embedding)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            
            logger.info(f"Added {len(chunks)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents in ChromaDB."""
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'chunk_id': results['metadatas'][0][i]['chunk_id']
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed in ChromaDB: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB."""
        try:
            # Get all chunk IDs for the document IDs
            chunk_ids_to_delete = []
            
            for doc_id in document_ids:
                results = self.collection.get(
                    where={"document_id": doc_id},
                    include=['metadatas']
                )
                
                if results['metadatas']:
                    for metadata in results['metadatas']:
                        chunk_ids_to_delete.append(metadata['chunk_id'])
            
            if chunk_ids_to_delete:
                self.collection.delete(ids=chunk_ids_to_delete)
                logger.info(f"Deleted {len(chunk_ids_to_delete)} chunks for {len(document_ids)} documents")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in ChromaDB."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get document count from ChromaDB: {e}")
            return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            
            # Get sample documents to understand structure
            sample = self.collection.peek(limit=1)
            
            return {
                'name': self.collection_name,
                'count': count,
                'has_documents': count > 0,
                'sample_metadata': sample['metadatas'][0] if sample['metadatas'] else None
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {'name': self.collection_name, 'count': 0, 'has_documents': False}


class PineconeVectorStore(VectorStoreInterface):
    """Pinecone implementation of vector store."""
    
    def __init__(self, config: Dict[str, Any]):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not installed. Install with: pip install pinecone-client")
        
        self.config = config
        self.api_key = config.get('api_key')
        self.environment = config.get('environment', 'us-west1-gcp')
        self.index_name = config.get('index_name', 'legal-research')
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        # Initialize Pinecone
        pinecone.init(api_key=self.api_key, environment=self.environment)
        
        # Create or connect to index
        self.index = self._get_or_create_index()
        
        logger.info(f"Initialized Pinecone vector store: {self.index_name}")
    
    def _get_or_create_index(self):
        """Get or create Pinecone index."""
        try:
            # Check if index exists
            if self.index_name in pinecone.list_indexes():
                index = pinecone.Index(self.index_name)
                logger.info(f"Connected to existing Pinecone index: {self.index_name}")
            else:
                # Create new index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine'
                )
                index = pinecone.Index(self.index_name)
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            return index
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingResult]) -> bool:
        """Add documents and their embeddings to Pinecone."""
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
            
            # Prepare vectors for upsert
            vectors = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Prepare metadata
                metadata = {
                    'content': chunk.content,
                    'document_id': chunk.document_id,
                    'document_type': chunk.document_type,
                    'filename': chunk.metadata.get('filename', ''),
                    'file_path': chunk.metadata.get('file_path', ''),
                    'section': chunk.section or '',
                    'page_number': chunk.page_number or 0,
                    'embedding_model': embedding.model_name,
                    'timestamp': embedding.timestamp
                }
                
                vector = {
                    'id': chunk.chunk_id,
                    'values': embedding.embedding,
                    'metadata': metadata
                }
                vectors.append(vector)
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Added {len(chunks)} documents to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents in Pinecone."""
        try:
            # Prepare filter
            pinecone_filter = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        pinecone_filter[key] = {"$in": value}
                    else:
                        pinecone_filter[key] = {"$eq": value}
            
            # Perform search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=pinecone_filter if pinecone_filter else None,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results['matches']:
                result = {
                    'content': match['metadata'].get('content', ''),
                    'metadata': match['metadata'],
                    'similarity_score': match['score'],
                    'chunk_id': match['id']
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed in Pinecone: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Pinecone."""
        try:
            # Query for all chunks belonging to these documents
            chunk_ids_to_delete = []
            
            for doc_id in document_ids:
                # Query to find all chunks for this document
                results = self.index.query(
                    vector=[0] * 1536,  # Dummy vector
                    top_k=10000,  # Large number to get all chunks
                    filter={"document_id": {"$eq": doc_id}},
                    include_metadata=True
                )
                
                for match in results['matches']:
                    chunk_ids_to_delete.append(match['id'])
            
            if chunk_ids_to_delete:
                # Delete in batches
                batch_size = 1000
                for i in range(0, len(chunk_ids_to_delete), batch_size):
                    batch = chunk_ids_to_delete[i:i + batch_size]
                    self.index.delete(ids=batch)
                
                logger.info(f"Deleted {len(chunk_ids_to_delete)} chunks for {len(document_ids)} documents")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from Pinecone: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in Pinecone."""
        try:
            stats = self.index.describe_index_stats()
            return stats['total_vector_count']
        except Exception as e:
            logger.error(f"Failed to get document count from Pinecone: {e}")
            return 0


class VectorStoreManager:
    """Manager for vector store operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', 'chromadb').lower()
        
        # Initialize appropriate vector store
        if self.provider == 'chromadb':
            chroma_config = config.get('chromadb', {})
            self.vector_store = ChromaVectorStore(chroma_config)
        elif self.provider == 'pinecone':
            pinecone_config = config.get('pinecone', {})
            self.vector_store = PineconeVectorStore(pinecone_config)
        else:
            raise ValueError(f"Unsupported vector store provider: {self.provider}")
        
        logger.info(f"Initialized vector store manager with provider: {self.provider}")
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingResult]) -> bool:
        """Add documents to the vector store."""
        return self.vector_store.add_documents(chunks, embeddings)
    
    def search(self, query_embedding: List[float], top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        return self.vector_store.search(query_embedding, top_k, filters)
    
    def hybrid_search(self, query_embedding: List[float], query_text: str, top_k: int = 5, 
                     semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search."""
        # Get semantic results
        semantic_results = self.vector_store.search(query_embedding, top_k * 2)  # Get more results for reranking
        
        # Perform keyword matching (simple implementation)
        query_terms = query_text.lower().split()
        
        # Score results based on keyword matching
        for result in semantic_results:
            content_lower = result['content'].lower()
            keyword_score = 0
            
            for term in query_terms:
                if term in content_lower:
                    keyword_score += content_lower.count(term) / len(query_terms)
            
            # Normalize keyword score
            keyword_score = min(keyword_score, 1.0)
            
            # Combine scores
            semantic_score = result['similarity_score']
            combined_score = semantic_weight * semantic_score + keyword_weight * keyword_score
            result['combined_score'] = combined_score
            result['keyword_score'] = keyword_score
        
        # Sort by combined score and return top_k
        semantic_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return semantic_results[:top_k]
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        return self.vector_store.delete_documents(document_ids)
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return self.vector_store.get_document_count()
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        info = {
            'provider': self.provider,
            'document_count': self.get_document_count()
        }
        
        # Add provider-specific info
        if hasattr(self.vector_store, 'get_collection_info'):
            info.update(self.vector_store.get_collection_info())
        
        return info
