"""
Retrieval system for legal document search.
Handles semantic, keyword, and hybrid search strategies.
"""

from typing import List, Dict, Any, Optional
import re
from loguru import logger

from .embedding_manager import EmbeddingManager
from .vector_store import VectorStoreManager


class RetrievalSystem:
    """Handles document retrieval using various search strategies."""
    
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStoreManager, config: Dict[str, Any]):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.config = config
        
        # Default search parameters
        self.default_top_k = config.get('top_k', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.75)
        self.hybrid_search_enabled = config.get('hybrid_search', True)
        self.semantic_weight = config.get('semantic_weight', 0.7)
        self.keyword_weight = config.get('keyword_weight', 0.3)
        
        logger.info("Retrieval system initialized")
    
    def semantic_search(self, query: str, top_k: int = None, filters: Dict[str, Any] = None, 
                       similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        
        top_k = top_k or self.default_top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_query(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.get('similarity_score', 0) >= similarity_threshold
            ]
            
            logger.info(f"Semantic search returned {len(filtered_results)} results for query: {query[:50]}...")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = None, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform keyword-based search."""
        
        top_k = top_k or self.default_top_k
        
        try:
            # For simplicity, we'll use semantic search but boost keyword matches
            # In a full implementation, this would use a traditional search index like Elasticsearch
            
            # Generate query embedding for initial retrieval
            query_embedding = self.embedding_manager.embed_query(query)
            
            # Get more results to re-rank
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 3,  # Get more results for keyword matching
                filters=filters
            )
            
            # Extract keywords from query
            query_keywords = self._extract_keywords(query)
            
            # Score results based on keyword matches
            scored_results = []
            for result in results:
                content = result.get('content', '').lower()
                keyword_score = self._calculate_keyword_score(content, query_keywords)
                
                result['keyword_score'] = keyword_score
                result['combined_score'] = keyword_score  # Pure keyword search
                
                scored_results.append(result)
            
            # Sort by keyword score and return top_k
            scored_results.sort(key=lambda x: x['keyword_score'], reverse=True)
            final_results = scored_results[:top_k]
            
            logger.info(f"Keyword search returned {len(final_results)} results for query: {query[:50]}...")
            return final_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = None, filters: Dict[str, Any] = None,
                     similarity_threshold: float = None, semantic_weight: float = None,
                     keyword_weight: float = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword approaches."""
        
        top_k = top_k or self.default_top_k
        similarity_threshold = similarity_threshold or self.similarity_threshold
        semantic_weight = semantic_weight or self.semantic_weight
        keyword_weight = keyword_weight or self.keyword_weight
        
        try:
            # Use the vector store's built-in hybrid search if available
            if hasattr(self.vector_store, 'hybrid_search'):
                query_embedding = self.embedding_manager.embed_query(query)
                results = self.vector_store.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=query,
                    top_k=top_k,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight
                )
            else:
                # Implement hybrid search manually
                results = self._manual_hybrid_search(
                    query, top_k, filters, similarity_threshold,
                    semantic_weight, keyword_weight
                )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.get('similarity_score', 0) >= similarity_threshold
            ]
            
            logger.info(f"Hybrid search returned {len(filtered_results)} results for query: {query[:50]}...")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _manual_hybrid_search(self, query: str, top_k: int, filters: Dict[str, Any],
                             similarity_threshold: float, semantic_weight: float,
                             keyword_weight: float) -> List[Dict[str, Any]]:
        """Manual implementation of hybrid search."""
        
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Get semantic results (more than needed for reranking)
        semantic_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            filters=filters
        )
        
        # Extract keywords from query
        query_keywords = self._extract_keywords(query)
        
        # Combine semantic and keyword scores
        scored_results = []
        for result in semantic_results:
            content = result.get('content', '').lower()
            
            # Get scores
            semantic_score = result.get('similarity_score', 0)
            keyword_score = self._calculate_keyword_score(content, query_keywords)
            
            # Combined score
            combined_score = semantic_weight * semantic_score + keyword_weight * keyword_score
            
            result['keyword_score'] = keyword_score
            result['combined_score'] = combined_score
            
            scored_results.append(result)
        
        # Sort by combined score and return top_k
        scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return scored_results[:top_k]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        
        # Basic keyword extraction
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as',
            'was', 'will', 'an', 'be', 'by', 'this', 'have', 'from', 'they',
            'know', 'want', 'been', 'now', 'were', 'said', 'each', 'that',
            'their', 'time', 'will', 'about', 'if', 'up', 'out', 'many',
            'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make',
            'like', 'into', 'him', 'has', 'two', 'more', 'very', 'what',
            'but', 'not', 'with', 'he', 'as', 'you', 'do', 'at', 'his',
            'but', 'def', 'if', 'while', 'he', 'my', 'than'
        }
        
        # Clean and split query
        cleaned_query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = cleaned_query.split()
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        return keywords
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """Calculate keyword matching score for content."""
        
        if not keywords:
            return 0.0
        
        total_score = 0.0
        content_words = content.lower().split()
        content_text = content.lower()
        
        for keyword in keywords:
            # Exact word matches (higher weight)
            word_matches = sum(1 for word in content_words if word == keyword)
            total_score += word_matches * 2.0
            
            # Partial matches (lower weight)
            if keyword in content_text:
                partial_matches = content_text.count(keyword)
                total_score += partial_matches * 0.5
        
        # Normalize by number of keywords and content length
        normalized_score = total_score / (len(keywords) * max(1, len(content_words) / 100))
        
        # Cap at 1.0
        return min(normalized_score, 1.0)
    
    def search_by_document_type(self, query: str, document_types: List[str], 
                               search_type: str = "hybrid", top_k: int = None) -> List[Dict[str, Any]]:
        """Search within specific document types."""
        
        filters = {'document_type': document_types}
        
        if search_type == "semantic":
            return self.semantic_search(query, top_k=top_k, filters=filters)
        elif search_type == "keyword":
            return self.keyword_search(query, top_k=top_k, filters=filters)
        else:
            return self.hybrid_search(query, top_k=top_k, filters=filters)
    
    def search_by_section(self, query: str, sections: List[str], 
                         search_type: str = "hybrid", top_k: int = None) -> List[Dict[str, Any]]:
        """Search within specific document sections."""
        
        # This would need to be implemented based on how sections are stored
        # For now, we'll do a post-processing filter
        
        if search_type == "semantic":
            results = self.semantic_search(query, top_k=top_k * 2)
        elif search_type == "keyword":
            results = self.keyword_search(query, top_k=top_k * 2)
        else:
            results = self.hybrid_search(query, top_k=top_k * 2)
        
        # Filter by sections
        filtered_results = []
        for result in results:
            section = result.get('metadata', {}).get('section', '').lower()
            if any(sec.lower() in section for sec in sections):
                filtered_results.append(result)
        
        return filtered_results[:top_k or self.default_top_k]
    
    def get_similar_documents(self, document_id: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        
        top_k = top_k or self.default_top_k
        
        try:
            # This would require getting the document content and finding similar ones
            # For now, return empty list as this requires more complex implementation
            logger.warning("Similar documents search not yet implemented")
            return []
            
        except Exception as e:
            logger.error(f"Similar documents search failed: {e}")
            return []
    
    def explain_results(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Provide explanation for search results."""
        
        if not results:
            return {"explanation": "No results found", "suggestions": ["Try broader search terms", "Check spelling"]}
        
        explanation = {
            "query": query,
            "num_results": len(results),
            "avg_similarity": sum(r.get('similarity_score', 0) for r in results) / len(results),
            "document_types": list(set(r.get('metadata', {}).get('document_type', 'unknown') for r in results)),
            "suggestions": []
        }
        
        # Add suggestions based on results
        if explanation["avg_similarity"] < 0.5:
            explanation["suggestions"].append("Results have low similarity - try more specific terms")
        
        if len(explanation["document_types"]) == 1:
            explanation["suggestions"].append(f"All results are from {explanation['document_types'][0]} documents")
        
        return explanation
