from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math

from .vector_store import VectorStore, KeywordSearch
from ..ingest.github_ingest import Document
from ..shared.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    document: Document
    score: float
    retrieval_method: str
    rank: int


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_search: KeywordSearch,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        if abs(vector_weight + keyword_weight - 1.0) > 1e-6:
            raise ValueError("Vector and keyword weights must sum to 1.0")
    
    def search(
        self,
        query: str,
        k: int = 10,
        vector_k: Optional[int] = None,
        keyword_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[SearchResult]:
        # Default to getting more results than needed for reranking
        vector_k = vector_k or min(k * 2, 50)
        keyword_k = keyword_k or min(k * 2, 50)
        
        logger.info(f"Hybrid search for: '{query}' (k={k})")
        
        # Get vector search results
        vector_results = self.vector_store.search(query, vector_k)
        logger.debug(f"Vector search returned {len(vector_results)} results")
        
        # Get keyword search results
        keyword_results = self.keyword_search.search(query, keyword_k)
        logger.debug(f"Keyword search returned {len(keyword_results)} results")
        
        # Apply filters if provided
        if filters:
            vector_results = self._apply_filters(vector_results, filters)
            keyword_results = self._apply_filters(keyword_results, filters)
        
        # Combine and rerank results
        if rerank:
            combined_results = self._reciprocal_rank_fusion(
                vector_results, keyword_results, k
            )
        else:
            combined_results = self._weighted_combination(
                vector_results, keyword_results, k
            )
        
        logger.info(f"Returning {len(combined_results)} results")
        return combined_results
    
    def _apply_filters(
        self, 
        results: List[Tuple[Document, float]], 
        filters: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        filtered = []
        
        for doc, score in results:
            include = True
            
            for filter_key, filter_value in filters.items():
                doc_value = doc.metadata.get(filter_key)
                
                if isinstance(filter_value, list):
                    if doc_value not in filter_value:
                        include = False
                        break
                elif doc_value != filter_value:
                    include = False
                    break
            
            if include:
                filtered.append((doc, score))
        
        return filtered
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        k: int,
        rrf_constant: int = 60
    ) -> List[SearchResult]:
        # Create document to score mapping
        doc_scores = {}
        doc_objects = {}
        
        # Add vector results with RRF scoring
        for rank, (doc, score) in enumerate(vector_results, 1):
            doc_id = doc.source
            rrf_score = 1.0 / (rrf_constant + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'vector_score': score,
                    'keyword_score': 0.0,
                    'rrf_score': 0.0,
                    'vector_rank': rank,
                    'keyword_rank': float('inf')
                }
                doc_objects[doc_id] = doc
            
            doc_scores[doc_id]['rrf_score'] += self.vector_weight * rrf_score
        
        # Add keyword results with RRF scoring
        for rank, (doc, score) in enumerate(keyword_results, 1):
            doc_id = doc.source
            rrf_score = 1.0 / (rrf_constant + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'vector_score': 0.0,
                    'keyword_score': score,
                    'rrf_score': 0.0,
                    'vector_rank': float('inf'),
                    'keyword_rank': rank
                }
                doc_objects[doc_id] = doc
            else:
                doc_scores[doc_id]['keyword_score'] = score
                doc_scores[doc_id]['keyword_rank'] = rank
            
            doc_scores[doc_id]['rrf_score'] += self.keyword_weight * rrf_score
        
        # Sort by RRF score and create results
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]['rrf_score'],
            reverse=True
        )
        
        results = []
        for rank, (doc_id, scores) in enumerate(sorted_docs[:k], 1):
            # Determine primary retrieval method
            if scores['vector_rank'] < scores['keyword_rank']:
                method = "vector"
            elif scores['keyword_rank'] < scores['vector_rank']:
                method = "keyword"
            else:
                method = "hybrid"
            
            result = SearchResult(
                document=doc_objects[doc_id],
                score=scores['rrf_score'],
                retrieval_method=method,
                rank=rank
            )
            results.append(result)
        
        return results
    
    def _weighted_combination(
        self,
        vector_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        k: int
    ) -> List[SearchResult]:
        # Normalize scores to [0, 1] range
        vector_results = self._normalize_scores(vector_results)
        keyword_results = self._normalize_scores(keyword_results)
        
        # Combine scores
        doc_scores = {}
        doc_objects = {}
        
        for doc, score in vector_results:
            doc_id = doc.source
            doc_scores[doc_id] = self.vector_weight * score
            doc_objects[doc_id] = doc
        
        for doc, score in keyword_results:
            doc_id = doc.source
            if doc_id in doc_scores:
                doc_scores[doc_id] += self.keyword_weight * score
            else:
                doc_scores[doc_id] = self.keyword_weight * score
                doc_objects[doc_id] = doc
        
        # Sort and create results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:k], 1):
            result = SearchResult(
                document=doc_objects[doc_id],
                score=score,
                retrieval_method="hybrid",
                rank=rank
            )
            results.append(result)
        
        return results
    
    def _normalize_scores(
        self, 
        results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        if not results:
            return results
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [(doc, 1.0) for doc, _ in results]
        
        normalized = []
        for doc, score in results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((doc, norm_score))
        
        return normalized
    
    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 4000,
        k: int = 10,
        **search_kwargs
    ) -> Tuple[str, List[SearchResult]]:
        # Get search results
        results = self.search(query, k=k, **search_kwargs)
        
        # Build context string within token limit
        context_parts = []
        current_tokens = 0
        
        for result in results:
            doc = result.document
            
            # Create document header
            source_info = f"Source: {doc.source}"
            if 'file_path' in doc.metadata:
                source_info += f" (File: {doc.metadata['file_path']})"
            
            doc_header = f"\n--- {source_info} ---\n"
            doc_content = f"{doc_header}{doc.content}\n"
            
            # Rough token estimation (1 token ≈ 4 characters)
            doc_tokens = len(doc_content) // 4
            
            if current_tokens + doc_tokens > max_tokens:
                break
            
            context_parts.append(doc_content)
            current_tokens += doc_tokens
        
        context = "\n".join(context_parts)
        return context, results[:len(context_parts)]