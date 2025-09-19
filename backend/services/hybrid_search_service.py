"""
Hybrid Search Service - Optimized for document-level retrieval
Implements MMR with high fetch_k to combat knowledge dilution
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

from langchain_milvus import Milvus
from langchain_core.documents import Document

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

class HybridSearchService:
    """
    Optimized hybrid search service focusing on document-level retrieval.
    Uses MMR with high fetch_k ratio to prevent knowledge dilution.
    """
    
    def __init__(self):
        """Initialize hybrid search service"""
        self.settings = get_settings()
        self.vectorstore = None
        self._performance_metrics = []
        logger.info("Hybrid Search Service initialized for document-level retrieval")
    
    def set_vectorstore(self, vectorstore: Milvus):
        """Set Milvus vectorstore instance"""
        self.vectorstore = vectorstore
        logger.info("Vectorstore set for hybrid search service")
    
    async def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 6,
        search_method: str = "mmr",
        threshold: float = 0.6,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with document-level optimization.
        
        Args:
            query: Search query
            collection_name: Milvus collection name
            top_k: Number of final documents to return
            search_method: Search method (mmr, similarity, similarity_score_threshold)
            threshold: Similarity threshold
            metadata_filter: Optional metadata filtering
            
        Returns:
            List of search results with document-level grouping
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not set. Call set_vectorstore() first.")
        
        start_time = time.time()
        
        try:
            search_params = self._get_search_params(search_method, top_k, threshold)
            
            if metadata_filter:
                search_params["expr"] = self._build_filter_expression(metadata_filter)
            
            retriever = self.vectorstore.as_retriever(**search_params)
            
            retrieved_docs = retriever.invoke(query)
            
            grouped_results = self._group_results_by_document(retrieved_docs)
            
            search_time = time.time() - start_time
            
            self._track_performance({
                "query": query[:100],  
                "search_method": search_method,
                "top_k": top_k,
                "results_count": len(retrieved_docs),
                "documents_count": len(grouped_results),
                "search_time": search_time,
                "timestamp": time.time()
            })
            
            logger.info(f"Search completed: {len(retrieved_docs)} chunks from {len(grouped_results)} documents in {search_time:.2f}s")
            
            return self._format_search_results(retrieved_docs, grouped_results)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    def _get_search_params(self, search_method: str, top_k: int, threshold: float) -> Dict[str, Any]:
        """
        Get optimized search parameters for focused context retrieval.
        Reduces context dilution by using tighter MMR parameters.
        """
        base_params = {
            "search_type": search_method,
            "search_kwargs": {
                "k": min(top_k, 6), 
                "score_threshold": max(threshold, 0.6)  
            }
        }
        
        if search_method == "mmr":
            fetch_k = min(top_k * 3, 18)  
            
            base_params["search_kwargs"].update({
                "fetch_k": fetch_k,          
                "lambda_mult": 0.6, 
            })
            logger.debug(f"MMR search optimized: k={top_k}, fetch_k={fetch_k}, lambda_mult=0.6")
            
        elif search_method == "similarity_score_threshold":
            base_params["search_kwargs"].update({
                "score_threshold": max(threshold, 0.7),  
                "k": min(top_k * 2, 10)  
            })
            
        return base_params
    
    def _build_filter_expression(self, metadata_filter: Dict[str, Any]) -> str:
        """Build Milvus filter expression from metadata dictionary"""
        expressions = []
        
        for key, value in metadata_filter.items():
            if isinstance(value, str):
                expressions.append(f'{key} == "{value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f'{key} == {value}')
            elif isinstance(value, list):
                if all(isinstance(v, str) for v in value):
                    value_str = '", "'.join(value)
                    expressions.append(f'{key} in ["{value_str}"]')
                else:
                    value_str = ', '.join(str(v) for v in value)
                    expressions.append(f'{key} in [{value_str}]')
        
        return " and ".join(expressions) if expressions else ""
    
    def _group_results_by_document(self, docs: List[Document]) -> Dict[str, List[Document]]:
        """
        Group search results by source document.
        Essential for document-level context assembly.
        """
        doc_groups = defaultdict(list)
        
        for doc in docs:
            doc_id = doc.metadata.get("doc_id", "unknown")
            doc_groups[doc_id].append(doc)
        
        for doc_id in doc_groups:
            doc_groups[doc_id].sort(key=lambda x: x.metadata.get("chunk_index", 0))
        
        return dict(doc_groups)
    
    def _format_search_results(
        self, 
        raw_docs: List[Document], 
        grouped_docs: Dict[str, List[Document]]
    ) -> List[Dict[str, Any]]:
        """Format search results with document-level information"""
        formatted_results = []
        
        for doc in raw_docs:
            result = {
                "content": doc.page_content,
                "metadata": dict(doc.metadata),
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "source": doc.metadata.get("source", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def _track_performance(self, metrics: Dict[str, Any]):
        """Track search performance metrics"""
        self._performance_metrics.append(metrics)
        
        if len(self._performance_metrics) > 100:
            self._performance_metrics = self._performance_metrics[-100:]
    
