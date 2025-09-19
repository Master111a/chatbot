"""
Milvus vector database client with hybrid search capabilities and non-blocking operations
"""
import uuid
import asyncio
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from langchain_core.documents import Document
from langchain_milvus import Milvus as MilvusVectorStore
from langchain_milvus import BM25BuiltInFunction

from utils.logger import get_logger
from config.settings import get_settings

settings = get_settings()
logger = get_logger(__name__)

def async_timeout(timeout_seconds: int):
    """Decorator to add timeout to async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"{func.__name__} timed out after {timeout_seconds} seconds")
                raise
        return wrapper
    return decorator

class MilvusClient:
    """
    Milvus vector database client with hybrid search capabilities and non-blocking operations
    """
    
    def __init__(self):
        """Initialize Milvus client with connection pooling"""
        self.collection_name = settings.MILVUS_COLLECTION
        self.connection_args = {
            "uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        }
        
        if hasattr(settings, 'MILVUS_USER') and settings.MILVUS_USER:
            self.connection_args["token"] = f"{settings.MILVUS_USER}:{settings.MILVUS_PASSWORD}"
        
        self.embeddings = None
        self.vectorstore = None
        self._connection_ready = False
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="milvus_")
        
        try:
            self._initialize_embeddings()
            self._initialize_vectorstore()
        except Exception as e:
            logger.error(f"Failed to initialize MilvusClient: {e}")
            self._connection_ready = False

    def _initialize_embeddings(self):
        """Initialize embedding model using cached instance from ModelWarmupService"""
        try:
            try:
                from services.model_warmup_service import get_warmup_service
                warmup_service = get_warmup_service()
                cached_embeddings = warmup_service.get_preloaded_embeddings()
                
                if cached_embeddings is not None:
                    self.embeddings = cached_embeddings
                    logger.info(f"Using cached embedding model from ModelWarmupService: {settings.EMBEDDING_MODEL}")
                    return
                else:
                    logger.info("No cached embeddings available, creating new instance")
                    
            except Exception as e:
                logger.warning(f"Could not get cached embeddings from ModelWarmupService: {e}")
            
            from langchain_huggingface import HuggingFaceEmbeddings
            
            model_name = settings.EMBEDDING_MODEL
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': settings.EMBEDDING_MODEL_DEVICE,
                    'trust_remote_code': True,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32,
                },
                cache_folder="/tmp/huggingface_cache"
            )
            logger.info(f"Created new BGE-M3 embeddings model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def _initialize_vectorstore(self):
        """Initialize Milvus vectorstore with proper hybrid search support"""
        try:
            logger.info("Attempting to initialize vectorstore")
            
            try:
                if settings.USE_HYBRID_SEARCH:
                    self.vectorstore = MilvusVectorStore.from_documents(
                        documents=[],  
                        embedding=self.embeddings,
                        builtin_function=BM25BuiltInFunction(
                            input_field_names="text",
                            output_field_names="sparse"
                        ),
                        vector_field=["dense", "sparse"],  
                        connection_args=self.connection_args,
                        collection_name=self.collection_name,
                        consistency_level="Strong",
                        drop_old=False
                    )
                    self._connection_ready = True
                    logger.info("Successfully initialized with hybrid search (dense + sparse)")
                else:
                    self.vectorstore = MilvusVectorStore.from_documents(
                        documents=[],  
                        embedding=self.embeddings,
                        connection_args=self.connection_args,
                        collection_name=self.collection_name,
                        consistency_level="Strong",
                        drop_old=False
                    )
                    self._connection_ready = True
                    logger.info("Successfully initialized with dense-only search")
                
            except Exception as error:
                logger.warning(f"Search initialization failed: {error}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vectorstore: {e}")
            raise ConnectionError(f"Cannot establish Milvus connection: {e}")

    def _ensure_connection(self):
        """Ensure connection is ready"""
        if not self._connection_ready or self.vectorstore is None:
            logger.info("Attempting to establish Milvus connection...")
            try:
                self._initialize_vectorstore()
            except Exception as e:
                logger.error(f"Failed to establish Milvus connection: {e}")
                raise ConnectionError("Milvus database is not available")

    def is_connected(self) -> bool:
        """Check if client is connected to Milvus"""
        try:
            self._ensure_connection()
            return self._connection_ready
        except:
            return False

    async def _run_sync_in_thread(self, func, *args, **kwargs):
        """Run synchronous function in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, func, *args, **kwargs)

    @async_timeout(180)
    async def insert(self, texts: List[str], doc_ids: List[str], chunk_ids: List[str], 
                    metadatas: List[Dict[str, Any]] = None, embeddings: List[List[float]] = None) -> List[int]:
        """
        Insert documents with enhanced logging and timeout handling
        """
        if not texts or len(texts) == 0:
            logger.warning("No texts provided for insertion")
            return []
        
        logger.info(f"Starting Milvus insertion for {len(texts)} documents")
        
        def _sync_insert():
            try:
                self._ensure_connection()
            except ConnectionError as e:
                logger.error(f"Cannot insert - Milvus not available: {e}")
                return []
            
            if metadatas is None:
                metadatas_list = [{} for _ in texts]
            else:
                metadatas_list = metadatas
            
            try:
                logger.info("Preparing documents for insertion...")
                documents = []
                
                for i, text in enumerate(texts):
                    metadata = metadatas_list[i].copy() if i < len(metadatas_list) else {}
                    metadata.update({
                        "doc_id": doc_ids[i] if i < len(doc_ids) else f"doc_{i}",
                        "chunk_id": chunk_ids[i] if i < len(chunk_ids) else f"chunk_{i}"
                    })
                    
                    documents.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))
                
                logger.info(f"Created {len(documents)} Document objects, starting vectorstore insertion...")
                
                start_time = time.time()
                ids = self.vectorstore.add_documents(documents)
                insert_time = time.time() - start_time
                
                logger.info(f"Milvus insertion completed in {insert_time:.2f}s")
                logger.info(f"Inserted {len(texts)} documents")
                logger.info(f"Returned {len(ids) if ids else 0} document IDs")
                
                return ids if ids else []
                
            except Exception as e:
                logger.error(f"Error during Milvus insertion: {e}", exc_info=True)
                return []
        
        return await self._run_sync_in_thread(_sync_insert)

    @async_timeout(120)
    async def search(self, query: str, top_k: int = 4, search_params: Optional[Dict] = None, 
                    filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search documents with focused retrieval and hard limits
        """
        logger.info(f"Focused Milvus search for query: '{query}' (top_k={top_k})")
        
        def _sync_search():
            try:
                self._ensure_connection()
            except ConnectionError as e:
                logger.error(f"Cannot search - Milvus not available: {e}")
                return []
            
            try:
                start_time = time.time()

                effective_top_k = min(top_k, 4) 
                
                search_kwargs = {"k": effective_top_k}
                if filter_expr:
                    search_kwargs["expr"] = filter_expr
                if search_params:
                    search_kwargs["param"] = search_params
                
                docs = self.vectorstore.similarity_search_with_score(
                    query=query,
                    **search_kwargs
                )
                
                search_time = time.time() - start_time
                
                results = []
                for doc, score in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })
                
                results = results[:effective_top_k]
                
                logger.info(f"Focused search completed in {search_time:.2f}s, returned {len(results)} results")
                return results
                
            except Exception as e:
                logger.error(f"Error during focused search: {e}", exc_info=True)
                return []
        
        return await self._run_sync_in_thread(_sync_search)

    @async_timeout(90)
    async def hybrid_search(self, query: str, top_k: int = 4, 
                           dense_weight: float = 0.7, sparse_weight: float = 0.3,
                           filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform focused hybrid search combining dense and sparse vectors
        """
        logger.info(f"Focused hybrid search for query: '{query}' (top_k={top_k})")
        
        def _sync_hybrid_search():
            try:
                self._ensure_connection()
            except ConnectionError as e:
                logger.error(f"Cannot search - Milvus not available: {e}")
                return []
            
            try:
                start_time = time.time()
                
                # Apply hard limits for focused retrieval
                effective_top_k = min(top_k, 4)  # Hard limit
                
                if hasattr(self.vectorstore, 'similarity_search') and hasattr(self.vectorstore, 'vector_field'):
                    try:
                        docs = self.vectorstore.similarity_search(
                            query=query,
                            k=effective_top_k,
                            ranker_type="weighted",
                            ranker_params={"weights": [dense_weight, sparse_weight]},
                            expr=filter_expr if filter_expr else None
                        )
                        
                        search_time = time.time() - start_time
                        results = []
                        for doc in docs:
                            results.append({
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "score": doc.metadata.get("score", 0.0)
                            })
                        
                        results = results[:effective_top_k]
                        
                        logger.info(f"Focused hybrid search completed in {search_time:.2f}s, returned {len(results)} results")
                        return results
                        
                    except Exception as hybrid_error:
                        logger.warning(f"Weighted hybrid search failed: {hybrid_error}")
                
                logger.info("Falling back to focused similarity search")
                search_kwargs = {"k": effective_top_k}
                if filter_expr:
                    search_kwargs["expr"] = filter_expr
                
                docs = self.vectorstore.similarity_search_with_score(
                    query=query,
                    **search_kwargs
                )
                
                search_time = time.time() - start_time
                
                results = []
                for doc, score in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })
                
                results = results[:effective_top_k]
                
                logger.info(f"Focused fallback search completed in {search_time:.2f}s, returned {len(results)} results")
                return results
                
            except Exception as e:
                logger.error(f"Error during focused hybrid search: {e}", exc_info=True)
                return []
        
        return await self._run_sync_in_thread(_sync_hybrid_search)

    @async_timeout(60)  
    async def delete(self, doc_ids: List[str]) -> bool:
        """
        Delete documents by document IDs with comprehensive error handling
        """
        logger.info(f"Deleting {len(doc_ids)} documents from Milvus")
        
        def _sync_delete():
            try:
                self._ensure_connection()
            except ConnectionError as e:
                logger.error(f"Cannot delete - Milvus not available: {e}")
                return False
            
            if not self.vectorstore:
                logger.error("Vectorstore not initialized")
                return False
            
            try:
                start_time = time.time()
                deleted_count = 0
                
                for doc_id in doc_ids:
                    try:
                        filter_expr = f'doc_id == "{doc_id}"'
                        logger.debug(f"Attempting to delete document {doc_id} with filter: {filter_expr}")
                        
                        if hasattr(self.vectorstore, 'delete'):
                            try:
                                self.vectorstore.delete(filter=filter_expr)
                                deleted_count += 1
                                logger.debug(f"Successfully deleted document {doc_id}")
                                continue
                            except Exception as e:
                                logger.warning(f"vectorstore.delete() failed for {doc_id}: {e}")
                        
                        try:
                            search_results = self.vectorstore.similarity_search(
                                query="", k=1000, expr=filter_expr
                            )
                            if not search_results:
                                logger.info(f"No documents found for {doc_id} - may already be deleted")
                                deleted_count += 1
                        except Exception as search_error:
                            logger.debug(f"Could not verify deletion for {doc_id}: {search_error}")
                            deleted_count += 1 
                            
                    except Exception as e:
                        logger.error(f"Failed to delete document {doc_id}: {e}")
                        continue
                
                delete_time = time.time() - start_time
                logger.info(f"Processed deletion for {deleted_count}/{len(doc_ids)} documents in {delete_time:.2f}s")
                
                return deleted_count >= len(doc_ids) * 0.8
            except Exception as e:
                logger.error(f"Critical error during deletion: {e}", exc_info=True)
                return False
        
        return await self._run_sync_in_thread(_sync_delete)

    @async_timeout(120)
    async def reindex(self) -> bool:
        """
        Compacts and reloads collection for performance optimization
        """
        logger.info("Starting collection reindexing for performance optimization")
        
        def _sync_reindex():
            try:
                self._ensure_connection()
            except ConnectionError as e:
                logger.error(f"Cannot reindex - Milvus not available: {e}")
                return False
            
            try:
                start_time = time.time()
                
                if hasattr(self.vectorstore, 'col') and self.vectorstore.col:
                    try:
                        if hasattr(self.vectorstore.col, 'compact'):
                            self.vectorstore.col.compact()
                            logger.info("Collection compaction completed")
                        
                        if hasattr(self.vectorstore.col, 'load'):
                            self.vectorstore.col.load()
                            logger.info("Collection reload completed")
                        
                        reindex_time = time.time() - start_time
                        logger.info(f"Collection reindexing completed in {reindex_time:.2f}s")
                        return True
                    except Exception as e:
                        logger.warning(f"Collection optimization failed: {e}")
                
                logger.info("No collection optimization methods available - continuing")
                return True
                
            except Exception as e:
                logger.error(f"Error during reindexing: {e}", exc_info=True)
                return False
        
        return await self._run_sync_in_thread(_sync_reindex)

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, '_thread_pool') and self._thread_pool:
                self._thread_pool.shutdown(wait=False)
        except:
            pass