"""
Document Service - Fixed with upload rollback and delete reindexing
Uses lazy initialization to prevent startup failures when Milvus is not ready
"""

import os
import asyncio
import uuid
import time
import json
from typing import Dict, List, Any, Optional
from utils import now
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from models.document import Document, DocumentChunk
from db.pg_manager import PostgresManager, PostgresDocumentStore
from utils.file_utils import file_storage, is_allowed_file_type
from utils.file_processor import FileProcessor
from utils.logger import get_logger
from config.settings import get_settings

settings = get_settings()
logger = get_logger(__name__)

class DocumentService:
    """
    Service for document processing, storage, and retrieval with MinIO + Milvus integration
    Fixed with proper rollback mechanisms and reindexing
    """
    
    def __init__(self):
        """Initialize the document service with lazy Milvus connection"""
        self.document_store = PostgresDocumentStore()
        self.pg_manager = PostgresManager()
        self.file_processor = FileProcessor()
        
        self._milvus_client = None
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="milvus_")
        
        self._initialized = False
        self._initialization_task = None

    @property
    def milvus_client(self):
        """Lazy property for MilvusClient - creates only when needed"""
        if self._milvus_client is None:
            try:
                from vector_store.milvus_client import MilvusClient
                self._milvus_client = MilvusClient()
                logger.info("MilvusClient initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MilvusClient: {e}")
                
        return self._milvus_client

    async def ensure_initialized(self):
        """Ensure the service is initialized, call before using the service"""
        if not self._initialized and self._initialization_task is None:
            self._initialization_task = asyncio.create_task(self._initialize_services())
        
        if self._initialization_task and not self._initialization_task.done():
            await self._initialization_task

    async def _initialize_services(self):
        """Initialize all services"""
        await self._initialize_db()
        
        try:
            from services.cleanup_service import cleanup_service
            await cleanup_service.start_background_cleanup()
            logger.info("Background cleanup service started")
        except Exception as e:
            logger.warning(f"Failed to start background cleanup service: {e}")
        
        self._initialized = True

    async def _initialize_db(self):
        """Initialize database"""
        try:
            await self.pg_manager.initialize()
            logger.info("PostgreSQL initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise

    @asynccontextmanager
    async def _rollback_context(self):
        """
        Context manager for rollback operations during upload
        """
        rollback_items = []
        
        try:
            yield rollback_items
        except Exception as e:
            logger.error(f"Error occurred, starting rollback: {e}")
            
            for item in reversed(rollback_items):
                try:
                    if item['type'] == 'document':
                        await self.document_store.delete_document(item['document_id'])
                        logger.info(f"Rollback: Deleted document {item['document_id']}")
                    
                    elif item['type'] == 'chunks':
                        await self.document_store.delete_chunks(item['document_id'])
                        logger.info(f"Rollback: Deleted chunks for document {item['document_id']}")
                    
                    elif item['type'] == 'minio_file':
                        await file_storage.delete_file(item['object_name'])
                        logger.info(f"Rollback: Deleted MinIO file {item['object_name']}")
                    
                    elif item['type'] == 'local_file':
                        if os.path.exists(item['file_path']):
                            os.remove(item['file_path'])
                            logger.info(f"Rollback: Deleted local file {item['file_path']}")
                    
                    elif item['type'] == 'vectors':
                        if self.milvus_client and self.milvus_client.is_connected():
                            await self.milvus_client.delete(doc_ids=[item['document_id']])
                            logger.info(f"Rollback: Deleted vectors for document {item['document_id']}")
                
                except Exception as rollback_error:
                    logger.error(f"Rollback failed for {item}: {rollback_error}")
            
            raise

    async def process_document(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        title: str,
        description: Optional[str] = None,
        language: Optional[str] = None,
        user_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process and store a document with complete rollback on failure
        
        Args:
            file_path: Path to the uploaded file
            file_name: Original file name
            file_type: MIME type
            title: Document title
            description: Optional description
            language: Document language
            user_id: ID of the uploader
            chunk_size: Text chunk size (optional)
            chunk_overlap: Chunk overlap size (optional)
            
        Returns:
            Result of the document processing
        """
        await self.ensure_initialized()
        
        if not is_allowed_file_type(file_type):
            raise ValueError(f"File type not supported: {file_name}")
        
        document_id = str(uuid.uuid4())
        
        async with self._rollback_context() as rollback_items:
            try:
                document = Document.create_new(
                    title=title,
                    file_name=file_name,
                    file_type=file_type,
                    file_size=os.path.getsize(file_path),
                    user_id=user_id,
                    language=language,
                    description=description,
                    chunk_size=chunk_size or 1500,
                    chunk_overlap=chunk_overlap or 200
                )
                
                logger.info(f"Starting document processing for: {file_name}")
                
                document_id = await self.document_store.create_document(document)
                rollback_items.append({'type': 'document', 'document_id': document_id})
                
                chunks = await self.file_processor.process_file(
                    file_path=file_path,
                    file_name=file_name,
                    doc_id=document_id,
                    metadata={
                        "title": title,
                        "description": description,
                        "language": language,
                        "user_id": user_id
                    }
                )
                
                if not chunks:
                    raise ValueError("No content could be extracted from the document")
                
                logger.info(f"Extracted {len(chunks)} chunks from {file_name}")
                
                stored_chunks = await self._store_chunks_non_blocking(document_id, chunks)
                rollback_items.append({'type': 'chunks', 'document_id': document_id})
                
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                file_info = await file_storage.save_file(
                    file_content=file_content,
                    file_name=file_name,
                    content_type=file_type,
                    user_id=user_id,
                    metadata={
                        "document_id": document_id,
                        "title": title,
                        "language": language or "vi"
                    }
                )
                rollback_items.append({'type': 'minio_file', 'object_name': file_info['object_name']})
                
                document.file_path = file_info['object_name']
                document.updated_at = now().isoformat()
                await self.document_store.update_document(document)
                
                document.status = "completed"
                document.chunks_count = len(stored_chunks)
                document.updated_at = now().isoformat()
                await self.document_store.update_document(document)
                
                logger.info(f"Document {document_id} processing completed with {len(stored_chunks)} chunks")
                
                return {
                    "document_id": document_id,
                    "title": title,
                    "chunks_count": len(stored_chunks),
                    "status": "completed"
                }
                
            except Exception as e:
                logger.error(f"Failed to process document {file_name}: {e}")
                raise

    async def _store_chunks_non_blocking(self, document_id: str, langchain_documents: List) -> List[DocumentChunk]:
        """
        Store chunks with non-blocking vector indexing to prevent timeout issues
        
        Args:
            document_id: Document ID
            langchain_documents: List of LangChain Document objects
            
        Returns:
            List of stored chunks
        """
        document_name = await self._get_document_name(document_id)
        stored_chunks = []
        
        texts = []
        doc_ids = []
        chunk_ids = []
        metadatas = []
        
        logger.info(f"Starting chunk storage for document {document_id} with {len(langchain_documents)} chunks")
        
        for i, langchain_doc in enumerate(langchain_documents):
            chunk_id = str(uuid.uuid4())
            chunk_text = langchain_doc.page_content
            
            prev_chunk_id = str(stored_chunks[-1].chunk_id) if stored_chunks else None
            next_chunk_id = str(uuid.uuid4()) if i < len(langchain_documents) - 1 else None
            
            chunk_metadata = {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": i,
                "total_chunks": len(langchain_documents),
                "created_at": now().isoformat()
            }
            chunk_metadata.update(langchain_doc.metadata if langchain_doc.metadata else {})
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content_path=f"chunks/{document_id}/{chunk_id}.txt",
                content=chunk_text,
                metadata=chunk_metadata,
                index=i,
                chunk_size=len(chunk_text),
                created_at=now().isoformat(),
                prev_chunk_id=prev_chunk_id,
                next_chunk_id=next_chunk_id
            )
            
            await self.document_store.add_chunk(chunk)
            stored_chunks.append(chunk)
            
            texts.append(chunk_text)
            doc_ids.append(document_id)
            chunk_ids.append(chunk_id)
            metadatas.append(chunk_metadata)
        
        logger.info(f"Stored {len(stored_chunks)} chunks in PostgreSQL for document {document_id}")
        
        asyncio.create_task(self._index_vectors_background(document_id, texts, doc_ids, chunk_ids, metadatas))
        
        return stored_chunks

    async def _index_vectors_background(self, document_id: str, texts: List[str], doc_ids: List[str], chunk_ids: List[str], metadatas: List[Dict[str, Any]]):
        """
        Index vectors in Milvus in background thread
        """
        logger.info(f"Starting background vector indexing for document {document_id}")
        
        try:
            if not self.milvus_client or not self.milvus_client.is_connected():
                logger.warning(f"Milvus not available for document {document_id}, marking as partial")
                document = await self.document_store.get_document(document_id)
                if document:
                    document.status = "partial"
                    document.updated_at = now().isoformat()
                    await self.document_store.update_document(document)
                return
            
            try:
                result = await asyncio.wait_for(
                    self.milvus_client.insert(
                        texts=texts,
                        doc_ids=doc_ids,
                        chunk_ids=chunk_ids,
                        metadatas=metadatas
                    ),
                    timeout=300
                )
                
                if result:
                    document.status = "completed"
                    document.updated_at = now().isoformat()
                    await self.document_store.update_document(document)
                    logger.info(f"Vector indexing completed for document {document_id}")
                else:
                    document.status = "partial"
                    document.updated_at = now().isoformat()
                    await self.document_store.update_document(document)
                    
            except asyncio.TimeoutError:
                logger.warning(f"Vector indexing timed out for document {document_id}")
                try:
                    document = await self.document_store.get_document(document_id)
                    if document:
                        document.status = "partial" 
                        document.updated_at = now().isoformat()
                        await self.document_store.update_document(document)
                except Exception as status_error:
                    logger.warning(f"Failed to update document status after timeout: {status_error}")
                    
            except Exception as insert_error:
                logger.error(f"Vector insertion failed for document {document_id}: {insert_error}")
                try:
                    document = await self.document_store.get_document(document_id)
                    if document:
                        document.status = "partial"
                        document.updated_at = now().isoformat()
                        await self.document_store.update_document(document)
                except Exception as status_error:
                    logger.warning(f"Failed to update document status after error: {status_error}")
                    
        except Exception as e:
            logger.error(f"Critical error during background vector indexing for document {document_id}: {e}", exc_info=True)
            try:
                document = await self.document_store.get_document(document_id)
                if document:
                    document.status = "partial"
                    document.updated_at = now().isoformat()
                    await self.document_store.update_document(document)
            except Exception as status_error:
                logger.warning(f"Failed to update document status after critical error: {status_error}")

    async def _get_document_name(self, document_id: str) -> str:
        """
        Get the document file name for metadata
        
        Args:
            document_id: Document ID
            
        Returns:
            Document file name
        """
        try:
            document_id_str = str(document_id)
            
            document = await self.document_store.get_document(document_id_str)
            return document.file_name if document else "Unknown document"
        except Exception as e:
            logger.warning(f"Could not get document name for {document_id}: {e}")
            return "Unknown document"

    async def delete_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a document and all its associated chunks with improved error handling.
        Process: DB Chunks -> DB Document -> MinIO -> Milvus -> Reindex -> Success
        
        Args:
            document_id: The ID of the document to delete.
            user_id: The ID of the user performing the deletion.
            
        Returns:
            True if deletion is successful, False if not found or an error occurs.
        """
        await self.ensure_initialized()
        
        logger.info(f"Starting deletion process for document {document_id}")
        
        document = None
        chunks_deleted = False
        document_deleted = False
        
        try:
            document = await self.document_store.get_document(document_id)
            if not document:
                logger.warning(f"Document {document_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            return False
        
        try:
            async with self.document_store.pg_manager.pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM document_chunks 
                    WHERE document_id = $1
                    ORDER BY index ASC
                ''', document_id)
                
                if rows:
                    chunks_deleted = await self.document_store.delete_chunks(document_id)
                    logger.info(f"Deleted chunks from PostgreSQL for document {document_id}: {chunks_deleted}")
                else:
                    chunks_deleted = True
                    logger.info(f"No chunks found for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting chunks from database: {e}")
        
        if chunks_deleted:
            try:
                document_deleted = await self.document_store.delete_document(document_id)
                if document_deleted:
                    logger.info(f"Successfully deleted document record from PostgreSQL: {document_id}")
                else:
                    logger.warning(f"Failed to delete document record from PostgreSQL: {document_id}")
                    
            except Exception as e:
                logger.error(f"Error deleting document from database: {e}")
                return False
        else:
            logger.error(f"Cannot delete document {document_id} because chunks deletion failed")
            return False
        
        try:
            if document and hasattr(document, 'file_path') and document.file_path:
                file_delete_success = await file_storage.delete_file(document.file_path)
                if file_delete_success:
                    logger.info(f"Successfully deleted file from MinIO: {document.file_path}")
                else:
                    logger.warning(f"Failed to delete file from MinIO: {document.file_path}")
            else:
                logger.info(f"No file path found for document {document_id} - skipping MinIO deletion")
        except Exception as e:
            logger.error(f"Failed to delete file from MinIO: {e}")
        
        asyncio.create_task(self._delete_vectors_with_reindex_background(document_id))
        
        return document_deleted

    async def _delete_vectors_with_reindex_background(self, document_id: str):
        """
        Delete vectors from Milvus in background with reindexing
        """
        logger.info(f"Starting background vector deletion with reindex for document {document_id}")
        
        try:
            if self.milvus_client and self.milvus_client.is_connected():
                async def delete_with_timeout():
                    loop = asyncio.get_event_loop()
                    
                    def sync_delete():
                        return asyncio.run_coroutine_threadsafe(
                            self.milvus_client.delete(doc_ids=[document_id]), loop
                        ).result()
                    
                    return await loop.run_in_executor(self._thread_pool, sync_delete)
                
                delete_success = await asyncio.wait_for(delete_with_timeout(), timeout=60)
                
                if delete_success:
                    logger.info(f"Successfully deleted vectors for document {document_id}")
                    
                    await self._trigger_reindex_background()
                    
                else:
                    logger.warning(f"Failed to delete vectors for document {document_id}")
                    
            else:
                logger.warning(f"Milvus not available for vector deletion of document {document_id}")
                
        except asyncio.TimeoutError:
            logger.warning(f"Vector deletion timed out for document {document_id}")
        except Exception as e:
            logger.error(f"Error during vector deletion for document {document_id}: {e}")
    
    async def _trigger_reindex_background(self):
        """
        Trigger background reindexing after deletion for performance optimization
        Uses the improved reindex method with proper error handling
        """
        try:
                
            logger.info("Starting background reindexing after deletion")
            
            try:
                reindex_success = await asyncio.wait_for(
                    self.milvus_client.reindex(),
                    timeout=180
                )
                
                if reindex_success:
                    logger.info("Background reindexing completed successfully")
                else:
                    logger.warning("Background reindexing completed with warnings")
                    
            except asyncio.TimeoutError:
                logger.warning("Background reindexing timed out after 180 seconds")
            except Exception as e:
                logger.warning(f"Background reindexing encountered error: {e}")
                
        except Exception as e:
            logger.warning(f"Error during background reindexing setup: {e}")
                        
    async def list_documents(
        self,
        page: int = 1,
        limit: int = 10,
        search: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        List documents with pagination
        
        Args:
            page: Page number (1-based)
            limit: Number of items per page
            search: Search by title or content
            user_id: Optional user ID filter
            
        Returns:
            Tuple of (documents list, total count)
        """
        await self.ensure_initialized()
        
        try:
            documents, total_count = await self.document_store.list_documents(
                user_id=user_id,
                page=page,
                page_size=limit,
                status_filter=None
            )
            
            formatted_docs = []
            for doc in documents:
                chunks_count = await self.get_chunks_count(doc.id)
                doc_dict = {
                    "document_id": doc.id,
                    "file_name": doc.file_name,
                    "file_size": doc.file_size,
                    "chunks_count": chunks_count,
                    "processing_status": doc.processing_status,
                    "upload_time": doc.upload_time,
                    "user_id": doc.user_id,
                    "processing_details": doc.processing_details
                }
                formatted_docs.append(doc_dict)
            
            return formatted_docs, total_count
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return [], 0

    async def get_chunks_count(self, document_id: str) -> int:
        """
        Get count of chunks for a document
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of chunks
        """
        await self.ensure_initialized()
        
        try:
            async with self.document_store.pg_manager.pool.acquire() as conn:
                result = await conn.fetchval('''
                    SELECT COUNT(*) FROM document_chunks 
                    WHERE document_id = $1
                ''', document_id)
                return result or 0
        except Exception as e:
            logger.error(f"Error getting chunks count for document {document_id}: {e}")
            return 0

    async def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata and details
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata dictionary
        """
        await self.ensure_initialized()
        
        try:
            document = await self.document_store.get_document(document_id)
            if not document:
                return None
                
            return {
                "document_id": document.id,
                "file_name": document.file_name,
                "file_size": document.file_size,
                "upload_time": document.upload_time,
                "processing_status": document.processing_status,
                "processing_details": document.processing_details,
                "user_id": document.user_id,
                "file_path": document.file_path
            }
            
        except Exception as e:
            logger.error(f"Error getting document metadata {document_id}: {e}")
            return None

    async def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of document chunks
        """
        await self.ensure_initialized()
        
        try:
            async with self.document_store.pg_manager.pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM document_chunks 
                    WHERE document_id = $1
                    ORDER BY index ASC
                ''', document_id)
                
                chunks = []
                for row in rows:
                    chunk_dict = dict(row)
                    chunk_dict['metadata'] = json.loads(chunk_dict['metadata']) if chunk_dict['metadata'] else {}
                    
                    if 'chunk_id' in chunk_dict and chunk_dict['chunk_id']:
                        chunk_dict['chunk_id'] = str(chunk_dict['chunk_id'])
                    if 'document_id' in chunk_dict and chunk_dict['document_id']:
                        chunk_dict['document_id'] = str(chunk_dict['document_id'])
                    if 'prev_chunk_id' in chunk_dict and chunk_dict['prev_chunk_id']:
                        chunk_dict['prev_chunk_id'] = str(chunk_dict['prev_chunk_id'])
                    if 'next_chunk_id' in chunk_dict and chunk_dict['next_chunk_id']:
                        chunk_dict['next_chunk_id'] = str(chunk_dict['next_chunk_id'])
                    
                    chunks.append(DocumentChunk.from_dict(chunk_dict))
                    
                return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks for document {document_id}: {e}")
            return []

    async def batch_upload_documents(self, files_data: List[Dict[str, Any]], user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple documents in batch with individual rollback
        Enhanced with per-file rollback handling
        
        Args:
            files_data: List of file data dictionaries
            user_id: Optional user ID
            
        Returns:
            Batch processing results
        """
        await self.ensure_initialized()
        
        successful_uploads = []
        failed_uploads = []
        total_size = 0
        
        for file_data in files_data:
            try:
                file_path = file_data.get("file_path")
                original_filename = file_data.get("original_filename")
                file_size = file_data.get("file_size", 0)
                
                total_size += file_size
                
                if not file_path or not original_filename:
                    failed_uploads.append({
                        "filename": original_filename or "unknown",
                        "error": "Missing file path or filename"
                    })
                    continue
                
                document_id = await self.process_document(file_path, original_filename, user_id)
                
                successful_uploads.append({
                    "document_id": document_id,
                    "filename": original_filename,
                    "file_size": file_size
                })
                
                logger.info(f"Successfully processed file in batch: {original_filename}")
                
            except Exception as e:
                logger.error(f"Failed to process file in batch: {file_data.get('original_filename', 'unknown')}: {e}")
                failed_uploads.append({
                    "filename": file_data.get("original_filename", "unknown"),
                    "error": str(e)
                })
        
        response_data = {
            "message": f"Batch upload completed: {len(successful_uploads)} successful, {len(failed_uploads)} failed",
            "total_files": len(files_data),
            "successful_count": len(successful_uploads),
            "failed_count": len(failed_uploads),
            "successful_uploads": successful_uploads,
            "failed_uploads": failed_uploads,
            "total_size_mb": round(total_size / 1024 / 1024, 2)
        }
        
        return response_data

    async def download_document_file(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Download document file content from storage
        
        Args:
            document_id: Document ID to download
            
        Returns:
            Dict with file content, filename, and content_type or None if not found
        """
        await self.ensure_initialized()
        
        try:
            document = await self.document_store.get_document(document_id)
            if not document:
                logger.error(f"Document not found: {document_id}")
                return None
            
            if not document.file_path:
                logger.error(f"No file path found for document: {document_id}")
                return None
            
            file_result = await file_storage.get_file(document.file_path)
            if not file_result:
                logger.error(f"Could not retrieve file content for document: {document_id}")
                return None
            
            file_content, storage_content_type = file_result
            
            return {
                "content": file_content,
                "filename": document.file_name,
                "content_type": document.file_type or storage_content_type
            }
            
        except Exception as e:
            logger.error(f"Error downloading document file {document_id}: {e}")
            return None

document_service = DocumentService()