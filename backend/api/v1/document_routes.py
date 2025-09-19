import time
import uuid
import os
import urllib.parse
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Path, Body
from fastapi.responses import Response, JSONResponse, RedirectResponse
from config.settings import get_settings
from services.document_service import document_service
from auth.otp_manager import verify_admin_headers
from models import DocumentUploadResponse, DocumentMetadataResponse
from utils.logger import get_logger
from utils import now

logger = get_logger(__name__)

router = APIRouter()
settings = get_settings()



@router.post("/upload", response_model=DocumentUploadResponse, summary="Upload new document")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    auth_data: tuple[str, str] = Depends(verify_admin_headers)
):
    """
    Upload a new document for the RAG system.
    Requires valid OTP and User-ID in headers with admin privileges.
    
    - Supports multiple formats (PDF, DOCX, TXT, XLSX, CSV)
    - Automatically splits document into chunks
    - Creates embedding vectors for each chunk
    - Requires valid OTP in header and User-ID with is_superuser=True
    """
    otp, user_id = auth_data
    
    start_time = time.time()
    
    try:
        filename = file.filename
        content_type = file.content_type
        file_bytes = await file.read()
        
        file_size = len(file_bytes)
        max_file_size = settings.MAX_FILE_SIZE
        if file_size > max_file_size:
            max_mb = max_file_size / (1024 * 1024)
            current_mb = file_size / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the {max_mb:.0f}MB limit (Current size: {current_mb:.2f}MB)"
            )
        
        supported_types = settings.ALLOWED_FILE_TYPES
        
        if content_type not in supported_types:
            type_names = []
            for mime_type in supported_types:
                if "pdf" in mime_type:
                    type_names.append("PDF")
                elif "wordprocessingml" in mime_type:
                    type_names.append("DOCX")
                elif "text/plain" in mime_type:
                    type_names.append("TXT")
                elif "spreadsheetml" in mime_type:
                    type_names.append("XLSX")
                elif "text/csv" in mime_type:
                    type_names.append("CSV")
                elif "presentationml" in mime_type:
                    type_names.append("PPTX")
                elif "powerpoint" in mime_type:
                    type_names.append("PPT")
            
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {content_type}. Supported types: {', '.join(set(type_names))}"
            )
        
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        temp_file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{filename}")
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)
        
        result = await document_service.process_document(
            file_path=temp_file_path,
            file_name=filename,
            file_type=content_type,
            title=title or os.path.splitext(filename)[0],
            description=description,
            language=language,
            user_id=user_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        process_time = time.time() - start_time
        logger.info(f"Document upload completed in {process_time:.2f}s - document_id: {result.get('document_id')} by user {user_id}")
        
        return {
            "document_id": result.get("document_id"),
            "title": result.get("title"),
            "file_name": filename,
            "file_type": content_type,
            "chunks_count": result.get("chunks_count"),
            "upload_timestamp": now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        try:
            if 'temp_file_path' in locals():
                os.remove(temp_file_path)
        except:
            pass
        
        process_time = time.time() - start_time
        logger.error(f"Error uploading document: {str(e)} (after {process_time:.2f}s)")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )


@router.post("/batch-upload")
async def batch_upload_documents(
    files: List[UploadFile] = File(..., description="Multiple files to upload"),
    description: Optional[str] = Form(None),
    language: Optional[str] = Form("vi"),
    chunk_size: Optional[int] = Form(512),
    chunk_overlap: Optional[int] = Form(128),
    auth_data: tuple[str, str] = Depends(verify_admin_headers)
):
    """
    Upload multiple documents in batch with strict validation.
    If any file fails validation, the entire batch is rejected.
    
    Args:
        files: List of files to upload
        description: Optional description for all documents
        language: Language code for processing
        chunk_size: Chunk size for text splitting
        chunk_overlap: Overlap between chunks
        auth_data: Authentication data
        
    Returns:
        Batch upload results with all document IDs
    """
    otp, user_id = auth_data
    
    logger.info(f"Batch upload started by user_id: {user_id} (type: {type(user_id)}, length: {len(user_id)})")
    
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided for batch upload"
        )
    
    if len(files) > 50: 
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 50 files allowed per batch."
        )
    
    logger.info(f"Starting batch validation for {len(files)} files")
    validation_results = []
    total_size = 0
    
    for i, file in enumerate(files):
        try:
            file_size = 0
            content = await file.read()
            file_size = len(content)
            await file.seek(0)  
            
            if file_size == 0:
                raise ValueError(f"File {file.filename} is empty")
            
            if file_size > 100 * 1024 * 1024: 
                raise ValueError(f"File {file.filename} exceeds 100MB limit")
            
            total_size += file_size
            
            file_type = file.content_type or "application/octet-stream"
            file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ""
            
            supported_types = {
                ".pdf": ["application/pdf"],
                ".doc": ["application/msword"],
                ".docx": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
                ".txt": ["text/plain"],
                ".md": ["text/markdown", "text/plain"],
                ".csv": ["text/csv", "application/csv"],
                ".xlsx": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
                ".xls": ["application/vnd.ms-excel"],
                ".pptx": ["application/vnd.openxmlformats-officedocument.presentationml.presentation"],
                ".ppt": ["application/vnd.ms-powerpoint"],
                ".json": ["application/json", "text/plain"],
                ".xml": ["application/xml", "text/xml"],
                ".html": ["text/html"],
                ".htm": ["text/html"]
            }
            
            if file_extension not in supported_types:
                raise ValueError(f"File {file.filename} has unsupported extension: {file_extension}")
            
            expected_types = supported_types[file_extension]
            if file_type not in expected_types:
                logger.warning(f"File {file.filename}: content-type {file_type} doesn't match extension {file_extension}, but proceeding")
            
            if not file.filename or len(file.filename.strip()) == 0:
                raise ValueError(f"File at position {i+1} has no filename")
            
            if len(file.filename) > 255:
                raise ValueError(f"File {file.filename} has filename too long (max 255 characters)")
            
            duplicate_count = sum(1 for f in files if f.filename == file.filename)
            if duplicate_count > 1:
                raise ValueError(f"Duplicate filename in batch: {file.filename}")
            
            validation_results.append({
                "filename": file.filename,
                "size": file_size,
                "type": file_type,
                "extension": file_extension,
                "valid": True
            })
            
        except Exception as e:
            logger.error(f"Validation failed for file {file.filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed for file '{file.filename}': {str(e)}"
            )
    
    if total_size > 500 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Total batch size ({total_size / 1024 / 1024:.1f}MB) exceeds 500MB limit"
        )
    
    logger.info(f"All {len(files)} files passed validation. Total size: {total_size / 1024 / 1024:.1f}MB")
    
    successful_uploads = []
    failed_uploads = []
    
    for i, file in enumerate(files):
        temp_file_path = None
        try:
            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}")
            
            filename = file.filename
            content_type = file.content_type
            file_bytes = await file.read()
            
            upload_dir = os.path.join(os.getcwd(), "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            temp_file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{filename}")
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
            
            result = await document_service.process_document(
                file_path=temp_file_path,
                file_name=filename,
                file_type=content_type,
                title=os.path.splitext(filename)[0],
                description=description,
                language=language,
                user_id=user_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            try:
                os.remove(temp_file_path)
            except:
                pass
            
            successful_uploads.append({
                "filename": filename,
                "document_id": result["document_id"],
                "title": result.get("title"),
                "chunks_count": result.get("chunks_count", 0),
                "file_size": validation_results[i]["size"]
            })
            
            logger.info(f"Successfully processed {filename} -> {result['document_id']}")
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            failed_uploads.append({
                "filename": file.filename,
                "error": str(e)
            })
            
            try:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except:
                pass
    
    response_data = {
        "message": f"Batch upload completed: {len(successful_uploads)} successful, {len(failed_uploads)} failed",
        "total_files": len(files),
        "successful_count": len(successful_uploads),
        "failed_count": len(failed_uploads),
        "successful_uploads": successful_uploads,
        "failed_uploads": failed_uploads,
        "total_size_mb": round(total_size / 1024 / 1024, 2)
    }
    
    if failed_uploads:
        logger.warning(f"Batch upload had {len(failed_uploads)} failures")
        return JSONResponse(
            status_code=207,
            content=response_data
        )
    else:
        logger.info(f"Batch upload fully successful: {len(successful_uploads)} files processed")
        return response_data 


@router.get("/list", summary="Get document list")
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Results per page"),
    search: Optional[str] = Query(None, description="Search by title or content"),
    user_id: Optional[str] = Query(None, description="Filter by user ID")
):
    """
    Get list of documents in the system.
    
    - Supports pagination
    - Search by title or content
    - Filter by user if user_id provided
    - No authentication required
    """
    try:
        documents, total = await document_service.list_documents(
            page=page,
            limit=limit,
            search=search,
            user_id=user_id
        )
        
        return {
            "documents": documents,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit
        }
        
    except Exception as e:
        logger.error(f"Error retrieving document list: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document list: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentMetadataResponse, summary="Get document information")
async def get_document(
    document_id: str = Path(..., description="Document ID")
):
    """
    Get detailed information about a document.
    
    - Document metadata
    - Chunk information
    - Current status
    - No authentication required
    """
    try:
        result = await document_service.get_document_metadata(document_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found with ID: {document_id}"
            )
        
        chunks_count = await document_service.get_chunks_count(document_id)
        result["chunks_count"] = chunks_count
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document information: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document information: {str(e)}"
        )
    

@router.delete("/{document_id}", summary="Delete document")
async def delete_document(
    document_id: str = Path(..., description="Document ID"),
    auth_data: tuple[str, str] = Depends(verify_admin_headers)
):
    """
    Delete document and all related chunks.
    Requires valid OTP and User-ID in headers with admin privileges.
    
    - Removes document from database
    - Removes chunks from vector store
    - Removes original file if it still exists
    - Requires valid OTP in header and User-ID with is_superuser=True
    """
    otp, user_id = auth_data

    try:
        result = await document_service.delete_document(document_id, user_id=user_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found with ID: {document_id}"
            )
        
        logger.info(f"Document deleted: {document_id} by user {user_id}")
        
        return {
            "document_id": document_id,
            "status": "deleted",
            "timestamp": now().isoformat(),
            "deleted_by": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/download/{document_id}", summary="Download document file")
async def download_document(
    document_id: str = Path(..., description="Document ID")
):
    """
    Download the original document file.
    """
    try:
        file_result = await document_service.download_document_file(document_id)
        
        if not file_result:
            raise HTTPException(
                status_code=404,
                detail=f"Document file not found for ID: {document_id}"
            )
        
        file_content = file_result.get("content")
        filename = file_result.get("filename", f"document_{document_id}")
        content_type = file_result.get("content_type", "application/octet-stream")
        
        if not file_content:
            raise HTTPException(
                status_code=404,
                detail="Document file content not available"
            )
        
        if not isinstance(file_content, bytes):
            logger.error(f"Invalid file content type: {type(file_content)} for document {document_id}")
            raise HTTPException(
                status_code=500,
                detail="Invalid file content format"
            )
        
        safe_filename = urllib.parse.quote(filename.encode('utf-8'))
        
        return Response(
            content=file_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}",
                "Content-Length": str(len(file_content)),
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document {document_id}: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading document: {str(e)}"
        )
