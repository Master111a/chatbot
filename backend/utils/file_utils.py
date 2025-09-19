from typing import Dict, Any, Optional, Tuple
import os
import uuid
import mimetypes
import io
import tempfile
from utils import now
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from utils.validate import sanitize_metadata
from datetime import timedelta
from urllib.parse import urlparse
from utils.logger import get_logger


logger = get_logger(__name__)


class FileStorage:
    """
    File storage service for handling files across local and MinIO storage.
    Manages file uploads, retrievals, and deletions with presigned URL support.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileStorage, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize file storage service."""
        if self.initialized:
            return
            
        from config.settings import get_settings
        self.settings = get_settings()
        
        self.minio_client = Minio(
            endpoint=self.settings.MINIO_ENDPOINT,
            access_key=self.settings.MINIO_ACCESS_KEY,
            secret_key=self.settings.MINIO_SECRET_KEY,
            secure=self.settings.MINIO_SECURE,
            region=self.settings.MINIO_REGION
        )
        
        self._ensure_bucket_exists()
        
        os.makedirs(self.settings.UPLOAD_DIR, exist_ok=True)
        
        self.initialized = True
        logger.info("FileStorage service initialized")
    
    def _ensure_bucket_exists(self):
        """Ensure that the storage bucket exists."""
        try:
            bucket_exists = self.minio_client.bucket_exists(self.settings.MINIO_BUCKET_NAME)
            if not bucket_exists:
                self.minio_client.make_bucket(self.settings.MINIO_BUCKET_NAME)
                logger.info(f"Bucket '{self.settings.MINIO_BUCKET_NAME}' created successfully")
        except S3Error as e:
            logger.error(f"Error checking/creating bucket: {e}")
    
    def generate_object_path(self, user_id: Optional[str], file_name: str) -> str:
        """
        Generate a unique path for storing an object in MinIO.
        
        Args:
            user_id: Optional user ID
            file_name: Original file name
            
        Returns:
            Unique object path
        """
        date_path = now().strftime("%Y/%m/%d")
        unique_id = str(uuid.uuid4())
        
        if user_id:
            return f"user_{user_id}/{date_path}/{unique_id}/{file_name}"
        else:
            return f"public/{date_path}/{unique_id}/{file_name}"
    
    async def save_file(
        self, 
        file_content: bytes, 
        file_name: str, 
        content_type: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Save file to MinIO and optionally to local storage.
        
        Args:
            file_content: File content as bytes
            file_name: File name
            content_type: File MIME type
            user_id: Optional user ID
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with file information
        """
        object_name = self.generate_object_path(user_id, file_name)
        
        try:
            file_size = len(file_content)
            file_stream = io.BytesIO(file_content)
            
            sanitized_metadata = sanitize_metadata(metadata)
            
            self.minio_client.put_object( 
                bucket_name=self.settings.MINIO_BUCKET_NAME,
                object_name=object_name,
                data=file_stream,
                length=file_size,
                content_type=content_type,
                metadata=sanitized_metadata
            )

            local_path = os.path.join(self.settings.UPLOAD_DIR, f"{uuid.uuid4()}_{file_name}")
            with open(local_path, "wb") as f:
                f.write(file_content)
            
            return {
                "file_name": file_name,
                "file_type": content_type,
                "file_size": file_size,
                "local_path": local_path,
                "object_name": object_name,
                "bucket_name": self.settings.MINIO_BUCKET_NAME,
                "user_id": user_id,
                "upload_timestamp": now().isoformat()
            }
            
        except S3Error as e:
            logger.error(f"Error saving file to MinIO: {e}")
            raise
    
    async def get_file(self, object_name: str) -> Tuple[bytes, str]:
        """
        Get file from MinIO.
        
        Args:
            object_name: Object name/path in MinIO
            
        Returns:
            Tuple of (file content, content type)
        """
        try:
            response = self.minio_client.get_object(
                bucket_name=self.settings.MINIO_BUCKET_NAME,
                object_name=object_name
            )
            
            file_content = response.read()
            content_type = response.headers.get('Content-Type', 'application/octet-stream')
            
            return file_content, content_type
            
        except S3Error as e:
            logger.error(f"Error retrieving file from MinIO: {e}")
            raise
        finally:
            if 'response' in locals():
                response.close()
                response.release_conn()
    
    async def delete_file(self, object_name: str) -> bool:
        """
        Delete file from MinIO.
        
        Args:
            object_name: Object name/path in MinIO
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            self.minio_client.remove_object(
                bucket_name=self.settings.MINIO_BUCKET_NAME,
                object_name=object_name
            )
            return True
        except S3Error as e:
            logger.error(f"Error deleting file from MinIO: {e}")
            return False
    
    async def get_download_url(self, object_name: str, expires: int = 3600) -> str:
        """
        Generate a presigned URL for downloading a file.
        MinIO client connects via internal endpoint but returns external URL for browser access.
        
        Args:
            object_name: Object name/path in MinIO
            expires: Expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL for download (accessible via external endpoint)
        """
        try:
            
            internal_presigned_url = self.minio_client.presigned_get_object(
                bucket_name=self.settings.MINIO_BUCKET_NAME,
                object_name=object_name,
                expires=timedelta(seconds=expires)
            )
            
            parsed_url = urlparse(internal_presigned_url)
            
            external_protocol = "https" if self.settings.MINIO_EXTERNAL_SECURE else "http"
            external_endpoint = self.settings.MINIO_EXTERNAL_ENDPOINT
            
            external_url = f"{external_protocol}://{external_endpoint}{parsed_url.path}"
            if parsed_url.query:
                external_url += f"?{parsed_url.query}"
            
            logger.info(f"Generated external MinIO URL: {external_protocol}://{external_endpoint}{parsed_url.path[:50]}...")
            
            return external_url
            
        except S3Error as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL: {e}")
            raise
    
    async def get_file_metadata(self, object_name: str) -> Dict[str, Any]:
        """
        Get file metadata from MinIO.
        
        Args:
            object_name: Object name/path in MinIO
            
        Returns:
            Dictionary with file metadata
        """
        try:
            stat = self.minio_client.stat_object(
                bucket_name=self.settings.MINIO_BUCKET_NAME,
                object_name=object_name
            )
            
            metadata = {
                "size": stat.size,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified.isoformat(),
                "etag": stat.etag
            }

            if hasattr(stat, 'metadata'):
                for key, value in stat.metadata.items():
                    if key.startswith('X-Amz-Meta-'):
                        clean_key = key[11:].lower()
                        metadata[clean_key] = value
            
            return metadata
        except S3Error as e:
            logger.error(f"Error getting file metadata: {e}")
            raise
    
    async def check_if_file_exists(self, object_name: str) -> bool:
        """
        Check if a file exists in MinIO.
        
        Args:
            object_name: Object name/path in MinIO
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.minio_client.stat_object(
                bucket_name=self.settings.MINIO_BUCKET_NAME,
                object_name=object_name
            )
            return True
        except S3Error:
            return False


file_storage = FileStorage()


def is_allowed_file_type(content_type: str) -> bool:
    """
    Check if a file type is allowed.
    
    Args:
        content_type: File MIME type
        
    Returns:
        True if allowed, False otherwise
    """
    from config.settings import get_settings
    settings = get_settings()
    return content_type in settings.ALLOWED_FILE_TYPES


def get_file_extension(file_name: str) -> str:
    """
    Get file extension from file name.
    
    Args:
        file_name: File name
        
    Returns:
        File extension (with dot)
    """
    return os.path.splitext(file_name)[1].lower()


def get_mime_type(file_path: str) -> str:
    """
    Get MIME type from file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing potentially unsafe characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    filename = os.path.basename(filename)
    
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    sanitized = ''.join(c if c in safe_chars else '_' for c in filename)
    
    max_length = 255
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext
    
    return sanitized


def create_temp_file(content: bytes, suffix: Optional[str] = None) -> str:
    """
    Create a temporary file with the given content.
    
    Args:
        content: File content
        suffix: Optional file extension (with dot)
        
    Returns:
        Path to temporary file
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
    except:
        os.unlink(temp_path)
        raise
        
    return temp_path