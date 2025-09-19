from typing import Dict, List, Any, Optional, Union
import uuid
from pydantic import BaseModel, Field
from utils.logger import get_logger
from utils import now

logger = get_logger(__name__)


class DocumentChunk(BaseModel):
    """
    Represents a chunk of a document with its metadata.
    Used for retrieval and context generation in the RAG system.
    Vector embeddings are stored only in Milvus.
    """
    chunk_id: str = Field(..., description="Unique ID for the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    content_path: str = Field(..., description="Path to text content of the chunk in MinIO")
    content: Optional[str] = Field(None, description="Cached text content, not stored in DB")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the chunk")
    index: int = Field(..., description="Index/position of chunk within the document")
    chunk_size: int = Field(..., description="Size of the chunk (token count)")
    created_at: str = Field(..., description="Creation timestamp")
    prev_chunk_id: Optional[str] = Field(None, description="ID of the previous chunk for context expansion")
    next_chunk_id: Optional[str] = Field(None, description="ID of the next chunk for context expansion")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content_path": self.content_path,
            "metadata": self.metadata,
            "index": self.index,
            "chunk_size": self.chunk_size,
            "created_at": self.created_at,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create instance from dictionary"""
        data.pop('embedding', None)
        return cls(**data)


class Document(BaseModel):
    """
    Represents a document with metadata and processing information.
    Container for document chunks and metadata about the document source.
    """
    document_id: str = Field(..., description="Unique document ID")
    title: str = Field(..., description="Document title")
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type/MIME type")
    file_size: int = Field(..., description="File size in bytes")
    user_id: Optional[str] = Field(None, description="ID of user who uploaded the document")
    language: Optional[str] = Field(None, description="Document language code")
    description: Optional[str] = Field(None, description="Document description")
    status: str = Field("processing", description="Processing status")
    chunks_count: int = Field(0, description="Number of chunks created")
    chunk_size: int = Field(0, description="Chunk size used for splitting")
    chunk_overlap: int = Field(0, description="Chunk overlap used for splitting")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    file_path: Optional[str] = Field(None, description="File path in MinIO storage")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "user_id": self.user_id,
            "language": self.language,
            "description": self.description,
            "status": self.status,
            "chunks_count": self.chunks_count,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "file_path": self.file_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create instance from dictionary"""
        return cls(**data)
    
    @classmethod
    def create_new(
        cls,
        title: str,
        file_name: str,
        file_type: str,
        file_size: int,
        user_id: Optional[str] = None,
        language: Optional[str] = None,
        description: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Document':
        """
        Create a new Document instance with generated ID and timestamps.
        
        Args:
            title: Document title
            file_name: Original file name
            file_type: File MIME type
            file_size: File size in bytes
            user_id: ID of user who uploaded the document
            language: Document language code
            description: Document description
            chunk_size: Chunk size for document splitting
            chunk_overlap: Chunk overlap for document splitting
            metadata: Additional metadata
            
        Returns:
            New Document instance
        """
        current_time = now().isoformat()
        
        return cls(
            document_id=str(uuid.uuid4()),
            title=title,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            user_id=user_id,
            language=language,
            description=description,
            status="processing",
            chunks_count=0,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata or {},
            created_at=current_time,
            updated_at=current_time
        )


class DocumentStore:
    """
    Interface for document and chunk storage operations.
    Handles CRUD operations for documents and their chunks.
    """
    
    async def create_document(self, document: Document) -> str:
        """
        Store a new document.
        
        Args:
            document: Document to store
            
        Returns:
            Document ID
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document or None if not found
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def update_document(self, document: Document) -> bool:
        """
        Update a document.
        
        Args:
            document: Document with updated fields
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def list_documents(
        self,
        user_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        search: Optional[str] = None
    ) -> List[Document]:
        """
        List documents with pagination and filtering.
        
        Args:
            user_id: Filter by user ID
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            search: Search term for title/description
            
        Returns:
            List of documents
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def add_chunk(self, chunk: DocumentChunk) -> str:
        """
        Add a document chunk.
        
        Args:
            chunk: Document chunk to add
            
        Returns:
            Chunk ID
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def get_chunks(
        self, 
        document_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentChunk]:
        """
        Get chunks for a document with pagination.
        
        Args:
            document_id: Document ID
            limit: Maximum number of chunks to return
            offset: Number of chunks to skip
            
        Returns:
            List of document chunks
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def delete_chunks(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
