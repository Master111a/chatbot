from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message content")
    session_id: Optional[str] = Field(None, description="Chat session ID (auto-generated if not provided)")
    language: Optional[str] = Field(None, description="Language code (vi, en, ja)")
    include_citations: Optional[bool] = Field(True, description="Whether to include citations in response")
    include_context: Optional[bool] = Field(True, description="Whether to include context information in response")
    use_tools: Optional[bool] = Field(True, description="Whether to use external tools")
    max_tokens: int = Field(8192, description="Maximum number of tokens for response")
    document_ids: Optional[List[str]] = Field(None, description="List of document IDs to query")

    class Config:
        extra = "ignore"
        json_schema_extra = {
            "example": {
                "message": "Xin chào, bạn có thể giúp tôi không?",
                "language": "vi"
            }
        }


class StreamChatRequest(BaseModel):
    """Stream chat request model - requires session_id"""
    message: str = Field(..., description="User message content")
    session_id: str = Field(..., description="Chat session ID (required for streaming)")
    max_tokens: int = Field(8192, description="Maximum number of tokens for response")
    class Config:
        extra = "ignore"
        json_schema_extra = {
            "example": {
                "message": "Xin chào, bạn có thể giúp tôi không?",
                "session_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }


class ChatResponse(BaseModel):
    """Chat response model"""
    response_text: str = Field(..., description="Response content")
    session_id: str = Field(..., description="Chat session ID")
    language: str = Field(..., description="Language code used")
    timestamp: str = Field(..., description="Response timestamp")
    citations: Optional[List[Dict[str, Any]]] = Field(None, description="Source citations (if any)")
    follow_up_questions: Optional[List[str]] = Field(None, description="Suggested follow-up questions")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context information")


class DocumentQueryRequest(BaseModel):
    """Document query request model"""
    query: str = Field(..., description="User query")
    document_ids: List[str] = Field(..., description="List of document IDs to query")
    language: Optional[str] = Field(None, description="Language code (vi, en, ja)")
    include_citations: bool = Field(True, description="Whether to include citations in response")


class SessionInfoResponse(BaseModel):
    """Chat session information model"""
    session_id: str = Field(..., description="Chat session ID")
    exists: bool = Field(..., description="Whether the session exists")
    messages_count: int = Field(0, description="Number of messages in the session")
    language: Optional[str] = Field(None, description="Primary language of the session")
    first_message: Optional[str] = Field(None, description="First message timestamp")
    last_message: Optional[str] = Field(None, description="Last message timestamp")
    topics: Optional[List[str]] = Field(None, description="Conversation topics")
    error: Optional[str] = Field(None, description="Error message (if any)")


class DocumentUploadResponse(BaseModel):
    """Document upload response model"""
    document_id: str = Field(..., description="Uploaded document ID")
    title: str = Field(..., description="Document title")
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type")
    chunks_count: int = Field(..., description="Number of chunks created")
    upload_timestamp: str = Field(..., description="Upload timestamp")
    status: str = Field(..., description="Processing status")
    error: Optional[str] = Field(None, description="Error message (if any)")


class DocumentMetadataResponse(BaseModel):
    """Document metadata information model"""
    document_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type")
    file_size: int = Field(..., description="File size (bytes)")
    chunks_count: int = Field(..., description="Number of chunks")
    upload_timestamp: str = Field(..., description="Upload timestamp")
    last_updated: str = Field(..., description="Last update timestamp")
    status: str = Field(..., description="Document status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AdminStatsResponse(BaseModel):
    """System statistics for admin model"""
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    total_chat_sessions: int = Field(..., description="Total number of chat sessions")
    total_messages: int = Field(..., description="Total number of messages")
    total_users: int = Field(..., description="Total number of users")
    system_uptime: str = Field(..., description="System uptime")
    system_version: str = Field(..., description="System version")
    timestamp: str = Field(..., description="Statistics timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error details")
    timestamp: str = Field(..., description="Error timestamp")
    error_code: Optional[str] = Field(None, description="Error code (if any)")
    path: Optional[str] = Field(None, description="API path where error occurred")


class DocumentQueryResponse(BaseModel):
    """Response model for document queries with MinIO URLs"""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents with metadata")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Results per page")
    pages: int = Field(..., description="Total number of pages")


# ========== SESSION MANAGEMENT MODELS ==========

class CreateSessionRequest(BaseModel):
    """Request model for creating new session"""
    query: str = Field(..., description="First query to generate session title")
    language: Optional[str] = Field(None, description="Language code (vi, en, ja)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Xin chào, bạn có thể giúp tôi về Python không?",
                "language": "vi"
            }
        }


class CreateSessionResponse(BaseModel):
    """Response model for session creation"""
    session_id: str = Field(..., description="Created session ID")
    title: str = Field(..., description="Generated session title")
    language: str = Field(..., description="Detected/provided language")
    created_at: str = Field(..., description="Session creation timestamp")
    user_id: Optional[str] = Field(None, description="User ID (null for anonymous)")
    is_anonymous: bool = Field(..., description="Whether session is anonymous")


class ChatMessage(BaseModel):
    """Individual chat message model"""
    message_id: str = Field(..., description="Unique message ID")
    message: str = Field(..., description="User message")
    response: str = Field(..., description="Assistant response")
    timestamp: str = Field(..., description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Message metadata (citations, etc.)")


class SessionDetailResponse(BaseModel):
    """Detailed session information with full chat history"""
    session_id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Session title")
    user_id: Optional[str] = Field(None, description="Owner user ID")
    is_anonymous: bool = Field(..., description="Whether session is anonymous")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Total number of messages")
    language: Optional[str] = Field(None, description="Primary session language")
    
    # Full chat history
    messages: List[ChatMessage] = Field(..., description="Complete chat history")
    
    # Session metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Session metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Python Programming Help",
                "user_id": "user123",
                "is_anonymous": False,
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T10:30:00Z",
                "message_count": 5,
                "language": "vi",
                "messages": [
                    {
                        "message_id": "msg1",
                        "message": "Xin chào",
                        "response": "Chào bạn! Tôi có thể giúp gì?",
                        "timestamp": "2024-01-01T10:00:00Z",
                        "metadata": {}
                    }
                ]
            }
        }


class SessionSummary(BaseModel):
    """Summary information for session list"""
    session_id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Session title")
    message_count: int = Field(..., description="Number of messages")
    last_activity: str = Field(..., description="Last activity timestamp")
    language: Optional[str] = Field(None, description="Primary language")
    is_empty: bool = Field(..., description="Whether session has no messages")


class SessionListResponse(BaseModel):
    """Response model for user session list"""
    sessions: List[SessionSummary] = Field(..., description="List of user sessions")
    total: int = Field(..., description="Total number of sessions")
    limit: int = Field(..., description="Results per page")
    offset: int = Field(..., description="Number of sessions skipped")
    is_anonymous: bool = Field(..., description="Whether user is anonymous")
    user_id: Optional[str] = Field(None, description="User ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sessions": [
                    {
                        "session_id": "123e4567-e89b-12d3-a456-426614174000",
                        "title": "Python Programming Help",
                        "message_count": 5,
                        "last_activity": "2024-01-01T10:30:00Z",
                        "language": "vi",
                        "is_empty": False
                    }
                ],
                "total": 10,
                "limit": 20,
                "offset": 0,
                "is_anonymous": False,
                "user_id": "user123"
            }
        }


class UpdateSessionTitleRequest(BaseModel):
    """Request model for updating session title"""
    title: str = Field(..., min_length=1, max_length=200, description="New session title")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Updated Session Title"
            }
        }


class UpdateSessionTitleResponse(BaseModel):
    """Response model for session title update"""
    session_id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Updated title")
    updated: bool = Field(..., description="Whether update was successful")
    timestamp: str = Field(..., description="Update timestamp")


class BuildIndexRequest(BaseModel):
    message: Optional[str] = None
    force_rebuild: Optional[bool] = False