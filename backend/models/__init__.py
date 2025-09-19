"""
Models module for the Agentic RAG system.
Contains all data models and schemas used throughout the application.
"""


from .document import Document, DocumentChunk, DocumentStore
from .evaluation import EvaluationQuery, BatchEvaluationRequest, SearchEvaluationQuery, SearchQualityEvaluation, SearchBenchmark
from .api_schemas import (
    ChatRequest,
    ChatResponse, 
    DocumentQueryRequest,
    SessionInfoResponse,
    DocumentUploadResponse,
    DocumentMetadataResponse,
    AdminStatsResponse,
    ErrorResponse, 
    DocumentQueryResponse,
    StreamChatRequest,
    UpdateSessionTitleRequest,
    UpdateSessionTitleResponse,
    ChatMessage,
    SessionDetailResponse,
    SessionListResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    BuildIndexRequest
)

__all__ = [
    
    # Document models  
    "Document",
    "DocumentChunk",
    "DocumentStore",
    
    # API schemas
    "ChatRequest",
    "ChatResponse",
    "DocumentQueryRequest", 
    "SessionInfoResponse",
    "DocumentUploadResponse",
    "DocumentMetadataResponse",
    "AdminStatsResponse",
    "ErrorResponse", 
    "DocumentQueryResponse",
    "StreamChatRequest",
    "UpdateSessionTitleRequest",
    "UpdateSessionTitleResponse",
    "ChatMessage",
    "SessionDetailResponse",
    "SessionListResponse",
    "CreateSessionRequest",
    "CreateSessionResponse",
    "BuildIndexRequest",
    "EvaluationQuery",
    "BatchEvaluationRequest",
    "SearchEvaluationQuery",
    "SearchQualityEvaluation",
    "SearchBenchmark"
]