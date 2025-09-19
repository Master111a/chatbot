from enum import Enum

class QueryType(str, Enum):
    """Enum for different query types"""
    RAG = "rag"
    TOOL = "tool_call"
    CHITCHAT = "chitchat"
    WORKFLOW = "workflow"
    SEMANTIC_CACHE_HIT = "semantic_cache_hit"
    DUPLICATE_WITH_ANSWER = "duplicate_with_answer"
    UNKNOWN = "unknown"

class ConversationRole(str, Enum):
    """Enum for different conversation roles"""
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"

class DocumentType(str, Enum):
    """Enum for different document types"""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    IMAGE = "image"
    URL = "url"
    UNKNOWN = "unknown"

class Language(str, Enum):
    """Enum for supported languages"""
    VI = "vi"
    EN = "en"
    JA = "ja"
    AUTO = "auto"