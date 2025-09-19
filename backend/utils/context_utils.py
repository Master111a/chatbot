from contextvars import ContextVar
from typing import Optional
from fastapi import Request
import os
from utils.logger import get_logger

logger = get_logger(__name__)

_request_context: ContextVar[Optional[Request]] = ContextVar('request_context', default=None)

def set_request_context(request: Request) -> None:
    """
    Store request in context variable for use across the application
    
    Args:
        request: FastAPI Request object
    """
    _request_context.set(request)

def get_request_context() -> Optional[Request]:
    """
    Get current request from context variable
    
    Returns:
        Current request or None if not set
    """
    return _request_context.get(None)

def get_base_url_from_context(request: Optional[Request] = None) -> str:
    """
    Get base URL from request context or environment variables
    Determines the correct scheme (http/https) and host/domain
    
    Args:
        request: Optional request object, if None will try to get from context
        
    Returns:
        Base URL string with scheme and host
    """
    try:
        if request is None:
            request = get_request_context()
        
        env_base_url = (
            os.getenv('PUBLIC_URL') or 
            os.getenv('FASTAPI_BASE_URL') or
            os.getenv('BASE_URL')
        )
        if env_base_url:
            base_url = env_base_url.rstrip('/')
            return base_url
        
        if request:
            scheme = getattr(request.url, 'scheme', 'http')
            hostname = getattr(request.url, 'hostname', 'localhost')
            port = getattr(request.url, 'port', None)
            
            if port and port not in [80, 443]:
                base_url = f"{scheme}://{hostname}:{port}"
            else:
                base_url = f"{scheme}://{hostname}"
            
            logger.debug(f"Constructed base URL from request: {base_url}")
            return base_url
        
        be_host = os.getenv('BE_HOST', 'localhost')
        be_port = os.getenv('BE_EXPORT_PORT', '17002')
        env = os.getenv('ENV', 'dev')
        
        scheme = 'https' if env == 'prod' else 'http'
        
        if be_port and be_port not in ['80', '443']:
            base_url = f"{scheme}://{be_host}:{be_port}"
        else:
            base_url = f"{scheme}://{be_host}"
        
        logger.debug(f"Using fallback base URL: {base_url}")
        return base_url
        
    except Exception as e:
        logger.warning(f"Error determining base URL: {e}")
        fallback_url = "http://localhost:17002"
        logger.debug(f"Using final fallback URL: {fallback_url}")
        return fallback_url

def build_download_url(document_id: str, request: Optional[Request] = None) -> str:
    """
    Build document download URL with proper base URL
    
    Args:
        document_id: Document ID
        request: Optional request object for context
        
    Returns:
        Complete download URL
    """
    base_url = get_base_url_from_context(request)
    
    if not document_id:
        return f"{base_url}/api/v1/documents/download/unknown"
    
    return f"{base_url}/api/v1/documents/download/{document_id}"

class RequestContextMiddleware:
    """
    Middleware to set request context for the entire request lifecycle
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            from fastapi import Request
            request = Request(scope, receive)
            set_request_context(request)
            logger.debug(f"Set request context: {request.url}")
        
        await self.app(scope, receive, send)