import time
from django.utils.deprecation import MiddlewareMixin
from .logger import get_logger

logger = get_logger(__name__)

class RequestLoggingMiddleware(MiddlewareMixin):

    def process_request(self, request):
        request._start_time = time.time()
        return None
    
    def process_response(self, request, response):
        if hasattr(request, '_start_time'):
            process_time = time.time() - request._start_time
            
            user_info = ""
            if hasattr(request, 'user') and request.user.is_authenticated:
                user_info = f" - User: {request.user.username}"
            
            logger.info(
                f"{request.method} {request.get_full_path()} "
                f"- Time: {process_time:.4f}s "
                f"- Status: {response.status_code}"
                f"{user_info}"
            )
        
        return response
    
    def process_exception(self, request, exception):
        if hasattr(request, '_start_time'):
            process_time = time.time() - request._start_time
            
            logger.error(
                f"{request.method} {request.get_full_path()} "
                f"- Time: {process_time:.4f}s "
                f"- Exception: {str(exception)}"
            )
        
        return None 