import logging
import logging.handlers
import sys
from typing import Dict
from pathlib import Path
from django.conf import settings

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        if hasattr(record, 'color') and record.color:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
            record.name = f"{color}{record.name}{reset}"
        
        return super().format(record)

class ContextFilter(logging.Filter):
    """Filter to add context information to log records"""
    
    def __init__(self):
        super().__init__()
        self.context = {}
    
    def filter(self, record):
        for key, value in self.context.items():
            setattr(record, key, value)
        return True
    
    def set_context(self, **kwargs):
        """Set context variables that will be added to all log records"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context variables"""
        self.context.clear()

class DjangoLoggerManager:
    """
    Centralized logger manager for Django application.
    Provides consistent logging configuration across all modules.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DjangoLoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.loggers: Dict[str, logging.Logger] = {}
            self.context_filter = ContextFilter()
            self._setup_logging()
            DjangoLoggerManager._initialized = True
    
    def _setup_logging(self):
        """Setup the logging configuration"""
        try:
            log_level = logging.DEBUG if settings.DEBUG else logging.INFO
            log_dir = Path(settings.BASE_DIR) / "logs"
            log_dir.mkdir(exist_ok=True)
        except Exception:
            log_level = logging.INFO
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
        
        # Console formatter with colors
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File formatter with more details
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(lambda record: setattr(record, 'color', True) or True)
        
        # File handler for general logs
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "django_app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        
        # Error handler for error logs only
        error_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "django_error.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add our handlers
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        root_logger.addFilter(self.context_filter)
        
        # Suppress noisy loggers
        logging.getLogger('django.db.backends').setLevel(logging.WARNING)
        logging.getLogger('django.request').setLevel(logging.INFO)
        logging.getLogger('django.server').setLevel(logging.INFO)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for the given name.
        Reuses existing loggers to avoid duplication.
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def set_context(self, **kwargs):
        """Set context variables for all loggers"""
        self.context_filter.set_context(**kwargs)
    
    def clear_context(self):
        """Clear context variables for all loggers"""
        self.context_filter.clear_context()
    
    def set_log_level(self, level: str):
        """Set log level for all loggers"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(log_level)
            elif isinstance(handler, logging.handlers.RotatingFileHandler):
                if 'error.log' not in handler.baseFilename:
                    handler.setLevel(log_level)

# Global instance
_logger_manager = DjangoLoggerManager()

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.
    This is the main function that should be used throughout the Django application.
    
    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("This is a log message")
    """
    return _logger_manager.get_logger(name)

def set_log_context(**kwargs):
    """
    Set context variables that will be added to all log records.
    
    Usage:
        set_log_context(user_id="123", session_id="abc")
    """
    _logger_manager.set_context(**kwargs)

def clear_log_context():
    """Clear all log context variables"""
    _logger_manager.clear_context()

def set_log_level(level: str):
    """
    Set the log level for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    _logger_manager.set_log_level(level)

class LoggingMixin:
    """
    Mixin class that provides logging capabilities to any Django class.
    
    Usage:
        class MyView(LoggingMixin, View):
            def get(self, request):
                self.logger.info("Processing GET request")
                return HttpResponse("OK")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)
        return self._logger

def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time.
    
    Usage:
        @log_function_call
        def my_function(param1, param2):
            return param1 + param2
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        logger.debug(f"Calling {func.__name__} with args={args[:3]}{'...' if len(args) > 3 else ''}, kwargs={list(kwargs.keys())}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    
    return wrapper

def log_performance(threshold_seconds: float = 1.0):
    """
    Decorator to log performance warnings for slow functions.
    
    Usage:
        @log_performance(threshold_seconds=2.0)
        def slow_function():
            time.sleep(3)  # Will log a performance warning
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time > threshold_seconds:
                    logger.warning(f"Performance warning: {func.__name__} took {execution_time:.3f}s (threshold: {threshold_seconds}s)")
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator 