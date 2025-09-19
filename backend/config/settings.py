from typing import Dict, Any, Optional, List, Union
import os
from functools import lru_cache
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    """
    Application configuration with optimized validation and defaults.
    Includes MinIO, OTP, and document processing settings.
    """
    
    ENV: str = Field(default="dev")
    DEBUG: bool = Field(default=True)
    
    APP_NAME: str = Field(default="Agentic RAG API")
    APP_HOST: str = Field(default="0.0.0.0")
    APP_PORT: int = Field(default=8000)
    BOT_NAME: str = Field(default="NewwaveBot")
    TIMEZONE: str = Field(default="Asia/Ho_Chi_Minh")
    
    # Production worker configuration
    WORKERS: int = Field(default=4)
    WORKER_CLASS: str = Field(default="uvicorn.workers.UvicornWorker")
    WORKER_CONNECTIONS: int = Field(default=1000)
    MAX_REQUESTS: int = Field(default=1000)
    MAX_REQUESTS_JITTER: int = Field(default=100)
    KEEPALIVE: int = Field(default=2)
    
    BACKEND_BASE_URL: str = Field(default="http://192.168.16.79:17002")
    
    SECRET_KEY: str = Field(default="secret_key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    ALLOWED_HOSTS: List[str] = Field(default=["*"])
    CORS_ORIGINS: List[str] = Field(default=["*"])
    ENABLE_DOCS: bool = Field(default=True)
    
    DATABASE_URL: str = Field(default="postgresql://postgres:postgres@db-postgres:5432/newwave_chatbot")
    
    MILVUS_HOST: str = Field(default="db-milvus")
    MILVUS_PORT: Union[str, int] = Field(default="19530")
    MILVUS_USER: str = Field(default="milvus")
    MILVUS_PASSWORD: str = Field(default="milvus")
    MILVUS_COLLECTION: str = Field(default="chatbot")
    
    MINIO_ENDPOINT: str = Field(default="minio:9000")
    MINIO_EXTERNAL_ENDPOINT: str = Field(default="localhost:9000")  
    MINIO_ACCESS_KEY: str = Field(default="minioadmin")
    MINIO_SECRET_KEY: str = Field(default="minioadmin")
    MINIO_SECURE: bool = Field(default=False)
    MINIO_EXTERNAL_SECURE: bool = Field(default=False)
    MINIO_BUCKET_NAME: str = Field(default="newwave-documents")
    MINIO_REGION: Optional[str] = Field(default=None)
    
    UPLOAD_DIR: str = Field(default="/app/uploads")
    VERSIONS_DIR: str = Field(default="/app/versions")
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024)
    ALLOWED_FILE_TYPES: List[str] = Field(default=[
        "application/pdf", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint"
    ])
    
    DEFAULT_LLM_PROVIDER: str = Field(default="gemini")
    FALLBACK_LLM_PROVIDER: str = Field(default="gemini")
    FUNCTION_CALLING_PROVIDER: str = Field(default="gemini")
    OLLAMA_API_URL: str = Field(default="http://192.168.200.57:11434")
    GEMINI_API_KEY: str = Field(default="")
    
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")
    EMBEDDING_MODEL_DEVICE: str = Field(default="cpu")
    EMBEDDING_MODEL_MAX_LENGTH: int = Field(default=8192)  
    PRELOAD_EMBEDDING_MODEL: bool = Field(default=True)  
    
    # Contextual Chunking Strategy 
    CHUNK_SIZE_SMALL: int = Field(default=800)   
    CHUNK_SIZE_MEDIUM: int = Field(default=1500) 
    CHUNK_SIZE_LARGE: int = Field(default=2000)  
    DEFAULT_CHUNK_SIZE: int = Field(default=1500)
    DEFAULT_CHUNK_OVERLAP: int = Field(default=200)
    
    DEFAULT_TOP_K: int = Field(default=6)         
    DEFAULT_FETCH_K: int = Field(default=30)     
    DEFAULT_LAMBDA_MULT: float = Field(default=0.4)  
    DEFAULT_THRESHOLD: float = Field(default=0.3)   
    
    # Advanced Hybrid Search
    USE_HYBRID_SEARCH: bool = Field(default=True)
    DENSE_WEIGHT: float = Field(default=0.7)   
    SPARSE_WEIGHT: float = Field(default=0.3)  
    
    GITLAB_CLIENT_ID: str = Field(default="")
    GITLAB_CLIENT_SECRET: str = Field(default="")
    GITLAB_REDIRECT_URI: str = Field(default="http://localhost:8000/auth/gitlab/callback")
    GITLAB_BASE_URL: str = Field(default="https://gitlab.com")
    
    OTP_SECRET_KEY: str = Field(default="JBSWY3DPEHPK3PXP")
    OTP_VALIDITY_SECONDS: int = Field(default=30)
    OTP_TOLERANCE_WINDOWS: int = Field(default=2)
    
    REFLECTION_MODEL: str = Field(default="gemini-2.0-flash")
    SEMANTIC_ROUTER_MODEL: str = Field(default="gemini-2.0-flash")
    RESPONSE_GENERATION_MODEL: str = Field(default="gemini-2.0-flash")
    FUNCTION_CALLING_MODEL: str = Field(default="gemini-2.0-flash")
    DEFAULT_CHAT_MODEL: str = Field(default="gemini-2.0-flash")
    
    OLLAMA_KEEP_ALIVE: int = Field(default=-1)
    OLLAMA_MAX_LOADED_MODELS: int = Field(default=2)
    OLLAMA_NUM_PARALLEL: int = Field(default=2)
    OLLAMA_FLASH_ATTENTION: bool = Field(default=True)
    
    NUM_FOLLOW_UP_QUESTIONS: int = Field(default=3)
    CHAT_SESSION_TITLE_MAX_WORDS: int = Field(default=10)
    CHAT_SESSION_TITLE_MAX_CHARS: int = Field(default=100)
    DEFAULT_MAX_TOKENS: int = Field(default=8192)
    DEFAULT_RAG_MAX_TOKENS: int = Field(default=8192)
    GEMINI_DEFAULT_MODEL: str = Field(default="gemini-2.0-flash")
    GEMINI_DEFAULT_MAX_TOKENS: int = Field(default=8192)

    @property
    def gemini_api_keys(self) -> List[str]:
        """Get list of Gemini API keys from comma-separated string"""
        if not self.GEMINI_API_KEY:
            return []
        return [key.strip() for key in self.GEMINI_API_KEY.split(",") if key.strip()]
    
    @property
    def get_workers_count(self) -> int:
        """Get optimal worker count based on environment and CPU cores"""
        if self.is_development():
            return 1
        
        if self.WORKERS > 1:
            return self.WORKERS
        
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            return min(max(2 * cpu_count + 1, 2), 8) 
        except:
            return 4 
    
    @field_validator("ENV")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed_envs = ["dev", "stg", "prod"]
        if v not in allowed_envs:
            raise ValueError(f"ENV must be one of {allowed_envs}")
        return v
    
    @field_validator("WORKERS")
    @classmethod
    def validate_workers(cls, v: int) -> int:
        if v < 1 or v > 16:
            raise ValueError("WORKERS must be between 1 and 16")
        return v
    
    @field_validator("WORKER_CONNECTIONS")
    @classmethod
    def validate_worker_connections(cls, v: int) -> int:
        if v < 100 or v > 10000:
            raise ValueError("WORKER_CONNECTIONS must be between 100 and 10000")
        return v
    
    @field_validator("ALLOWED_HOSTS", "CORS_ORIGINS")
    @classmethod
    def validate_hosts(cls, v: List[str], info) -> List[str]:
        env = os.getenv("ENV", "dev")
        if env != "dev" and "*" in v:
            if info.field_name == "ALLOWED_HOSTS":
                return [os.getenv("APP_DOMAIN", "api.example.com")]
            elif info.field_name == "CORS_ORIGINS":
                return [f"https://{os.getenv('FRONTEND_DOMAIN', 'app.example.com')}"]
        return v
    
    @field_validator("ENABLE_DOCS")
    @classmethod
    def validate_enable_docs(cls, v: bool) -> bool:
        env = os.getenv("ENV", "dev")
        return v if env == "dev" else False
    
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string")
        return v
    
    @field_validator("MAX_FILE_SIZE")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_FILE_SIZE must be positive")
        if v > 1024 * 1024 * 1024:
            raise ValueError("MAX_FILE_SIZE cannot exceed 1GB")
        return v
    
    @field_validator("DEFAULT_TOP_K")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v <= 0 or v > 100:
            raise ValueError("DEFAULT_TOP_K must be between 1 and 100")
        return v
    
    @field_validator("DEFAULT_THRESHOLD")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("DEFAULT_THRESHOLD must be between 0.0 and 1.0")
        return v
    
    @field_validator("OTP_SECRET_KEY")
    @classmethod
    def validate_otp_secret(cls, v: str) -> str:
        if len(v) < 16:
            raise ValueError("OTP_SECRET_KEY must be at least 16 characters")
        return v.upper()
    
    @field_validator("OTP_VALIDITY_SECONDS")
    @classmethod
    def validate_otp_validity(cls, v: int) -> int:
        if v < 15 or v > 300:
            raise ValueError("OTP_VALIDITY_SECONDS must be between 15 and 300")
        return v
    
    @field_validator("GEMINI_API_KEY")
    @classmethod
    def validate_gemini_api_key(cls, v: str) -> str:
        if not v:
            if os.getenv("DEFAULT_LLM_PROVIDER") == "gemini":
                raise ValueError("GEMINI_API_KEY is required when using Gemini as default provider")
        return v
    
    @field_validator("GEMINI_DEFAULT_MODEL")
    @classmethod
    def validate_gemini_model(cls, v: str) -> str:
        valid_models = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-pro"
        ]
        if v not in valid_models:
            import logging
            logging.warning(f"GEMINI_DEFAULT_MODEL '{v}' is not in known valid models: {valid_models}")
        return v
    
    def model_post_init(self, __context) -> None:
        """Post-init validation and directory setup"""
        try:
            os.makedirs(self.UPLOAD_DIR, exist_ok=True)
            os.makedirs(self.VERSIONS_DIR, exist_ok=True)
        except PermissionError:
            if self.is_development():
                import tempfile
                temp_dir = tempfile.gettempdir()
                self.UPLOAD_DIR = os.path.join(temp_dir, "uploads")
                self.VERSIONS_DIR = os.path.join(temp_dir, "versions")
                os.makedirs(self.UPLOAD_DIR, exist_ok=True)
                os.makedirs(self.VERSIONS_DIR, exist_ok=True)
            else:
                raise
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENV == "prod"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENV == "dev"
    
    def get_minio_config(self) -> Dict[str, Any]:
        """Get MinIO configuration"""
        return {
            "endpoint": self.MINIO_ENDPOINT,
            "access_key": self.MINIO_ACCESS_KEY,
            "secret_key": self.MINIO_SECRET_KEY,
            "secure": self.MINIO_SECURE,
            "bucket_name": self.MINIO_BUCKET_NAME,
            "region": self.MINIO_REGION
        }
    
    def get_otp_config(self) -> Dict[str, Any]:
        """Get OTP configuration"""
        return {
            "secret_key": self.OTP_SECRET_KEY,
            "validity_seconds": self.OTP_VALIDITY_SECONDS,
            "tolerance_windows": self.OTP_TOLERANCE_WINDOWS
        }
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """Get uvicorn configuration for production deployment"""
        config = {
            "host": self.APP_HOST,
            "port": self.APP_PORT,
            "workers": self.get_workers_count,
            "worker_class": self.WORKER_CLASS,
            "worker_connections": self.WORKER_CONNECTIONS,
            "max_requests": self.MAX_REQUESTS,
            "max_requests_jitter": self.MAX_REQUESTS_JITTER,
            "keepalive": self.KEEPALIVE,
            "access_log": not self.is_production(),
            "reload": self.is_development()
        }
        
        if self.is_production():
            config.update({
                "log_level": "info",
                "access_log": False,
                "reload": False
            })
        
        return config
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance for optimal performance.
    
    Returns:
        Settings instance
    """
    return Settings()

def get_environment_info() -> Dict[str, Any]:
    """
    Get environment information for system diagnostics.
    
    Returns:
        Environment information dictionary
    """
    settings = get_settings()
    return {
        "environment": settings.ENV,
        "debug": settings.DEBUG,
        "app_name": settings.APP_NAME,
        "python_version": os.sys.version,
        "providers_available": {
            "ollama": bool(settings.OLLAMA_API_URL),
            "gemini": bool(settings.GEMINI_API_KEY),
            "gitlab": bool(settings.GITLAB_CLIENT_ID and settings.GITLAB_CLIENT_SECRET),
            "minio": bool(settings.MINIO_ACCESS_KEY and settings.MINIO_SECRET_KEY)
        },
        "storage_config": {
            "upload_dir": settings.UPLOAD_DIR,
            "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
            "allowed_file_types": len(settings.ALLOWED_FILE_TYPES),
            "minio_bucket": settings.MINIO_BUCKET_NAME
        },
        "otp_config": {
            "validity_seconds": settings.OTP_VALIDITY_SECONDS,
            "tolerance_windows": settings.OTP_TOLERANCE_WINDOWS
        }
    }