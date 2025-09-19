import time
import uvicorn
import os
import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from config.settings import get_settings
from api.v1 import chat_routes, document_routes
from auth import auth_routes
from utils.logger import get_logger
from db.pg_manager import PostgresManager
from services.model_warmup_service import get_warmup_service
from utils.context_utils import RequestContextMiddleware

logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None
)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

start_time = time.time()
model_warmup_service = get_warmup_service()

app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(chat_routes.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(document_routes.router, prefix="/api/v1/documents", tags=["Documents"])

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"- Time: {process_time:.4f}s "
        f"- Status: {response.status_code}"
    )
    
    return response

@app.on_event("startup")
async def startup_event():
    """App startup event - preload models và initialize services"""
    logger.info("🚀 Starting Agentic RAG API...")
    
    try:
        pg_manager = PostgresManager()
        await pg_manager.initialize()
        logger.info("Database connection initialized")
        
        from services.document_service import document_service
        await document_service.ensure_initialized()
        logger.info("Document service initialized")
        
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs("/app/data", exist_ok=True)
        
        logger.info("Starting comprehensive model warmup...")
        warmup_results = await model_warmup_service.warmup_all_models()
        
        for model_key, success in warmup_results.items():
            status = "✅" if success else "❌"
            logger.info(f"{status} {model_key}: {'Success' if success else 'Failed'}")
        
        if model_warmup_service.needs_llm_warmup:
            logger.info("Starting LLM keep-alive task for local providers...")
            asyncio.create_task(model_warmup_service.keep_llm_models_alive())
        
        model_info = model_warmup_service.get_model_info()
        logger.info(f"Model Info: {model_info}")
        
        logger.info("Agentic RAG API startup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise

@app.on_event("shutdown")  
async def shutdown_event():
    """App shutdown event"""
    logger.info("Shutting down Agentic RAG API...")
    
    try:
        pg_manager = PostgresManager()
        await pg_manager.close()
        logger.info("Database connections closed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    
    logger.info("Agentic RAG API shutdown completed")

@app.get("/", tags=["System"])
async def root():
    """System check endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "uptime": f"{(time.time() - start_time):.2f} s"
    }

@app.get("/health/models", tags=["System"])
async def model_health():
    """Model health check endpoint"""
    return {
        "status": "healthy" if model_warmup_service.is_initialized() else "initializing",
        "model_info": model_warmup_service.get_model_info(),
        "initialized": model_warmup_service.is_initialized()
    }

@app.get("/health/detailed", tags=["System"])
async def detailed_health_check():
    """Detailed system health check"""
    health_status = {
        "status": "healthy",
        "uptime": f"{(time.time() - start_time):.2f} s",
        "environment": settings.ENV,
        "models": model_warmup_service.get_model_info(),
        "services": {
            "ollama": settings.OLLAMA_API_URL,
            "milvus": f"{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            "postgres": settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "configured"
        },
        "configuration": {
            "reflection_model": settings.REFLECTION_MODEL,
            "semantic_router_model": settings.SEMANTIC_ROUTER_MODEL,
            "response_generation_model": settings.RESPONSE_GENERATION_MODEL,
            "default_chat_model": settings.DEFAULT_CHAT_MODEL
        }
    }
    
    return health_status

@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "uptime": f"{(time.time() - start_time):.2f} s",
        "environment": settings.ENV
    }

if settings.ENABLE_DOCS:
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Swagger UI endpoint"""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        """ReDoc endpoint"""
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - ReDoc",
            redoc_js_url="/static/redoc.standalone.js",
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )