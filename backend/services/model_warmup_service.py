"""
Model Warmup Service - Fixed embedding configuration
Supports both embedding models and LLM providers without conflicts
"""

import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from threading import Lock

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

class ModelWarmupService:
    """
    Model warmup service supporting:
    - BGE-M3 embedding preloading (for performance)
    - LLM provider warmup (Ollama, cloud providers)
    - Singleton pattern for efficient reuse
    """
    
    _instance: Optional['ModelWarmupService'] = None
    _lock = Lock()
    
    def __new__(cls) -> 'ModelWarmupService':
        """Singleton pattern to ensure single instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize warmup service - only once due to singleton"""
        if hasattr(self, '_initialized'):
            return
            
        self.settings = get_settings()
        
        self._preloaded_models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        
        self.warmup_models = list(set([
            self.settings.REFLECTION_MODEL,
            self.settings.SEMANTIC_ROUTER_MODEL,
            self.settings.RESPONSE_GENERATION_MODEL,
            self.settings.FUNCTION_CALLING_MODEL,
            self.settings.DEFAULT_CHAT_MODEL,
        ]))
        self.default_provider = self.settings.DEFAULT_LLM_PROVIDER
        self.cloud_providers = {"gemini"}
        self.needs_llm_warmup = self.default_provider not in self.cloud_providers
        
        self._initialized = False
        
        logger.info("ModelWarmupService singleton initialized")
    
    async def preload_embedding_model(self, force_reload: bool = False) -> HuggingFaceEmbeddings:
        """Preload BGE-M3 embedding model with fixed configuration"""
        model_name = self.settings.EMBEDDING_MODEL
        
        if self._embeddings is not None and not force_reload:
            logger.info(f"Using cached embedding model: {model_name}")
            return self._embeddings
        
        try:
            start_time = time.time()
            logger.info(f"Preloading embedding model: {model_name}")
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': self.settings.EMBEDDING_MODEL_DEVICE,
                    'trust_remote_code': True,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32,
                },
                cache_folder="/tmp/huggingface_cache"
            )
            
            test_texts = ["Test query for model warmup", "Embedding model preloading verification"]
            logger.info("Testing embedding model with sample texts...")
            
            _ = self._embeddings.embed_documents(test_texts)
            _ = self._embeddings.embed_query("Test warmup query")
            
            load_time = time.time() - start_time
            logger.info(f"Successfully preloaded {model_name} in {load_time:.2f}s")
            
            return self._embeddings
            
        except Exception as e:
            logger.error(f"Failed to preload embedding model {model_name}: {e}")
            try:
                logger.info("Attempting fallback embedding configuration...")
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': self.settings.EMBEDDING_MODEL_DEVICE}
                )
                
                _ = self._embeddings.embed_query("Fallback test")
                logger.info(f"Fallback embedding configuration successful for {model_name}")
                return self._embeddings
                
            except Exception as fallback_error:
                logger.error(f"Fallback embedding configuration also failed: {fallback_error}")
                raise
    
    async def preload_tokenizer(self, model_name: Optional[str] = None) -> AutoTokenizer:
        """Preload HuggingFace tokenizer"""
        if model_name is None:
            model_name = self.settings.EMBEDDING_MODEL
        
        if model_name in self._tokenizers:
            logger.info(f"Using cached tokenizer: {model_name}")
            return self._tokenizers[model_name]
        
        try:
            start_time = time.time()
            logger.info(f"Preloading tokenizer: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir="/tmp/huggingface_cache"
            )
            
            test_text = "Test tokenization for warmup"
            tokens = tokenizer.encode(test_text)
            _ = tokenizer.decode(tokens)
            
            self._tokenizers[model_name] = tokenizer
            
            load_time = time.time() - start_time
            logger.info(f"Successfully preloaded tokenizer {model_name} in {load_time:.2f}s")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to preload tokenizer {model_name}: {e}")
            logger.warning("Using fallback tokenizer")
            try:
                fallback_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                self._tokenizers[model_name] = fallback_tokenizer
                return fallback_tokenizer
            except Exception as fallback_error:
                logger.error(f"Fallback tokenizer also failed: {fallback_error}")
                raise

    async def warmup_llm_providers(self) -> Dict[str, bool]:
        """Warmup LLM providers - Ollama and cloud providers"""
        results = {}
        
        if not self.needs_llm_warmup:
            logger.info(f"Provider '{self.default_provider}' is cloud-based, testing connection...")
            await self._test_cloud_provider_connection()
            for model in self.warmup_models:
                results[f"llm_{model}"] = True
            return results
        
        logger.info(f"Starting LLM warmup for local provider: {self.default_provider}")
        
        from llm.llm_router import get_llm_router
        llm_router = get_llm_router() 
        
        for model in self.warmup_models:
            try:
                logger.info(f"Warming up LLM: {model} via {self.default_provider}")
                
                client = llm_router.get_client(self.default_provider, model=model)
                if not client:
                    logger.warning(f"Could not get client for {self.default_provider} with model {model}")
                    results[f"llm_{model}"] = False
                    continue
                
                await client.generate_text(
                    prompt="warmup test",
                    max_tokens=1,
                    model=model
                )
                
                results[f"llm_{model}"] = True
                logger.info(f"LLM {model} warmed up successfully")
                
            except Exception as e:
                logger.error(f"Failed to warmup LLM {model}: {e}")
                results[f"llm_{model}"] = False
        
        return results
    
    async def _test_cloud_provider_connection(self):
        """Test connection to cloud provider"""
        try:
            from llm.llm_router import get_llm_router
            llm_router = get_llm_router() 
            
            client = llm_router.get_client(self.default_provider)
            if not client:
                logger.error(f"Could not get client for cloud provider {self.default_provider}")
                return
                
            response = await client.generate_text(
                prompt="test",
                max_tokens=1,
                temperature=0.1
            )
            
            logger.info(f"Connection test successful for {self.default_provider}")
            
        except Exception as e:
            logger.error(f"Connection test failed for {self.default_provider}: {e}")
    
    async def keep_llm_models_alive(self):
        """Background task to ping models every 5 minutes (only for local providers)"""
        if not self.needs_llm_warmup:
            logger.info(f"Provider '{self.default_provider}' is cloud-based, skipping keep-alive task")
            return
            
        from llm.llm_router import get_llm_router
        llm_router = get_llm_router()  
        
        while True:
            try:
                await asyncio.sleep(300)  
                
                for model in self.warmup_models:
                    try:
                        client = llm_router.get_client(self.default_provider, model=model)
                        if not client:
                            continue
                            
                        await client.generate_text(
                            prompt="ping",
                            max_tokens=1, 
                            model=model
                        )
                        
                    except Exception as e:
                        logger.warning(f"Failed to ping model {model}: {e}")
                        
            except Exception as e:
                logger.error(f"Error in keep_models_alive: {e}")

    async def warmup_all_models(self) -> Dict[str, bool]:
        """Warmup both embeddings and LLM providers"""
        results = {}
        
        if self.settings.PRELOAD_EMBEDDING_MODEL:
            try:
                await self.preload_embedding_model()
                results["embeddings"] = True
                logger.info("Embedding model warmed up successfully")
            except Exception as e:
                logger.error(f"Failed to warmup embedding model: {e}")
                results["embeddings"] = False
            
            try:
                await self.preload_tokenizer()
                results["tokenizer"] = True
                logger.info("Tokenizer warmed up successfully")
            except Exception as e:
                logger.error(f"Failed to warmup tokenizer: {e}")
                results["tokenizer"] = False
        else:
            logger.info("Embedding model preloading disabled")
            results["embeddings"] = True  
            results["tokenizer"] = True
        
        try:
            llm_results = await self.warmup_llm_providers()
            results.update(llm_results)
        except Exception as e:
            logger.error(f"Failed to warmup LLM providers: {e}")
            results["llm_providers"] = False
        
        self._initialized = True
        
        total_success = sum(1 for v in results.values() if v)
        total_models = len(results)
        
        logger.info(f"Model warmup completed: {total_success}/{total_models} successful")
        return results

    def get_preloaded_embeddings(self) -> Optional[HuggingFaceEmbeddings]:
        """Get cached embedding model if available"""
        return self._embeddings
    
    def get_preloaded_tokenizer(self, model_name: Optional[str] = None) -> Optional[AutoTokenizer]:
        """Get cached tokenizer if available"""
        if model_name is None:
            model_name = self.settings.EMBEDDING_MODEL
        return self._tokenizers.get(model_name)
    
    def is_initialized(self) -> bool:
        """Check if warmup service is fully initialized"""
        return self._initialized
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about preloaded models"""
        return {
            "embedding_model": self.settings.EMBEDDING_MODEL,
            "device": self.settings.EMBEDDING_MODEL_DEVICE,
            "max_length": self.settings.EMBEDDING_MODEL_MAX_LENGTH,
            "embeddings_loaded": self._embeddings is not None,
            "tokenizers_loaded": list(self._tokenizers.keys()),
            "default_llm_provider": self.default_provider,
            "needs_llm_warmup": self.needs_llm_warmup,
            "warmup_models": self.warmup_models,
            "initialized": self._initialized,
            "preload_enabled": self.settings.PRELOAD_EMBEDDING_MODEL
        }

_warmup_service_instance = None

def get_warmup_service() -> ModelWarmupService:
    """Get global ModelWarmupService instance"""
    global _warmup_service_instance
    if _warmup_service_instance is None:
        _warmup_service_instance = ModelWarmupService()
    return _warmup_service_instance