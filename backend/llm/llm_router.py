from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import os
import time
from enum import Enum
from threading import Lock
from utils.logger import get_logger
import asyncio

logger = get_logger(__name__)


class LLMProvider(Enum):
    """Enum for LLM providers"""
    OLLAMA = "ollama"
    GEMINI = "gemini"
    MISTRAL = "mistral"

class LLMRouter:
    """
    Router to manage and route between different LLM clients.
    Central class for communicating with LLM models in the system.
    Implemented as singleton to avoid multiple instances and share client cache.
    """
    
    _instance: Optional['LLMRouter'] = None
    _lock = Lock()
    
    def __new__(cls) -> 'LLMRouter':
        """Singleton pattern to ensure single LLMRouter instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize router - only once due to singleton"""
        if hasattr(self, '_initialized'):
            return
            
        self._registry = None
        self._providers_info = {}
        self._active_clients: Dict[str, Any] = {}

        from config.settings import get_settings
        settings = get_settings()
        
        self.default_provider = settings.DEFAULT_LLM_PROVIDER
        self.fallback_provider = settings.FALLBACK_LLM_PROVIDER
        self.semantic_router_model = settings.SEMANTIC_ROUTER_MODEL
        self.response_generation_model = settings.RESPONSE_GENERATION_MODEL
        self.default_chat_model = settings.DEFAULT_CHAT_MODEL
        
        self.language_provider_map = {
            "vi": self.default_provider,
            "en": self.default_provider,
            "ja": self.default_provider
        }
        
        self._initialized = True
        logger.info("LLMRouter singleton initialized")
    
    @property
    def registry(self):
        """Lazy load registry"""
        if self._registry is None:
            from .base_client import LLMRegistry
            self._registry = LLMRegistry()
            self._load_providers_info()
        return self._registry
    
    def _load_providers_info(self):
        """Load available LLM providers information"""
        available_providers = self.registry.list_available_providers()
        
        self._providers_info = {
            provider: self.registry.get_provider_info(provider)
            for provider in available_providers
        }
    
    def get_provider_for_language(self, language: str) -> str:
        """Get the most suitable provider for the specified language"""
        return self.language_provider_map.get(language, self.default_provider)

    def get_semantic_router_client(self, **kwargs):
        """Get LLM client specifically for semantic routing tasks"""
        return self.get_client(self.default_provider, model=self.semantic_router_model, **kwargs)

    def get_chat_client(self, **kwargs):
        """Get LLM client specifically for chat conversations"""
        return self.get_client(self.default_provider, model=self.default_chat_model, **kwargs)
    
    def get_client(self, provider: Optional[str] = None, **kwargs):
        """Get LLM client for the specified provider. Utilizes a cache for active clients."""
        if not provider:
            provider = self.default_provider

        relevant_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, type(None)))}
        if "model" not in relevant_kwargs and "model" in kwargs:
            relevant_kwargs["model"] = kwargs["model"]
            
        sorted_kwargs_tuple = tuple(sorted(relevant_kwargs.items()))
        cache_key_parts = [provider] + [f"{k}={v}" for k, v in sorted_kwargs_tuple]
        cache_key = "_".join(cache_key_parts)

        if cache_key in self._active_clients:
            return self._active_clients[cache_key]
            
        client = self.registry.get_client(provider, **kwargs)
        
        if client:
            self._active_clients[cache_key] = client
        elif provider != self.fallback_provider:
            logger.warning(f"Client creation failed for primary provider {provider} with args {kwargs}. Attempting fallback: {self.fallback_provider}")
           
            fallback_cache_key_parts = [self.fallback_provider] + [f"{k}={v}" for k, v in sorted_kwargs_tuple]
            fallback_cache_key = "_".join(fallback_cache_key_parts)

            if fallback_cache_key in self._active_clients:
                return self._active_clients[fallback_cache_key]
            
            client = self.registry.get_client(self.fallback_provider, **kwargs) 
            if client:
                self._active_clients[fallback_cache_key] = client
            else:
                logger.error(f"Fallback client creation also failed for {self.fallback_provider} with args {kwargs}")
        else:
            logger.error(f"Client creation failed for {provider} (fallback or no other fallback) with args {kwargs}")
            
        return client
    
    async def generate_text(
        self,
        prompt: str,
        language: str = "vi",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt using appropriate LLM"""
        start_time = time.time()
        
        if not provider:
            provider = self.get_provider_for_language(language)
        
        try:
            client = self.get_client(provider, model=model)
            
            if not client:
                raise ValueError(f"LLM client not found for provider: {provider}")
            
            response = await client.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                language=language,
                model=model,
                **kwargs
            )
            
            logger.debug(f"Generated text using {provider} in {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating text with {provider}: {e}")
            if provider != self.fallback_provider:
                logger.info(f"Attempting fallback to {self.fallback_provider}")
                return await self.generate_text(
                    prompt=prompt,
                    language=language,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    provider=self.fallback_provider,
                    model=model,
                    **kwargs
                )
            else:
                raise Exception(f"All LLM providers failed. Last error: {str(e)}")

    async def generate_text_stream(
        self,
        prompt: str,
        language: str = "vi",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text stream from prompt using appropriate LLM"""
        if not provider:
            provider = self.get_provider_for_language(language)
        
        try:
            client = self.get_client(provider, model=model)
            
            if not client:
                raise ValueError(f"LLM client not found for provider: {provider}")
            
            async for token in client.generate_text_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                language=language,
                **kwargs
            ):
                yield token
                
        except Exception as e:
            logger.error(f"Error generating text stream with {provider}: {e}")
            if provider != self.fallback_provider:
                logger.info(f"Attempting fallback to {self.fallback_provider}")
                async for token in self.generate_text_stream(
                    prompt=prompt,
                    language=language,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    provider=self.fallback_provider,
                    model=model,
                    **kwargs
                ):
                    yield token
            else:
                error_message = self._get_error_message(language)
                words = error_message.split()
                for i, word in enumerate(words):
                    if i < len(words) - 1:
                        yield word + " "
                    else:
                        yield word
                    await asyncio.sleep(0.02)

    def _get_error_message(self, language: str) -> str:
        """Get error message based on language"""
        messages = {
            "en": "I'm sorry, but I encountered an error while processing your request. Please try again.",
            "ja": "申し訳ありませんが、リクエストの処理中にエラーが発生しました。もう一度お試しください。",
            "vi": "Tôi xin lỗi, nhưng tôi đã gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."
        }
        return messages.get(language, messages["vi"])


_llm_router_instance = None

def get_llm_router() -> LLMRouter:
    """Get global LLMRouter singleton instance"""
    global _llm_router_instance
    if _llm_router_instance is None:
        _llm_router_instance = LLMRouter()
    return _llm_router_instance