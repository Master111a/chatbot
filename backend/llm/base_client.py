from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """
    Abstract interface for LLM clients.
    All concrete LLM implementations should inherit from this class.
    """
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        language: str = "vi",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from a prompt"""
        pass
    
    async def generate_text_stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        language: str = "vi"
    ) -> AsyncGenerator[str, None]:
        """Generate text from a prompt as a stream. Default implementation using simulate streaming"""
        response = await self.generate_text(prompt, max_tokens, temperature, language)
        
        import asyncio
        words = response.split()
        for i, word in enumerate(words):
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word
            await asyncio.sleep(0.02)
    
    async def generate_function_call(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        model: Optional[str] = None,
        language: str = "vi",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Generate function call based on the prompt"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the model"""
        pass
    
    @property
    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling. Default is False"""
        return False
    
    @property
    def supports_streaming(self) -> bool:
        """Check if the model supports streaming. Default is False"""
        return False
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """Get the maximum context length for the model"""
        pass
    
    @property
    @abstractmethod
    def preferred_languages(self) -> List[str]:
        """Get the languages preferred by this model"""
        pass


class LLMRegistry:
    """
    Registry for LLM clients.
    Manages loading and accessing different LLM implementations.
    """
    
    def __init__(self):
        self._clients: Dict[str, type] = {}
        self._instances: Dict[str, LLMInterface] = {}
        self._load_clients()
    
    def _load_clients(self):
        """Load available LLM clients"""
        try:
            from .ollama_client import OllamaClient
            self.register_client("ollama", OllamaClient)
        except ImportError:
            pass
        
        try:
            from .gemini_client import GeminiClient
            self.register_client("gemini", GeminiClient)
        except ImportError:
            pass
    
    def register_client(self, name: str, client_class: type):
        """Register an LLM client class"""
        self._clients[name] = client_class
    
    def get_client(self, provider: str, **kwargs) -> Optional[LLMInterface]:
        """Get an LLM client instance for the specified provider"""
        if provider in self._instances and not kwargs:
            return self._instances[provider]
        
        if provider not in self._clients:
            return None
        
        try:
            client_class = self._clients[provider]
            instance = client_class(**kwargs)
            
            if not kwargs:
                self._instances[provider] = instance
            
            return instance
        except Exception:
            return None
    
    def list_available_providers(self) -> List[str]:
        """List available LLM providers"""
        return list(self._clients.keys())
    
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get information about a provider"""
        if provider not in self._clients:
            return {"name": provider, "available": False}
        
        try:
            client = self.get_client(provider)
            if not client:
                return {"name": provider, "available": False}
            
            return {
                "name": provider,
                "available": True,
                "model_name": client.model_name,
                "supports_function_calling": client.supports_function_calling,
                "supports_streaming": client.supports_streaming,
                "max_context_length": client.max_context_length,
                "preferred_languages": client.preferred_languages
            }
        except Exception:
            return {"name": provider, "available": False}