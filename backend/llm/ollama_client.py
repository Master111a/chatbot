from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import json
import aiohttp
from llm import LLMInterface
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


settings = get_settings()

class OllamaClient(LLMInterface):
    """
    Client for Ollama API - allows using local LLM models.
    Supports multiple models like Mistral, Llama, Phi, etc.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Ollama Client.
        
        Args:
            base_url: Base URL for Ollama API
            default_model: Default model name
            model: Specific model to use (overrides default_model)
            api_key: API key (not used for local Ollama)
            **kwargs: Additional parameters
        """
        self._base_url = base_url or settings.OLLAMA_API_URL
        self._default_model = model or default_model or settings.DEFAULT_CHAT_MODEL
        
        logger.info(f"Ollama Client initialized with URL: {self._base_url}, model: {self._default_model}")
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        language: str = "vi",
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        model = kwargs.get("model", self._default_model)
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            async with aiohttp.ClientSession() as session:
                api_url = f"{self._base_url}/api/generate"
                async with session.post(api_url, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        raise Exception(f"Ollama API returned error: {response.status}")
                    
                    result = await response.json()
                    return result.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e!r}", exc_info=True)
            raise
    
    async def generate_text_stream(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        language: str = "vi",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate text from prompt as a stream.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            language: Language code
            **kwargs: Additional parameters
            
        Yields:
            Generated text tokens sequentially
        """
        model = kwargs.get("model", self._default_model)
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            async with aiohttp.ClientSession() as session:
                api_url = f"{self._base_url}/api/generate"
                async with session.post(api_url, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API stream error: {response.status} - {error_text}")
                        raise Exception(f"Ollama API returned error: {response.status}")
                    
                    async for line in response.content:
                        if not line:
                            continue
                        
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON from stream: {line}")
                            continue
    
                        if chunk.get("done", False):
                            break
        except Exception as e:
            logger.error(f"Error calling Ollama API stream: {e!r}", exc_info=True)
            raise
    
    async def generate_function_calls(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        language: str = "vi"
    ) -> List[Dict[str, Any]]:
        """
        Generate function calls based on prompt.
        
        Args:
            prompt: Input prompt
            functions: List of function definitions
            language: Language code
            
        Returns:
            List of function calls
        
        Note:
            Ollama doesn't support function calling directly.
            This method uses to generate JSON-formatted function calls.
        """
        logger.warning("Ollama doesn't support function calling directly, using alternative solution.")
        
        function_descriptions = []
        for fn in functions:
            function_descriptions.append(f"""
Name: {fn.get('name')}
Description: {fn.get('description')}
Parameters: {json.dumps(fn.get('parameters', {}), indent=2)}
""")
        
        functions_text = "\n".join(function_descriptions)
        
        if language == "en":
            system_prompt = f"""You are an AI assistant that can determine which tools to use based on user queries. You have access to the following tools:

{functions_text}

Analyze the user query and determine which tools to use, if any. Respond ONLY with a JSON array of tool calls in this exact format:
```json
[
  {{
    "tool_name": "name_of_tool",
    "parameters": {{
      "param1": "value1",
      "param2": "value2"
    }}
  }}
]
```
If no tools are needed, respond with an empty array: []

Be precise with parameter names and values. Only use tools that are actually needed to answer the query.
"""
        elif language == "ja":
            system_prompt = f"""あなたはユーザーのクエリに基づいてどのツールを使用するかを決定できるAIアシスタントです。以下のツールにアクセスできます：

{functions_text}

ユーザークエリを分析し、使用するツールを決定してください。次の正確な形式のJSONツールコール配列のみで応答してください：
```json
[
  {{
    "tool_name": "ツール名",
    "parameters": {{
      "パラメータ1": "値1",
      "パラメータ2": "値2"
    }}
  }}
]
```
ツールが必要ない場合は、空の配列で応答してください：[]

パラメータ名と値を正確に記述してください。クエリに答えるために実際に必要なツールのみを使用してください。
"""
        else:
            system_prompt = f"""Bạn là trợ lý AI có thể xác định công cụ nào cần sử dụng dựa trên truy vấn của người dùng. Bạn có quyền truy cập vào các công cụ sau:

{functions_text}

Phân tích truy vấn của người dùng và xác định công cụ nào cần sử dụng, nếu có. Chỉ trả lời với mảng JSON gọi công cụ theo định dạng chính xác này:
```json
[
  {{
    "tool_name": "tên_công_cụ",
    "parameters": {{
      "tham_số1": "giá_trị1",
      "tham_số2": "giá_trị2"
    }}
  }}
]
```
Nếu không cần công cụ nào, hãy trả lời với mảng rỗng: []

Hãy chính xác với tên và giá trị tham số. Chỉ sử dụng những công cụ thực sự cần thiết để trả lời truy vấn.
"""
        
        response = await self.generate_text(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.1,
            language=language,
            system_prompt=system_prompt
        )
        
        try:
            json_str = response.strip()
            import re
            json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_blocks:
                json_str = json_blocks[0].strip()
            elif not (json_str.startswith('[') and json_str.endswith(']')):
                json_arrays = re.findall(r'\[(.*?)\]', response, re.DOTALL)
                if json_arrays:
                    json_str = f"[{json_arrays[0]}]"
            
            function_calls = json.loads(json_str)
            
            if isinstance(function_calls, list):
                valid_calls = []
                for call in function_calls:
                    if isinstance(call, dict) and "tool_name" in call:
                        if "parameters" not in call:
                            call["parameters"] = {}
                        valid_calls.append(call)
                    else:
                        logger.warning(f"Invalid function call format: {call}")
                
                logger.info(f"Generated {len(valid_calls)} valid function calls using llama3.1:8b")
                return valid_calls
            else:
                logger.error(f"Function calls response is not an array: {function_calls}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting function calls from llama3.1:8b response: {e!r}", exc_info=True)
            return []
    
    @property
    def model_name(self) -> str:
        """
        Get the model name.
        
        Returns:
            Model name
        """
        return self._default_model
    
    @property
    def supports_function_calling(self) -> bool:
        """
        Check function calling support.
        
        Returns:
            False because Ollama doesn't natively support function calling
        """
        return False
    
    @property
    def max_context_length(self) -> int:
        """
        Get maximum context length for the model.
        
        Returns:
            Maximum context length
        """
        model_context_lengths = {
            "mistral": 8192,
            "llama3.1:8b": 8192,
            "phi3": 4096,
            "qwen": 8192,
            "gemma": 8192
        }
        
        return model_context_lengths.get(self._default_model, 4096)
    
    @property
    def preferred_languages(self) -> List[str]:
        """
        Get list of preferred languages for the model.
        
        Returns:
            List of language codes
        """
        return ["en", "vi", "ja"] 