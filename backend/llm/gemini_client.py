import google.generativeai as genai
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import json
import asyncio
import re
import time
import random
from llm.base_client import LLMInterface
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

class GeminiClient(LLMInterface):
    """
    Client for interacting with Google Gemini models.
    Implements LLMInterface for compatibility with the LLM routing system.
    Supports multiple API keys with automatic rotation on rate limits.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        default_model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Gemini Client.
        
        Args:
            api_key: Google API key for Gemini (can be comma-separated list)
            model: Specific model to use
            default_model: Default model name
            **kwargs: Additional parameters
        """
        if api_key:
            self.api_keys = [key.strip() for key in api_key.split(",") if key.strip()]
        else:
            self.api_keys = settings.gemini_api_keys
            
        if not self.api_keys:
            raise ValueError("No Gemini API keys provided")
        
        self.current_key_index = 0
        self.rate_limit_tracker = {}  
        self.key_failure_count = {}  
        
        self._configure_api_key(self.api_keys[0])
        
        self._default_model = (
            model or 
            default_model or 
            settings.GEMINI_DEFAULT_MODEL or 
            "gemini-2.0-flash"
        )
        
        if not self._default_model:
            raise ValueError("No valid Gemini model specified")
            
        logger.info(f"GeminiClient initialized with {len(self.api_keys)} API key(s), model: {self._default_model}")

    def _configure_api_key(self, api_key: str):
        """Configure Google AI with specific API key"""
        genai.configure(api_key=api_key)
        logger.debug(f"Configured Gemini with API key: {api_key[:8]}...")
    
    def _get_next_api_key(self) -> Optional[str]:
        """Get next available API key for rotation with improved selection"""
        if len(self.api_keys) <= 1:
            return None
        
        original_index = self.current_key_index
        current_time = time.time()
        
        for i in range(len(self.api_keys)):
            test_index = (self.current_key_index + i + 1) % len(self.api_keys)
            test_key = self.api_keys[test_index]
            
            if test_key in self.rate_limit_tracker:
                if current_time < self.rate_limit_tracker[test_key]:
                    continue  
                else:
                    del self.rate_limit_tracker[test_key]
            
            failure_count = self.key_failure_count.get(test_key, 0)
            if failure_count >= 3:  
                continue
            
            self.current_key_index = test_index
            logger.info(f"Rotated to API key {test_index + 1}/{len(self.api_keys)}: {test_key[:8]}...")
            return test_key
        
        logger.warning("No available API keys found, all are rate limited or failed")
        return None
    
    def _handle_rate_limit(self, api_key: str, retry_delay_seconds: int = 60):
        """Handle rate limit for specific API key with improved tracking"""
        self.rate_limit_tracker[api_key] = time.time() + retry_delay_seconds
        self.key_failure_count[api_key] = self.key_failure_count.get(api_key, 0) + 1
        logger.warning(f"API key {api_key[:8]}... rate limited for {retry_delay_seconds}s (failures: {self.key_failure_count[api_key]})")

    def _handle_success(self, api_key: str):
        """Reset failure count on successful request"""
        if api_key in self.key_failure_count:
            del self.key_failure_count[api_key]

    def _is_rate_limit_error(self, error_str: str) -> bool:
        """Improved detection of rate limit errors"""
        rate_limit_indicators = [
            "429",
            "quota",
            "rate limit",
            "rate_limit",
            "too many requests",
            "resource_exhausted",
            "requests per minute",
            "requests per day"
        ]
        return any(indicator in error_str.lower() for indicator in rate_limit_indicators)

    def _extract_retry_delay(self, error_str: str) -> int:
        """Extract retry delay from error message with improved parsing"""
        default_delay = 60
        
        patterns = [
            r'retry_delay.*?seconds?:?\s*(\d+)',
            r'retry.*?after.*?(\d+)\s*seconds?',
            r'wait.*?(\d+)\s*seconds?',
            r'back.*?off.*?(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    delay = int(match.group(1))
                    return min(delay, 300) 
                except (ValueError, IndexError):
                    continue
        
        return default_delay

    async def _retry_with_backoff(self, func, *args, max_retries: int = 3, **kwargs):
        """Generic retry mechanism with exponential backoff"""
        for attempt in range(max_retries):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                base_delay = 2 ** attempt
                jitter = random.uniform(0.5, 1.5)
                delay = base_delay * jitter
                
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)[:100]}")
                await asyncio.sleep(delay)
        
        raise Exception("Max retries exceeded")

    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = 0.7,
        language: str = "vi",
        **kwargs
    ) -> str:
        """
        Generate text from a prompt using Gemini model with improved retry logic.
        """
        if max_tokens is None:
            max_tokens = settings.GEMINI_DEFAULT_MAX_TOKENS
            
        model_name = kwargs.get("model") or self._default_model
        if not model_name:
            model_name = "gemini-2.0-flash"
            logger.warning(f"Model name was None, using fallback: {model_name}")
            
        top_p = kwargs.get("top_p")
        top_k = kwargs.get("top_k")
        stop_sequences = kwargs.get("stop_sequences")
        system_prompt = kwargs.get("system_prompt")
        
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=stop_sequences,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        max_key_attempts = len(self.api_keys) * 2 
        attempt_count = 0
        
        while attempt_count < max_key_attempts:
            current_key = self.api_keys[self.current_key_index]
            attempt_count += 1
            
            try:
                logger.debug(f"Request to Gemini model: {model_name} (Key: {current_key[:8]}..., attempt: {attempt_count}/{max_key_attempts})")
                
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    system_instruction=system_prompt
                )
                
                response = await asyncio.wait_for(
                    model.generate_content_async(prompt), 
                    timeout=30.0 
                )
                
                if not response.candidates:
                    block_reason = "Unknown"
                    if response.prompt_feedback:
                        block_reason = response.prompt_feedback.block_reason
                    logger.warning(f"Gemini API returned no candidates. Block reason: {block_reason}")
                    return ""
                    
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                
                if finish_reason == 2 or finish_reason == "MAX_TOKENS":
                    logger.warning(f"Gemini response was truncated due to max_tokens limit ({max_tokens}). Consider increasing max_tokens.")
                elif finish_reason == 3 or finish_reason == "SAFETY":
                    logger.warning(f"Gemini response was blocked by safety filters")
                elif finish_reason == 4 or finish_reason == "RECITATION":
                    logger.warning(f"Gemini response was blocked due to recitation concerns")
                elif finish_reason not in [1, "STOP"]:
                    logger.warning(f"Gemini generation did not finish normally. Reason: {finish_reason}")

                generated_text = ""
                if candidate.content and candidate.content.parts:
                    generated_text = "".join(part.text for part in candidate.content.parts)
                
                self._handle_success(current_key)
                logger.debug(f"Gemini response received: {generated_text[:100]}...")
                return generated_text

            except Exception as e:
                error_str = str(e)
                
                if self._is_rate_limit_error(error_str):
                    retry_delay = self._extract_retry_delay(error_str)
                    logger.warning(f"Rate limit detected for key {current_key[:8]}..., marking unavailable for {retry_delay}s")
                    self._handle_rate_limit(current_key, retry_delay)
                    
                    next_key = self._get_next_api_key()
                    if next_key:
                        self._configure_api_key(next_key)
                        logger.info(f"Immediately switched to next API key: {next_key[:8]}...")
                        continue 
                    else:
                        logger.warning("No available keys, waiting 5s for rate limits to expire...")
                        await asyncio.sleep(5)  
                        continue
                
                elif isinstance(e, asyncio.TimeoutError):
                    logger.error(f"Gemini API request timed out after 30 seconds (Key: {current_key[:8]}..., attempt {attempt_count})")
                    next_key = self._get_next_api_key()
                    if next_key:
                        self._configure_api_key(next_key)
                        logger.info(f"Timeout, switched to next API key: {next_key[:8]}...")
                        continue
                    else:
                        await asyncio.sleep(2)  
                        continue
                
                else:
                    logger.error(f"Error generating text with Gemini model {model_name}: {e}")
                    next_key = self._get_next_api_key()
                    if next_key:
                        self._configure_api_key(next_key)
                        continue
                    else:
                        await asyncio.sleep(1) 
                        continue
        
        raise Exception(f"Gemini API error: All attempts failed after {max_key_attempts} tries")

    async def generate_text_stream(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = 0.7,
        language: str = "vi",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate text stream from prompt using native Gemini streaming with improved retry logic.
        """
        if max_tokens is None:
            max_tokens = settings.GEMINI_DEFAULT_MAX_TOKENS
            
        model_name = kwargs.get("model") or self._default_model
        if not model_name:
            model_name = "gemini-2.0-flash"
            logger.warning(f"Model name was None, using fallback: {model_name}")
            
        top_p = kwargs.get("top_p")
        top_k = kwargs.get("top_k")
        stop_sequences = kwargs.get("stop_sequences")
        system_prompt = kwargs.get("system_prompt")
        
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=stop_sequences,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        max_key_attempts = len(self.api_keys) * 2 
        attempt_count = 0
        
        while attempt_count < max_key_attempts:
            current_key = self.api_keys[self.current_key_index]
            attempt_count += 1
            
            try:
                logger.debug(f"Streaming request to Gemini model: {model_name} (Key: {current_key[:8]}..., attempt: {attempt_count}/{max_key_attempts})")
                
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    system_instruction=system_prompt
                )
                
                stream_response = model.generate_content(
                    prompt, 
                    stream=True
                )
                
                for chunk in stream_response:  
                    if chunk.candidates and chunk.candidates[0].content:
                        candidate = chunk.candidates[0]
                        
                        if candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    yield part.text
                                    await asyncio.sleep(0)
                        
                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                            if candidate.finish_reason not in [1, "STOP"]:
                                logger.debug(f"Stream finished: {candidate.finish_reason}")
                            break
                
                self._handle_success(current_key)
                return

            except Exception as e:
                error_str = str(e)
            
                if self._is_rate_limit_error(error_str):
                    retry_delay = self._extract_retry_delay(error_str)
                    logger.warning(f"Rate limit in streaming for key {current_key[:8]}..., marking unavailable for {retry_delay}s")
                    self._handle_rate_limit(current_key, retry_delay)
                    
                    next_key = self._get_next_api_key()
                    if next_key:
                        self._configure_api_key(next_key)
                        logger.info(f"Immediately switched to next API key for streaming: {next_key[:8]}...")
                        continue 
                    else:
                        logger.warning("No available keys for streaming, waiting 5s...")
                        await asyncio.sleep(5)
                        continue
                
                elif isinstance(e, asyncio.TimeoutError):
                    logger.error(f"Gemini streaming timed out (Key: {current_key[:8]}..., attempt {attempt_count})")
                    next_key = self._get_next_api_key()
                    if next_key:
                        self._configure_api_key(next_key)
                        logger.info(f"Timeout in streaming, switched to next API key: {next_key[:8]}...")
                        continue
                    else:
                        await asyncio.sleep(2)
                        continue
                
                else:
                    logger.error(f"Error in Gemini streaming: {e}")
                    next_key = self._get_next_api_key()
                    if next_key:
                        self._configure_api_key(next_key)
                        continue
                    else:
                        await asyncio.sleep(1)
                        continue
    
        error_message = "Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."

        if language == "en":
            error_message = "Sorry, I encountered an error while processing your request. Please try again later."
        elif language == "ja":
            error_message = "申し訳ありません。あなたのリクエストを処理する際にエラーが発生しました。後でもう一度お試しください。"
    
        for char in error_message:
            yield char
            await asyncio.sleep(0.01)

    async def generate_function_calls(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        language: str = "vi"
    ) -> List[Dict[str, Any]]:
        """
        Generate function calls based on prompt using Gemini.
        
        Args:
            prompt: Input prompt
            functions: List of function definitions
            language: Language code
            
        Returns:
            List of function calls
        """
        logger.info("Generating function calls using Gemini")
        
        function_descriptions = []
        for fn in functions:
            function_descriptions.append(f"""
Name: {fn.get('name')}
Description: {fn.get('description')}
Parameters: {json.dumps(fn.get('parameters', {}), indent=2)}
""")
        
        functions_text = "\n".join(function_descriptions)
        
        if language == "en":
            system_prompt = f"""You are an AI assistant that determines which tools to use based on user queries. You have access to the following tools:

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
        
        try:
            response = await self.generate_text(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.1,
                language=language,
                system_prompt=system_prompt
            )
            
            return self._parse_function_calls_from_response(response)
            
        except Exception as e:
            logger.error(f"Error generating function calls with Gemini: {e}", exc_info=True)
            return []

    def _parse_function_calls_from_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse function calls from Gemini response.
        Handles various response formats and extracts valid JSON.
        """
        try:
            json_blocks = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_blocks:
                json_str = json_blocks[0].strip()
            else:
                json_str = response.strip()
                
                if not (json_str.startswith('[') and json_str.endswith(']')):
                    array_matches = re.findall(r'\[(.*?)\]', response, re.DOTALL)
                    if array_matches:
                        json_str = f"[{array_matches[0]}]"
                    else:
                        return []
            
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
                
                logger.info(f"Generated {len(valid_calls)} valid function calls using Gemini")
                return valid_calls
            else:
                logger.error(f"Function calls response is not an array: {function_calls}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error when parsing function calls: {e}")
            logger.debug(f"Response was: {response}")
            return []
        except Exception as e:
            logger.error(f"Error parsing function calls from Gemini response: {e}", exc_info=True)
            return []

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._default_model

    @property
    def supports_function_calling(self) -> bool:
        """Check function calling support."""
        return True

    @property
    def supports_streaming(self) -> bool:
        """Check streaming support."""
        return True

    @property
    def max_context_length(self) -> int:
        """Get maximum context length for the model."""
        if "2.0" in self._default_model:
            return 1000000
        elif "1.5" in self._default_model:
            return 128000
        else:
            return 32768  

    @property
    def preferred_languages(self) -> List[str]:
        """Get list of preferred languages for the model."""
        return ["en", "vi", "ja"]

_gemini_client_instance = None

def get_gemini_client() -> GeminiClient:
    """
    Get a singleton instance of the GeminiClient.
    """
    global _gemini_client_instance
    if _gemini_client_instance is None:
        _gemini_client_instance = GeminiClient()
    return _gemini_client_instance