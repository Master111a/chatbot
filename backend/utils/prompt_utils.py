import json
import re
from typing import Dict, Any, Optional, List
from utils.logger import get_logger

logger = get_logger(__name__)


class PromptUtils:
    """
    Simplified utility class for basic prompt operations.
    Focus: Simple & Practical
    """
    
    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """
        Parse a JSON response from an LLM, with more robust extraction.
        Enhanced to handle Gemini and other LLM response formats.
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed JSON as dictionary or empty dict if parsing fails
        """
        if not response or not isinstance(response, str):
            logger.warning("Attempted to parse an empty or non-string response.")
            return {}

        cleaned_response = response.strip()
        
        patterns_to_remove = [
            r'<think.*?>.*?</think>',
            r'<think.*?/>',
            r'<think>',
            r'</think>',
            r'```(?:json)?$',
            r'^```(?:json)?',
        ]
        
        for pattern in patterns_to_remove:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.DOTALL).strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            pass

        try:
            json_blocks = re.findall(r'```json\s*(\{.*?\}|\[.*?\])\s*```', cleaned_response, re.DOTALL)
            if json_blocks:
                return json.loads(json_blocks[0])
            
            code_blocks = re.findall(r'```\s*(\{.*?\}|\[.*?\])\s*```', cleaned_response, re.DOTALL)
            if code_blocks:
                return json.loads(code_blocks[0])
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"Error during markdown JSON extraction: {e}") 

        json_patterns = [
            r'(?:response|result|json|answer)[:：]\s*(\{.*?\}|\[.*?\])',
            r'(?:here\'s|here is).*?[:：]\s*(\{.*?\}|\[.*?\])',
            r'(?:output|data)[:：]\s*(\{.*?\}|\[.*?\])',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    continue

        start_char, end_char = None, None
        
        first_brace = cleaned_response.find('{')
        first_bracket = cleaned_response.find('[')

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_char = '{'
            end_char = '}'
            start_index = first_brace
        elif first_bracket != -1:
            start_char = '['
            end_char = ']'
            start_index = first_bracket
        else:
            logger.warning(f"Failed to parse JSON response (no { '{' } or [ found): {cleaned_response[:100]}...")
            return {}

        balance = 0
        end_index = -1
        in_string = False
        escape_next = False
        
        for i in range(start_index, len(cleaned_response)):
            char = cleaned_response[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == start_char:
                    balance += 1
                elif char == end_char:
                    balance -= 1
                
                if balance == 0:
                    end_index = i
                    break
        
        if end_index != -1:
            json_str_candidate = cleaned_response[start_index : end_index + 1]
            try:
                result = json.loads(json_str_candidate)
                logger.debug(f"Successfully parsed JSON from response using fallback method")
                return result
            except json.JSONDecodeError as e_final:
                logger.warning(f"Failed to parse JSON response (final attempt on substring '{json_str_candidate[:50]}...'): {e_final}. Original response: {cleaned_response[:100]}...")
                return {}
        else:
            logger.warning(f"Failed to parse JSON response (could not find matching '{end_char}'): {cleaned_response[:100]}...")
            return {}
    

prompt_utils = PromptUtils()

__all__ = ["prompt_utils", "PromptUtils"] 