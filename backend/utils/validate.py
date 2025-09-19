import uuid
import re
import unicodedata
from typing import Dict, Optional


def is_valid_uuid(uuid_str: str) -> bool:
    """
    Validate if a string is a valid UUID format.
    
    Args:
        uuid_str: String to validate
        
    Returns:
        True if valid UUID format, False otherwise
    """
    if not uuid_str or not isinstance(uuid_str, str):
        return False
    
    if len(uuid_str) not in [32, 36]:
        return False
    
    try:
        uuid.UUID(uuid_str)
        return True
    except (ValueError, AttributeError):
        return False

def format_sse_event(data: str) -> str:
    """
    Format data according to SSE (Server-Sent Events) standard.
    
    Args:
        data: Data to be sent
        
    Returns:
        Properly formatted SSE string
    """
    return f"data: {data}\n\n"



def sanitize_metadata(metadata: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Sanitize metadata for MinIO compatibility.
    MinIO only supports US-ASCII encoded characters in metadata.
    
    Args:
        metadata: Original metadata dictionary
        
    Returns:
        Sanitized metadata dictionary with ASCII-only values
    """
    if not metadata:
        return metadata
    
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            continue
            
        str_value = str(value)
        
        normalized = unicodedata.normalize('NFD', str_value)
        
        ascii_value = ''.join(
            char for char in normalized 
            if unicodedata.category(char) != 'Mn'
        )
        
        ascii_value = ascii_value.encode('ascii', 'ignore').decode('ascii')
        
        replacements = {
            'đ': 'd', 'Đ': 'D',
            'ă': 'a', 'Ă': 'A',
            'â': 'a', 'Â': 'A', 
            'ê': 'e', 'Ê': 'E',
            'ô': 'o', 'Ô': 'O',
            'ơ': 'o', 'Ơ': 'O',
            'ư': 'u', 'Ư': 'U'
        }
        
        for vietnamese, replacement in replacements.items():
            ascii_value = ascii_value.replace(vietnamese, replacement)
        
        ascii_value = re.sub(r'[^\x20-\x7E]', '', ascii_value).strip()
        
        if ascii_value:
            sanitized[key] = ascii_value
            
    return sanitized
