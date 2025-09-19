from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional
from .jwt_manager import jwt_manager
from config.settings import get_settings

security = HTTPBearer()
settings = get_settings()

class AdminManager:
    """
    Manager for admin authentication and authorization.
    """
    
    def __init__(self):
        """
        Initialize the AdminManager.
        """
        pass
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """
        Get current user information from JWT token (login required).
        """
        token = credentials.credentials
        payload = jwt_manager.verify_token(token)
        return payload
    
    async def optional_user(self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[Dict[str, Any]]:
        """
        Get user information if token is present (login not required).
        """
        if not credentials:
            return None
        
        try:
            token = credentials.credentials
            payload = jwt_manager.verify_token(token)
            return payload
        except HTTPException:
            return None
    
    async def require_admin(self, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        """
        Require user to be an admin.
        """
        if not current_user.get("is_superuser", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can perform this action"
            )
        return current_user

admin_manager = AdminManager() 