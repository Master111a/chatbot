import jwt
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.settings import get_settings
from utils.logger import get_logger
from utils import now

logger = get_logger(__name__)
settings = get_settings()
security = HTTPBearer()

class JWTManager:
    """
    JWT token manager with GitLab OAuth integration.
    """
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """
        Create JWT access token.
        """
        to_encode = data.copy()
        expire = now() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire.timestamp()})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify JWT token.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def get_gitlab_access_token(self, code: str) -> str:
        """
        Exchange authorization code for access token from GitLab.
        """
        token_url = f"{settings.GITLAB_BASE_URL}/oauth/token"
        
        data = {
            "client_id": settings.GITLAB_CLIENT_ID,
            "client_secret": settings.GITLAB_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": settings.GITLAB_REDIRECT_URI
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not retrieve access token from GitLab"
                )
            
            token_data = response.json()
            return token_data["access_token"]
    
    async def get_gitlab_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Retrieve user information from GitLab API.
        """
        user_url = f"{settings.GITLAB_BASE_URL}/api/v4/user"
        
        headers = {"Authorization": f"Bearer {access_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(user_url, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not retrieve user information from GitLab"
                )
            
            return response.json()
    
    async def create_user_token(self, user_info: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Create JWT token for user from GitLab information and save to DB.
        """
        from db.pg_manager import UserManager
        
        user_manager = UserManager()
        
        user_data = {
            "email": user_info["email"],
            "username": user_info["username"],
            "name": user_info["name"],
            "avatar_url": user_info.get("avatar_url"),
            "gitlab_id": user_info["id"]
        }
        
        db_user = await user_manager.create_or_update_user(user_data)
        
        payload = {
            "user_id": db_user["user_id"],
            "email": db_user["email"],
            "username": db_user["username"],
            "name": db_user["name"],
            "avatar_url": db_user["avatar_url"],
            "is_superuser": db_user["is_superuser"],
            "gitlab_id": db_user["gitlab_id"]
        }
        
        token = self.create_access_token(payload)
        return token, db_user

jwt_manager = JWTManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Get current user information from JWT token (login required).
    """
    token = credentials.credentials
    payload = jwt_manager.verify_token(token)
    return payload

async def optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[Dict[str, Any]]:
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

async def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Require user to be an admin.
    """
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can perform this action"
        )
    return current_user