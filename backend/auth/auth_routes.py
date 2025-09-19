from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import RedirectResponse
from typing import Dict, Any
from auth.jwt_manager import jwt_manager, get_current_user
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()

@router.get("/gitlab/login")
async def gitlab_login():
    """
    Redirect user to GitLab for login.
    """
    if not settings.GITLAB_CLIENT_ID:
        raise HTTPException(status_code=400, detail="GitLab OAuth not configured")
    
    auth_url = (
        f"{settings.GITLAB_BASE_URL}/oauth/authorize?"
        f"client_id={settings.GITLAB_CLIENT_ID}&"
        f"redirect_uri={settings.GITLAB_REDIRECT_URI}&"
        f"response_type=code&"
        f"scope=read_user"
    )
    
    return RedirectResponse(url=auth_url)

@router.get("/gitlab/callback")
async def gitlab_callback(
    code: str = Query(...), 
    error: str = Query(None),
    session_id: str = Query(None)
):
    """
    Handle callback from GitLab after user login.
    Can link anonymous session to user account.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"GitLab OAuth error: {error}")
    
    try:
        gitlab_access_token = await jwt_manager.get_gitlab_access_token(code)
        user_info = await jwt_manager.get_gitlab_user_info(gitlab_access_token)
        
        access_token, db_user = await jwt_manager.create_user_token(user_info)
        
        if session_id:
            from db.pg_manager import ChatHistory
            chat_history = ChatHistory()
            await chat_history.link_anonymous_session_to_user(session_id, db_user["user_id"])
            logger.info(f"Linked session {session_id} to user {db_user['email']}")
        
        logger.info(f"User login successful: {user_info['email']}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "user_id": db_user["user_id"],
                "email": db_user["email"],
                "username": db_user["username"],
                "name": db_user["name"],
                "avatar_url": db_user.get("avatar_url"),
                "is_superuser": db_user["is_superuser"]
            },
            "session_linked": session_id is not None
        }
        
    except Exception as e:
        logger.error(f"Error during GitLab OAuth: {str(e)}")
        raise HTTPException(status_code=400, detail="GitLab login failed")

@router.get("/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user information.
    """
    return {
        "user_id": current_user["user_id"],
        "email": current_user["email"],
        "username": current_user["username"],
        "name": current_user["name"],
        "avatar_url": current_user.get("avatar_url"),
        "is_superuser": current_user.get("is_superuser", False)
    }

@router.post("/logout")
async def logout():
    """
    Logout (client side token removal).
    """
    return {"message": "Logout successful"}

@router.get("/sessions")
async def get_user_sessions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get list of sessions for the current user.
    """
    from db.pg_manager import ChatHistory
    
    chat_history = ChatHistory()
    offset = (page - 1) * limit
    
    sessions = await chat_history.get_user_sessions(
        user_id=current_user["user_id"],
        limit=limit,
        offset=offset
    )
    
    return {
        "sessions": sessions,
        "page": page,
        "limit": limit,
        "total": len(sessions)
    }

@router.post("/link-session")
async def link_current_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Link current session to user account.
    """
    from db.pg_manager import ChatHistory
    
    chat_history = ChatHistory()
    success = await chat_history.link_anonymous_session_to_user(
        session_id=session_id,
        user_id=current_user["user_id"]
    )
    
    if success:
        return {"message": "Session linked to account", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session does not exist or is already linked")