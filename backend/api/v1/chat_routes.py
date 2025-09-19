from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import json
from utils import now
import asyncio

from config.settings import get_settings
from db.pg_manager import ChatHistory
from models import (
    CreateSessionRequest, CreateSessionResponse,
    SessionDetailResponse, SessionListResponse, 
    UpdateSessionTitleRequest, UpdateSessionTitleResponse,
    ChatMessage
)
from models.api_schemas import StreamChatRequest
from auth.jwt_manager import JWTManager, optional_user
from utils.logger import get_logger
from utils.validate import is_valid_uuid
from services.language_detector import LanguageDetector
from agents.orchestrator import Orchestrator
from services.chat_service import chat_service

logger = get_logger(__name__)

router = APIRouter()
settings = get_settings()
chat_history = ChatHistory()
jwt_manager = JWTManager()
orchestrator = Orchestrator()

@router.post("/create-session", response_model=CreateSessionResponse, summary="Create new chat session with title")
async def create_session(
    request: CreateSessionRequest,
    user: Optional[Dict[str, Any]] = Depends(optional_user)
):
    """
    Create new chat session and generate title from first query
    
    Creates new session in database and generates meaningful title
    based on the provided query using LLM
    
    Args:
        request: Session creation request with initial query
        user: Optional authenticated user information
        
    Returns:
        New session information with generated title
    """
    try:
        user_id = user.get("user_id") if user else None
        
        session = await chat_history.create_session(user_id=user_id)
        session_id = session["session_id"]
        
        title = await chat_service.generate_title_from_message(request.query)
        
        if title and title != request.query:
            await chat_history.update_session_title(session_id, title)
            logger.info(f"Generated title for new session {session_id}: {title}")
        else:
            title = request.query[:50] + "..." if len(request.query) > 50 else request.query
            await chat_history.update_session_title(session_id, title)
        
        language_detector = LanguageDetector()
        language = await language_detector.detect_language(request.query)
        
        return CreateSessionResponse(
            session_id=str(session_id),
            title=title,
            language=language,
            created_at=session.get("created_at"),
            user_id=user_id,
            is_anonymous=user_id is None
        )
        
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating session: {str(e)}"
        )


@router.post("/stream", summary="Stream message response")
async def stream_message(
    request: Request,
    chat_request: StreamChatRequest,
    user: Optional[Dict[str, Any]] = Depends(optional_user)
):
    """
    Stream message response with real-time effect through Orchestrator
    
    This endpoint provides the main chat functionality:
    1. Validates session exists
    2. Routes through Orchestrator for processing
    3. Returns streaming response via Server-Sent Events (SSE)
    4. Compatible with frontend streamChat function format
    
    Args:
        request: FastAPI request object
        chat_request: Chat request with message and session_id
        user: Optional authenticated user information
        
    Returns:
        StreamingResponse with SSE format tokens
    """
    
    try:
        user_id = user.get("user_id") if user else None
        
        if not chat_request.session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required. Use /create-session endpoint to create a new session first."
            )
        
        if not is_valid_uuid(chat_request.session_id):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session_id format: {chat_request.session_id}"
            )
        
        session_info = await chat_history.get_session_info(chat_request.session_id)
        if not session_info:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {chat_request.session_id}. Use /create-session endpoint to create a new session."
            )
        
        session_id = chat_request.session_id
        logger.info(f"Starting streaming chat for session: {session_id}")
            
        async def event_generator():
            """
            Generator function for Server-Sent Events streaming
            Yields properly formatted SSE tokens compatible with frontend
            """
            try:
                yield f"data: {json.dumps({'type': 'init', 'session_id': session_id, 'timestamp': now().isoformat()})}\n\n"
                
                orchestrator_response = orchestrator.process_chat_message(
                    message=chat_request.message,
                    session_id=session_id,
                    user_id=user_id,
                    streaming=True,
                    max_tokens=chat_request.max_tokens
                )
                
                async for item in orchestrator_response:
                    if isinstance(item, dict) and item.get("type") in ["rag_stream", "chitchat_stream"]:
                        token_buffer = ""
                        last_send_time = asyncio.get_event_loop().time()
                        buffer_delay = 0.1 
                        
                        async for token in chat_service.stream_llm_response(
                            prompt=item["prompt"],
                            language=item["language"],
                            max_tokens=item["max_tokens"],
                            temperature=item["temperature"],
                            session_id=item["session_id"],
                            message=item["message"],
                            user_id=item["user_id"],
                            metadata=item["metadata"]
                        ):
                            if token.startswith('{"type"'):
                                try:
                                    metadata_obj = json.loads(token)
                                    if metadata_obj.get("type") == "completion_metadata":
                                        
                                        if token_buffer.strip():
                                            inner_token = {
                                                "type": "response_token",
                                                "token": token_buffer
                                            }
                                            
                                            outer_token = {
                                                "type": "token",
                                                "text": json.dumps(inner_token, ensure_ascii=False)
                                            }
                                            
                                            yield f"data: {json.dumps(outer_token, ensure_ascii=False)}\n\n"
                                            token_buffer = ""
                                        
                                        yield f"data: {json.dumps({'type': 'complete', 'message_id': metadata_obj.get('message_id'), 'metadata': metadata_obj.get('metadata')})}\n\n"
                                    elif metadata_obj.get("type") == "error":
                                        yield f"data: {json.dumps(metadata_obj)}\n\n"
                                except json.JSONDecodeError:
                                    pass
                            else:
                                # Add token to buffer
                                token_buffer += token
                                current_time = asyncio.get_event_loop().time()
                                
                                # Send buffer if enough time has passed or buffer is getting large
                                if (current_time - last_send_time >= buffer_delay) or len(token_buffer) >= 100:
                                    if token_buffer.strip():
                                        inner_token = {
                                            "type": "response_token",
                                            "token": token_buffer
                                        }
                                        
                                        outer_token = {
                                            "type": "token",
                                            "text": json.dumps(inner_token, ensure_ascii=False)
                                        }
                                        
                                        yield f"data: {json.dumps(outer_token, ensure_ascii=False)}\n\n"
                                        await asyncio.sleep(0.02)
                                        
                                        token_buffer = ""
                                        last_send_time = current_time
                        
                        # Send any remaining tokens in buffer
                        if token_buffer.strip():
                            inner_token = {
                                "type": "response_token",
                                "token": token_buffer
                            }
                            
                            outer_token = {
                                "type": "token",
                                "text": json.dumps(inner_token, ensure_ascii=False)
                            }
                            
                            yield f"data: {json.dumps(outer_token, ensure_ascii=False)}\n\n"
                        
                        stream_end_inner = {
                            "type": "stream_end"
                        }
                        
                        stream_end_outer = {
                            "type": "token",
                            "text": json.dumps(stream_end_inner, ensure_ascii=False)
                        }
                        
                        yield f"data: {json.dumps(stream_end_outer, ensure_ascii=False)}\n\n"
                    
                    elif isinstance(item, str) and item.startswith("data: "):
                        yield item
                
                yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in event generator: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'timestamp': now().isoformat()})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stream_message: {str(e)}")
        
        async def error_generator():
            """Error generator for streaming error responses"""
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'timestamp': now().isoformat()})}\n\n"
        
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )
    

@router.get("/session/{session_id}", response_model=SessionDetailResponse, summary="Get chat session with full history")
async def get_session_detail(
    session_id: str,
    user: Optional[Dict[str, Any]] = Depends(optional_user)
):
    """
    Get detailed information about a chat session including full chat history
    
    Provides complete session information including:
    - Session metadata and information
    - Complete message history with responses
    - User authorization check for privacy
    
    Args:
        session_id: Unique session identifier
        user: Optional authenticated user information
        
    Returns:
        Complete session details with message history
    """
    try:
        session_info = await chat_history.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        user_id = user.get("user_id") if user else None
        if not session_info.get("is_anonymous") and user_id != session_info.get("user_id"):
            raise HTTPException(
                status_code=403,
                detail="Not authorized to access this session"
            )
        
        chat_messages = await chat_history.get_history(
            session_id=session_id,
            limit=1000,
            offset=0
        )
        
        message_models = []
        primary_language = None
        
        for msg in chat_messages:
            if not primary_language and msg.get("metadata", {}).get("language"):
                primary_language = msg["metadata"]["language"]
            
            message_models.append(ChatMessage(
                message_id=str(msg["message_id"]),
                message=msg["message"],
                response=msg["response"],
                timestamp=msg["timestamp"],
                metadata=msg.get("metadata", {})
            ))
        
        return SessionDetailResponse(
            session_id=str(session_info["session_id"]),
            title=session_info.get("title", "Untitled Chat"),
            user_id=session_info.get("user_id"),
            is_anonymous=session_info.get("is_anonymous", True),
            created_at=session_info.get("created_at"),
            updated_at=session_info.get("updated_at"),
            message_count=len(message_models),
            language=primary_language,
            messages=message_models,
            metadata=session_info.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session detail: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving session detail: {str(e)}"
        )

@router.put("/session/{session_id}/title", response_model=UpdateSessionTitleResponse, summary="Update session title")
async def update_session_title(
    session_id: str,
    request: UpdateSessionTitleRequest,
    user: Optional[Dict[str, Any]] = Depends(optional_user)
):
    """
    Update title for a chat session
    
    Allows users to customize their session titles for better organization
    
    Args:
        session_id: Session identifier to update
        request: New title information
        user: Optional authenticated user information
        
    Returns:
        Update confirmation with new title
    """
    try:
        session_info = await chat_history.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        user_id = user.get("user_id") if user else None
        
        if not session_info["is_anonymous"] and user_id != session_info["user_id"]:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to update this session"
            )
        
        success = await chat_history.update_session_title(session_id, request.title)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update session title"
            )
        
        logger.info(f"Updated title for session {session_id}: '{request.title}'")
        
        return UpdateSessionTitleResponse(
            session_id=session_id,
            title=request.title,
            updated=True,
            timestamp=now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session title: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating session title: {str(e)}"
        )

@router.delete("/session/{session_id}", summary="Clear chat session history")
async def clear_session(
    session_id: str,
    user: Optional[Dict[str, Any]] = Depends(optional_user)
):
    """
    Clear chat session history while keeping session active
    
    Removes all messages from the session but keeps the session
    itself so new messages can still be added
    
    Args:
        session_id: Session identifier to clear
        user: Optional authenticated user information
        
    Returns:
        Operation status and confirmation
    """
    try:
        result = await chat_service.clear_session(session_id)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        
        logger.info(f"Cleared session history: {session_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing session: {str(e)}"
        )

@router.delete("/session/{session_id}/delete", summary="Delete entire chat session")
async def delete_session(
    session_id: str,
    user: Optional[Dict[str, Any]] = Depends(optional_user)
):
    """
    Delete entire chat session and all its messages permanently
    
    Completely removes the session from the database including
    all messages and metadata. Cannot be undone.
    
    Args:
        session_id: Session identifier to delete
        user: Optional authenticated user information
        
    Returns:
        Deletion confirmation
    """
    try:
        session_info = await chat_history.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        user_id = user.get("user_id") if user else None
        
        if not session_info.get("is_anonymous") and user_id != session_info.get("user_id"):
            raise HTTPException(
                status_code=403,
                detail="Not authorized to delete this session"
            )
        
        success = await chat_history.delete_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete session"
            )
        
        logger.info(f"Deleted session {session_id} by user {user_id}")
        
        return {
            "session_id": session_id,
            "status": "deleted",
            "timestamp": now().isoformat(),
            "deleted_by": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )

@router.get("/sessions", response_model=SessionListResponse, summary="Get user chat sessions list")
async def get_user_sessions(
    limit: int = Query(20, description="Maximum number of sessions to return", ge=1, le=100),
    offset: int = Query(0, description="Number of sessions to skip", ge=0),
    include_empty: bool = Query(False, description="Include sessions with no messages"),
    user: Optional[Dict[str, Any]] = Depends(optional_user)
):
    """
    Get paginated list of chat sessions for the authenticated user
    
    Returns sessions ordered by last activity (most recent first)
    Anonymous users will receive empty list for privacy
    
    Args:
        limit: Maximum number of sessions per page
        offset: Number of sessions to skip for pagination
        include_empty: Whether to include sessions with no messages
        user: Optional authenticated user information
        
    Returns:
        Paginated list of user sessions with basic information
    """
    try:
        user_id = user.get("user_id") if user else None
        
        if not user_id:
            return SessionListResponse(
                sessions=[],
                total=0,
                limit=limit,
                offset=offset,
                is_anonymous=True,
                user_id=None
            )
        
        sessions = await chat_history.get_user_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset,
            include_empty=include_empty
        )
        
        total_count = await chat_history.count_user_sessions(
            user_id=user_id,
            include_empty=include_empty
        )
        
        session_summaries = []
        for session in sessions:
            session_summaries.append({
                "session_id": str(session.get("session_id")),
                "title": session.get("title", "Untitled Chat"),
                "message_count": session.get("message_count", 0),
                "last_activity": session.get("last_activity"),
                "language": session.get("language"),
                "is_empty": session.get("message_count", 0) == 0
            })
        
        return SessionListResponse(
            sessions=session_summaries,
            total=total_count,
            limit=limit,
            offset=offset,
            is_anonymous=False,
            user_id=user_id
        )
        
    except Exception as e:
        logger.error(f"Error getting user sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting user sessions: {str(e)}"
        )
