

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from utils.logger import get_logger
from utils import now
from db.pg_manager import ChatHistory
from services.language_detector import LanguageDetector
from llm.llm_router import get_llm_router
from config.settings import get_settings

logger = get_logger(__name__)

class ChatService:
    """
    Chat service with utility methods for Orchestrator
    Contains reusable methods that Orchestrator calls when needed
    """
    
    def __init__(self):
        """Initialize chat service utilities"""
        self.chat_history = ChatHistory()
        self.language_detector = LanguageDetector()
        self.llm_router = get_llm_router()  
        self.settings = get_settings()
        
        logger.info("ChatService utilities initialized with singleton LLM router")
    
    async def save_chat_message(
        self,
        session_id: str,
        message: str,
        response: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save chat message to database
        
        Args:
            session_id: Chat session ID
            message: User message
            response: AI response
            user_id: Optional user ID
            metadata: Optional metadata (citations, etc.)
            
        Returns:
            Saved message data
        """
        try:
            saved_message = await self.chat_history.add_message(
                session_id=session_id,
                message=message,
                response=response,
                user_id=user_id,
                metadata=metadata
            )
            
            logger.info(f"Saved chat message to session {session_id}")
            return saved_message
            
        except Exception as e:
            logger.error(f"Error saving chat message: {e}")
            raise
    
    async def update_message_metadata(
        self,
        message_id: str,
        metadata_update: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a specific message
        
        Args:
            message_id: Message ID to update
            metadata_update: New metadata to merge
            
        Returns:
            Success status
        """
        try:
            return await self.chat_history.update_message_metadata(
                message_id=message_id,
                metadata_update=metadata_update
            )
        except Exception as e:
            logger.error(f"Error updating message metadata: {e}")
            return False
    
    async def generate_title_from_message(
        self,
        message: str,
        language: str = "vi"
    ) -> str:
        """
        Generate session title from first message
        
        Args:
            message: First message in session
            language: Language for title generation
            
        Returns:
            Generated title string
        """
        try:
            title_prompts = {
                "vi": f"""Tạo một tiêu đề ngắn gọn (tối đa {self.settings.CHAT_SESSION_TITLE_MAX_WORDS} từ) cho cuộc trò chuyện dựa trên tin nhắn này:

"{message}"

Chỉ trả về tiêu đề, không giải thích thêm.""",

                "en": f"""Create a concise title (max {self.settings.CHAT_SESSION_TITLE_MAX_WORDS} words) for this conversation based on this message:

"{message}"

Return only the title, no additional explanation.""",

                "ja": f"""このメッセージに基づいて、会話の簡潔なタイトル（最大{self.settings.CHAT_SESSION_TITLE_MAX_WORDS}語）を作成してください：

"{message}"

タイトルのみを返してください。追加の説明は不要です。"""
            }
            
            prompt = title_prompts.get(language, title_prompts["vi"])
            
            title = await self.llm_router.generate_text(
                prompt=prompt,
                language=language,
                max_tokens=50,
                temperature=0.1
            )
            
            title = title.strip().strip('"').strip("'")
            
            if len(title) > self.settings.CHAT_SESSION_TITLE_MAX_CHARS:
                title = title[:self.settings.CHAT_SESSION_TITLE_MAX_CHARS-3] + "..."
            
            if not title or len(title.split()) > self.settings.CHAT_SESSION_TITLE_MAX_WORDS + 2:
                title = message[:50] + "..." if len(message) > 50 else message
            
            return title
            
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            fallback_title = message[:50] + "..." if len(message) > 50 else message
            return fallback_title
    
    async def generate_follow_up_questions(
        self,
        query: str,
        response: str,
        language: str = "vi"
    ) -> List[str]:
        """
        Generate follow-up questions based on query and response
        
        Args:
            query: Original user query
            response: AI response
            language: Language for questions
            
        Returns:
            List of follow-up questions
        """
        try:
            follow_up_prompts = {
                "vi": f"""Dựa trên câu hỏi và câu trả lời sau, hãy tạo ra {self.settings.NUM_FOLLOW_UP_QUESTIONS} câu hỏi tiếp theo hữu ích mà người dùng có thể quan tâm.

Câu hỏi gốc: {query}
Câu trả lời: {response}

Trả về danh sách câu hỏi, mỗi câu một dòng, không đánh số.""",

                "en": f"""Based on the following question and answer, create {self.settings.NUM_FOLLOW_UP_QUESTIONS} useful follow-up questions that the user might be interested in.

Original question: {query}
Answer: {response}

Return a list of questions, one per line, without numbering.""",

                "ja": f"""以下の質問と回答に基づいて、ユーザーが興味を持ちそうな{self.settings.NUM_FOLLOW_UP_QUESTIONS}つの有用なフォローアップ質問を作成してください。

元の質問: {query}
回答: {response}

質問のリストを返してください。1行に1つずつ、番号は付けないでください。"""
            }
            
            prompt = follow_up_prompts.get(language, follow_up_prompts["vi"])
            
            follow_up_text = await self.llm_router.generate_text(
                prompt=prompt,
                language=language,
                max_tokens=300,
                temperature=0.7
            )
            
            questions = []
            for line in follow_up_text.split('\n'):
                line = line.strip()
                if line and len(line) > 10:  
                    line = line.lstrip('-•*1234567890. ')
                    questions.append(line)
            
            questions = questions[:self.settings.NUM_FOLLOW_UP_QUESTIONS]
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []
    
    async def create_streaming_response(
        self,
        response_text: str,
        language: str = "vi"
    ) -> AsyncGenerator[str, None]:
        """
        Convert response text to streaming tokens compatible with frontend
        Used only for chitchat responses. RAG responses now stream directly from LLM.
        Preserves markdown formatting by streaming in character chunks
        
        Args:
            response_text: Complete response text to stream (may contain markdown)
            language: Language for streaming context
            
        Yields:
            Streaming response tokens in SSE format that preserve markdown
        """
        try:
            if not response_text:
                return
            
            chunk_size = 20 
            text_length = len(response_text)
            
            for i in range(0, text_length, chunk_size):
                chunk = response_text[i:i + chunk_size]
                
                inner_token = {
                    "type": "response_token",
                    "token": chunk
                }
                
                outer_token = {
                    "type": "token", 
                    "text": json.dumps(inner_token, ensure_ascii=False)
                }
                
                sse_event = f"data: {json.dumps(outer_token, ensure_ascii=False)}\n\n"

                yield sse_event
                
                await asyncio.sleep(0.03)
            
            stream_end_inner = {
                "type": "stream_end"
            }
            
            stream_end_outer = {
                "type": "token",
                "text": json.dumps(stream_end_inner, ensure_ascii=False)
            }
            
            end_event = f"data: {json.dumps(stream_end_outer, ensure_ascii=False)}\n\n"
            yield end_event
                
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            error_data = {
                "type": "error",
                "message": f"Streaming error: {str(e)}"
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    async def clear_session(self, session_id: str) -> Dict[str, Any]:
        """
        Clear chat session history
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            Operation result
        """
        try:
            success = await self.chat_history.clear_history(session_id)
            
            if success:
                logger.info(f"Cleared session history: {session_id}")
                return {
                    "session_id": session_id,
                    "status": "cleared",
                    "timestamp": now().isoformat()
                }
            else:
                return {
                    "session_id": session_id,
                    "status": "error",
                    "message": "Failed to clear session",
                    "timestamp": now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            return {
                "session_id": session_id,
                "status": "error", 
                "message": str(e),
                "timestamp": now().isoformat()
            }
    
    def get_default_responses(self, language: str) -> Dict[str, str]:
        """
        Get default responses for different scenarios
        
        Args:
            language: Language code
            
        Returns:
            Dictionary of default responses
        """
        responses = {
            "vi": {
                "greeting": f"Xin chào! Tôi là {self.settings.BOT_NAME}, trợ lý AI của bạn. Tôi có thể giúp gì cho bạn hôm nay?",
                "error": "Đã xảy ra lỗi trong quá trình xử lý yêu cầu của bạn.",
                "no_info": "Tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn.",
                "refused": "Tôi không thể xử lý yêu cầu này."
            },
            "en": {
                "greeting": f"Hello! I'm {self.settings.BOT_NAME}, your AI assistant. How can I help you today?",
                "error": "An error occurred while processing your request.",
                "no_info": "I couldn't find information related to your question.",
                "refused": "I cannot process this request."
            },
            "ja": {
                "greeting": f"こんにちは！私は{self.settings.BOT_NAME}、あなたのAIアシスタントです。今日はどのようにお手伝いできますか？",
                "error": "リクエストの処理中にエラーが発生しました。",
                "no_info": "ご質問に関連する情報が見つかりませんでした。",
                "refused": "このリクエストを処理できません。"
            }
        }
        
        return responses.get(language, responses["vi"])
    
    def format_response_metadata(
        self,
        orchestrator_result: Dict[str, Any],
        processing_time: float
    ) -> Dict[str, Any]:
        """
        Format metadata for chat response
        
        Args:
            orchestrator_result: Result from orchestrator
            processing_time: Time taken to process
            
        Returns:
            Formatted metadata
        """
        metadata = {
            "processing_time": round(processing_time, 3),
            "query_type": orchestrator_result.get("type"),
            "routing_reason": orchestrator_result.get("routing_reason"),
            "timestamp": now().isoformat()
        }
        
        if orchestrator_result.get("metadata"):
            metadata.update(orchestrator_result["metadata"])
        
        return metadata

    async def stream_llm_response(
        self,
        prompt: str,
        language: str,
        max_tokens: int,
        temperature: float,
        session_id: str,
        message: str,
        user_id: Optional[str],
        metadata: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from LLM prompt (works for both RAG and chitchat)
        Saves message only after streaming is complete with full response
        
        Args:
            prompt: LLM prompt to generate from
            language: Language for generation
            max_tokens: Maximum tokens
            temperature: Generation temperature
            session_id: Session ID for saving
            message: Original user message
            user_id: Optional user ID
            metadata: Message metadata
            
        Yields:
            Raw tokens from LLM for chat_routes to format
        """
        try:
            response_text = ""
            
            # Stream tokens directly from LLM
            async for token in self.llm_router.generate_text_stream(
                prompt=prompt,
                language=language,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                response_text += token
                yield token
            
            # Save complete message to database after streaming
            saved_message = await self.save_chat_message(
                session_id=session_id,
                message=message,
                response=response_text,
                user_id=user_id,
                metadata=metadata
            )
            
            # Send completion metadata
            yield json.dumps({
                "type": "completion_metadata",
                "message_id": saved_message.get("message_id"),
                "metadata": metadata
            })
            
        except Exception as e:
            logger.error(f"Error in LLM streaming: {e}")
            error_data = {
                "type": "error",
                "message": f"LLM streaming error: {str(e)}"
            }
            yield json.dumps(error_data)

chat_service = ChatService()