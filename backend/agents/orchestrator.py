"""
Orchestrator - Central coordination hub for chat processing
Manages the complete chat pipeline: Query -> Semantic Router -> RAG/Chitchat -> Streaming Response
"""

import json
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from utils.logger import get_logger
from services.semantic_router_service import SemanticRouterService
from agents.rag_agent import RAGAgentInput
from services.types import QueryType
from services.chat_service import ChatService
import asyncio

logger = get_logger(__name__)

class Orchestrator:
    """
    Central orchestrator for chat pipeline coordination
    
    Handles the complete flow:
    Chat Route -> Orchestrator -> Semantic Router -> Refined Query ->
    Chitchat -> SSE Response OR RAG -> Query -> SSE Response
    """
    
    def __init__(self):
        """Initialize Orchestrator with required dependencies"""
        self._agent_registry: Dict[str, Any] = {}
        self._agent_instances: Dict[str, Any] = {}
        self.semantic_router_service = SemanticRouterService()
        
        self._chat_service = None
        
        self._register_default_agents()
        
        logger.info("Orchestrator initialized as central coordination hub")

    @property
    def chat_service(self):
        """Lazy property for ChatService to avoid circular imports"""
        if self._chat_service is None:
            self._chat_service = ChatService()
        return self._chat_service

    def _register_default_agents(self):
        """Register default agents available to the orchestrator"""
        try:
            from agents.rag_agent import RAGAgent
            self._agent_registry['rag'] = RAGAgent
            logger.info("Successfully registered RAG agent")
        except Exception as e:
            logger.error(f"Failed to register default agents: {e}")
            raise

    async def process_chat_message(
        self,
        message: str,
        session_id: str,
        user_id: Optional[str] = None,
        language: Optional[str] = None,
        streaming: bool = False,
        max_tokens: int = 8192
    ) -> AsyncGenerator[str, None]:
        """
        Main entry point for processing chat messages with full pipeline and streaming
        
        This method handles the complete chat flow:
        1. Language detection if not provided
        2. Query routing through semantic router
        3. Execution via appropriate agent (RAG/Chitchat)
        4. Direct streaming from LLM for RAG responses
        5. Message persistence to database
        
        Args:
            message: User input message
            session_id: Unique chat session identifier
            user_id: Optional authenticated user ID
            language: Language code (auto-detected if None)
            streaming: Enable streaming response (default False)
            max_tokens: Maximum tokens for response generation
            
        Yields:
            Streaming response tokens in SSE format compatible with frontend
        """
        start_time = time.time()
        
        try:
            if not language:
                from services.language_detector import LanguageDetector
                language_detector = LanguageDetector()
                language = await language_detector.detect_language(message)
            
            logger.info(f"Processing chat message: session={session_id}, language={language}, streaming={streaming}")
            
            if streaming:
                yield self._format_sse_token({
                    "type": "status",
                    "message": "Processing query..." if language == "en" else ("Đang phân tích..." if language == "vi" else "処理中...")
                })
            
            orchestrator_result = await self.route_and_execute(
                query=message,
                language=language,
                session_id=session_id,
                max_tokens=max_tokens,
                streaming=streaming
            )
            
            processing_time = time.time() - start_time
            
            if orchestrator_result.get("type") == "rag" and orchestrator_result.get("prompt"):
                if streaming:
                    metadata = self.chat_service.format_response_metadata(orchestrator_result, processing_time)
                    if orchestrator_result.get("metadata", {}).get("citations"):
                        metadata["citations"] = orchestrator_result["metadata"]["citations"]
                    
                    yield {
                        "type": "rag_stream",
                        "prompt": orchestrator_result["prompt"],
                        "max_tokens": orchestrator_result.get("max_tokens", max_tokens),
                        "temperature": orchestrator_result.get("temperature", 0.1),
                        "language": language,
                        "session_id": session_id,
                        "message": message,
                        "user_id": user_id,
                        "metadata": metadata
                    }
                else:
                    from llm.llm_router import get_llm_router
                    llm_router = get_llm_router()
                    
                    response_text = await llm_router.generate_text(
                        prompt=orchestrator_result["prompt"],
                        language=language,
                        max_tokens=orchestrator_result.get("max_tokens", max_tokens),
                        temperature=orchestrator_result.get("temperature", 0.1)
                    )
                    
                    metadata = self.chat_service.format_response_metadata(orchestrator_result, processing_time)
                    if orchestrator_result.get("metadata", {}).get("citations"):
                        metadata["citations"] = orchestrator_result["metadata"]["citations"]
                    
                    saved_message = await self.chat_service.save_chat_message(
                        session_id=session_id,
                        message=message,
                        response=response_text,
                        user_id=user_id,
                        metadata=metadata
                    )
                    
                    yield self._format_sse_token({
                        "type": "response_complete",
                        "text": response_text
                    })
                    
                    yield self._format_sse_token({
                        "type": "complete",
                        "message_id": saved_message.get("message_id"),
                        "metadata": metadata
                    })
            elif orchestrator_result.get("type") == "chitchat_stream" and orchestrator_result.get("prompt"):
                
                if streaming:
                    metadata = self.chat_service.format_response_metadata(orchestrator_result, processing_time)
                    
                    yield {
                        "type": "chitchat_stream",
                        "prompt": orchestrator_result["prompt"],
                        "max_tokens": orchestrator_result.get("max_tokens", 300),
                        "temperature": orchestrator_result.get("temperature", 0.7),
                        "language": language,
                        "session_id": session_id,
                        "message": message,
                        "user_id": user_id,
                        "metadata": metadata
                    }
                else:
                    from llm.llm_router import get_llm_router
                    llm_router = get_llm_router()
                    
                    response_text = await llm_router.generate_text(
                        prompt=orchestrator_result["prompt"],
                        language=language,
                        max_tokens=orchestrator_result.get("max_tokens", 300),
                        temperature=orchestrator_result.get("temperature", 0.7)
                    )
                    
                    metadata = self.chat_service.format_response_metadata(orchestrator_result, processing_time)
                    
                    saved_message = await self.chat_service.save_chat_message(
                        session_id=session_id,
                        message=message,
                        response=response_text,
                        user_id=user_id,
                        metadata=metadata
                    )
                    
                    yield self._format_sse_token({
                        "type": "response_complete",
                        "text": response_text
                    })
                    
                    yield self._format_sse_token({
                        "type": "complete",
                        "message_id": saved_message.get("message_id"),
                        "metadata": metadata
                    })
            else:
                response_text = orchestrator_result.get("response", "")
                
                if streaming and response_text:
                    async for token in self.chat_service.create_streaming_response(response_text, language):
                        yield token
                else:
                    yield self._format_sse_token({
                        "type": "response_complete",
                        "text": response_text
                    })
                
                metadata = self.chat_service.format_response_metadata(orchestrator_result, processing_time)
                
                saved_message = await self.chat_service.save_chat_message(
                    session_id=session_id,
                    message=message,
                    response=response_text,
                    user_id=user_id,
                    metadata=metadata
                )
                
                yield self._format_sse_token({
                    "type": "complete",
                    "message_id": saved_message.get("message_id"),
                    "metadata": metadata
                })
            
            logger.info(f"Successfully processed chat message in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in process_chat_message: {e}", exc_info=True)
            
            error_response = self.chat_service.get_default_responses(language or "vi")["error"]
            
            yield self._format_sse_token({
                "type": "error",
                "message": error_response,
                "error_details": str(e)
            })
    
    async def route_and_execute(
        self,
        query: str,
        language: str = "vi",
        session_id: Optional[str] = None,
        max_tokens: int = 8192,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Core routing and execution logic
        Query -> Semantic Router -> RAG/Chitchat execution
        
        Args:
            query: User query
            language: Language code
            session_id: Chat session ID
            max_tokens: Max tokens
            streaming: Streaming flag
            
        Returns:
            Execution result with response and metadata
        """
        original_query = query
        start_time = time.time()
        
        try:
            route_result = await self.semantic_router_service.route_query(
                session_id=session_id,
                query=query,
                language=language
            )
            
            query_type = route_result.get("query_type")
            refined_query = route_result.get("refined_query", query)
            
            logger.info(f"Semantic routing: {query_type} | Refined: '{refined_query[:50]}...'")
            
            if route_result.get("response"):
                return {
                    "status": "refused" if route_result.get("must_refuse") else "success",
                    "type": "chitchat",
                    "response": route_result.get("response"),
                    "original_query": original_query,
                    "refined_query": refined_query,
                    "routing_reason": route_result.get("reason"),
                    "execution_time": time.time() - start_time
                }
            
            if query_type == QueryType.CHITCHAT:
                if route_result.get("response"):
                    return {
                        "status": "refused" if route_result.get("must_refuse") else "success",
                        "type": "chitchat",
                        "response": route_result.get("response"),
                        "original_query": original_query,
                        "refined_query": refined_query,
                        "routing_reason": route_result.get("reason"),
                        "execution_time": time.time() - start_time
                    }
                else:
                    chitchat_template = self._get_chitchat_template(language)
                    chitchat_prompt = chitchat_template.format(
                        user_query=refined_query,
                        conversation_context="Đây là cuộc trò chuyện mới." if language == "vi" else ("This is a new conversation." if language == "en" else "これは新しい会話です。")
                    )
                    
                    return {
                        "status": "success",
                        "type": "chitchat_stream",
                        "prompt": chitchat_prompt,
                        "max_tokens": 300,
                        "temperature": 0.7,
                        "original_query": original_query,
                        "refined_query": refined_query,
                        "routing_reason": route_result.get("reason"),
                        "execution_time": time.time() - start_time
                    }
            
            else:
                rag_result = await self.execute_agent(
                    query=refined_query,
                    agent_name="rag",
                    language=language,
                    session_id=session_id,
                    max_tokens=max_tokens
                )
                
                if rag_result["status"] != "success":
                    default_responses = self.chat_service.get_default_responses(language)
                    return {
                        "status": "error",
                        "type": "rag",
                        "response": default_responses["error"],
                        "original_query": original_query,
                        "refined_query": refined_query,
                        "routing_reason": route_result.get("reason"),
                        "error": rag_result.get("error"),
                        "execution_time": time.time() - start_time
                    }
                
                rag_agent_output = rag_result["result"]
                
                return {
                    "status": "success",
                    "type": "rag",
                    "prompt": rag_agent_output.prompt,
                    "max_tokens": rag_agent_output.max_tokens,
                    "temperature": rag_agent_output.temperature,
                    "original_query": original_query,
                    "refined_query": refined_query,
                    "routing_reason": route_result.get("reason"),
                    "metadata": {
                        "citations": rag_agent_output.citations,
                        "chunks_used": rag_agent_output.chunks_used,
                        "search_results": rag_agent_output.search_results,
                        "citation_summary": rag_agent_output.citation_summary,
                        **rag_agent_output.metadata
                    },
                    "execution_time": time.time() - start_time
                }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in route_and_execute: {e}", exc_info=True)
            
            return {
                "status": "error",
                "error": str(e),
                "execution_time": execution_time,
                "original_query": original_query
            }

    async def execute_agent(
        self,
        query: str,
        agent_name: str = "rag",
        language: str = "vi",
        session_id: Optional[str] = None,
        max_tokens: int = 8192
    ) -> Dict[str, Any]:
        """
        Execute specific agent with given parameters
        
        Args:
            query: Query to process
            agent_name: Name of agent to execute (default: "rag")
            language: Language for processing
            session_id: Optional session ID for context
            max_tokens: Maximum tokens for generation
            
        Returns:
            Execution result with status and agent output
        """
        start_time = time.time()
        
        try:
            if agent_name not in self._agent_instances:
                if agent_name not in self._agent_registry:
                    raise ValueError(f"Agent '{agent_name}' not registered")
                
                agent_class = self._agent_registry[agent_name]
                self._agent_instances[agent_name] = agent_class()
                logger.info(f"Created new instance of {agent_name} agent")
            
            agent_instance = self._agent_instances[agent_name]

            input_data = RAGAgentInput(
                query=query,
                language=language,
                session_id=session_id,
                max_tokens=max_tokens
            )
            
            result = await agent_instance.process(input_data)
            execution_time = time.time() - start_time
            
            logger.info(f"Agent {agent_name} execution completed in {execution_time:.2f}s")
            
            return {
                "status": "success",
                "agent": agent_name,
                "query": query,
                "result": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Agent execution failed for {agent_name}: {e}", exc_info=True)
            
            return {
                "status": "error",
                "agent": agent_name,
                "error": str(e),
                "execution_time": execution_time
            }

    def _get_chitchat_template(self, language: str) -> str:
        """
        Get chitchat template with company information for newwave.vn
        
        Args:
            language: Language code (vi, en, ja)
            
        Returns:
            Formatted template string with company context
        """
        if language == "vi":
            return """Bạn là trợ lý AI nội bộ của NewWave Solutions, được NewWave Solutions phát triển.

Mục tiêu:
- Hỗ trợ nhân viên truy cập nhanh chóng thông tin nội bộ, quy trình, chính sách và tài liệu công ty.

Nguyên tắc bảo mật:
- Mọi thông tin chỉ dành cho nội bộ, tuyệt đối không chia sẻ ra ngoài.
- Nếu câu hỏi vượt phạm vi cho phép, hãy lịch sự hướng dẫn người hỏi liên hệ bộ phận phù hợp.

Thông tin công ty:
- NewWave Solutions là công ty công nghệ hàng đầu Việt Nam chuyên về phát triển phần mềm, AI/ML và tự động hóa quy trình.
- Sứ mệnh: Tạo ra giá trị thực tế thông qua công nghệ, tối ưu hóa hiệu quả kinh doanh cho khách hàng và chính nội bộ.

Vai trò của bạn:
- Trả lời câu hỏi về quy trình, chính sách, dự án, nhân sự, hệ thống IT, văn phòng...
- Hỗ trợ nhân viên tìm kiếm tài liệu, liên hệ phòng ban, giải thích quy định.
- Bạn có thể trò chuyện về cuộc sống, công việc, chuyện phiếm,.... để hỗ trợ tinh thần nhân viên

Ngữ cảnh cuộc trò chuyện: {conversation_context}

Câu hỏi của nhân viên: {user_query}

Vui lòng trả lời ngắn gọn, chính xác, thân thiện, chuyên nghiệp và đảm bảo tính bảo mật."""

        elif language == "en":
            return """You are an internal AI assistant for NewWave Solutions, developed by NewWave Solutions.

Purpose:
- Help employees quickly access internal information, processes, policies, and documents.

Confidentiality rules:
- All content is strictly for internal use and must not be shared externally.
- If a question is outside the permitted scope, politely guide the asker to the appropriate department.

Company overview:
- NewWave Solutions is a leading Vietnamese technology company specializing in software development, AI/ML, and process automation.
- Mission: Deliver real value through technology, optimize business efficiency for both clients and internal teams.

Your role:
- Answer questions about internal processes, policies, projects, HR, IT systems, office matters, etc.
- Support employees in finding documents, contacting departments, and explaining regulations.
- You can chat about life, work, office matters, etc. to support employee morale.

Conversation context: {conversation_context}

Employee question: {user_query}

Please answer concisely, accurately, friendly, professionally, and keep information confidential."""

        else:  # Japanese
            return """あなたはNewWave Solutions の社内向けインテリジェント AI アシスタントです。 NewWave Solutions が開発しました。

目的:
- 社員が社内情報、プロセス、ポリシー、ドキュメントに迅速にアクセスできるよう支援します。

機密保持ルール:
- すべての内容は社内限定であり、外部への共有を禁止します。
- 範囲外の質問があった場合は、丁寧に適切な部署への連絡を案内してください。

会社概要:
- NewWave Solutions はソフトウェア開発、AI/ML、業務プロセス自動化を専門とするベトナムのリーディングカンパニーです。
- ミッション: テクノロジーを通じて実際の価値を創出し、顧客および社内のビジネス効率を最適化すること。

あなたの役割:
- 社内プロセス、ポリシー、プロジェクト、人事、IT システム、オフィス関連などの質問に回答します。
- 社員がドキュメントを見つけたり、部署と連絡を取ったり、規定を理解したりするのをサポートします。
- 社員とのコミュニケーションを通じて、社内のモチベーションをサポートします。

会話の文脈: {conversation_context}

社員からの質問: {user_query}

簡潔で正確、親しみやすく、プロフェッショナルに、機密を守って回答してください。"""

    def _format_sse_token(self, data: Dict[str, Any]) -> str:
        """
        Format data as Server-Sent Events token
        
        Args:
            data: Dictionary to format as SSE token
            
        Returns:
            Formatted SSE token string
        """
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent names"""
        return list(self._agent_registry.keys())
    
    def register_agent(self, name: str, agent_class: Any):
        """
        Register a new agent type
        
        Args:
            name: Agent name identifier
            agent_class: Agent class to register
        """
        try:
            self._agent_registry[name] = agent_class
            logger.info(f"Successfully registered agent: {name} -> {agent_class.__name__}")
        except Exception as e:
            logger.error(f"Failed to register agent {name}: {e}")
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific agent
        
        Args:
            agent_name: Name of agent to get info for
            
        Returns:
            Agent information dictionary or None if not found
        """
        if agent_name not in self._agent_registry:
            return None
        
        agent_class = self._agent_registry[agent_name]
        return {
            "name": agent_name,
            "class_name": agent_class.__name__,
            "module": agent_class.__module__,
            "doc": agent_class.__doc__ or "No documentation available"
        }
