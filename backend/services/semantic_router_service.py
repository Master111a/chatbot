import os
import json
import re
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from utils.logger import get_logger
from services.types import QueryType, ConversationRole
from utils.prompt_utils import prompt_utils
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class SemanticRouterService:
    """
    Service for semantically routing queries and refining them in single LLM call.
    
    Workflow:
    1. Single LLM call with chat history and original query
    2. Returns: type, original_query, refined_query
    """
    
    def __init__(self):
        from llm.llm_router import get_llm_router
        from services.language_detector import LanguageDetector
        
        self.llm_router = get_llm_router()  # Use singleton
        self.language_detector = LanguageDetector()
        self.prompt_utils = prompt_utils
        
        logger.info("SemanticRouterService initialized with singleton LLM router")
        
    async def route_query(self, session_id: str, query: str, language: Optional[str] = None, history_limit: int = 5) -> Dict[str, Any]:
        """
        Route and refine query using single LLM call with configurable history limit.
        
        Args:
            session_id: Session identifier
            query: User query to process
            language: Optional language (auto-detected if not provided)
            history_limit: Number of recent messages to include in context (default: 5)
            
        Returns:
            Dict with: query_type, original_query, refined_query, reason
        """
        if not query:
            return self._create_route_response(QueryType.CHITCHAT, "Empty query treated as chitchat", query, query)

        if not language:
            language = await self.language_detector.detect_language(query)

        chat_history = []

        if session_id:
            chat_history = await self._get_chat_history_from_session(session_id, limit=history_limit)
        
        return await self._single_llm_call(query, language, chat_history)
    
    async def refine_query(self, query: str, language: Optional[str] = None, session_id: Optional[str] = None, history_limit: int = 5) -> str:
        """
        Refine query with configurable history limit (compatibility method).
        
        Args:
            query: Query to refine
            language: Optional language (auto-detected if not provided)
            session_id: Optional session for context
            history_limit: Number of recent messages to include in context (default: 5)
            
        Returns:
            Refined query string
        """
        try:
            if not language:
                language = await self.language_detector.detect_language(query)
            
            chat_history = []
            if session_id:
                chat_history = await self._get_chat_history_from_session(session_id, limit=history_limit)
            
            result = await self._single_llm_call(query, language, chat_history)
            return result.get("refined_query", query)
            
        except Exception as e:
            logger.error(f"Error in refinement: {e}")
            return query
    
    async def _get_chat_history_from_session(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get chat history from session in correct chronological order with configurable limit.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of recent messages to retrieve (default: 5)
            
        Returns:
            List of chat history messages in chronological order (oldest first)
        """
        try:
            from db.pg_manager import ChatHistory
            chat_history_manager = ChatHistory()
            
            history = await chat_history_manager.get_recent_history(session_id, limit=limit)
            
            history.reverse()
            
            return history
        except Exception as e:
            logger.error(f"Error getting chat history from session {session_id} with limit {limit}: {e}")
            return []
        
    async def _single_llm_call(self, query: str, language: str, chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Single LLM call for both classification and refinement using 2-step process.
        """
        conversation_context_summary = ""
        if chat_history:
            conversation_context_summary = self._build_conversation_summary(chat_history)
        
        prompt = self._create_combined_prompt(
            query=query, 
            language=language, 
            conversation_context_summary=conversation_context_summary
        )
        
        try:
            semantic_router_client = self.llm_router.get_semantic_router_client()
            response_text = await semantic_router_client.generate_text(
                prompt=prompt,
                language=language,
                temperature=0.1, 
                max_tokens=500
            )
            
            llm_result = self.prompt_utils.parse_json_response(response_text)
            
            if not llm_result or not isinstance(llm_result, dict):
                logger.warning(f"Failed to parse LLM response: {response_text[:200]}...")
                return self._create_route_response(QueryType.RAG, "Failed to parse LLM response, defaulting to RAG", query, query)
            
            step1_analysis = llm_result.get("step1_analysis", "")
            refined_query = llm_result.get("refined_query", query).strip()
            step2_analysis = llm_result.get("step2_analysis", "")
            query_type_str = llm_result.get("query_type", "").lower()
            confidence = llm_result.get("confidence", 0.0)
            reasoning = llm_result.get("reasoning", "No reasoning provided")
            must_refuse = llm_result.get("must_refuse", False)
            refusal_message = llm_result.get("refusal_message")
            
            if not refined_query or len(refined_query) > len(query) * 3:
                refined_query = query
            
            logger.info(f"Step 1 Analysis: {step1_analysis}")
            logger.info(f"Step 2 Analysis: {step2_analysis}")
            logger.info(f"2-Step LLM Result - Type: '{query_type_str}', Original: '{query}', Refined: '{refined_query}', Confidence: {confidence}")

            if query_type_str == QueryType.CHITCHAT.value:
                if must_refuse and refusal_message:
                    return self._create_route_response(
                        QueryType.CHITCHAT,
                        f"2-Step Analysis - Step1: {step1_analysis} | Step2: {step2_analysis} | Refusal: {reasoning}",
                        query,
                        refined_query,
                        response=refusal_message,
                    )
                return self._create_route_response(
                    QueryType.CHITCHAT, 
                    f"2-Step Analysis - Step1: {step1_analysis} | Step2: {step2_analysis} | Reasoning: {reasoning}",
                    query,
                    refined_query
                )
            elif query_type_str == QueryType.RAG.value:
                return self._create_route_response(
                    QueryType.RAG,
                    f"2-Step Analysis - Step1: {step1_analysis} | Step2: {step2_analysis} | Reasoning: {reasoning}",
                    query,
                    refined_query
                )
            else: 
                 logger.warning(f"LLM returned unknown query_type '{query_type_str}'. Converting to RAG")
                 return self._create_route_response(
                    QueryType.RAG,
                    f"Unknown type '{query_type_str}' converted to RAG. Step1: {step1_analysis} | Step2: {step2_analysis}",
                    query,
                    refined_query
                 )
                
        except Exception as e:
            logger.error(f"Error in 2-step LLM call for query '{query}': {e}", exc_info=True)
            return self._create_route_response(QueryType.RAG, f"Error in 2-step LLM call: {str(e)}", query, query)

    def _create_route_response(self, query_type: Union[QueryType, str], reason: str, original_query: str, refined_query: str, response: Optional[str] = None) -> Dict[str, Any]:
        """
        Helper to create a standardized route response object.
        """
        qt_value = query_type.value if hasattr(query_type, 'value') else query_type 
        resp_obj = {
            "query_type": qt_value,
            "original_query": original_query,
            "refined_query": refined_query,
            "reason": reason,
            "type": "classification",
            "name": str(qt_value),
        }
        if response is not None:
            resp_obj["response"] = response
        return resp_obj
    
    def _create_combined_prompt(self, query: str, language: str, conversation_context_summary: str = "") -> str:
        """
        Create single prompt for both classification and refinement in 2 clear steps.
        """
        
        if language == "vi":
            instruction_header = "XỬ LÝ TRUY VẤN 2 BƯỚC"
            system_role_description = "Bạn là một trợ lý AI chuyên xử lý truy vấn của người dùng theo quy trình 2 bước rõ ràng."
            
            query_type_description = """'rag': Truy vấn cần tìm kiếm thông tin trong cơ sở kiến thức nội bộ.
'chitchat': Truy vấn là một lời chào hỏi, cảm ơn, hoặc các cuộc trò chuyện thông thường không cần tìm kiếm thông tin."""
            
            disallowed_guidance = """
QUY TẮC ĐẠO ĐỨC (nội dung mã nguồn / hack / vi phạm chuẩn mực)
Nếu truy vấn yêu cầu hoặc ám chỉ:
  • Viết, chia sẻ hoặc sửa **bất kỳ mã nguồn (code) nào**, bao gồm ví dụ, snippet, demo nhỏ
  • Hướng dẫn hack, tấn công, mã độc, SQL-Injection hoặc nội dung vi phạm pháp luật, đạo đức

THÌ:
1. Đặt `query_type` = **"chitchat"** và `must_refuse` = **true**.
2. Trả về `refusal_message` là một câu từ chối ngắn gọn bằng tiếng Việt.
3. Đặt `refined_query` = truy vấn gốc.
"""
            
            step1_instructions = """
BƯỚC 1: LÀM RÕ TRUY VẤN
- Phân tích truy vấn gốc và ngữ cảnh lịch sử chat
- Tập trung vào từ khóa cốt lõi
- Loại bỏ từ ngữ thừa, không cần thiết mà vẫn giữ nguyên ý nghĩa  
- Sử dụng ngữ cảnh (lịch sử chat) để làm rõ ý nghĩa
- Nếu câu hỏi đã rõ ràng → giữ nguyên
- KHÔNG thay đổi ý nghĩa gốc
- Ví dụ 1: 
    - các câu hỏi trước đó:
        - 'anh X là ai' 
        - 'chi tiết' -> 'chi tiết về anh X'
    - truy vấn gốc hiện tại:
        - 'còn gì nữa không' 
    - truy vấn đã làm rõ:
        - 'chi tiết về anh X'
- Ví dụ 2: 
    - các câu hỏi trước đó:
        - 'Y có nghĩa là gì' 
        - 'có mấy giá trị' -> 'Y có mấy giá trị'
    - truy vấn gốc hiện tại:
        - 'liệt kê' 
    - truy vấn đã làm rõ:
        - 'Liệt kê các giá trị của Y'
"""

            step2_instructions = """
BƯỚC 2: PHÂN LOẠI TRUY VẤN ĐÃ LÀM RÕ
- Dựa vào truy vấn đã làm rõ ở Bước 1
- Phân loại thành 'rag' hoặc 'chitchat'
- Đánh giá độ tin cậy của phân loại (0.0 - 1.0)
- Giải thích lý do phân loại
"""

        elif language == "ja":
            instruction_header = "2ステップクエリ処理"
            system_role_description = "あなたは明確な2ステップのプロセスでユーザーのクエリを処理する専門のAIアシスタントです。"
            
            query_type_description = """'rag': 内部ナレッジベースでの情報検索が必要な問い合わせ。
'chitchat': 挨拶、感謝、または情報検索を必要としない一般的な会話の問い合わせ。"""
            
            disallowed_guidance = """
【倫理ルール】（コード作成・違法行為）
次のいずれかを要求・示唆する問い合わせの場合：
  • いかなるソースコード（例・スニペットを含む）の作成・共有・修正
  • ハッキング、攻撃手法、マルウェア等の違法または非倫理的内容

対応:
1. `query_type` を **"chitchat"** に、`must_refuse` を **true** に設定
2. `refusal_message` として日本語の簡潔な拒否文を返す
3. `refined_query` を元のクエリに設定
"""
            
            step1_instructions = """
ステップ1: クエリの明確化
- 元のクエリと会話履歴のコンテキストを分析
- コアキーワードに集中
- 冗長で不要な言葉を削除しつつ、元の意味を保持
- コンテキスト（チャット履歴）を使用して意味を明確化
- 質問が既に明確な場合 → そのまま保持
- 元の意味を変更しない
- 例1:
    - 以前の質問:
        - 'Xさんは誰ですか'
        - '詳細' -> 'Xさんの詳細'
    - 現在の元のクエリ:
        - '他にありますか'
    - 明確化されたクエリ:
        - 'Xさんの詳細'
- 例2:
    - 以前の質問:
        - 'Yとは何ですか'
        - '値はいくつありますか' -> 'Yの値はいくつありますか'
    - 現在の元のクエリ:
        - '一覧'
    - 明確化されたクエリ:
        - 'Yの値の一覧を表示'
"""

            step2_instructions = """
ステップ2: 明確化されたクエリの分類
- ステップ1で明確化されたクエリに基づく
- 'rag'または'chitchat'に分類
- 分類の信頼度を評価 (0.0 - 1.0)
- 分類の理由を説明
"""

        else: # en
            instruction_header = "2-STEP QUERY PROCESSING"
            system_role_description = "You are an AI assistant specialized in processing user queries through a clear 2-step process."
            
            query_type_description = """'rag': The query requires information retrieval from internal knowledge base.
'chitchat': The query is a greeting, thank you, or general conversation not needing information retrieval."""
            
            disallowed_guidance = """
ETHICAL RULE (Code, Illegal Activities)
If the query requests or implies:
  • Writing, sharing, or modifying **any source code** (including examples or small snippets)
  • Instructions for hacking, attacks, malware, or any illegal/unethical activity

THEN:
1. Set `query_type` = **"chitchat"** and `must_refuse` = **true**
2. Return `refusal_message` as a short refusal in English
3. Set `refined_query` to the original query
"""
            
            step1_instructions = """
STEP 1: CLARIFY QUERY
- Analyze original query and chat history context
- Focus on core keywords
- Remove redundant, unnecessary words while preserving original meaning
- Use context (chat history) to clarify meaning
- If question is already clear → keep as is
- Do NOT change original meaning
- Example 1:
    - previous questions:
        - 'who is X'
        - 'detail' -> 'detail of X'
    - original query:
        - 'what else'
    - clarified query:
        - 'detail of X'
- Example 2:
    - previous questions:
        - 'what does Y mean'
        - 'how many values' -> 'how many values of Y'
    - original query:
        - 'list'
    - clarified query:
        - 'list the values of Y'
"""

            step2_instructions = """
STEP 2: CLASSIFY CLARIFIED QUERY
- Based on the clarified query from Step 1
- Classify as 'rag' or 'chitchat'
- Assess classification confidence (0.0 - 1.0)
- Explain reasoning for classification
"""

        general_context_str = "Không có lịch sử cuộc trò chuyện trước đó." if language == "vi" else ("以前の会話履歴はありません。" if language == "ja" else "No previous conversation history.")
        if conversation_context_summary:
            if language == 'vi':
                general_context_str = f"Ngữ cảnh cuộc trò chuyện:\n{conversation_context_summary}"
            elif language == 'ja':
                general_context_str = f"会話のコンテキスト:\n{conversation_context_summary}"
            else: # en
                general_context_str = f"Conversation context:\n{conversation_context_summary}"

        prompt = f"""SYSTEM ROLE: {system_role_description}

{instruction_header}

{disallowed_guidance}

QUY TRÌNH XỬ LÝ:

{step1_instructions}

{step2_instructions}

ĐỊNH NGHĨA PHÂN LOẠI:
{query_type_description}

---
{general_context_str}
---

TRUY VẤN GỐC: "{query}"

HÃY THỰC HIỆN 2 BƯỚC TRÊN VÀ TRẢ VỀ JSON:
{{
  "step1_analysis": "(phân tích ngắn gọn về việc làm rõ truy vấn)",
  "refined_query": "(truy vấn đã được làm rõ từ Bước 1)",
  "step2_analysis": "(phân tích ngắn gọn về việc phân loại)",
  "query_type": "(chitchat|rag)",
  "confidence": (float, 0.0 to 1.0, độ tin cậy phân loại),
  "reasoning": "(giải thích ngắn gọn cho phân loại)",
  "must_refuse": (boolean, true nếu vi phạm quy tắc đạo đức),
  "refusal_message": "(string, tin nhắn từ chối nếu must_refuse = true)"
}}

JSON Response:
"""
        return prompt
    
    def _build_conversation_summary(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Build conversation summary from chat history in proper chronological order.
        Enhanced to handle different message structures and provide better context.
        
        Args:
            chat_history: List of chat messages in chronological order
            
        Returns:
            Formatted conversation summary string
        """
        try:
            if not chat_history:
                return ""
            
            context_lines = []
            
            for i, msg in enumerate(chat_history):
                role = msg.get("role")
                timestamp = msg.get("timestamp", "")
                
                if role == "user":
                    content = msg.get("query", msg.get("message", ""))
                    if content.strip():
                        context_lines.append(f"User: {content.strip()}")
                
                elif role == "assistant":
                    content = msg.get("response", "")
                    if content:
                        try:
                            parsed_content = json.loads(content)
                            response_text = parsed_content.get("response", str(parsed_content))
                        except (json.JSONDecodeError, TypeError):
                            response_text = content
                        
                        if len(response_text) > 100:
                            response_text = response_text[:97] + "..."
                        
                        if response_text.strip():
                            context_lines.append(f"Assistant: {response_text.strip()}")
                
                elif not role:
                    user_msg = msg.get("message", "")
                    assistant_msg = msg.get("response", "")
                    
                    if user_msg.strip():
                        context_lines.append(f"User: {user_msg.strip()}")
                    
                    if assistant_msg:
                        try:
                            parsed_content = json.loads(assistant_msg)
                            response_text = parsed_content.get("response", str(parsed_content))
                        except (json.JSONDecodeError, TypeError):
                            response_text = assistant_msg
                        
                        if len(response_text) > 100:
                            response_text = response_text[:97] + "..."
                        
                        if response_text.strip():
                            context_lines.append(f"Assistant: {response_text.strip()}")
            
            result = "\n".join(context_lines)
            return result
            
        except Exception as e:
            logger.error(f"Error building conversation summary: {e}")
            return ""