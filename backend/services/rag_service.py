"""
RAG Service - Agentic RAG implementation with proper Milvus integration
Implements document-level context assembly, advanced hybrid search, and multi-stage query processing
"""
import time
import json
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from llm.llm_router import get_llm_router
from config.settings import get_settings
from services.citation_service import CitationService
from vector_store.milvus_client import MilvusClient
from services.hybrid_search_service import HybridSearchService
from utils.logger import get_logger

logger = get_logger(__name__)

class RAGService:
    """
    Agentic RAG service implementing:
    - Document-level context assembly
    - Integration with MilvusClient and HybridSearchService
    - Multi-stage query processing pipeline
    - Graceful fallbacks for all components
    """
    
    def __init__(self):
        """Initialize RAG service with existing search components""" 
        self.settings = get_settings()
        self.llm_router = get_llm_router() 
        self.milvus_client = MilvusClient()
        self.hybrid_search_service = HybridSearchService()
        
        self._milvus_available = False
        self._initialize_search_services()
        
        logger.info("RAG Service initialized with singleton LLM router and existing search services")
    
    def _initialize_search_services(self):
        """Initialize search services and check availability"""
        try:
            if hasattr(self.milvus_client, 'vectorstore') and self.milvus_client.vectorstore:
                self.hybrid_search_service.set_vectorstore(self.milvus_client.vectorstore)
                self._milvus_available = True
                logger.info("Search services initialized successfully")
            else:
                logger.warning("MilvusClient vectorstore not available")
                self._milvus_available = False
        except Exception as e:
            logger.error(f"Failed to initialize search services: {e}")
            self._milvus_available = False

    async def retrieve_and_generate(
        self,
        query: str,
        language: str = "vi",
        collection_name: Optional[str] = None,
        top_k: int = 4, 
        threshold: float = 0.6,
        include_citations: bool = True,
        use_hybrid_search: bool = False,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main RAG pipeline: retrieve relevant documents and prepare prompt for generation.
        Uses MMR search with focused retrieval to prevent context dilution.
        
        Args:
            query: User query string
            language: Response language (vi, en, ja)
            collection_name: Milvus collection name (optional)
            top_k: Maximum number of chunks to retrieve (default: 4, hard limit: 4)
            threshold: Similarity threshold for filtering (default: 0.6, minimum: 0.6)
            include_citations: Whether to include document citations
            use_hybrid_search: Ignored - always uses MMR for anti-dilution
            max_tokens: Maximum tokens for generation
            
        Returns:
            Dict containing prepared prompt, citations, and metadata for streaming generation
        """
        start_time = time.time()
        
        if not self._milvus_available:
            raise RuntimeError("Search system is currently unavailable. Please try again later.")
        
        try:
            processed_query = self._preprocess_query(query, language)
            
            relevant_docs = await self._search_documents(
                processed_query,
                top_k=top_k,
                threshold=threshold
            )
            
            if not relevant_docs:
                raise ValueError(f"No relevant information found for the question: '{query}'")
            
            context_data = self._create_direct_context(relevant_docs)
            chunks_actually_used = context_data["chunks_used_ids"]
            
            if include_citations:
                citation_service = CitationService()
                
                documents_for_citation = []
                for doc in relevant_docs:
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        doc_dict = {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                            "document_id": doc.metadata.get("document_id", ""),
                            "file_name": doc.metadata.get("source", "")
                        }
                        documents_for_citation.append(doc_dict)
                    elif isinstance(doc, dict):
                        documents_for_citation.append(doc)
                    else:
                        logger.warning(f"Unexpected document format: {type(doc)}")
                        continue
                
                citation_data = await citation_service.process_documents_for_citations(
                    used_documents=documents_for_citation,
                    chunks_actually_used=chunks_actually_used,
                    language=language
                )
            
            prompt_template = self._get_prompt_template_with_citations(language)
            prompt_text = prompt_template.format(
                context=context_data["context"],
                question=query,
                citation_format=citation_data.get("citation_format", ""),
                reference_header=citation_data.get("reference_header", ""),
                documents_info=json.dumps(citation_data.get("documents_info", []), ensure_ascii=False, indent=2)
            )
            
            search_results = []
            for doc in relevant_docs:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    search_results.append({
                        "content": doc.page_content, 
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', 0.0),
                        "document_id": doc.metadata.get("document_id", ""),
                        "file_name": doc.metadata.get("source", "")
                    })
                elif isinstance(doc, dict):
                    search_results.append({
                        "content": doc.get("page_content", doc.get("content", "")), 
                        "metadata": doc.get("metadata", {}),
                        "score": doc.get("score", 0.0),
                        "document_id": doc.get("document_id", doc.get("metadata", {}).get("document_id", "")),
                        "file_name": doc.get("file_name", doc.get("metadata", {}).get("source", ""))
                    })
            
            return {
                "type": "rag_prompt",
                "prompt": prompt_text,
                "citations": citation_data.get("citations_metadata", []),
                "chunks_used": chunks_actually_used,
                "search_results": search_results,
                "citation_summary": citation_data.get("citation_summary", ""),
                "max_tokens": max_tokens or self.settings.DEFAULT_RAG_MAX_TOKENS,
                "temperature": 0.1,
                "metadata": {
                    "search_time": time.time() - start_time,
                    "chunks_found": len(relevant_docs),
                    "documents_grouped": context_data.get("documents_count", 1),
                    "language": language,
                    "processed_query": processed_query,
                    "retrieval_method": "mmr_anti_dilution",
                    "search_method_used": "mmr"
                }
            }
            
        except Exception as e:
            logger.error(f"RAG processing error: {e}")
            raise RuntimeError(f"An error occurred during processing: {str(e)}")
    
    def _create_direct_context(self, relevant_docs: List[Document]) -> Dict[str, Any]:
        """
        Group chunks by unique document_id to eliminate document dilution.
        Creates single title per document instead of per chunk.
        Returns context with properly structured document sections.
        
        Args:
            relevant_docs: List of retrieved document chunks
            
        Returns:
            Dict containing context string, chunk IDs, document count and total chunks
        """
        try:
            context_parts = []
            chunks_used_ids = []
            document_groups = {}
            
            for i, doc in enumerate(relevant_docs, 1):
                if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                    metadata = doc.metadata
                    content = doc.page_content
                elif isinstance(doc, dict):
                    metadata = doc.get("metadata", {})
                    content = doc.get("page_content", doc.get("content", ""))
                else:
                    logger.warning(f"Unexpected document format in context creation: {type(doc)}")
                    continue
                
                doc_id = metadata.get("document_id", f"unknown_{i}")
                doc_title = metadata.get("source", f"Document {i}")
                chunk_id = metadata.get("chunk_id", f"chunk_{doc_id}_{i}")
                chunk_index = metadata.get("chunk_index", i)
                
                if doc_id not in document_groups:
                    document_groups[doc_id] = {
                        "title": doc_title,
                        "chunks": []
                    }
                
                document_groups[doc_id]["chunks"].append({
                    "content": content,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index
                })
                
                chunks_used_ids.append(chunk_id)
            
            doc_number = 1
            for doc_id, doc_info in document_groups.items():
                doc_title = doc_info["title"]
                chunks = doc_info["chunks"]
                
                chunks.sort(key=lambda x: x.get("chunk_index", 0))
                
                doc_header = f"## Document {doc_number}: {doc_title}\n\n"
                
                chunk_contents = []
                for chunk in chunks:
                    content = chunk["content"].strip()
                    if content:
                        chunk_contents.append(content)
                
                combined_content = "\n\n".join(chunk_contents)
                full_doc_content = doc_header + combined_content
                
                context_parts.append(full_doc_content)
                doc_number += 1
            
            full_context = "\n\n" + "="*50 + "\n\n".join([""] + context_parts)
            
            return {
                "context": full_context,
                "chunks_used_ids": chunks_used_ids,
                "documents_count": len(document_groups),
                "total_chunks": len(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in direct context creation: {e}")
            
            simple_context = "\n\n".join([f"Document {i}: {getattr(doc, 'page_content', str(doc))}" 
                                        for i, doc in enumerate(relevant_docs, 1)])
            chunks_used_ids = [f"chunk_{i}" for i in range(len(relevant_docs))]
            
            return {
                "context": simple_context,
                "chunks_used_ids": chunks_used_ids,
                "documents_count": len(relevant_docs),
                "total_chunks": len(relevant_docs)
            }

    def _preprocess_query(self, query: str, language: str) -> str:
        """
        Preprocess user query for better retrieval performance.
        Handles query normalization and language-specific processing.
        
        Args:
            query: Raw user query
            language: Query language
            
        Returns:
            Processed query string
        """
        try:
            processed = query.strip()
            
            if language == "vi":
                processed = processed.replace("?", "").replace(".", "")
                processed = " ".join(processed.split())
            elif language == "en":
                processed = processed.lower().replace("?", "").replace(".", "")
                processed = " ".join(processed.split())
            elif language == "ja":
                processed = processed.replace("？", "").replace("。", "")
                processed = " ".join(processed.split())
            
            return processed if processed else query
            
        except Exception as e:
            logger.warning(f"Query preprocessing failed: {e}")
            return query

    async def _search_documents(
        self,
        query: str,
        top_k: int = 4,  # Reduced default
        threshold: float = 0.6
    ) -> List[Document]:
        """
        Search for relevant documents with focused retrieval.
        Uses MMR with optimized parameters to prevent context dilution.
        
        Args:
            query: Processed query string
            top_k: Maximum number of documents to retrieve (hard limit: 4)
            threshold: Similarity threshold for filtering (minimum: 0.6)
            
        Returns:
            List of highly relevant Document objects
        """
        try:
            if not self._milvus_available:
                logger.warning("Milvus not available for search")
                return []
            
            # Apply hard limits to prevent over-retrieval
            effective_top_k = min(top_k, 4)  # Hard limit at 4
            effective_threshold = max(threshold, 0.6)  # Minimum quality threshold
            
            search_results = await self.hybrid_search_service.search(
                query=query,
                collection_name=self.settings.MILVUS_COLLECTION,
                top_k=effective_top_k,
                search_method="mmr",
                threshold=effective_threshold
            )
            
            search_results = search_results[:effective_top_k]
            
            documents = []
            for result in search_results:
                doc = Document(
                    page_content=result["content"],
                    metadata=result["metadata"]
                )
                if "score" in result:
                    doc.metadata["score"] = result["score"]
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} focused documents (requested: {top_k}, effective: {effective_top_k})")
            return documents
            
        except Exception as e:
            logger.error(f"Focused document search failed: {e}")
            return []

    def _get_prompt_template_with_citations(self, language: str) -> str:
        """
        Get prompt template for LLM with citation formatting instructions.
        Includes document information for proper citation generation.
        
        Args:
            language: Language code (vi, en, ja)
            
        Returns:
            Formatted prompt template string
        """
        if language == "vi":
            return """Dựa trên thông tin từ các tài liệu sau, hãy trả lời câu hỏi của người dùng một cách chính xác và chi tiết:

{context}

Câu hỏi: {question}

Hướng dẫn:
1. Trả lời câu hỏi dựa trên thông tin từ các tài liệu được cung cấp, KHÔNG SỬ DỤNG đánh số page trong context
2. Sử dụng thông tin từ nhiều tài liệu nếu cần thiết để đưa ra câu trả lời đầy đủ
3. Kết thúc câu trả lời bằng phần tài liệu tham khảo:

{reference_header}

HÃY TẠO TỪNG DÒNG CITATION CHO MỖI TÀI LIỆU SỬ DỤNG ĐỊNH DẠNG SAU:
{citation_format}

Trong đó:
- {{number}} = số thứ tự tài liệu (1, 2, 3...)  
- {{title}} = tên tài liệu (KHÔNG bao gồm đuôi .pdf, .docx...)
- {{url}} = đường dẫn download

Thông tin tài liệu để tạo citations:
{documents_info}

VÍ DỤ CITATION ĐÚNG:
[1].[ Nội quy lao động](http://example.com/doc1)
[2].[ Quy định chấm công](http://example.com/doc2)

Trả lời:"""
    
        elif language == "en":
            return """Based on the information from the following documents, answer the user's question accurately and comprehensively:

{context}

Question: {question}

Instructions:
1. Answer the question based on information from the provided documents, DO NOT USE PAGE NUMBERS IN CONTEXT
2. Use information from multiple documents if necessary to provide a complete answer
3. End your answer with a references section:

{reference_header}

CREATE EACH CITATION LINE FOR EACH DOCUMENT USING THE FORMAT:
{citation_format}

Where:
- {{number}} = document sequence number (1, 2, 3...)
- {{title}} = document title (WITHOUT file extensions like .pdf, .docx...)
- {{url}} = download link

Document information for citations:
{documents_info}

CORRECT CITATION EXAMPLE:
[1].[ Labor Regulations](http://example.com/doc1)
[2].[ Attendance Policy](http://example.com/doc2)

Answer:"""
    
        else:
            return """以下の文書の情報に基づいて、ユーザーの質問に正確かつ包括的に答えてください：

{context}

質問: {question}

指示:
1. 提供された文書の情報に基づいて質問に答える, コンテキストにページ番号は使用しない
2. 完全な回答を提供するために必要に応じて複数の文書の情報を使用する
3. 答えの最後に参考文献セクションを追加:

{reference_header}

各文書に対して次の形式で引用行を作成してください:
{citation_format}

形式説明:
- {{number}} = 文書の連番 (1, 2, 3...)
- {{title}} = 文書タイトル (.pdf, .docx等の拡張子は含めない)
- {{url}} = ダウンロードリンク

引用のための文書情報:
{documents_info}

正しい引用例:
[1].[ 労働規則](http://example.com/doc1)
[2].[ 出勤規定](http://example.com/doc2)

回答:"""