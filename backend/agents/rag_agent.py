"""
RAG Agent - Simplified implementation with document-level context assembly
Eliminates chunk concatenation and focuses on retrieval quality
"""

import logging
from typing import Dict, Any, List, Optional

from agents.agent_base import AgentBase, AgentInput, AgentOutput
from services.rag_service import RAGService
from config.settings import get_settings
from pydantic import BaseModel, Field
from utils.logger import get_logger

logger = get_logger(__name__)

class RAGAgentInput(AgentInput):
    """Input model for RAG Agent with simplified parameters"""
    query: str = Field(..., description="User query for RAG processing")
    language: str = Field(default="vi", description="Response language")
    collection_name: Optional[str] = Field(default=None, description="Milvus collection name")
    max_chunks: int = Field(default=6, description="Maximum number of document chunks to retrieve")
    search_threshold: float = Field(default=0.6, description="Similarity search threshold")
    include_citations: bool = Field(default=True, description="Include citations in response")
    use_hybrid_search: bool = Field(default=False, description="Use hybrid search (dense + sparse)")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for generation")

class RAGAgentOutput(AgentOutput):
    """Output model for RAG Agent with document-level information"""
    response: str = Field(..., description="Generated response")
    prompt: Optional[str] = Field(default="", description="Prepared prompt for streaming generation")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Document-level citations")
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    chunks_used: List[str] = Field(default_factory=list, description="Chunk IDs used")
    citation_summary: str = Field(default="", description="Summary of sources used")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for generation")
    temperature: Optional[float] = Field(default=None, description="Temperature for generation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")

class RAGAgent(AgentBase):
    """
    RAG Agent implementing document-level context assembly.
    Focuses on retrieval quality and eliminates knowledge dilution.
    """
    
    def __init__(self):
        """Initialize RAG Agent with RAG service"""
        super().__init__()
        self.rag_service = RAGService()
        self.agent_type = "RAGAgent"
        logger.info("RAG Agent initialized with document-level assembly")
    
    async def process(self, input_data: RAGAgentInput) -> RAGAgentOutput:
        """
        Process RAG query with document-level context assembly.
        Returns prompt and metadata for streaming generation.
        
        Args:
            input_data: RAG input with query and parameters
            
        Returns:
            RAG output with prompt and document-level citations for streaming
        """
        logger.info(f"Processing RAG query: '{input_data.query[:50]}...' in {input_data.language}")
        
        settings = get_settings()
        max_tokens = input_data.max_tokens or settings.DEFAULT_RAG_MAX_TOKENS

        rag_result = await self.rag_service.retrieve_and_generate(
            query=input_data.query,
            language=input_data.language,
            collection_name=input_data.collection_name,
            top_k=input_data.max_chunks,
            threshold=input_data.search_threshold,
            include_citations=input_data.include_citations,
            use_hybrid_search=input_data.use_hybrid_search,
            max_tokens=max_tokens
        )
        
        return RAGAgentOutput(
            response="", 
            prompt=rag_result.get("prompt", ""),
            citations=rag_result.get("citations", []),
            search_results=rag_result.get("search_results", []),
            chunks_used=rag_result.get("chunks_used", []),
            citation_summary=rag_result.get("citation_summary", ""),
            max_tokens=rag_result.get("max_tokens", max_tokens),
            temperature=rag_result.get("temperature", 0.1),
            metadata={
                **rag_result.get("metadata", {}),
                "processing_agent": "RAGAgent",
                "requires_streaming": True
            }
        )