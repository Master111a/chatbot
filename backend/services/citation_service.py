from typing import Dict, List, Any, Union
from fastapi import Request

from utils.logger import get_logger
from utils.context_utils import build_download_url

logger = get_logger(__name__)

class CitationService:
    """
    Service for generating and formatting citations with MinIO download URLs.
    Essential for RAG systems to provide transparency about information sources.
    """
    
    def __init__(self):
        self.citation_styles = self._load_citation_styles()
        logger.info("CitationService initialized")
        self._document_service = None 
    
    def _load_citation_styles(self) -> Dict[str, Dict[str, str]]:
        """Load citation styles for different languages"""
        return {
            "vi": {
                "file_reference": " [{number}].[ {title}]({url})",
                "reference_header": "**Tài liệu tham khảo:**",
                "url_format": "Tải về: {url}"
            },
            "en": {
                "file_reference": " [{number}].[ {title}]({url})",
                "reference_header": "**References:**",
                "url_format": "Download: {url}"
            },
            "ja": {
                "file_reference": " [{number}].[ {title}]({url})",
                "reference_header": "**参考文献:**",
                "url_format": "ダウンロード: {url}"
            }
        }
    
    @property
    def document_service(self):
        """Lazy initialization of DocumentService to avoid multiple instantiations"""
        if self._document_service is None:
            from services.document_service import DocumentService
            self._document_service = DocumentService()
            logger.info("CitationService: Lazily initialized DocumentService")
        return self._document_service

    async def _generate_document_url(self, document_id: str, file_name: str, request: Request = None) -> str:
        """
        Generate download URL for document with proper URL generation
        Returns: full URL with scheme and host/domain
        """
        try:
            if not document_id:
                logger.warning(f"Empty document_id for file: {file_name}")
                return build_download_url("unknown", request)
           
            return build_download_url(document_id, request)
            
        except Exception as e:
            logger.error(f"Error generating download URL for {document_id}: {e}")
            return build_download_url(document_id, request)



    async def get_name_and_url(self, search_results: List[Dict[str, Any]], chunks_actually_used: List[str]) -> List[Dict[str, Any]]:
        """
        Get name and URL from search results documents that were actually used
        
        Args:
            search_results: List of documents with used chunks
            chunks_actually_used: List of chunk IDs that were actually used
            
        Returns:
            List of citations with name and url
        """
        try:
            citations = []
            
            document_chunks = {}
            for result in search_results:
                metadata = result.get("metadata", {})
                chunk_id = metadata.get("chunk_id", "")
                
                if chunk_id in chunks_actually_used:
                    document_id = (
                        result.get("document_id") or 
                        metadata.get("document_id") or
                        metadata.get("doc_id", "")
                    )
                    
                    if document_id not in document_chunks:
                        document_chunks[document_id] = {
                            "result": result,
                            "chunks": []
                        }
                    document_chunks[document_id]["chunks"].append(chunk_id)
            
            for document_id, doc_data in document_chunks.items():
                result = doc_data["result"]
                chunks_count = len(doc_data["chunks"])  
                
                metadata = result.get("metadata", {})
                file_name = (
                    result.get("file_name") or
                    metadata.get("source") or
                    metadata.get("file_name") or
                    f"Document {document_id}" if document_id else "Unknown Document"
                )
                
                url = await self._generate_document_url(document_id, file_name)
                
                citation = {
                    "name": file_name,
                    "url": url,
                    "doc_id": document_id,
                    "source": file_name,
                    "title": file_name,
                    "chunks_used": chunks_count 
                }
                
                citations.append(citation)
            
            total_chunks = sum(len(doc_data["chunks"]) for doc_data in document_chunks.values())
            logger.info(f"Generated {len(citations)} citations from {len(document_chunks)} documents using {total_chunks} chunks")
            return citations
            
        except Exception as e:
            logger.error(f"Error generating citations: {e}")
            return []

    async def process_documents_for_citations(
        self, 
        used_documents: List[Union[Dict[str, Any], Any]], 
        chunks_actually_used: List[str],
        language: str = "vi"
    ) -> Dict[str, Any]:
        """
        Process used documents to create citations and format for LLM prompt
        Fixed: Unique document grouping and proper URL generation
        
        Args:
            used_documents: List of documents that were actually used in RAG
            chunks_actually_used: List of chunk IDs that were actually used
            language: Language for citation formatting
            
        Returns:
            Dict containing formatted citations for LLM and metadata
        """
        try:
            citations_metadata = []
            documents_info = []
            document_chunks = {}
            
            for doc in used_documents:
                if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                    metadata = doc.metadata if hasattr(doc.metadata, 'get') else {}
                    document_id = metadata.get("document_id") or metadata.get("doc_id", "")
                    chunk_id = metadata.get("chunk_id", "")
                elif isinstance(doc, dict):
                    metadata = doc.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    document_id = (
                        doc.get("document_id") or 
                        metadata.get("document_id") or 
                        metadata.get("doc_id", "")
                    )
                    chunk_id = metadata.get("chunk_id", "")
                else:
                    logger.warning(f"Unsupported document format: {type(doc)}")
                    continue
                
                if document_id and chunk_id in chunks_actually_used:
                    if document_id not in document_chunks:
                        document_chunks[document_id] = {
                            "doc": doc,
                            "chunks": []
                        }
                    document_chunks[document_id]["chunks"].append(chunk_id)

            for i, (document_id, doc_data) in enumerate(document_chunks.items(), 1):
                doc = doc_data["doc"]
                chunks_count = len(doc_data["chunks"])
                
                if hasattr(doc, 'metadata'):
                    metadata = doc.metadata if hasattr(doc.metadata, 'get') else {}
                elif isinstance(doc, dict):
                    metadata = doc.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                else:
                    metadata = {}
                
                doc_title = (
                    metadata.get("source") or
                    metadata.get("file_name") or
                    doc.get("file_name", "") if isinstance(doc, dict) else "" or
                    f"Document {i}"
                )
                
                download_url = await self._generate_document_url(document_id, doc_title)
                
                citation_metadata = {
                    "doc_id": document_id,
                    "source": doc_title,
                    "title": doc_title,
                    "url": download_url,
                    "chunks_used": chunks_count
                }
                citations_metadata.append(citation_metadata)
                
                doc_info = {
                    "number": i,
                    "title": doc_title,
                    "document_id": document_id,
                    "url": download_url,
                    "source": doc_title,
                    "chunks_used": chunks_count 
                }
                documents_info.append(doc_info)
            
            style = self.citation_styles.get(language, self.citation_styles.get("vi", {}))
            citation_format = style.get("file_reference", "[{number}] {title}({url})")
            reference_header = style.get("reference_header", "**Tài liệu tham khảo:**")
            
            total_chunks = sum(len(doc_data["chunks"]) for doc_data in document_chunks.values())
            
            return {
                "citations_metadata": citations_metadata,
                "documents_info": documents_info,
                "citation_format": citation_format,
                "reference_header": reference_header,
                "citation_summary": f"Used information from {len(document_chunks)} documents, {total_chunks} chunks"
            }
            
        except Exception as e:
            logger.error(f"Error processing documents for citations: {e}")
            return {
                "citations_metadata": [],
                "documents_info": [],
                "citation_format": "[{number}] {title}({url})",
                "reference_header": "**Tài liệu tham khảo:**",
                "citation_summary": ""
            }