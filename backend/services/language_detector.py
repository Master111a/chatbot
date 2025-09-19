from typing import Dict, List, Any, Optional, Union
import os
import json
from pydantic import BaseModel
from langdetect import detect, DetectorFactory, LangDetectException
import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    """
    Service for detecting and handling different languages in the system.
    Supports Vietnamese, English, and Japanese as specified in the requirements.
    """
    
    def __init__(self, default_language="vi"):
        self.default_language = default_language
        self.supported_languages = {"en", "vi", "ja"}
        DetectorFactory.seed = 0
        
        self._load_language_resources()
    
    def _load_language_resources(self) -> None:
        """
        Load language-specific resources like stopwords, templates, etc.
        """
        self.language_resources = {}
        
        for lang_code in self.supported_languages:
            self.language_resources[lang_code] = {
                "stopwords": self._load_stopwords(lang_code),
                "templates": self._load_templates(lang_code)
            }
    
    def _load_stopwords(self, language_code: str) -> List[str]:
        """
        Load stopwords for a specific language.
        
        Args:
            language_code: ISO language code
            
        Returns:
            List of stopwords for the language
        """
        file_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            "resources",
            f"stopwords_{language_code}.txt"
        )
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        else:
            if language_code == "vi":
                return ["của", "và", "các", "có", "được", "là", "trong", "cho", "đã", "với"]
            elif language_code == "en":
                return ["the", "and", "is", "in", "of", "to", "a", "for", "that", "with"]
            elif language_code == "ja":
                return ["の", "に", "は", "を", "が", "と", "た", "で", "て", "も"]
            else:
                return []
    
    def _load_templates(self, language_code: str) -> Dict[str, str]:
        """
        Load templates for a specific language.
        
        Args:
            language_code: ISO language code
            
        Returns:
            Dictionary of templates for the language
        """
        file_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            "resources",
            f"templates_{language_code}.json"
        )
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            if language_code == "vi":
                return {
                    "greeting": "Xin chào! Tôi có thể giúp gì cho bạn?",
                    "not_found": "Tôi không tìm thấy thông tin liên quan đến truy vấn của bạn.",
                    "query_refinement": "Để tôi có thể giúp bạn tốt hơn, bạn có thể cung cấp thêm thông tin về {query}?"
                }
            elif language_code == "en":
                return {
                    "greeting": "Hello! How can I help you?",
                    "not_found": "I couldn't find information related to your query.",
                    "query_refinement": "To help you better, could you provide more information about {query}?"
                }
            elif language_code == "ja":
                return {
                    "greeting": "こんにちは！どのようにお手伝いできますか？",
                    "not_found": "お問い合わせに関連する情報が見つかりませんでした。",
                    "query_refinement": "より良くお手伝いするために、{query}についてもう少し情報を提供していただけますか？"
                }
            else:
                return {}
    
    async def detect_language(self, text: str) -> str:
        """
        Detects the language of a given text.

        Args:
            text: The input text.

        Returns:
            The detected language code ('en', 'vi', 'ja') or the default language.
        """
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Language detection on empty or invalid text. Falling back to default.")
            return self.default_language

        words = text.strip().split()
        if len(words) == 1 and len(words[0]) < 15:
            logger.info(f"Short text ('{text}') detected. Using LLM for more reliable language detection.")
            return await self._detect_language_with_llm(text)

        try:
            detected_lang = detect(text)
            logger.info(f"Detected language code: '{detected_lang}' for text: '{text[:50]}...'")
            
            if detected_lang in self.supported_languages:
                return detected_lang
            else:
                logger.warning(f"Detected language '{detected_lang}' is not supported. Falling back to default '{self.default_language}'.")
                return self.default_language

        except LangDetectException:
            logger.warning(f"Could not detect language for text: '{text[:50]}...'. Falling back to default.")

            return await self._detect_language_with_llm(text)

    async def _detect_language_with_llm(self, text: str) -> str:
        """
        Use the LLM to detect the language of a short text.
        """
        from llm.llm_router import get_llm_router
        llm_router = get_llm_router()  # Use singleton
        
        prompt = f"""
        What is the language of the following text?
        Respond with only the two-letter ISO 639-1 language code (e.g., "en", "vi", "ja").
        Supported codes: en, vi, ja.
        If the language is not one of the supported languages, respond with "en".

        Text: "{text}"
        
        Language code:
        """
        
        try:
            response = await llm_router.generate_text(prompt, "en") 
            lang_code = response.strip().lower()
            
            if lang_code in self.supported_languages:
                logger.info(f"LLM detected language as '{lang_code}' for text: '{text}'")
                return lang_code
            else:
                logger.warning(f"LLM returned unsupported language code '{lang_code}'. Defaulting to 'en'.")
                return "en"
        except Exception as e:
            logger.error(f"Error using LLM for language detection: {e}. Falling back to default.")
            return self.default_language

    async def get_language_resources(self, language_code: str) -> Dict[str, Any]:
        """
        Get resources for a specific language.
        
        Args:
            language_code: ISO language code
            
        Returns:
            Dictionary of resources for the language
        """
        if language_code not in self.supported_languages:
            language_code = self.default_language
        
        return self.language_resources.get(language_code, {})
    
    async def translate_query(
        self,
        query: str,
        source_language: Optional[str] = None,
        target_language: str = "en"
    ) -> str:
        """
        Translate a query between languages.
        
        Args:
            query: Query to translate
            source_language: Source language code (auto-detect if None)
            target_language: Target language code
            
        Returns:
            Translated query
        """
        if not source_language:
            source_language = await self.detect_language(query)
        
        if source_language == target_language:
            return query
        
        from llm.llm_router import get_llm_router
        
        llm_router = get_llm_router()
        
        prompt = f"""Please translate the following text from {source_language} to {target_language}:

Text: {query}

Translation:"""
        
        try:
            translation = await llm_router.generate_text(prompt, target_language)
            return translation
        except Exception as e:
            print(f"Error translating query: {e}")
            return query
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get the name of a language from its code.
        
        Args:
            language_code: ISO language code
            
        Returns:
            Full name of the language
        """
        return {
            "en": "English",
            "vi": "Vietnamese",
            "ja": "Japanese"
        }.get(language_code, "Unknown")