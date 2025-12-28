"""
Document Parser Module

Provides document text extraction for the Legal RAG system.
Supports: PDF, DOCX, TXT, MD, HTML, JSON, CSV, XML, RTF, Images (OCR), URLs

This module is initialized at system startup alongside the pipeline.

File: document_parser/__init__.py
"""

from typing import Optional, Dict, Any, List, Tuple
from .parser import DocumentParser
from .storage import DocumentStorage
from .context_builder import DocumentContextBuilder
from .exceptions import (
    DocumentParserError,
    UnsupportedFormatError,
    ExtractionError,
    FileTooLargeError,
    DocumentLimitExceededError,
    URLExtractionError,
    URLValidationError,
    URLBlockedError
)
from .extractors.url_extractor import (
    URLExtractor,
    find_urls,
    extract_urls_from_prompt
)

# Module-level instance (initialized at startup)
_parser_instance: Optional[DocumentParser] = None
_storage_instance: Optional[DocumentStorage] = None
_url_extractor: Optional[URLExtractor] = None


def initialize(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize document parser module.
    Called during system startup (alongside pipeline initialization).
    
    Args:
        config: Optional configuration overrides (uses DOCUMENT_PARSER_CONFIG by default)
        
    Returns:
        True if successful
    """
    global _parser_instance, _storage_instance, _url_extractor
    
    from utils.logger_utils import get_logger
    logger = get_logger("DocumentParser")
    
    try:
        logger.info("Initializing document parser module...")
        
        # Merge with default config from config.py
        from config import DOCUMENT_PARSER_CONFIG, URL_EXTRACTION_CONFIG
        merged_config = {**DOCUMENT_PARSER_CONFIG, **(config or {})}
        
        # Initialize storage
        _storage_instance = DocumentStorage(merged_config)
        _storage_instance.initialize()
        
        # Initialize parser
        _parser_instance = DocumentParser(merged_config)
        _parser_instance.initialize()
        
        # Initialize URL extractor if enabled
        if URL_EXTRACTION_CONFIG.get('enabled', True):
            _url_extractor = URLExtractor(
                timeout=URL_EXTRACTION_CONFIG.get('timeout', 10),
                max_size_bytes=URL_EXTRACTION_CONFIG.get('max_size_bytes', 5 * 1024 * 1024),
                allowed_domains=URL_EXTRACTION_CONFIG.get('allowed_domains'),
                user_agent=URL_EXTRACTION_CONFIG.get('user_agent', 'LegalRAG-Bot/1.0')
            )
            logger.info("URL extraction enabled")
        else:
            logger.info("URL extraction disabled")
        
        logger.success("Document parser module initialized")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize document parser: {e}")
        return False


def get_parser() -> DocumentParser:
    """Get the initialized parser instance"""
    if _parser_instance is None:
        raise RuntimeError("Document parser not initialized. Call initialize() first.")
    return _parser_instance


def get_storage() -> DocumentStorage:
    """Get the initialized storage instance"""
    if _storage_instance is None:
        raise RuntimeError("Document storage not initialized. Call initialize() first.")
    return _storage_instance


def get_url_extractor() -> Optional[URLExtractor]:
    """Get the URL extractor instance (None if disabled)"""
    return _url_extractor


def extract_from_url(url: str, session_id: str) -> Dict[str, Any]:
    """
    Extract content from URL and store it.
    
    Args:
        url: URL to extract from
        session_id: Session to associate with
        
    Returns:
        Document info dict with extracted content
    """
    if _url_extractor is None:
        raise RuntimeError("URL extraction is disabled")
    
    # Extract content
    result = _url_extractor.extract(url)
    
    # Store in database
    storage = get_storage()
    
    doc_info = storage.store_document(
        session_id=session_id,
        filename=url,
        extracted_text=result['text'],
        format='url',
        original_size_bytes=len(result['text'].encode()),
        page_count=result.get('page_count', 1),
        extraction_method=result.get('method', 'url')
    )
    
    doc_info['preview'] = result['text'][:500] + ('...' if len(result['text']) > 500 else '')
    doc_info['url_metadata'] = result.get('metadata', {})
    
    return doc_info


def is_initialized() -> bool:
    """Check if module is initialized"""
    return _parser_instance is not None and _storage_instance is not None


def is_url_extraction_enabled() -> bool:
    """Check if URL extraction is enabled"""
    return _url_extractor is not None


def shutdown():
    """Cleanup resources"""
    global _parser_instance, _storage_instance, _url_extractor
    
    if _storage_instance:
        _storage_instance.cleanup_expired()
        _storage_instance = None
    
    if _parser_instance:
        _parser_instance = None
    
    _url_extractor = None


__all__ = [
    # Main classes
    'DocumentParser',
    'DocumentStorage', 
    'DocumentContextBuilder',
    'URLExtractor',
    
    # Module functions
    'initialize',
    'get_parser',
    'get_storage',
    'get_url_extractor',
    'extract_from_url',
    'is_initialized',
    'is_url_extraction_enabled',
    'shutdown',
    
    # URL utilities
    'find_urls',
    'extract_urls_from_prompt',
    
    # Exceptions
    'DocumentParserError',
    'UnsupportedFormatError',
    'ExtractionError',
    'FileTooLargeError',
    'DocumentLimitExceededError',
    'URLExtractionError',
    'URLValidationError',
    'URLBlockedError'
]

