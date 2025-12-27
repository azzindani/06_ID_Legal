"""
Document Parser - Main document parsing class

Orchestrates document validation, extraction, and storage.

File: document_parser/parser.py
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, BinaryIO

from utils.logger_utils import get_logger
from security.file_protection import FileValidator, is_safe_filename, check_file_header


class DocumentParser:
    """
    Main document parser class.
    
    Handles file validation, text extraction, and storage orchestration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize document parser.
        
        Args:
            config: Configuration overrides
        """
        self.logger = get_logger("DocumentParser")
        self.config = config or {}
        
        # Settings with defaults
        self.max_file_size_mb = self.config.get('max_file_size_mb', 5)
        self.max_chars_per_document = self.config.get('max_chars_per_document', 50000)
        self.ocr_provider = self.config.get('ocr_provider', 'tesseract')
        self.ocr_languages = self.config.get('ocr_languages', ['ind', 'eng'])
        
        # Temp directory for uploads
        self.temp_dir = self.config.get('temp_upload_dir', 'uploads/temp')
        
        # File validator
        self._validator = None
        self._extractors = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize parser components"""
        try:
            # Create temp directory
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Initialize file validator
            from .extractors import get_supported_extensions
            supported = get_supported_extensions()
            
            self._validator = FileValidator(
                max_size_mb=self.max_file_size_mb,
                allowed_extensions=supported,
                check_mime_type=True
            )
            
            self._initialized = True
            self.logger.info(f"Parser initialized. Supported formats: {len(supported)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize parser: {e}")
            return False
    
    def parse_file(
        self, 
        file_path: str, 
        session_id: str,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse a document file and store extracted text.
        
        Args:
            file_path: Path to the file (temporary or permanent)
            session_id: Session to associate document with
            filename: Original filename (if different from path)
            
        Returns:
            Document info dict with id, filename, char_count, etc.
        """
        if not self._initialized:
            raise RuntimeError("Parser not initialized. Call initialize() first.")
        
        path = Path(file_path)
        original_filename = filename or path.name
        
        self.logger.info(f"Parsing document: {original_filename}")
        
        # Validate filename
        if not is_safe_filename(original_filename):
            from .exceptions import DocumentParserError
            raise DocumentParserError(f"Unsafe filename: {original_filename}")
        
        # Validate file
        is_valid, error = self._validator.validate(file_path)
        if not is_valid:
            from .exceptions import DocumentParserError
            raise DocumentParserError(f"Validation failed: {error}")
        
        # Check file header (magic bytes)
        extension = path.suffix.lower() if not filename else Path(filename).suffix.lower()
        if not check_file_header(file_path, extension):
            self.logger.warning(f"File header mismatch for {original_filename}")
            # Continue anyway - some files may have non-standard headers
        
        # Get extractor for this file type
        from .extractors import get_extractor
        extractor = get_extractor(extension)
        
        if not extractor.is_available():
            from .exceptions import ExtractionError
            raise ExtractionError(
                original_filename, 
                f"Extractor for {extension} not available"
            )
        
        # Extract text
        try:
            result = extractor.extract(file_path)
        except Exception as e:
            from .exceptions import ExtractionError
            raise ExtractionError(original_filename, str(e))
        
        extracted_text = result.get('text', '')
        page_count = result.get('page_count', 1)
        extraction_method = result.get('method', 'unknown')
        
        # Truncate if too long
        if len(extracted_text) > self.max_chars_per_document:
            self.logger.warning(
                f"Truncating {original_filename}: {len(extracted_text)} -> {self.max_chars_per_document} chars"
            )
            extracted_text = extracted_text[:self.max_chars_per_document]
            extracted_text += "\n\n[... Document truncated due to size limits ...]"
        
        # Sanitize text for prompt injection prevention
        extracted_text = self._sanitize_text(extracted_text)
        
        # Store in database
        from . import get_storage
        storage = get_storage()
        
        doc_info = storage.store_document(
            session_id=session_id,
            filename=original_filename,
            extracted_text=extracted_text,
            format=extension.lstrip('.'),
            original_size_bytes=path.stat().st_size,
            page_count=page_count,
            extraction_method=extraction_method
        )
        
        # Add preview to response
        doc_info['preview'] = extracted_text[:500] + ('...' if len(extracted_text) > 500 else '')
        doc_info['extraction_metadata'] = result.get('metadata', {})
        
        self.logger.info(
            f"Parsed {original_filename}: {len(extracted_text)} chars, {page_count} pages"
        )
        
        return doc_info
    
    def parse_upload(
        self,
        file_content: bytes,
        filename: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Parse uploaded file content directly.
        
        Saves to temp file, parses, then cleans up.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            session_id: Session ID
            
        Returns:
            Document info dict
        """
        # Create temp file
        temp_path = os.path.join(
            self.temp_dir, 
            f"upload_{session_id[:8]}_{filename}"
        )
        
        try:
            # Write to temp
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            # Parse
            return self.parse_file(temp_path, session_id, filename)
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def parse_gradio_file(
        self,
        gradio_file: Any,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Parse file from Gradio file component.
        
        Args:
            gradio_file: Gradio file object (has .name attribute with temp path)
            session_id: Session ID
            
        Returns:
            Document info dict
        """
        # Gradio files have .name attribute with temp path
        if hasattr(gradio_file, 'name'):
            temp_path = gradio_file.name
            original_filename = Path(temp_path).name
        elif isinstance(gradio_file, str):
            temp_path = gradio_file
            original_filename = Path(gradio_file).name
        else:
            from .exceptions import DocumentParserError
            raise DocumentParserError("Invalid file object")
        
        return self.parse_file(temp_path, session_id, original_filename)
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize extracted text for prompt injection prevention.
        
        Removes/escapes potentially harmful patterns.
        """
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove common prompt injection patterns
        # These are patterns that might try to override system prompts
        dangerous_patterns = [
            'ignore previous instructions',
            'ignore all previous',
            'forget previous instructions',
            'new system prompt:',
            'system:',
            'assistant:',
            '```python\nimport os',
            '```bash\nrm -rf',
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in text_lower:
                self.logger.warning(f"Suspicious pattern found in document: {pattern[:30]}")
                # Don't block, just log - legitimate documents might contain these
        
        return text
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        from .extractors import get_supported_extensions
        return get_supported_extensions()
    
    def get_extraction_capabilities(self) -> Dict[str, Any]:
        """Get info about available extractors"""
        from .extractors import EXTRACTORS
        
        capabilities = {}
        for ext, extractor_class in EXTRACTORS.items():
            extractor = extractor_class()
            capabilities[ext] = {
                'available': extractor.is_available(),
                'name': extractor.EXTRACTOR_NAME
            }
        
        return capabilities
    
    def cleanup_temp_files(self, max_age_seconds: int = 300) -> int:
        """Clean up old temp files"""
        import time
        
        cleaned = 0
        cutoff = time.time() - max_age_seconds
        
        try:
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath):
                    if os.path.getmtime(filepath) < cutoff:
                        os.remove(filepath)
                        cleaned += 1
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp files: {e}")
        
        if cleaned > 0:
            self.logger.info(f"Cleaned {cleaned} temp files")
        
        return cleaned
