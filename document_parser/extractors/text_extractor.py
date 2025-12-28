"""
Text Extractor - Extract text from plain text files

Supports: .txt, .md, .rtf (basic)

File: document_parser/extractors/text_extractor.py
"""

from typing import Dict, Any
from pathlib import Path
from .base import BaseExtractor
from utils.logger_utils import get_logger


class TextExtractor(BaseExtractor):
    """Extract text from plain text files"""
    
    SUPPORTED_EXTENSIONS = ['.txt', '.md', '.rtf']
    EXTRACTOR_NAME = "text"
    
    # Encodings to try in order
    ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("TextExtractor")
    
    def _check_dependencies(self) -> bool:
        """Text extraction has no dependencies"""
        return True
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from plain text files.
        
        Tries multiple encodings if UTF-8 fails.
        """
        self.validate_file(file_path)
        
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Read file with encoding detection
        text = self._read_with_encoding(file_path)
        
        # Handle RTF specially
        if extension == '.rtf':
            text = self._strip_rtf(text)
        
        # Count lines for page estimate
        line_count = text.count('\n') + 1
        page_count = max(1, line_count // 50)  # ~50 lines per page
        
        return {
            'text': text,
            'page_count': page_count,
            'method': 'text',
            'metadata': {
                'extension': extension,
                'line_count': line_count,
                'char_count': len(text)
            }
        }
    
    def _read_with_encoding(self, file_path: str) -> str:
        """Try multiple encodings to read file"""
        last_error = None
        
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError as e:
                last_error = e
                continue
        
        # Last resort: read as binary and decode with errors ignored
        try:
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='ignore')
        except Exception as e:
            from ..exceptions import ExtractionError
            raise ExtractionError(file_path, f"Could not decode file: {last_error or e}")
    
    def _strip_rtf(self, text: str) -> str:
        """
        Basic RTF stripping.
        
        For full RTF support, use striprtf library.
        """
        # Try striprtf first
        try:
            from striprtf.striprtf import rtf_to_text
            return rtf_to_text(text)
        except ImportError:
            pass
        
        # Basic manual stripping
        import re
        
        # Remove RTF control words
        text = re.sub(r'\\[a-z]+\d*\s?', '', text)
        # Remove groups
        text = re.sub(r'[{}]', '', text)
        # Remove font tables etc
        text = re.sub(r'\\fonttbl.*?;', '', text, flags=re.DOTALL)
        text = re.sub(r'\\colortbl.*?;', '', text, flags=re.DOTALL)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
