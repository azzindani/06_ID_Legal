"""
HTML Extractor - Extract text from HTML documents

Also handles XML files.

File: document_parser/extractors/html_extractor.py
"""

from typing import Dict, Any
from pathlib import Path
from .base import BaseExtractor
from utils.logger_utils import get_logger


class HTMLExtractor(BaseExtractor):
    """Extract text from HTML and XML documents"""
    
    SUPPORTED_EXTENSIONS = ['.html', '.htm', '.xml']
    EXTRACTOR_NAME = "html"
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("HTMLExtractor")
    
    def _check_dependencies(self) -> bool:
        """Check if BeautifulSoup is available"""
        try:
            from bs4 import BeautifulSoup
            return True
        except ImportError:
            return False
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from HTML/XML file.
        
        Uses BeautifulSoup to strip tags and extract text.
        """
        self.validate_file(file_path)
        
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Use BeautifulSoup if available
        if self.is_available():
            text = self._extract_with_beautifulsoup(content, extension)
        else:
            text = self._extract_basic(content)
        
        # Estimate page count
        line_count = text.count('\n') + 1
        page_count = max(1, line_count // 50)
        
        return {
            'text': text,
            'page_count': page_count,
            'method': 'beautifulsoup' if self.is_available() else 'regex',
            'metadata': {
                'extension': extension,
                'original_length': len(content),
                'extracted_length': len(text)
            }
        }
    
    def _extract_with_beautifulsoup(self, content: str, extension: str) -> str:
        """Extract text using BeautifulSoup"""
        from bs4 import BeautifulSoup
        
        # Choose parser
        parser = 'lxml' if extension == '.xml' else 'html.parser'
        try:
            soup = BeautifulSoup(content, parser)
        except:
            soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'head', 'meta', 'link']):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        
        return '\n'.join(lines)
    
    def _extract_basic(self, content: str) -> str:
        """Basic HTML stripping without BeautifulSoup"""
        import re
        
        # Remove script and style content
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all tags
        content = re.sub(r'<[^>]+>', '\n', content)
        
        # Decode HTML entities
        content = self._decode_entities(content)
        
        # Clean whitespace
        lines = [line.strip() for line in content.splitlines()]
        lines = [line for line in lines if line]
        
        return '\n'.join(lines)
    
    def _decode_entities(self, text: str) -> str:
        """Decode HTML entities"""
        import html
        try:
            return html.unescape(text)
        except:
            # Manual fallback for common entities
            replacements = {
                '&amp;': '&',
                '&lt;': '<',
                '&gt;': '>',
                '&quot;': '"',
                '&nbsp;': ' ',
                '&#39;': "'",
            }
            for entity, char in replacements.items():
                text = text.replace(entity, char)
            return text
