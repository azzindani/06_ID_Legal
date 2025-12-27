"""
URL Extractor - Fetch and extract text content from URLs

Supports:
- Web pages (HTML)
- Direct PDF links
- Direct DOCX links
- JSON API responses

Security:
- Blocks private IPs (SSRF prevention)
- Timeout limits
- Size limits
- Content-Type validation

File: document_parser/extractors/url_extractor.py
"""

import re
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path
import socket

from .base import BaseExtractor
from utils.logger_utils import get_logger


class URLExtractor(BaseExtractor):
    """Extract text content from URLs"""
    
    SUPPORTED_EXTENSIONS = []  # URLs don't have extensions in the traditional sense
    EXTRACTOR_NAME = "url"
    
    # Content types we can handle
    SUPPORTED_CONTENT_TYPES = {
        'text/html': 'html',
        'application/pdf': 'pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/json': 'json',
        'text/plain': 'txt',
        'text/markdown': 'md',
        'text/csv': 'csv',
        'application/xml': 'xml',
        'text/xml': 'xml',
    }
    
    # Blocked private IP ranges (SSRF prevention)
    BLOCKED_IP_PREFIXES = [
        '127.',         # Localhost
        '10.',          # Private Class A
        '172.16.', '172.17.', '172.18.', '172.19.',  # Private Class B
        '172.20.', '172.21.', '172.22.', '172.23.',
        '172.24.', '172.25.', '172.26.', '172.27.',
        '172.28.', '172.29.', '172.30.', '172.31.',
        '192.168.',     # Private Class C
        '0.',           # Invalid
        '169.254.',     # Link-local
    ]
    
    def __init__(
        self,
        timeout: int = 10,
        max_size_bytes: int = 5 * 1024 * 1024,  # 5MB
        allowed_domains: Optional[List[str]] = None,
        user_agent: str = "LegalRAG-Bot/1.0"
    ):
        """
        Initialize URL extractor.
        
        Args:
            timeout: Request timeout in seconds
            max_size_bytes: Maximum content size to download
            allowed_domains: Optional whitelist of allowed domains (None = allow all public)
            user_agent: User agent string for requests
        """
        super().__init__()
        self.logger = get_logger("URLExtractor")
        self.timeout = timeout
        self.max_size_bytes = max_size_bytes
        self.allowed_domains = allowed_domains
        self.user_agent = user_agent
    
    def _check_dependencies(self) -> bool:
        """Check if requests library is available"""
        try:
            import requests
            return True
        except ImportError:
            return False
    
    def is_valid_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate URL for safety.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)
            
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid URL format"
            
            # Only http/https
            if parsed.scheme not in ('http', 'https'):
                return False, f"Unsupported scheme: {parsed.scheme}"
            
            # Check domain whitelist if configured
            if self.allowed_domains:
                domain = parsed.netloc.lower()
                allowed = any(
                    domain == d or domain.endswith('.' + d)
                    for d in self.allowed_domains
                )
                if not allowed:
                    return False, f"Domain not in whitelist: {domain}"
            
            # Resolve hostname to check for private IPs
            try:
                hostname = parsed.netloc.split(':')[0]
                ip = socket.gethostbyname(hostname)
                
                for prefix in self.BLOCKED_IP_PREFIXES:
                    if ip.startswith(prefix):
                        return False, f"Private/blocked IP address: {ip}"
                        
            except socket.gaierror:
                return False, f"Cannot resolve hostname: {parsed.netloc}"
            
            return True, ""
            
        except Exception as e:
            return False, f"URL validation error: {e}"
    
    def extract(self, url: str) -> Dict[str, Any]:
        """
        Fetch and extract text from URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Dict with extracted text and metadata
        """
        if not self.is_available():
            from ..exceptions import ExtractionError
            raise ExtractionError(url, "requests library not available")
        
        # Validate URL
        is_valid, error = self.is_valid_url(url)
        if not is_valid:
            from ..exceptions import ExtractionError
            raise ExtractionError(url, error)
        
        import requests
        
        try:
            # Make request with streaming to check size
            response = requests.get(
                url,
                timeout=self.timeout,
                stream=True,
                headers={'User-Agent': self.user_agent},
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_size_bytes:
                from ..exceptions import FileTooLargeError
                raise FileTooLargeError(url, int(content_length), self.max_size_bytes)
            
            # Get content type
            content_type = response.headers.get('content-type', '').split(';')[0].strip()
            
            # Determine file type
            file_type = self.SUPPORTED_CONTENT_TYPES.get(content_type)
            
            if not file_type:
                # Try to guess from URL extension
                path = urlparse(url).path.lower()
                if path.endswith('.pdf'):
                    file_type = 'pdf'
                elif path.endswith('.docx'):
                    file_type = 'docx'
                elif path.endswith('.json'):
                    file_type = 'json'
                elif path.endswith('.csv'):
                    file_type = 'csv'
                else:
                    file_type = 'html'  # Default to HTML
            
            # Download content (with size limit)
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > self.max_size_bytes:
                    from ..exceptions import FileTooLargeError
                    raise FileTooLargeError(url, len(content), self.max_size_bytes)
            
            # Extract based on type
            if file_type == 'html':
                return self._extract_html(content, url)
            elif file_type in ('pdf', 'docx'):
                return self._extract_binary(content, file_type, url)
            elif file_type == 'json':
                return self._extract_json(content, url)
            elif file_type in ('txt', 'md', 'csv', 'xml'):
                return self._extract_text(content, file_type, url)
            else:
                from ..exceptions import ExtractionError
                raise ExtractionError(url, f"Unsupported content type: {content_type}")
                
        except requests.RequestException as e:
            from ..exceptions import ExtractionError
            raise ExtractionError(url, f"Request failed: {e}")
    
    def _extract_html(self, content: bytes, url: str) -> Dict[str, Any]:
        """Extract text from HTML content"""
        try:
            from bs4 import BeautifulSoup
            
            # Decode content
            try:
                html = content.decode('utf-8')
            except:
                html = content.decode('latin-1')
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style
            for element in soup(['script', 'style', 'head', 'meta', 'link', 'nav', 'footer']):
                element.decompose()
            
            # Get title
            title = soup.title.string if soup.title else ''
            
            # Get main content
            # Try common content containers first
            main_content = None
            for selector in ['article', 'main', '.content', '#content', '.post']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                text = main_content.get_text(separator='\n')
            else:
                text = soup.get_text(separator='\n')
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if line]
            text = '\n'.join(lines)
            
            return {
                'text': text,
                'page_count': 1,
                'method': 'url_html',
                'metadata': {
                    'url': url,
                    'title': title,
                    'content_type': 'text/html',
                    'char_count': len(text)
                }
            }
            
        except ImportError:
            # Fallback without BeautifulSoup
            import re
            try:
                html = content.decode('utf-8')
            except:
                html = content.decode('latin-1')
            
            # Basic HTML stripping
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', '\n', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return {
                'text': text,
                'page_count': 1,
                'method': 'url_html_basic',
                'metadata': {'url': url}
            }
    
    def _extract_binary(self, content: bytes, file_type: str, url: str) -> Dict[str, Any]:
        """Extract text from binary files (PDF, DOCX)"""
        # Save to temp file and use appropriate extractor
        suffix = f'.{file_type}'
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            if file_type == 'pdf':
                from .pdf_extractor import PDFExtractor
                extractor = PDFExtractor()
            elif file_type == 'docx':
                from .docx_extractor import DOCXExtractor
                extractor = DOCXExtractor()
            else:
                from ..exceptions import ExtractionError
                raise ExtractionError(url, f"No extractor for {file_type}")
            
            result = extractor.extract(temp_path)
            result['metadata']['url'] = url
            result['method'] = f'url_{file_type}'
            return result
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _extract_json(self, content: bytes, url: str) -> Dict[str, Any]:
        """Extract text from JSON content"""
        import json
        
        try:
            data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            from ..exceptions import ExtractionError
            raise ExtractionError(url, f"Invalid JSON: {e}")
        
        # Format as readable text
        from .json_extractor import JSONExtractor
        extractor = JSONExtractor()
        text = extractor._format_json_as_text(data)
        
        return {
            'text': text,
            'page_count': 1,
            'method': 'url_json',
            'metadata': {
                'url': url,
                'content_type': 'application/json'
            }
        }
    
    def _extract_text(self, content: bytes, file_type: str, url: str) -> Dict[str, Any]:
        """Extract text from plain text content"""
        try:
            text = content.decode('utf-8')
        except:
            text = content.decode('latin-1')
        
        return {
            'text': text,
            'page_count': 1,
            'method': f'url_{file_type}',
            'metadata': {
                'url': url,
                'content_type': f'text/{file_type}'
            }
        }


# URL detection utilities

URL_PATTERN = re.compile(
    r'https?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
    r'localhost|'  # localhost
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)',
    re.IGNORECASE
)


def find_urls(text: str) -> List[str]:
    """
    Find all URLs in text.
    
    Args:
        text: Text to search
        
    Returns:
        List of URLs found
    """
    return URL_PATTERN.findall(text)


def extract_urls_from_prompt(prompt: str) -> Tuple[str, List[str]]:
    """
    Extract URLs from prompt and return clean prompt.
    
    Args:
        prompt: User prompt
        
    Returns:
        Tuple of (prompt_without_urls, list_of_urls)
    """
    urls = find_urls(prompt)
    
    # Remove URLs from prompt
    clean_prompt = prompt
    for url in urls:
        clean_prompt = clean_prompt.replace(url, '')
    
    # Clean up extra whitespace
    clean_prompt = re.sub(r'\s+', ' ', clean_prompt).strip()
    
    return clean_prompt, urls
