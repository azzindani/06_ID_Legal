"""
PDF Extractor - Extract text from PDF documents

Supports: pypdf2 (default) and pdfplumber (better tables)

File: document_parser/extractors/pdf_extractor.py
"""

from typing import Dict, Any
from .base import BaseExtractor
from utils.logger_utils import get_logger


class PDFExtractor(BaseExtractor):
    """Extract text from PDF documents"""
    
    SUPPORTED_EXTENSIONS = ['.pdf']
    EXTRACTOR_NAME = "pdf"
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("PDFExtractor")
        self._pypdf2_available = None
        self._pdfplumber_available = None
    
    def _check_dependencies(self) -> bool:
        """Check if PDF libraries are available"""
        # Try pypdf2 first (lighter)
        try:
            import pypdf
            self._pypdf2_available = True
        except ImportError:
            try:
                import PyPDF2
                self._pypdf2_available = True
            except ImportError:
                self._pypdf2_available = False
        
        # Try pdfplumber as fallback (better tables)
        try:
            import pdfplumber
            self._pdfplumber_available = True
        except ImportError:
            self._pdfplumber_available = False
        
        return self._pypdf2_available or self._pdfplumber_available
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF file.
        
        Uses pypdf2 by default, falls back to pdfplumber for better table support.
        """
        self.validate_file(file_path)
        
        if not self.is_available():
            from ..exceptions import ExtractionError
            raise ExtractionError(
                file_path, 
                "No PDF library available. Install: pip install pypdf2"
            )
        
        # Try pypdf2 first
        if self._pypdf2_available:
            try:
                return self._extract_with_pypdf2(file_path)
            except Exception as e:
                # Fall back to pdfplumber
                if self._pdfplumber_available:
                    return self._extract_with_pdfplumber(file_path)
                raise
        
        # Use pdfplumber if pypdf2 not available
        if self._pdfplumber_available:
            return self._extract_with_pdfplumber(file_path)
        
        from ..exceptions import ExtractionError
        raise ExtractionError(file_path, "No PDF extraction method available")
    
    def _extract_with_pypdf2(self, file_path: str) -> Dict[str, Any]:
        """Extract using pypdf2/pypdf"""
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        
        text_parts = []
        page_count = 0
        
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            page_count = len(reader.pages)
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return {
            'text': '\n\n'.join(text_parts),
            'page_count': page_count,
            'method': 'pypdf2',
            'metadata': {
                'library': 'pypdf2/pypdf',
                'pages_processed': len(text_parts)
            }
        }
    
    def _extract_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Extract using pdfplumber (better table support)"""
        import pdfplumber
        
        text_parts = []
        page_count = 0
        tables_found = 0
        
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            
            for page in pdf.pages:
                # Extract regular text
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    tables_found += 1
                    table_text = self._format_table(table)
                    if table_text:
                        text_parts.append(table_text)
        
        return {
            'text': '\n\n'.join(text_parts),
            'page_count': page_count,
            'method': 'pdfplumber',
            'metadata': {
                'library': 'pdfplumber',
                'pages_processed': page_count,
                'tables_found': tables_found
            }
        }
    
    def _format_table(self, table: list) -> str:
        """Format table data as text"""
        if not table:
            return ""
        
        lines = []
        for row in table:
            if row:
                # Clean None values
                cells = [str(cell) if cell else '' for cell in row]
                lines.append(' | '.join(cells))
        
        return '\n'.join(lines)
