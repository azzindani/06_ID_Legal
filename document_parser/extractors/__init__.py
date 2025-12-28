"""
Base Extractor - Abstract base class for document text extractors

File: document_parser/extractors/__init__.py
"""

from .base import BaseExtractor
from .pdf_extractor import PDFExtractor
from .docx_extractor import DOCXExtractor
from .text_extractor import TextExtractor
from .html_extractor import HTMLExtractor
from .json_extractor import JSONExtractor
from .csv_extractor import CSVExtractor
from .image_extractor import ImageExtractor
from .url_extractor import URLExtractor, find_urls, extract_urls_from_prompt

# Format to extractor mapping
EXTRACTORS = {
    '.pdf': PDFExtractor,
    '.docx': DOCXExtractor,
    '.doc': DOCXExtractor,  # Try docx parser, may fail for old .doc
    '.txt': TextExtractor,
    '.md': TextExtractor,
    '.html': HTMLExtractor,
    '.htm': HTMLExtractor,
    '.json': JSONExtractor,
    '.csv': CSVExtractor,
    '.xml': HTMLExtractor,  # Use HTML extractor for XML (tag stripping)
    '.rtf': TextExtractor,  # Basic RTF support
    '.png': ImageExtractor,
    '.jpg': ImageExtractor,
    '.jpeg': ImageExtractor,
    '.tiff': ImageExtractor,
    '.bmp': ImageExtractor,
}

def get_extractor(extension: str) -> BaseExtractor:
    """Get appropriate extractor for file extension"""
    ext_lower = extension.lower()
    if ext_lower not in EXTRACTORS:
        from ..exceptions import UnsupportedFormatError
        raise UnsupportedFormatError(extension, list(EXTRACTORS.keys()))
    return EXTRACTORS[ext_lower]()

def get_url_extractor(**kwargs) -> URLExtractor:
    """Get URL extractor instance"""
    return URLExtractor(**kwargs)

def get_supported_extensions() -> list:
    """Get list of supported file extensions"""
    return list(EXTRACTORS.keys())

__all__ = [
    'BaseExtractor',
    'PDFExtractor',
    'DOCXExtractor', 
    'TextExtractor',
    'HTMLExtractor',
    'JSONExtractor',
    'CSVExtractor',
    'ImageExtractor',
    'URLExtractor',
    'EXTRACTORS',
    'get_extractor',
    'get_url_extractor',
    'get_supported_extensions',
    'find_urls',
    'extract_urls_from_prompt'
]
