"""
Base Extractor - Abstract base class for document text extractors

File: document_parser/extractors/base.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
from utils.logger_utils import get_logger


class BaseExtractor(ABC):
    """
    Abstract base class for document text extractors.
    
    All extractors must implement the extract() method.
    """
    
    # Override in subclasses
    SUPPORTED_EXTENSIONS: list = []
    EXTRACTOR_NAME: str = "base"
    
    def __init__(self):
        self._available = None
        # Subclasses should set their own logger
    
    @abstractmethod
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with:
                - text: Extracted text content
                - page_count: Number of pages (if applicable)
                - metadata: Additional metadata
                - method: Extraction method used
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if required dependencies are available.
        Override in subclasses to check for optional dependencies.
        """
        if self._available is not None:
            return self._available
        self._available = self._check_dependencies()
        return self._available
    
    def _check_dependencies(self) -> bool:
        """Check if required libraries are installed"""
        return True
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic file information"""
        path = Path(file_path)
        return {
            'filename': path.name,
            'extension': path.suffix.lower(),
            'size_bytes': path.stat().st_size if path.exists() else 0
        }
    
    def validate_file(self, file_path: str) -> bool:
        """Basic file validation"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        return True
    
    def __repr__(self):
        return f"<{self.__class__.__name__} available={self.is_available()}>"
