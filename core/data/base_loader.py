# core/data/base_loader.py
"""
Abstract base class for data loaders.
Defines the interface that all loaders must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
from utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    All loaders must implement:
    - load(): Load data from source
    - get_statistics(): Return dataset statistics
    - validate(): Validate loaded data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base loader.
        
        Args:
            config: Configuration dictionary with loader settings
        """
        self.config = config
        self.loaded = False
        self.data = None
        self.metadata = {}
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def load(self, **kwargs) -> bool:
        """
        Load data from source.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """
        Validate loaded data.
        
        Returns:
            Validation report with issues/warnings
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self.loaded
    
    def get_config(self) -> Dict[str, Any]:
        """Get loader configuration."""
        return self.config.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get loader metadata."""
        return self.metadata.copy()
    
    def __repr__(self) -> str:
        status = "loaded" if self.loaded else "not loaded"
        return f"{self.__class__.__name__}(status={status})"