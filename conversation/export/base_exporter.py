"""
Base Exporter - Abstract Base Class for Conversation Export

File: conversation/export/base_exporter.py
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from logger_utils import get_logger


class BaseExporter(ABC):
    """
    Abstract base class for conversation exporters

    Provides common functionality for all export formats.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize exporter

        Args:
            config: Optional configuration
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}

        # Export options
        self.include_metadata = self.config.get('include_metadata', True)
        self.include_sources = self.config.get('include_sources', True)
        self.include_timing = self.config.get('include_timing', True)
        self.include_thinking = self.config.get('include_thinking', False)

        self.logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def export(self, session_data: Dict[str, Any]) -> str:
        """
        Export session data to string

        Args:
            session_data: Complete session data from ConversationManager

        Returns:
            Exported content as string
        """
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """
        Get file extension for this format

        Returns:
            File extension (e.g., '.md', '.json', '.html')
        """
        pass

    def save_to_file(
        self,
        content: str,
        filename: Optional[str] = None,
        directory: str = 'exports'
    ) -> Path:
        """
        Save exported content to file

        Args:
            content: Exported content string
            filename: Optional filename (auto-generated if None)
            directory: Output directory

        Returns:
            Path to saved file
        """
        # Create directory if needed
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conversation_{timestamp}{self.get_file_extension()}"

        # Ensure correct extension
        if not filename.endswith(self.get_file_extension()):
            filename += self.get_file_extension()

        file_path = dir_path / filename

        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.info(f"Exported to {file_path}", {
            "size_bytes": len(content)
        })

        return file_path

    def export_and_save(
        self,
        session_data: Dict[str, Any],
        filename: Optional[str] = None,
        directory: str = 'exports'
    ) -> Path:
        """
        Export session and save to file

        Args:
            session_data: Session data
            filename: Optional filename
            directory: Output directory

        Returns:
            Path to saved file
        """
        content = self.export(session_data)
        return self.save_to_file(content, filename, directory)

    def _format_timestamp(self, timestamp: str) -> str:
        """Format ISO timestamp to readable format"""
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return timestamp

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to readable format"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
