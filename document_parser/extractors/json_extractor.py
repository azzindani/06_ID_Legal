"""
JSON Extractor - Extract text from JSON files

Formats JSON content as readable text.

File: document_parser/extractors/json_extractor.py
"""

import json
from typing import Dict, Any
from .base import BaseExtractor
from utils.logger_utils import get_logger


class JSONExtractor(BaseExtractor):
    """Extract and format text from JSON files"""
    
    SUPPORTED_EXTENSIONS = ['.json']
    EXTRACTOR_NAME = "json"
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("JSONExtractor")
    
    def _check_dependencies(self) -> bool:
        """JSON is built-in"""
        return True
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from JSON file.
        
        Formats JSON as readable key-value pairs.
        """
        self.validate_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            from ..exceptions import ExtractionError
            raise ExtractionError(file_path, f"Invalid JSON: {e}")
        
        # Format as readable text
        text = self._format_json_as_text(data)
        
        # Count items for metadata
        item_count = self._count_items(data)
        
        return {
            'text': text,
            'page_count': 1,
            'method': 'json',
            'metadata': {
                'type': type(data).__name__,
                'item_count': item_count
            }
        }
    
    def _format_json_as_text(self, data: Any, indent: int = 0) -> str:
        """
        Format JSON data as readable text.
        
        Converts nested structures to indented text.
        """
        lines = []
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._format_json_as_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
                    
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i+1}]")
                    lines.append(self._format_json_as_text(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
                    
        else:
            lines.append(f"{prefix}{data}")
        
        return '\n'.join(lines)
    
    def _count_items(self, data: Any) -> int:
        """Count total items in JSON structure"""
        if isinstance(data, dict):
            count = len(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    count += self._count_items(value)
            return count
        elif isinstance(data, list):
            count = len(data)
            for item in data:
                if isinstance(item, (dict, list)):
                    count += self._count_items(item)
            return count
        return 1
