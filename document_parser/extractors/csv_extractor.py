"""
CSV Extractor - Extract text from CSV files

Formats CSV as readable tables.

File: document_parser/extractors/csv_extractor.py
"""

import csv
from typing import Dict, Any, List
from .base import BaseExtractor
from utils.logger_utils import get_logger


class CSVExtractor(BaseExtractor):
    """Extract and format text from CSV files"""
    
    SUPPORTED_EXTENSIONS = ['.csv']
    EXTRACTOR_NAME = "csv"
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("CSVExtractor")
    
    def _check_dependencies(self) -> bool:
        """CSV is built-in"""
        return True
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from CSV file.
        
        Formats as readable table.
        """
        self.validate_file(file_path)
        
        rows = []
        
        # Detect dialect and read
        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            # Sniff the dialect
            sample = f.read(4096)
            f.seek(0)
            
            try:
                dialect = csv.Sniffer().sniff(sample)
            except:
                dialect = csv.excel  # Default
            
            reader = csv.reader(f, dialect)
            for row in reader:
                rows.append(row)
        
        if not rows:
            return {
                'text': '(Empty CSV file)',
                'page_count': 1,
                'method': 'csv',
                'metadata': {'row_count': 0, 'column_count': 0}
            }
        
        # Format as table
        text = self._format_as_table(rows)
        
        # Metadata
        row_count = len(rows)
        col_count = max(len(row) for row in rows) if rows else 0
        
        return {
            'text': text,
            'page_count': max(1, row_count // 50),
            'method': 'csv',
            'metadata': {
                'row_count': row_count,
                'column_count': col_count
            }
        }
    
    def _format_as_table(self, rows: List[List[str]]) -> str:
        """Format rows as a readable ASCII table"""
        if not rows:
            return ""
        
        # Get max width for each column
        col_count = max(len(row) for row in rows)
        col_widths = [0] * col_count
        
        for row in rows:
            for i, cell in enumerate(row):
                if i < col_count:
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Cap column widths for readability
        col_widths = [min(w, 50) for w in col_widths]
        
        lines = []
        
        for row_idx, row in enumerate(rows):
            # Pad cells
            cells = []
            for i in range(col_count):
                cell = row[i] if i < len(row) else ''
                cell = str(cell)[:50]  # Truncate
                cells.append(cell.ljust(col_widths[i]))
            
            line = ' | '.join(cells)
            lines.append(line)
            
            # Add separator after header
            if row_idx == 0:
                separator = '-+-'.join('-' * w for w in col_widths)
                lines.append(separator)
        
        return '\n'.join(lines)
