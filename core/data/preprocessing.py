# core/data/preprocessing.py
"""
Data cleaning and preprocessing utilities.
"""
import re
from typing import Dict, Any, List, Optional
import pandas as pd
from utils.logging_config import get_logger, LogBlock

logger = get_logger(__name__)

class DataPreprocessor:
    """
    Clean and validate legal document data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.cleaning_stats = {
            'total_processed': 0,
            'cleaned': 0,
            'invalid': 0,
            'duplicates': 0
        }
        
        logger.info("DataPreprocessor initialized")
    
    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single record.
        
        Args:
            record: Raw record dictionary
            
        Returns:
            Cleaned record
        """
        cleaned = record.copy()
        
        try:
            # Clean text fields
            for field in ['about', 'content', 'regulation_type', 'enacting_body']:
                if field in cleaned and cleaned[field]:
                    cleaned[field] = self._clean_text(str(cleaned[field]))
            
            # Normalize regulation type
            if 'regulation_type' in cleaned:
                cleaned['regulation_type'] = self._normalize_regulation_type(
                    cleaned['regulation_type']
                )
            
            # Clean year
            if 'year' in cleaned:
                cleaned['year'] = self._clean_year(cleaned['year'])
            
            # Ensure required fields
            required_fields = ['global_id', 'regulation_type', 'regulation_number', 'year']
            for field in required_fields:
                if field not in cleaned or not cleaned[field]:
                    logger.warning(f"Missing required field: {field} in record {record.get('global_id', 'unknown')}")
                    return None
            
            self.cleaning_stats['cleaned'] += 1
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning record {record.get('global_id', 'unknown')}: {e}")
            self.cleaning_stats['invalid'] += 1
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text field."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except common punctuation
        text = re.sub(r'[^\w\s.,;:()\-/]', '', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def _normalize_regulation_type(self, reg_type: str) -> str:
        """Normalize regulation type naming."""
        reg_type_lower = reg_type.lower().strip()
        
        # Mapping to standard names
        type_map = {
            'uu': 'Undang-Undang',
            'undang-undang': 'Undang-Undang',
            'undang undang': 'Undang-Undang',
            'pp': 'Peraturan Pemerintah',
            'peraturan pemerintah': 'Peraturan Pemerintah',
            'perpres': 'Peraturan Presiden',
            'peraturan presiden': 'Peraturan Presiden',
            'permen': 'Peraturan Menteri',
            'peraturan menteri': 'Peraturan Menteri',
            'perda': 'Peraturan Daerah',
            'peraturan daerah': 'Peraturan Daerah',
        }
        
        return type_map.get(reg_type_lower, reg_type)
    
    def _clean_year(self, year: Any) -> str:
        """Extract and validate year."""
        year_str = str(year).strip()
        
        # Extract 4-digit year
        match = re.search(r'(19|20)\d{2}', year_str)
        if match:
            return match.group(0)
        
        return year_str
    
    def remove_duplicates(self, records: List[Dict]) -> List[Dict]:
        """
        Remove duplicate records based on regulation identifiers.
        
        Args:
            records: List of records
            
        Returns:
            Deduplicated list
        """
        logger.info(f"Removing duplicates from {len(records):,} records")
        
        seen_keys = set()
        unique_records = []
        
        for record in records:
            # Create unique key
            key = (
                record.get('regulation_type', ''),
                record.get('regulation_number', ''),
                record.get('year', ''),
                record.get('chunk_id', '')
            )
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_records.append(record)
            else:
                self.cleaning_stats['duplicates'] += 1
        
        logger.info(
            f"Removed {self.cleaning_stats['duplicates']} duplicates, "
            f"{len(unique_records):,} unique records remaining"
        )
        
        return unique_records
    
    def get_stats(self) -> Dict[str, int]:
        """Get cleaning statistics."""
        return self.cleaning_stats.copy()