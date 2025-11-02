# core/data/validators.py
"""
Data validation rules and checks.
"""
from typing import Dict, Any, List
from utils.logging_config import get_logger

logger = get_logger(__name__)

class DataValidator:
    """
    Validate dataset integrity and quality.
    """
    
    def __init__(self):
        self.validation_report = {
            'total_records': 0,
            'valid_records': 0,
            'warnings': [],
            'errors': []
        }
        
        logger.info("DataValidator initialized")
    
    def validate_record(self, record: Dict[str, Any], record_id: int) -> Dict[str, Any]:
        """
        Validate a single record.
        
        Args:
            record: Record to validate
            record_id: Record index for error reporting
            
        Returns:
            Validation result with issues
        """
        issues = {'errors': [], 'warnings': []}
        
        # Check required fields
        required_fields = [
            'global_id', 'regulation_type', 'regulation_number',
            'year', 'about', 'content'
        ]
        
        for field in required_fields:
            if field not in record or not record[field]:
                issues['errors'].append(f"Missing required field: {field}")
        
        # Check data types
        if 'global_id' in record:
            try:
                int(record['global_id'])
            except (ValueError, TypeError):
                issues['errors'].append(f"Invalid global_id: {record.get('global_id')}")
        
        # Check year format
        if 'year' in record:
            year_str = str(record['year'])
            if not (year_str.isdigit() and 1900 <= int(year_str) <= 2100):
                issues['warnings'].append(f"Suspicious year: {year_str}")
        
        # Check content length
        if 'content' in record:
            content_len = len(str(record['content']))
            if content_len < 10:
                issues['warnings'].append(f"Very short content: {content_len} chars")
            elif content_len > 100000:
                issues['warnings'].append(f"Very long content: {content_len} chars")
        
        # Check KG fields
        if 'kg_authority_score' in record:
            score = record['kg_authority_score']
            if not (0 <= score <= 1):
                issues['warnings'].append(f"Authority score out of range: {score}")
        
        return issues
    
    def validate_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate entire dataset.
        
        Args:
            records: List of all records
            
        Returns:
            Comprehensive validation report
        """
        logger.info(f"Validating dataset: {len(records):,} records")
        
        self.validation_report['total_records'] = len(records)
        
        for i, record in enumerate(records):
            issues = self.validate_record(record, i)
            
            if issues['errors']:
                self.validation_report['errors'].extend([
                    f"Record {i}: {error}" for error in issues['errors']
                ])
            else:
                self.validation_report['valid_records'] += 1
            
            if issues['warnings']:
                self.validation_report['warnings'].extend([
                    f"Record {i}: {warning}" for warning in issues['warnings']
                ])
        
        # Summary
        error_count = len(self.validation_report['errors'])
        warning_count = len(self.validation_report['warnings'])
        valid_rate = (
            self.validation_report['valid_records'] / 
            self.validation_report['total_records'] * 100
        ) if self.validation_report['total_records'] > 0 else 0
        
        logger.info(
            f"Validation complete: {valid_rate:.1f}% valid, "
            f"{error_count} errors, {warning_count} warnings"
        )
        
        return self.validation_report