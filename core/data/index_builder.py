# core/data/index_builder.py
"""
Build specialized indexes for fast searching.
"""
from typing import List, Dict, Any, Set
from collections import defaultdict
import numpy as np
from utils.logging_config import get_logger, LogBlock

logger = get_logger(__name__)

class IndexBuilder:
    """
    Build various indexes for fast document retrieval.
    """
    
    def __init__(self, records: List[Dict[str, Any]]):
        """
        Initialize index builder.
        
        Args:
            records: List of all records
        """
        self.records = records
        logger.info(f"IndexBuilder initialized with {len(records):,} records")
    
    def build_all_indexes(self) -> Dict[str, Any]:
        """
        Build all indexes at once.
        
        Returns:
            Dictionary with all indexes
        """
        logger.info("Building all indexes...")
        
        with LogBlock(logger, "Index building"):
            indexes = {
                'regulation_index': self.build_regulation_index(),
                'year_index': self.build_year_index(),
                'domain_index': self.build_domain_index(),
                'authority_index': self.build_authority_index(),
                'temporal_index': self.build_temporal_index(),
                'article_index': self.build_article_index(),
                'keyword_index': self.build_keyword_index()
            }
        
        logger.info("✅ All indexes built successfully")
        return indexes
    
    def build_regulation_index(self) -> Dict[str, List[int]]:
        """
        Build index: (regulation_type, number, year) -> [record_indices]
        """
        logger.debug("Building regulation index...")
        
        index = defaultdict(list)
        
        for i, record in enumerate(self.records):
            key = (
                record['regulation_type'],
                record['regulation_number'],
                record['year']
            )
            index[key].append(i)
        
        logger.debug(f"Regulation index: {len(index)} unique regulations")
        return dict(index)
    
    def build_year_index(self) -> Dict[str, List[int]]:
        """Build index: year -> [record_indices]"""
        logger.debug("Building year index...")
        
        index = defaultdict(list)
        
        for i, record in enumerate(self.records):
            year = record['year']
            index[year].append(i)
        
        logger.debug(f"Year index: {len(index)} unique years")
        return dict(index)
    
    def build_domain_index(self) -> Dict[str, List[int]]:
        """Build index: domain -> [record_indices]"""
        logger.debug("Building domain index...")
        
        index = defaultdict(list)
        
        for i, record in enumerate(self.records):
            domain = record.get('kg_primary_domain', 'Unknown')
            index[domain].append(i)
        
        logger.debug(f"Domain index: {len(index)} unique domains")
        return dict(index)
    
    def build_authority_index(self) -> Dict[int, List[int]]:
        """Build index: authority_tier -> [record_indices]"""
        logger.debug("Building authority index...")
        
        index = defaultdict(list)
        
        for i, record in enumerate(self.records):
            tier = int(record.get('kg_authority_score', 0.5) * 10)
            tier = max(0, min(10, tier))
            index[tier].append(i)
        
        logger.debug(f"Authority index: {len(index)} tiers")
        return dict(index)
    
    def build_temporal_index(self) -> Dict[int, List[int]]:
        """Build index: temporal_tier -> [record_indices]"""
        logger.debug("Building temporal index...")
        
        index = defaultdict(list)
        
        for i, record in enumerate(self.records):
            tier = int(record.get('kg_temporal_score', 0.5) * 10)
            tier = max(0, min(10, tier))
            index[tier].append(i)
        
        logger.debug(f"Temporal index: {len(index)} tiers")
        return dict(index)
    
    def build_article_index(self) -> Dict[str, List[int]]:
        """Build index: article_number -> [record_indices]"""
        logger.debug("Building article index...")
        
        index = defaultdict(list)
        
        for i, record in enumerate(self.records):
            article = record.get('article', 'N/A')
            if article != 'N/A':
                index[article].append(i)
        
        logger.debug(f"Article index: {len(index)} unique articles")
        return dict(index)
    
    def build_keyword_index(self) -> Dict[str, List[int]]:
        """
        Build inverted index: keyword -> [record_indices]
        Simple tokenization of about + content.
        """
        logger.debug("Building keyword index...")
        
        index = defaultdict(set)
        
        # Indonesian stopwords
        stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada',
            'dengan', 'adalah', 'ini', 'itu', 'atau', 'jika', 'maka'
        }
        
        for i, record in enumerate(self.records):
            # Tokenize about + content
            text = f"{record['about']} {record['content']}"
            tokens = text.lower().split()
            
            # Add to index
            for token in tokens:
                # Clean token
                token = ''.join(c for c in token if c.isalnum())
                
                # Skip stopwords and short tokens
                if len(token) > 2 and token not in stopwords:
                    index[token].add(i)
        
        # Convert sets to lists
        index = {k: list(v) for k, v in index.items()}
        
        logger.debug(f"Keyword index: {len(index):,} unique keywords")
        return index
    
    def build_composite_index(
        self,
        fields: List[str]
    ) -> Dict[tuple, List[int]]:
        """
        Build composite index from multiple fields.
        
        Args:
            fields: List of field names to index
            
        Returns:
            Dictionary with composite keys
        """
        logger.debug(f"Building composite index on {fields}")
        
        index = defaultdict(list)
        
        for i, record in enumerate(self.records):
            key = tuple(record.get(field) for field in fields)
            index[key].append(i)
        
        logger.debug(f"Composite index: {len(index)} unique combinations")
        return dict(index)