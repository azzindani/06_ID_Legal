"""
Query Detection and Context Analysis Module
Analyzes user queries to determine search strategy and team composition
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from utils.logger_utils import get_logger
from config import (
    QUERY_PATTERNS,
    QUERY_TEAM_COMPOSITIONS,
    REGULATION_TYPE_PATTERNS,
    YEAR_SEPARATORS,
    REGULATION_PRONOUNS,
    FOLLOWUP_INDICATORS,
    CLARIFICATION_INDICATORS,
    CONTENT_QUERY_KEYWORDS,
    INDONESIAN_STOPWORDS
)

# Import comprehensive legal vocabulary from standalone file
from config.legal_vocab import (
    INDONESIAN_LEGAL_SYNONYMS,
    LEGAL_DOMAINS
)


class QueryDetector:
    """
    Detects query type, extracts key information, and determines optimal search strategy
    """
    
    def __init__(self):
        self.logger = get_logger("QueryDetector")
        self.logger.info("QueryDetector initialized")
    
    def analyze_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze query to determine type, extract entities, and configure search strategy
        
        Args:
            query: User query string
            conversation_history: Optional conversation context
            
        Returns:
            Dictionary with query analysis results
        """
        self.logger.info("Analyzing query", {"query_length": len(query)})
        
        query_lower = query.lower()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        self.logger.debug("Query type detected", {"type": query_type})
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        self.logger.debug("Entities extracted", {"count": len(entities)})
        
        # Detect context indicators
        is_followup = self._is_followup_query(query_lower)
        is_clarification = self._is_clarification(query_lower)
        
        # Determine team composition
        team_composition = self._get_team_composition(query_type)
        self.logger.debug("Team composition determined", {
            "team_size": len(team_composition),
            "members": ", ".join(team_composition)
        })
        
        # Get priority weights for this query type
        priority_weights = QUERY_PATTERNS[query_type]['priority_weights']
        
        # Determine search phases to enable
        enabled_phases = self._determine_phases(query_type, entities)
        
        result = {
            'query_type': query_type,
            'entities': entities,
            'team_composition': team_composition,
            'priority_weights': priority_weights,
            'enabled_phases': enabled_phases,
            'is_followup': is_followup,
            'is_clarification': is_clarification,
            'has_specific_article': bool(entities.get('article_references')),
            'has_regulation_ref': bool(entities.get('regulation_type') or entities.get('regulation_number')),
            'complexity_score': self._calculate_complexity(query_lower, entities)
        }
        
        self.logger.info("Query analysis completed", {
            "type": query_type,
            "complexity": f"{result['complexity_score']:.2f}",
            "team_size": len(team_composition)
        })
        
        return result
    
    def _detect_query_type(self, query_lower: str) -> str:
        """Detect the type of query based on patterns"""
        
        # Check each query pattern
        for query_type, pattern_config in QUERY_PATTERNS.items():
            if query_type == 'general':
                continue
                
            indicators = pattern_config['indicators']
            for indicator in indicators:
                if indicator in query_lower:
                    return query_type
        
        return 'general'
    
    def _extract_entities(self, query_lower: str) -> Dict[str, Any]:
        """Extract legal entities from query"""
        
        entities = {
            'regulation_type': None,
            'regulation_number': None,
            'year': None,
            'article_references': [],
            'keywords': [],
            'regulation_mentions': []
        }
        
        # Extract regulation type
        for reg_type, patterns in REGULATION_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    entities['regulation_type'] = reg_type
                    entities['regulation_mentions'].append(pattern)
                    break
            if entities['regulation_type']:
                break
        
        # Extract article references
        article_patterns = [
            r'pasal\s+(\d+[a-z]?)',
            r'ayat\s+\((\d+)\)',
            r'huruf\s+([a-z])',
            r'angka\s+(\d+)'
        ]
        
        for pattern in article_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                entities['article_references'].extend(matches)
        
        # Extract year
        year_pattern = r'(?:tahun|th\.?|nomor)\s*(\d{4})'
        year_matches = re.findall(year_pattern, query_lower)
        if year_matches:
            entities['year'] = year_matches[0]
        
        # Extract regulation number
        number_pattern = r'(?:nomor|no\.?|nummer)\s*(\d+(?:/[A-Z]+)?(?:/\d{4})?)'
        number_matches = re.findall(number_pattern, query_lower)
        if number_matches:
            entities['regulation_number'] = number_matches[0]
        
        # Extract keywords (non-stopwords)
        words = re.findall(r'\b\w+\b', query_lower)
        entities['keywords'] = [w for w in words if w not in INDONESIAN_STOPWORDS and len(w) > 2]
        
        return entities
    
    def _is_followup_query(self, query_lower: str) -> bool:
        """Check if query is a follow-up question"""
        
        # Check for regulation pronouns
        for pronoun in REGULATION_PRONOUNS:
            if pronoun in query_lower:
                return True
        
        # Check for follow-up indicators
        for indicator in FOLLOWUP_INDICATORS:
            if indicator in query_lower:
                return True
        
        return False
    
    def _is_clarification(self, query_lower: str) -> bool:
        """Check if query is asking for clarification"""
        
        for indicator in CLARIFICATION_INDICATORS:
            if indicator in query_lower:
                return True
        
        return False
    
    def _get_team_composition(self, query_type: str) -> List[str]:
        """Get optimal team composition for query type"""
        
        return QUERY_TEAM_COMPOSITIONS.get(query_type, QUERY_TEAM_COMPOSITIONS['general'])
    
    def _determine_phases(self, query_type: str, entities: Dict) -> List[str]:
        """Determine which search phases should be enabled"""
        
        phases = []
        
        # Always start with initial scan
        phases.append('initial_scan')
        
        if query_type == 'specific_article' and entities.get('article_references'):
            # For specific articles, focus on targeted phases
            phases.extend(['focused_review', 'verification'])
        elif query_type == 'procedural':
            # For procedural queries, need comprehensive search
            phases.extend(['focused_review', 'deep_analysis', 'verification'])
        elif query_type == 'definitional':
            # For definitions, focus on authoritative sources
            phases.extend(['focused_review', 'verification'])
        elif query_type == 'sanctions':
            # For sanctions, need thorough analysis
            phases.extend(['focused_review', 'deep_analysis', 'verification', 'expert_review'])
        else:
            # General queries use all standard phases
            phases.extend(['focused_review', 'deep_analysis', 'verification'])
        
        return phases
    
    def _calculate_complexity(self, query_lower: str, entities: Dict) -> float:
        """Calculate query complexity score (0-1)"""
        
        complexity = 0.3  # Base complexity
        
        # Add complexity for specific references
        if entities.get('article_references'):
            complexity += 0.2
        
        if entities.get('regulation_number'):
            complexity += 0.15
        
        # Add complexity for multiple concepts
        keyword_count = len(entities.get('keywords', []))
        if keyword_count > 5:
            complexity += 0.2
        elif keyword_count > 3:
            complexity += 0.1
        
        # Add complexity for legal terms
        legal_terms = ['sanksi', 'pidana', 'denda', 'prosedur', 'tata cara', 'ketentuan']
        for term in legal_terms:
            if term in query_lower:
                complexity += 0.05
        
        return min(1.0, complexity)
    
    def enhance_query(self, original_query: str, analysis: Dict[str, Any]) -> str:
        """
        Enhance query with extracted entities for better search
        
        Args:
            original_query: Original user query
            analysis: Query analysis results
            
        Returns:
            Enhanced query string
        """
        enhanced = original_query
        entities = analysis['entities']
        
        # Add regulation type if found
        if entities.get('regulation_type'):
            if entities['regulation_type'] not in enhanced.lower():
                enhanced += f" {entities['regulation_type']}"
        
        # Add year if found
        if entities.get('year'):
            if entities['year'] not in enhanced:
                enhanced += f" tahun {entities['year']}"
        
        self.logger.debug("Query enhanced", {
            "original_length": len(original_query),
            "enhanced_length": len(enhanced)
        })

        return enhanced
    
    def expand_query_with_synonyms(self, query: str) -> List[str]:
        """
        Expand query with Indonesian legal synonyms for better keyword matching
        
        Args:
            query: Original query string
            
        Returns:
            List of query variations with synonyms (up to 5 variations)
        """
        query_lower = query.lower()
        expanded_terms = [query]  # Original query first
        
        for term, synonyms in INDONESIAN_LEGAL_SYNONYMS.items():
            if term in query_lower:
                # Add variations with synonyms
                for synonym in synonyms[1:]:  # Skip original term
                    expanded_variant = query_lower.replace(term, synonym)
                    if expanded_variant not in expanded_terms and expanded_variant != query_lower:
                        expanded_terms.append(expanded_variant)
                        if len(expanded_terms) >= 5:
                            break
            if len(expanded_terms) >= 5:
                break
        
        self.logger.debug("Query expanded with synonyms", {
            "original": query,
            "variations": len(expanded_terms)
        })
        
        return expanded_terms[:5]
    
    def classify_query_intent(self, query: str) -> str:
        """
        Classify query intent for intent-aware search strategy
        
        Intent Types:
        - specific_regulation: Query mentions specific regulation number
        - rights_benefits: Query about rights, allowances, benefits
        - procedural: Query about procedures, requirements
        - topic_search: General topic search
        
        Args:
            query: Query string
            
        Returns:
            Intent type string
        """
        query_lower = query.lower()
        
        # Specific regulation (highest priority)
        if re.search(r'(uu|pp|perpres|permen|perda).*?\d+.*?tahun', query_lower):
            return 'specific_regulation'
        
        # Rights/benefits intent
        rights_keywords = ['hak', 'berhak', 'tunjangan', 'benefit', 'insentif', 'gaji', 'upah', 
                          'penghasilan', 'kesejahteraan', 'jaminan']
        if any(keyword in query_lower for keyword in rights_keywords):
            return 'rights_benefits'
        
        # Procedural intent
        procedural_keywords = ['bagaimana', 'cara', 'prosedur', 'syarat', 'ketentuan', 
                               'tata cara', 'mekanisme', 'tahapan']
        if any(keyword in query_lower for keyword in procedural_keywords):
            return 'procedural'
        
        # Default: topic search
        return 'topic_search'


if __name__ == "__main__":
    from utils.logger_utils import initialize_logging
    initialize_logging(enable_file_logging=False)

    print("=" * 60)
    print("QUERY DETECTION TEST")
    print("=" * 60)

    detector = QueryDetector()

    # Test queries
    test_queries = [
        ("Apa itu hukum ketenagakerjaan?", "definitional"),
        ("Bagaimana prosedur pengajuan cuti?", "procedural"),
        ("Apa sanksi dalam UU ITE pasal 27?", "sanctions"),
        ("Jelaskan pasal 156 UU Ketenagakerjaan", "specific_article"),
        ("Peraturan tentang upah minimum", "general"),
    ]

    print("\nQuery Type Detection:")
    print("-" * 60)

    for query, expected in test_queries:
        result = detector.analyze_query(query)
        status = "✓" if result['query_type'] == expected else "✗"
        print(f"\n{status} Query: {query}")
        print(f"   Type: {result['query_type']} (expected: {expected})")
        print(f"   Team: {', '.join(result['team_composition'])}")
        print(f"   Complexity: {result['complexity_score']:.2f}")

        if result['entities']:
            print(f"   Entities: {result['entities']}")

    # Test follow-up detection
    print("\n" + "-" * 60)
    print("Follow-up Detection:")
    print("-" * 60)

    followup_queries = [
        "Lanjutkan penjelasan tadi",
        "Maksudnya bagaimana?",
        "Bisa dijelaskan lebih detail?",
    ]

    for query in followup_queries:
        result = detector.analyze_query(query)
        status = "✓" if result['is_followup'] or result['is_clarification'] else "✗"
        print(f"  {status} '{query}' -> followup={result['is_followup']}, clarification={result['is_clarification']}")

    # Test query enhancement
    print("\n" + "-" * 60)
    print("Query Enhancement:")
    print("-" * 60)

    query = "sanksi pasal 27"
    result = detector.analyze_query(query)
    enhanced = detector.enhance_query(query, result)
    print(f"  Original: {query}")
    print(f"  Enhanced: {enhanced}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)