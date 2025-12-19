"""
Advanced Query Analyzer - Multi-Strategy Query Analysis

Analyzes queries to determine optimal search strategy and confidence levels.
Supports keyword_first, semantic_first, and hybrid_balanced strategies.
"""

from typing import Dict, Any, List, Tuple
import re
from utils.logger_utils import get_logger

logger = get_logger(__name__)


class AdvancedQueryAnalyzer:
    """
    Advanced query analyzer that determines optimal search strategy.

    Analyzes query characteristics to recommend:
    - keyword_first: For specific regulation references
    - semantic_first: For conceptual/thematic queries
    - hybrid_balanced: For mixed queries
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Keywords that suggest keyword-first strategy
        self.keyword_indicators = [
            r'\b(?:UU|PP|Perpres|Permen|Perda)\b',
            r'\bNo\.?\s*\d+',
            r'\bTahun\s*\d{4}',
            r'\bPasal\s+\d+',
            r'\bayat\s+\(\d+\)',
            r'"[^"]+"',  # Quoted phrases
        ]

        # Keywords that suggest semantic-first strategy
        self.semantic_indicators = [
            r'\b(?:bagaimana|apa|mengapa|kapan|siapa|dimana)\b',
            r'\b(?:prosedur|mekanisme|cara|syarat|ketentuan)\b',
            r'\b(?:hak|kewajiban|tanggung\s*jawab|sanksi)\b',
            r'\b(?:contoh|ilustrasi|penjelasan)\b',
            r'\b(?:hubungan|kaitan|perbedaan|persamaan)\b',
        ]

        # Domain-specific terms that boost confidence
        self.domain_terms = {
            'ketenagakerjaan': ['pekerja', 'buruh', 'upah', 'phk', 'cuti', 'kontrak kerja'],
            'perdata': ['perjanjian', 'kontrak', 'ganti rugi', 'wanprestasi'],
            'pidana': ['sanksi', 'pidana', 'denda', 'penjara', 'pelanggaran'],
            'administrasi': ['perizinan', 'izin', 'pendaftaran', 'prosedur'],
            'keuangan': ['pajak', 'cukai', 'bea', 'anggaran'],
        }

        logger.info("AdvancedQueryAnalyzer initialized")

    def analyze(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Analyze query and determine optimal search strategy.

        Args:
            query: User query
            conversation_history: Optional conversation context

        Returns:
            Analysis dict with strategy, confidence, and metadata
        """
        query_lower = query.lower()

        # Count indicators
        keyword_score = self._count_keyword_indicators(query)
        semantic_score = self._count_semantic_indicators(query_lower)

        # Extract key phrases
        key_phrases = self._extract_key_phrases(query)

        # Detect domains
        domains = self._detect_domains(query_lower)

        # Calculate strategy and confidence
        strategy, confidence = self._determine_strategy(keyword_score, semantic_score)

        # Check for regulation references
        has_regulation_ref = bool(re.search(
            r'(?:UU|PP|Perpres|Permen|Perda)\s*(?:No\.?\s*)?\d+',
            query, re.IGNORECASE
        ))

        # Check for specific article reference
        has_article_ref = bool(re.search(
            r'Pasal\s+\d+(?:\s+ayat\s+\(\d+\))?',
            query, re.IGNORECASE
        ))

        # Determine query type
        query_type = self._determine_query_type(
            query_lower, has_regulation_ref, has_article_ref
        )

        # Build analysis result
        result = {
            'query': query,
            'strategy': strategy,
            'confidence': confidence,
            'query_type': query_type,
            'key_phrases': key_phrases,
            'domains': domains,
            'has_regulation_reference': has_regulation_ref,
            'has_article_reference': has_article_ref,
            'scores': {
                'keyword': keyword_score,
                'semantic': semantic_score
            },
            'recommended_weights': self._get_recommended_weights(strategy),
            'context_dependent': self._is_context_dependent(query_lower, conversation_history)
        }

        logger.info("Query analyzed", {
            "strategy": strategy,
            "confidence": f"{confidence:.2f}",
            "query_type": query_type,
            "domains": domains
        })

        return result

    def _count_keyword_indicators(self, query: str) -> float:
        """Count keyword-first indicators in query"""
        count = 0
        for pattern in self.keyword_indicators:
            matches = re.findall(pattern, query, re.IGNORECASE)
            count += len(matches)

        # Normalize to 0-1 range
        return min(1.0, count * 0.25)

    def _count_semantic_indicators(self, query_lower: str) -> float:
        """Count semantic-first indicators in query"""
        count = 0
        for pattern in self.semantic_indicators:
            if re.search(pattern, query_lower):
                count += 1

        # Normalize to 0-1 range
        return min(1.0, count * 0.2)

    def _determine_strategy(
        self,
        keyword_score: float,
        semantic_score: float
    ) -> Tuple[str, float]:
        """Determine optimal search strategy based on scores"""

        # Calculate difference
        diff = keyword_score - semantic_score

        if diff > 0.3:
            # Strong keyword preference
            strategy = 'keyword_first'
            confidence = 0.7 + min(0.3, diff)
        elif diff < -0.3:
            # Strong semantic preference
            strategy = 'semantic_first'
            confidence = 0.7 + min(0.3, abs(diff))
        else:
            # Balanced approach
            strategy = 'hybrid_balanced'
            confidence = 0.6 + (1 - abs(diff)) * 0.3

        return strategy, min(1.0, confidence)

    def _extract_key_phrases(self, query: str) -> List[str]:
        """Extract key phrases from query"""
        phrases = []

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        phrases.extend(quoted)

        # Extract regulation references
        reg_refs = re.findall(
            r'(?:UU|PP|Perpres|Permen|Perda)\s*(?:No\.?\s*)?\d+(?:\s*Tahun\s*\d{4})?',
            query, re.IGNORECASE
        )
        phrases.extend(reg_refs)

        # Extract article references
        article_refs = re.findall(
            r'Pasal\s+\d+(?:\s+ayat\s+\(\d+\))?',
            query, re.IGNORECASE
        )
        phrases.extend(article_refs)

        # Extract domain keywords (2+ words)
        domain_phrases = re.findall(
            r'\b(?:hak\s+\w+|kewajiban\s+\w+|prosedur\s+\w+|syarat\s+\w+)\b',
            query.lower()
        )
        phrases.extend(domain_phrases)

        return list(set(phrases))

    def _detect_domains(self, query_lower: str) -> List[str]:
        """Detect legal domains in query"""
        detected = []

        for domain, keywords in self.domain_terms.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(domain)

        return detected

    def _determine_query_type(
        self,
        query_lower: str,
        has_regulation_ref: bool,
        has_article_ref: bool
    ) -> str:
        """Determine the type of legal query"""

        if has_article_ref:
            return 'specific_article'
        elif has_regulation_ref:
            return 'specific_regulation'
        elif any(word in query_lower for word in ['prosedur', 'cara', 'langkah', 'tahapan']):
            return 'procedural'
        elif any(word in query_lower for word in ['syarat', 'ketentuan', 'persyaratan']):
            return 'requirements'
        elif any(word in query_lower for word in ['sanksi', 'hukuman', 'denda', 'pidana']):
            return 'sanctions'
        elif any(word in query_lower for word in ['hak', 'kewajiban']):
            return 'rights_obligations'
        elif any(word in query_lower for word in ['apa', 'definisi', 'pengertian']):
            return 'definitional'
        elif any(word in query_lower for word in ['bagaimana', 'mengapa']):
            return 'explanatory'
        else:
            return 'general'

    def _get_recommended_weights(self, strategy: str) -> Dict[str, float]:
        """Get recommended search weights based on strategy"""

        if strategy == 'keyword_first':
            return {
                'keyword_precision': 0.35,
                'semantic_match': 0.20,
                'knowledge_graph': 0.20,
                'authority_hierarchy': 0.15,
                'temporal_relevance': 0.10
            }
        elif strategy == 'semantic_first':
            return {
                'semantic_match': 0.35,
                'keyword_precision': 0.15,
                'knowledge_graph': 0.25,
                'authority_hierarchy': 0.15,
                'temporal_relevance': 0.10
            }
        else:  # hybrid_balanced
            return {
                'semantic_match': 0.25,
                'keyword_precision': 0.25,
                'knowledge_graph': 0.20,
                'authority_hierarchy': 0.15,
                'temporal_relevance': 0.15
            }

    def _is_context_dependent(
        self,
        query_lower: str,
        conversation_history: List[Dict] = None
    ) -> bool:
        """Check if query depends on conversation context"""

        # Pronouns and references
        context_indicators = [
            r'\b(?:ini|itu|tersebut|dimaksud)\b',
            r'\b(?:peraturan|regulasi|undang-undang)\s+(?:ini|itu|tersebut)\b',
            r'\b(?:yang|seperti)\s+(?:sudah|telah|baru)\s+(?:disebut|disebutkan)\b',
        ]

        for pattern in context_indicators:
            if re.search(pattern, query_lower):
                return True

        # Short query with history likely needs context
        if conversation_history and len(query_lower.split()) < 5:
            return True

        return False

    def get_strategy_description(self, strategy: str) -> str:
        """Get human-readable description of strategy"""
        descriptions = {
            'keyword_first': 'Prioritas pencarian kata kunci untuk referensi spesifik',
            'semantic_first': 'Prioritas pencarian semantik untuk konsep dan tema',
            'hybrid_balanced': 'Pendekatan seimbang antara kata kunci dan semantik'
        }
        return descriptions.get(strategy, strategy)
