# core/search/scoring.py
"""
Candidate scoring with KG enhancement.
Extracted from EnhancedKGSearchEngine._score_candidates_with_enhanced_kg()
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional
from utils.logging_config import get_logger, log_performance, LogBlock

logger = get_logger(__name__)

class CandidateScorer:
    """
    Scores search candidates using semantic, keyword, KG, and domain signals.
    """
    
    def __init__(self, knowledge_graph, records):
        """
        Initialize scorer.
        
        Args:
            knowledge_graph: KnowledgeGraph instance
            records: List of all document records
        """
        self.kg = knowledge_graph
        self.records = records
        logger.info(f"CandidateScorer initialized with {len(records):,} records")
    
    @log_performance(logger)
    def score_candidates(
        self,
        semantic_sims: np.ndarray,
        keyword_sims: np.ndarray,
        query_entities: List[str],
        query_type: str,
        search_style: Dict[str, float],
        phase_config: Dict[str, Any],
        persona: Dict[str, Any],
        regulation_filter: Optional[Dict] = None,
        query_analysis: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Score candidates with comprehensive multi-signal approach.
        
        Args:
            semantic_sims: Semantic similarity scores (N,)
            keyword_sims: Keyword similarity scores (N,)
            query_entities: Extracted query entities
            query_type: Type of query (specific_article, procedural, etc.)
            search_style: Researcher search preferences
            phase_config: Current search phase configuration
            persona: Researcher persona
            regulation_filter: Optional filter for specific regulation
            query_analysis: Query analysis results
        
        Returns:
            List of scored candidates sorted by composite score
        """
        logger.info(f"Scoring candidates for query_type={query_type}, phase={phase_config.get('description', 'unknown')}")
        logger.debug(f"Search style: {search_style}")
        
        with LogBlock(logger, "Candidate scoring"):
            candidates = []
            
            # Handle missing keyword scores
            if keyword_sims is None:
                keyword_sims = np.zeros_like(semantic_sims)
                logger.warning("No keyword similarities available, using zeros")
            
            # Prepare query entity set for fast lookup
            query_entity_set = set(str(e).lower() for e in query_entities) if query_entities else set()
            logger.debug(f"Query entities: {query_entity_set}")
            
            # Extract key phrases from query analysis
            key_phrases_set = set()
            if query_analysis and query_analysis.get('key_phrases'):
                key_phrases_set = {p['phrase'].lower() for p in query_analysis['key_phrases']}
                logger.info(f"Key phrases detected: {key_phrases_set}")
            
            # Score each candidate
            scored_count = 0
            filtered_count = 0
            
            for i, (sem_score, key_score) in enumerate(zip(
                semantic_sims.tolist() if hasattr(semantic_sims, 'tolist') else semantic_sims,
                keyword_sims
            )):
                try:
                    # Early filtering by thresholds
                    if (sem_score < phase_config['semantic_threshold'] * 0.8 and 
                        key_score < phase_config['keyword_threshold'] * 0.8):
                        filtered_count += 1
                        continue
                    
                    record = self.records[i]
                    
                    # Calculate base scores
                    scores = self._calculate_base_scores(
                        sem_score, key_score, record, search_style,
                        query_entities, query_type, key_phrases_set, query_analysis
                    )
                    
                    # Apply bonuses
                    bonuses = self._calculate_bonuses(
                        record, query_type, scores['kg_score'],
                        persona, query_entity_set, regulation_filter
                    )
                    
                    # Composite score
                    composite_score = scores['composite'] + bonuses['total']
                    composite_score = min(1.0, composite_score)
                    
                    # Create candidate
                    candidate_data = {
                        'record': record,
                        'composite_score': composite_score,
                        'semantic_score': scores['semantic'],
                        'keyword_score': scores['keyword'],
                        'kg_score': scores['kg_score'],
                        'researcher_bias_applied': persona['name'],
                        'query_strategy': query_analysis.get('search_strategy') if query_analysis else 'default',
                        **bonuses  # Include all bonus details
                    }
                    
                    candidates.append(candidate_data)
                    scored_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing candidate {i}: {e}")
                    continue
            
            logger.info(f"Scored {scored_count:,} candidates, filtered {filtered_count:,}")
            
            # Sort by composite score
            candidates.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Log top candidates
            if candidates:
                logger.debug(f"Top candidate: score={candidates[0]['composite_score']:.4f}, "
                           f"doc={candidates[0]['record']['regulation_type']} "
                           f"{candidates[0]['record']['regulation_number']}")
            
            return candidates
    
    def _calculate_base_scores(
        self, sem_score, key_score, record, search_style,
        query_entities, query_type, key_phrases_set, query_analysis
    ) -> Dict[str, float]:
        """Calculate base scoring components."""
        # Normalize scores
        norm_sem_score = max(0, min(1, sem_score))
        norm_key_score = max(0, min(1, (key_score + 1) / 2))
        
        # KG score
        kg_score = self.kg.calculate_enhanced_kg_score(query_entities, record, query_type)
        
        # Key phrase matching bonus
        key_phrase_bonus = self._calculate_key_phrase_bonus(
            record, key_phrases_set
        )
        
        # Dynamic keyword weight based on query analysis
        keyword_weight = search_style.get('semantic_weight', 0.25)
        if query_analysis and query_analysis['search_strategy'] == 'keyword_first':
            keyword_weight = 0.35
            logger.debug("Using keyword-first strategy, increased keyword weight")
        
        # Composite calculation
        composite_score = (
            norm_sem_score * search_style['semantic_weight'] +
            norm_key_score * keyword_weight +
            record['kg_authority_score'] * search_style['authority_weight'] +
            record['kg_temporal_score'] * search_style['temporal_weight'] +
            kg_score * search_style['kg_weight'] +
            record['kg_legal_richness'] * 0.08 +
            record.get('kg_pagerank', 0.0) * 0.05 +
            record['kg_completeness_score'] * 0.07 +
            key_phrase_bonus
        )
        
        return {
            'semantic': norm_sem_score,
            'keyword': norm_key_score,
            'kg_score': kg_score,
            'composite': composite_score,
            'key_phrase_bonus': key_phrase_bonus
        }
    
    def _calculate_key_phrase_bonus(self, record, key_phrases_set) -> float:
        """Calculate bonus for exact key phrase matches."""
        if not key_phrases_set:
            return 0.0
        
        bonus = 0.0
        content_lower = record.get('content', '').lower()
        about_lower = record.get('about', '').lower()
        
        for phrase in key_phrases_set:
            if phrase in content_lower:
                bonus += 0.25
                logger.debug(f"Key phrase '{phrase}' found in content")
            elif phrase in about_lower:
                bonus += 0.20
                logger.debug(f"Key phrase '{phrase}' found in about")
        
        return min(0.5, bonus)  # Cap at 0.5
    
    def _calculate_bonuses(
        self, record, query_type, kg_score, persona,
        query_entity_set, regulation_filter
    ) -> Dict[str, float]:
        """Calculate all scoring bonuses."""
        bonuses = {
            'domain_bonus': self._apply_domain_bonus(record, query_type, kg_score),
            'expertise_bonus': self._apply_expertise_bonus(record, persona, query_type),
            'entity_overlap_bonus': self._apply_entity_overlap_bonus(
                record, query_entity_set
            ),
            'contextual_boost': self._apply_contextual_boost(
                record, regulation_filter
            ),
            'law_name_bonus': 0.0  # Placeholder
        }
        
        bonuses['total'] = sum(bonuses.values())
        
        # Log significant bonuses
        for bonus_type, value in bonuses.items():
            if value > 0.05:
                logger.debug(f"Applied {bonus_type}: +{value:.3f}")
        
        return bonuses
    
    def _apply_domain_bonus(self, record, query_type, kg_score) -> float:
        """Apply domain-specific bonuses."""
        bonus = 0.0
        
        domain = record.get('kg_primary_domain', '').lower()
        domain_confidence = record.get('kg_domain_confidence', 0)
        
        if domain_confidence > 0.7:
            bonus += 0.05
        
        # Query-type specific bonuses
        if query_type == 'sanctions':
            if 'criminal' in domain or 'administrative' in domain:
                bonus += 0.08
            if record.get('kg_has_prohibitions', False):
                bonus += 0.10
        elif query_type == 'procedural':
            if 'administrative' in domain or 'procedural' in domain:
                bonus += 0.08
            if record.get('kg_has_obligations', False):
                bonus += 0.10
        elif query_type == 'specific_article':
            if record.get('article', 'N/A') != 'N/A':
                bonus += 0.12
        
        # KG enhancement bonus
        if kg_score > 0.5:
            bonus += 0.06
        
        return min(0.20, bonus)
    
    def _apply_expertise_bonus(self, record, persona, query_type) -> float:
        """Apply researcher expertise bonus."""
        bonus = 0.0
        specialties = persona.get('specialties', [])
        
        if query_type == 'specific_article' and 'precedent_analysis' in specialties:
            bonus += 0.05
        elif query_type == 'procedural' and 'procedural_law' in specialties:
            bonus += 0.06
        elif query_type == 'definitional' and 'constitutional_law' in specialties:
            bonus += 0.04
        elif 'knowledge_graphs' in specialties:
            if record.get('kg_entity_count', 0) > 5:
                bonus += 0.05
        
        # Authority preference bonus
        if persona.get('bias_towards') == 'established_precedents':
            if record.get('kg_authority_score', 0) > 0.8:
                bonus += 0.04
        
        return min(0.10, bonus)
    
    def _apply_entity_overlap_bonus(self, record, query_entity_set) -> float:
        """Calculate entity overlap bonus."""
        if not query_entity_set or record.get('kg_entity_count', 0) == 0:
            return 0.0
        
        doc_entities_json = record.get('kg_entities_json', '[]')
        if doc_entities_json == '[]':
            return 0.0
        
        try:
            doc_entities = json.loads(doc_entities_json)
            doc_entity_set = set()
            
            for entity in doc_entities[:10]:
                if isinstance(entity, dict):
                    doc_entity_set.add(str(entity.get('text', '')).lower())
                else:
                    doc_entity_set.add(str(entity).lower())
            
            overlap = query_entity_set & doc_entity_set
            if overlap:
                bonus = min(0.15, len(overlap) * 0.05)
                logger.debug(f"Entity overlap: {len(overlap)} entities matched")
                return bonus
        except Exception as e:
            logger.warning(f"Error parsing doc entities: {e}")
        
        return 0.0
    
    def _apply_contextual_boost(self, record, regulation_filter) -> float:
        """Apply contextual boost for regulation filter."""
        if not regulation_filter:
            return 0.0
        
        try:
            rec_type = str(record.get('regulation_type', '')).lower()
            rec_number = str(record.get('regulation_number', ''))
            rec_year = str(record.get('year', ''))
            
            filter_type = str(regulation_filter.get('type') or 
                            regulation_filter.get('regulation_type', '')).lower()
            filter_number = str(regulation_filter.get('number') or 
                               regulation_filter.get('regulation_number', ''))
            filter_year = str(regulation_filter.get('year', ''))
            
            # Type match
            from config.search_config import REGULATION_TYPE_PATTERNS
            type_match = any(
                p in rec_type 
                for p in REGULATION_TYPE_PATTERNS.get(filter_type, [filter_type])
            )
            
            # Number match
            number_match = (filter_number == rec_number)
            
            # Year match
            year_match = (not filter_year or filter_year == rec_year)
            
            if type_match and number_match and year_match:
                logger.info(f"Contextual boost: exact regulation match")
                return 0.25
            elif type_match and number_match:
                logger.debug(f"Contextual boost: type+number match")
                return 0.15
        except Exception as e:
            logger.warning(f"Error applying contextual boost: {e}")
        
        return 0.0