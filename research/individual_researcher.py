# research/individual_researcher.py
"""
Individual researcher simulation with persona-based search.
Extracted from EnhancedKGSearchEngine._conduct_individual_research()
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from utils.logging_config import get_logger, log_performance, LogBlock

logger = get_logger(__name__)

class IndividualResearcher:
    """
    Simulates a single researcher conducting legal research.
    Each researcher has unique expertise and search preferences.
    """
    
    def __init__(
        self,
        researcher_id: str,
        persona: Dict[str, Any],
        embedding_model,
        knowledge_graph,
        scorer,
        diversity_filter,
        records,
        embeddings
    ):
        """
        Initialize researcher.
        
        Args:
            researcher_id: Unique researcher identifier
            persona: Researcher persona configuration
            embedding_model: Model for query embedding
            knowledge_graph: KG instance
            scorer: CandidateScorer instance
            diversity_filter: DiversityFilter instance
            records: All document records
            embeddings: Document embeddings tensor
        """
        self.researcher_id = researcher_id
        self.persona = persona
        self.embedding_model = embedding_model
        self.kg = knowledge_graph
        self.scorer = scorer
        self.diversity_filter = diversity_filter
        self.records = records
        self.embeddings = embeddings
        
        logger.info(
            f"Researcher initialized: {persona['name']} "
            f"(exp: {persona['experience_years']}y, "
            f"speed: {persona['speed_multiplier']}x)"
        )
    
    @log_performance(logger)
    def conduct_research(
        self,
        query: str,
        query_type: str,
        query_embedding: torch.Tensor,
        query_entities: List[str],
        config: Dict[str, Any],
        regulation_filter: Optional[Dict] = None,
        query_analysis: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Conduct research based on researcher's expertise and style.
        
        Args:
            query: User query
            query_type: Query classification
            query_embedding: Embedded query vector
            query_entities: Extracted entities
            config: Search configuration
            regulation_filter: Optional regulation filter
            query_analysis: Query analysis results
        
        Returns:
            Research results with phase breakdowns
        """
        logger.info(
            f"🔍 {self.persona['name']} researching: "
            f"query_type={query_type}, filter={regulation_filter is not None}"
        )
        
        with LogBlock(logger, f"Research by {self.persona['name']}"):
            try:
                # Check for strict metadata mode
                exact_regulation_ref = None
                exact_match_mode = False
                metadata_candidates = []
                
                # STRICT METADATA DETECTION
                if self.kg and regulation_filter:
                    regulation_refs = self.kg.extract_regulation_references_with_confidence(query)
                    
                    if regulation_refs:
                        best_ref = regulation_refs[0]
                        
                        if (best_ref['confidence'] >= 0.9 and
                            best_ref['specificity'] == 'complete' and
                            best_ref['regulation'].get('type') and
                            best_ref['regulation'].get('number') and
                            best_ref['regulation'].get('year')):
                            
                            exact_regulation_ref = best_ref['regulation']
                            exact_match_mode = True
                            
                            logger.info(
                                f"🎯 STRICT METADATA MODE: "
                                f"{exact_regulation_ref['type']} "
                                f"{exact_regulation_ref['number']}/"
                                f"{exact_regulation_ref['year']}"
                            )
                            
                            # Execute metadata-only search
                            metadata_candidates = self._metadata_only_search(
                                exact_regulation_ref
                            )
                
                # Normal semantic search if no metadata matches
                if not exact_match_mode or not metadata_candidates:
                    if exact_match_mode and not metadata_candidates:
                        logger.warning(
                            "⚠️ Metadata mode but no matches found, "
                            "falling back to semantic"
                        )
                    
                    phase_results = self._execute_search_phases(
                        query, query_type, query_embedding, query_entities,
                        config, regulation_filter, query_analysis
                    )
                    
                    all_candidates = []
                    for phase_data in phase_results.values():
                        all_candidates.extend(phase_data['candidates'])
                else:
                    # Metadata-only results
                    phase_results = {
                        f"{self.researcher_id}_metadata_strict": {
                            'phase': 'metadata_strict',
                            'researcher': self.researcher_id,
                            'researcher_name': self.persona['name'],
                            'candidates': metadata_candidates,
                            'confidence': 1.0,
                            'exact_metadata_mode': True
                        }
                    }
                    all_candidates = metadata_candidates
                
                logger.info(
                    f"✅ {self.persona['name']} found {len(all_candidates)} candidates "
                    f"across {len(phase_results)} phases"
                )
                
                return {
                    'researcher_id': self.researcher_id,
                    'phase_results': phase_results,
                    'all_candidates': all_candidates,
                    'exact_match_mode': bool(metadata_candidates),
                    'query_type': query_type,
                    'persona_metadata': {
                        'adjusted': self.persona.get('_adjusted', False),
                        'success_rate': self.persona.get('_success_rate', 0),
                        'learned_adjustment': self.persona.get('_learned_adjustment', 0)
                    }
                }
                
            except Exception as e:
                logger.error(
                    f"❌ Research failed for {self.persona['name']}: {e}",
                    exc_info=True
                )
                return {
                    'researcher_id': self.researcher_id,
                    'phase_results': {},
                    'all_candidates': [],
                    'error': str(e)
                }
    
    def _metadata_only_search(self, regulation_ref: Dict) -> List[Dict]:
        """
        Execute strict metadata-only search.
        
        Args:
            regulation_ref: Regulation reference with type, number, year
        
        Returns:
            List of exactly matching candidates with perfect scores
        """
        logger.info("🔒 Executing STRICT metadata filter")
        
        ref_type = regulation_ref['type'].lower()
        ref_number = regulation_ref['number']
        ref_year = regulation_ref['year']
        
        matching_candidates = []
        
        from config.search_config import REGULATION_TYPE_PATTERNS
        
        for i, record in enumerate(self.records):
            try:
                rec_type = str(record.get('regulation_type', '')).lower()
                rec_number = str(record.get('regulation_number', ''))
                rec_year = str(record.get('year', ''))
                
                # Type match (flexible pattern matching)
                type_match = False
                for patterns in REGULATION_TYPE_PATTERNS.values():
                    if (any(p in rec_type for p in patterns) and 
                        any(p in ref_type for p in patterns)):
                        type_match = True
                        break
                
                if not type_match:
                    continue
                
                # Number match (EXACT)
                if ref_number != rec_number:
                    continue
                
                # Year match (EXACT)
                if ref_year != rec_year:
                    continue
                
                # ✅ TRIPLE MATCH - PERFECT SCORE OVERRIDE
                logger.info(
                    f"✅ EXACT MATCH: {rec_type} {rec_number}/{rec_year}"
                )
                
                candidate = {
                    'record': record,
                    'composite_score': 1.0,  # Perfect score
                    'semantic_score': 0.0,
                    'keyword_score': 0.0,
                    'kg_score': float(record.get('kg_connectivity_score', 0.5)),
                    'metadata_match': True,
                    'exact_metadata_mode': True,
                    'match_type': 'STRICT_TRIPLE_MATCH',
                    'researcher_bias_applied': self.persona['name'],
                    'perfect_score_override': True
                }
                
                matching_candidates.append(candidate)
                
            except Exception as e:
                logger.warning(f"Error processing record {i}: {e}")
                continue
        
        logger.info(f"📊 Metadata filter: {len(matching_candidates)} exact matches")
        
        return matching_candidates
    
    def _execute_search_phases(
        self,
        query: str,
        query_type: str,
        query_embedding: torch.Tensor,
        query_entities: List[str],
        config: Dict[str, Any],
        regulation_filter: Optional[Dict],
        query_analysis: Optional[Dict]
    ) -> Dict[str, Dict]:
        """
        Execute multiple search phases based on researcher preferences.
        
        Returns:
            Dictionary of phase results
        """
        # Adjust search weights based on query analysis
        search_style = self._adjust_search_style(query_analysis)
        
        # Get preferred phases
        phase_preference = self.persona['phases_preference']
        speed_mult = self.persona['speed_multiplier']
        
        search_phases = config.get('search_phases', {})
        
        phase_results = {}
        
        logger.info(
            f"📋 Executing {len(phase_preference)} phases: {phase_preference}"
        )
        
        for phase_name in phase_preference:
            if phase_name not in search_phases:
                logger.debug(f"Phase {phase_name} not in config, skipping")
                continue
            
            phase_config = search_phases[phase_name]
            if not phase_config.get('enabled', True):
                logger.debug(f"Phase {phase_name} disabled, skipping")
                continue
            
            # Adjust candidate count by speed multiplier
            adjusted_config = phase_config.copy()
            adjusted_config['candidates'] = int(
                phase_config['candidates'] * speed_mult
            )
            
            logger.debug(
                f"🔹 Starting phase: {phase_name} "
                f"(candidates: {adjusted_config['candidates']}, "
                f"semantic_threshold: {adjusted_config['semantic_threshold']})"
            )
            
            # Calculate semantic similarities
            semantic_sims = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                self.embeddings,
                dim=1
            ).cpu().numpy()
            
            # Calculate keyword similarities (if available)
            keyword_sims = self._calculate_keyword_similarities(query)
            
            # Score candidates
            phase_candidates = self.scorer.score_candidates(
                semantic_sims,
                keyword_sims,
                query_entities,
                query_type,
                search_style,
                adjusted_config,
                self.persona,
                regulation_filter=regulation_filter,
                query_analysis=query_analysis
            )
            
            # Apply diversity filter
            diverse_candidates = self.diversity_filter.apply_diversity(
                phase_candidates,
                adjusted_config['candidates']
            )
            
            # Calculate confidence
            confidence = self._calculate_phase_confidence(
                diverse_candidates,
                phase_config
            )
            
            phase_key = f"{self.researcher_id}_{phase_name}"
            phase_results[phase_key] = {
                'phase': phase_name,
                'researcher': self.researcher_id,
                'researcher_name': self.persona['name'],
                'candidates': diverse_candidates,
                'confidence': confidence,
                'persona_adjusted': self.persona.get('_adjusted', False),
                'learned_bonus': self.persona.get('_learned_adjustment', 0.0),
                'query_strategy': query_analysis.get('search_strategy') if query_analysis else 'default',
                'exact_match_mode': False
            }
            
            logger.info(
                f"✅ Phase {phase_name} complete: "
                f"{len(diverse_candidates)} candidates, confidence={confidence:.2%}"
            )
        
        return phase_results
    
    def _adjust_search_style(
        self,
        query_analysis: Optional[Dict]
    ) -> Dict[str, float]:
        """
        Adjust search weights based on query analysis.
        
        Args:
            query_analysis: Query analysis results
        
        Returns:
            Adjusted search style weights
        """
        search_style = self.persona['search_style'].copy()
        
        if not query_analysis:
            return search_style
        
        strategy = query_analysis.get('search_strategy', 'default')
        
        if strategy == 'keyword_first':
            # Boost keyword weight
            keyword_boost = query_analysis.get('keyword_boost', 0.30)
            semantic_boost = query_analysis.get('semantic_boost', 0.15)
            
            search_style['semantic_weight'] = semantic_boost
            search_style['kg_weight'] = search_style.get('kg_weight', 0.25) + 0.10
            
            logger.debug(
                f"Adjusted for keyword-first: "
                f"semantic={semantic_boost}, kg={search_style['kg_weight']}"
            )
            
        elif strategy == 'semantic_first':
            # Boost semantic weight
            semantic_boost = query_analysis.get('semantic_boost', 0.35)
            search_style['semantic_weight'] = semantic_boost
            search_style['authority_weight'] = max(
                0.10,
                search_style.get('authority_weight', 0.25) - 0.05
            )
            
            logger.debug(f"Adjusted for semantic-first: semantic={semantic_boost}")
        
        return search_style
    
    def _calculate_keyword_similarities(self, query: str):
        """Calculate keyword similarities using TF-IDF."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Access dataset loader's TF-IDF components
            # This assumes you have access to tfidf_vectorizer and tfidf_matrix
            # You'll need to pass these in __init__ or via a data manager
            
            # Placeholder - you'll need to implement this based on your setup
            logger.debug("Calculating keyword similarities")
            return None
            
        except Exception as e:
            logger.warning(f"Keyword similarity calculation failed: {e}")
            return None
    
    def _calculate_phase_confidence(
        self,
        candidates: List[Dict],
        phase_config: Dict
    ) -> float:
        """
        Calculate confidence score for phase results.
        
        Args:
            candidates: Scored candidates
            phase_config: Phase configuration
        
        Returns:
            Confidence score (0-1)
        """
        if not candidates:
            return 0.0
        
        # Average score above threshold
        avg_score = sum(c.get('composite_score', 0) for c in candidates) / len(candidates)
        
        # Normalize by threshold
        sem_threshold = phase_config.get('semantic_threshold', 0.3)
        confidence = min(1.0, avg_score / (sem_threshold + 0.1))
        
        logger.debug(
            f"Phase confidence: {confidence:.2%} "
            f"(avg_score={avg_score:.3f}, threshold={sem_threshold})"
        )
        
        return confidence