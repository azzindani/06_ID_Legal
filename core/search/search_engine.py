# core/search/search_engine.py
"""
Main search engine orchestrator - modular version.
Coordinates all search components with comprehensive logging.
"""
from typing import List, Dict, Any, Optional
from utils.logging_config import get_logger, log_performance, LogBlock
from core.search.scoring import CandidateScorer
from core.search.diversity import DiversityFilter
from research.individual_researcher import IndividualResearcher

logger = get_logger(__name__)

class EnhancedSearchEngine:
    """
    Modular search engine for Indonesian legal documents.
    
    This is the REFACTORED version of EnhancedKGSearchEngine.
    All components are now separate, testable modules.
    """
    
    def __init__(
        self,
        records: List[Dict],
        embeddings,
        embedding_model,
        knowledge_graph,
        config: Dict[str, Any]
    ):
        """
        Initialize modular search engine.
        
        Args:
            records: All document records
            embeddings: Document embeddings tensor
            embedding_model: Embedding model instance
            knowledge_graph: KG instance
            config: Search configuration
        """
        logger.info("=" * 80)
        logger.info("Initializing Enhanced Search Engine (Modular)")
        logger.info("=" * 80)
        
        self.records = records
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.kg = knowledge_graph
        self.config = config
        
        # Initialize components
        logger.info("🔧 Initializing search components...")
        
        self.scorer = CandidateScorer(knowledge_graph, records)
        logger.info("  ✅ CandidateScorer initialized")
        
        self.diversity_filter = DiversityFilter()
        logger.info("  ✅ DiversityFilter initialized")
        
        # Initialize researchers
        self.researchers = {}
        self._initialize_research_team()
        
        # Tracking
        self.all_phase_results = {}
        self.research_session_log = []
        
        logger.info(f"✅ Search engine ready: {len(records):,} documents indexed")
        logger.info("=" * 80)
    
    def _initialize_research_team(self):
        """Initialize research team with personas."""
        from config.search_config import RESEARCH_TEAM_PERSONAS
        
        logger.info("👥 Initializing research team...")
        
        for researcher_id, persona in RESEARCH_TEAM_PERSONAS.items():
            self.researchers[researcher_id] = IndividualResearcher(
                researcher_id=researcher_id,
                persona=persona,
                embedding_model=self.embedding_model,
                knowledge_graph=self.kg,
                scorer=self.scorer,
                diversity_filter=self.diversity_filter,
                records=self.records,
                embeddings=self.embeddings
            )
        
        logger.info(f"  ✅ {len(self.researchers)} researchers initialized")
    
    @log_performance(logger)
    def search(
        self,
        query: str,
        query_type: str = 'general',
        top_k: int = 10,
        regulation_filter: Optional[Dict] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Execute modular search with research team.
        
        Args:
            query: User query
            query_type: Query classification
            top_k: Number of results
            regulation_filter: Optional regulation filter
            progress_callback: Progress reporting function
        
        Returns:
            List of top results
        """
        logger.info("=" * 80)
        logger.info(f"🔍 SEARCH REQUEST")
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"Type: {query_type}, Top-K: {top_k}")
        logger.info("=" * 80)
        
        with LogBlock(logger, "Complete search process"):
            # Step 1: Analyze query
            if progress_callback:
                progress_callback("🧠 Analyzing query...")
            
            query_analysis = self._analyze_query(query)
            logger.info(
                f"Query strategy: {query_analysis.get('search_strategy', 'unknown')} "
                f"(confidence: {query_analysis.get('confidence', 0):.0%})"
            )
            
            # Step 2: Extract entities
            if progress_callback:
                progress_callback("🏷️ Extracting entities...")
            
            query_entities = self._extract_query_entities(query)
            logger.info(f"Extracted {len(query_entities)} entities: {query_entities[:3]}")
            
            # Step 3: Embed query
            if progress_callback:
                progress_callback("🔢 Embedding query...")
            
            query_embedding = self._embed_query(query, query_type)
            
            # Step 4: Assemble research team
            if progress_callback:
                progress_callback("👥 Assembling research team...")
            
            team_members = self._select_research_team(query_type)
            logger.info(f"Team assembled: {len(team_members)} researchers")
            
            # Step 5: Parallel research
            if progress_callback:
                progress_callback("🔬 Conducting parallel research...")
            
            individual_results = self._conduct_parallel_research(
                team_members,
                query,
                query_type,
                query_embedding,
                query_entities,
                regulation_filter,
                query_analysis,
                progress_callback
            )
            
            # Step 6: Build consensus
            if progress_callback:
                progress_callback("🤝 Building team consensus...")
            
            consensus_results = self._build_team_consensus(
                individual_results,
                team_members,
                query_type
            )
            
            # Step 7: Final selection
            final_results = consensus_results[:top_k]
            
            logger.info("=" * 80)
            logger.info(f"✅ SEARCH COMPLETE: {len(final_results)} final results")
            logger.info("=" * 80)
            
            return final_results
    
    def _analyze_query(self, query: str) -> Dict:
        """Analyze query using query analyzer."""
        # This would use your QueryAnalyzer class
        # For now, returning basic structure
        logger.debug("Analyzing query structure and intent")
        
        return {
            'search_strategy': 'semantic_first',
            'confidence': 0.7,
            'key_phrases': [],
            'reasoning': 'Balanced hybrid search'
        }
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query using KG."""
        logger.debug("Extracting legal entities from query")
        
        entities = [
            entity for entity, _ in 
            self.kg.extract_entities_from_text(query)
        ]
        
        logger.debug(f"Found {len(entities)} entities")
        return entities
    
    def _embed_query(self, query: str, query_type: str):
        """Embed query with context."""
        logger.debug(f"Embedding query for type: {query_type}")
        
        # Add context based on query type
        context_map = {
            'specific_article': 'pasal dan ayat spesifik',
            'procedural': 'prosedur dan tata cara',
            'definitional': 'definisi dan pengertian',
            'sanctions': 'sanksi dan hukuman',
            'general': 'informasi hukum'
        }
        
        context = context_map.get(query_type, 'informasi hukum')
        enhanced_query = f"Mencari {context}: {query}"
        
        return self.embedding_model.embed(enhanced_query)
    
    def _select_research_team(self, query_type: str) -> List[str]:
        """Select optimal team for query type."""
        from config.search_config import QUERY_TEAM_COMPOSITIONS
        
        team_composition = QUERY_TEAM_COMPOSITIONS.get(
            query_type,
            QUERY_TEAM_COMPOSITIONS['general']
        )
        
        team_size = self.config.get('research_team_size', 4)
        selected = team_composition[:team_size]
        
        # Add devil's advocate if enabled
        if (self.config.get('enable_devil_advocate', True) and
            'devils_advocate' not in selected and len(selected) < 5):
            selected.append('devils_advocate')
        
        logger.info(f"Selected team: {[self.researchers[r].persona['name'] for r in selected]}")
        
        return selected
    
    def _conduct_parallel_research(
        self,
        team_members: List[str],
        query: str,
        query_type: str,
        query_embedding,
        query_entities: List[str],
        regulation_filter: Optional[Dict],
        query_analysis: Dict,
        progress_callback: Optional[callable]
    ) -> Dict:
        """
        Conduct parallel research with multiple researchers.
        
        Returns:
            Dictionary of individual research results
        """
        logger.info(f"🔬 Starting parallel research with {len(team_members)} researchers")
        
        individual_results = {}
        
        for researcher_id in team_members:
            researcher = self.researchers[researcher_id]
            
            if progress_callback:
                progress_callback(f"  📋 {researcher.persona['name']} researching...")
            
            logger.info(f"  🔹 {researcher.persona['name']} starting research")
            
            try:
                result = researcher.conduct_research(
                    query=query,
                    query_type=query_type,
                    query_embedding=query_embedding,
                    query_entities=query_entities,
                    config=self.config,
                    regulation_filter=regulation_filter,
                    query_analysis=query_analysis
                )
                
                individual_results[researcher_id] = result
                
                # Log to all_phase_results for compatibility
                for phase_key, phase_data in result.get('phase_results', {}).items():
                    self.all_phase_results[phase_key] = phase_data
                
                logger.info(
                    f"  ✅ {researcher.persona['name']}: "
                    f"{len(result['all_candidates'])} candidates found"
                )
                
            except Exception as e:
                logger.error(
                    f"  ❌ {researcher.persona['name']} research failed: {e}",
                    exc_info=True
                )
                individual_results[researcher_id] = {
                    'researcher_id': researcher_id,
                    'phase_results': {},
                    'all_candidates': [],
                    'error': str(e)
                }
        
        logger.info(f"✅ Parallel research complete: {len(individual_results)} researchers")
        
        return individual_results
    
    def _build_team_consensus(
        self,
        individual_results: Dict,
        team_members: List[str],
        query_type: str
    ) -> List[Dict]:
        """
        Build consensus from individual research results.
        
        Args:
            individual_results: Results from each researcher
            team_members: List of researcher IDs
            query_type: Query classification
        
        Returns:
            Consensus-ranked candidates
        """
        logger.info("🤝 Building team consensus")
        
        with LogBlock(logger, "Consensus building"):
            # Collect all candidates with attributions
            all_candidates_with_attribution = {}
            
            for researcher_id, results in individual_results.items():
                researcher_persona = self.researchers[researcher_id].persona
                
                for candidate in results.get('all_candidates', []):
                    doc_id = candidate['record']['global_id']
                    
                    if doc_id not in all_candidates_with_attribution:
                        all_candidates_with_attribution[doc_id] = {
                            'candidate': candidate,
                            'researcher_scores': {},
                            'supporting_researchers': [],
                            'researcher_types': set()
                        }
                    
                    attribution = all_candidates_with_attribution[doc_id]
                    attribution['researcher_scores'][researcher_id] = candidate['composite_score']
                    attribution['supporting_researchers'].append(researcher_id)
                    attribution['researcher_types'].add(researcher_persona['approach'])
            
            logger.info(f"Collected {len(all_candidates_with_attribution)} unique documents")
            
            # Calculate weighted consensus scores
            final_candidates = []
            consensus_threshold = self.config.get('consensus_threshold', 0.6)
            
            for doc_id, attribution in all_candidates_with_attribution.items():
                try:
                    # Weighted average by experience
                    total_weight = 0
                    weighted_score = 0
                    
                    for researcher_id, score in attribution['researcher_scores'].items():
                        persona = self.researchers[researcher_id].persona
                        weight = (persona['experience_years'] / 15.0) + persona.get('accuracy_bonus', 0)
                        weighted_score += score * weight
                        total_weight += weight
                    
                    final_weighted_score = weighted_score / total_weight if total_weight > 0 else 0
                    
                    # Apply consensus bonuses
                    if len(attribution['supporting_researchers']) > 1:
                        consensus_bonus = min(0.10, 0.03 * (len(attribution['supporting_researchers']) - 1))
                        final_weighted_score += consensus_bonus
                        logger.debug(
                            f"Applied consensus bonus: +{consensus_bonus:.3f} "
                            f"({len(attribution['supporting_researchers'])} researchers)"
                        )
                    
                    if len(attribution['researcher_types']) > 1:
                        final_weighted_score += 0.05
                        logger.debug("Applied diversity bonus: +0.05 (multiple approaches)")
                    
                    # Check threshold
                    if final_weighted_score >= consensus_threshold:
                        candidate = attribution['candidate'].copy()
                        candidate['final_consensus_score'] = min(1.0, final_weighted_score)
                        candidate['team_consensus'] = True
                        candidate['supporting_researchers'] = attribution['supporting_researchers']
                        candidate['researcher_agreement'] = len(attribution['supporting_researchers'])
                        
                        final_candidates.append(candidate)
                
                except Exception as e:
                    logger.warning(f"Error processing candidate consensus: {e}")
                    continue
            
            # Sort by consensus score
            final_candidates.sort(key=lambda x: x.get('final_consensus_score', 0), reverse=True)
            
            logger.info(
                f"✅ Consensus complete: {len(final_candidates)} candidates above threshold "
                f"(threshold: {consensus_threshold:.0%})"
            )
            
            return final_candidates