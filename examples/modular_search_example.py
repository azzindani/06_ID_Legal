# examples/modular_search_example.py
"""
Example of using the modular search system.
Shows how to use individual components and full pipeline.
"""
from utils.logging_config import get_logger, setup_logger
from core.search.search_engine import EnhancedSearchEngine
from config.search_config import DEFAULT_RAG_CONFIG, RESEARCH_TEAM_PERSONAS
import numpy as np

# Setup logging
logger = setup_logger('example', level=logging.INFO, console_level=logging.INFO)

def example_full_search():
    """Example: Full search pipeline."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Full Search Pipeline")
    logger.info("=" * 80)
    
    # Mock components (replace with real instances)
    class MockEmbeddingModel:
        def embed(self, text):
            return np.random.randn(768)
    
    class MockKG:
        def extract_entities_from_text(self, text):
            return [('entity1', 'type'), ('entity2', 'type')]
        
        def calculate_enhanced_kg_score(self, entities, record, query_type):
            return 0.7
        
        def extract_regulation_references_with_confidence(self, text):
            return []
    
    # Create mock data
    records = [
        {
            'global_id': i,
            'regulation_type': 'Undang-Undang',
            'regulation_number': str(i),
            'year': '2020',
            'about': f'Tentang {i}',
            'content': f'Isi dokumen {i}',
            'kg_authority_score': 0.8,
            'kg_temporal_score': 0.7,
            'kg_legal_richness': 0.6,
            'kg_completeness_score': 0.75,
            'kg_pagerank': 0.01,
            'kg_primary_domain': 'law',
            'kg_domain_confidence': 0.8,
            'kg_entity_count': 3,
            'kg_entities_json': '[]',
            'kg_hierarchy_level': 2,
            'article': 'N/A'
        }
        for i in range(100)
    ]
    
    embeddings = np.random.randn(100, 768)
    
    # Initialize search engine
    search_engine = EnhancedSearchEngine(
        records=records,
        embeddings=embeddings,
        embedding_model=MockEmbeddingModel(),
        knowledge_graph=MockKG(),
        config=DEFAULT_RAG_CONFIG
    )
    
    # Execute search
    query = "Bagaimana prosedur ketenagakerjaan?"
    
    logger.info(f"Searching for: {query}")
    
    results = search_engine.search(
        query=query,
        query_type='procedural',
        top_k=5,
        progress_callback=lambda msg: logger.info(f"  📋 {msg}")
    )
    
    logger.info(f"✅ Found {len(results)} results")
    
    for i, result in enumerate(results, 1):
        logger.info(
            f"  {i}. {result['record']['regulation_type']} "
            f"{result['record']['regulation_number']} "
            f"(score: {result.get('final_consensus_score', 0):.3f})"
        )
    
    return results


def example_individual_components():
    """Example: Using individual components separately."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Individual Components")
    logger.info("=" * 80)
    
    # Example 1: Just scoring
    from core.search.scoring import CandidateScorer
    
    class MockKG:
        def calculate_enhanced_kg_score(self, entities, record, query_type):
            return 0.6
    
    records = [{'global_id': 1, 'regulation_type': 'UU', 'kg_authority_score': 0.8}]
    scorer = CandidateScorer(MockKG(), records)
    
    logger.info("✅ Scorer created independently")
    
    # Example 2: Just diversity filtering
    from core.search.diversity import DiversityFilter
    
    candidates = [
        {'record': {'regulation_type': 'UU'}, 'composite_score': 0.9},
        {'record': {'regulation_type': 'PP'}, 'composite_score': 0.8}
    ]
    
    filter = DiversityFilter()
    diverse = filter.apply_diversity(candidates, target_count=2)
    
    logger.info(f"✅ Diversity filter: {len(diverse)} diverse results")


def example_custom_researcher():
    """Example: Creating custom researcher persona."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Custom Researcher Persona")
    logger.info("=" * 80)
    
    custom_persona = {
        'name': '🤖 AI Legal Assistant',
        'experience_years': 5,
        'specialties': ['ai_law', 'technology_law'],
        'approach': 'ai_augmented',
        'strengths': ['fast_search', 'pattern_recognition'],
        'weaknesses': ['subjective_interpretation'],
        'bias_towards': 'data_driven_decisions',
        'search_style': {
            'semantic_weight': 0.40,
            'authority_weight': 0.20,
            'kg_weight': 0.30,
            'temporal_weight': 0.10
        },
        'phases_preference': ['initial_scan', 'deep_analysis'],
        'speed_multiplier': 1.5,
        'accuracy_bonus': 0.05
    }
    
    logger.info(f"Created custom persona: {custom_persona['name']}")
    logger.info(f"  Specialties: {custom_persona['specialties']}")
    logger.info(f"  Speed: {custom_persona['speed_multiplier']}x")


if __name__ == '__main__':
    import logging
    
    # Run examples
    try:
        results = example_full_search()
        print("\n")
        
        example_individual_components()
        print("\n")
        
        example_custom_researcher()
        
        logger.info("=" * 80)
        logger.info("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}", exc_info=True)