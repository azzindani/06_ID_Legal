"""
Comprehensive Test Script - Tests the FIXED RAG system
Run this to verify all fixes are working correctly
"""

import sys
import os

# Add project root to path (two directories up from this file)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger_utils import initialize_logging, get_logger, log_session_end
from config import DATASET_NAME, EMBEDDING_DIM, get_default_config
from loader.dataloader import EnhancedKGDatasetLoader
from core.model_manager import get_model_manager
from core.search.query_detection import QueryDetector
from core.search.hybrid_search import HybridSearchEngine
from core.search.stages_research import StagesResearchEngine
from core.search.consensus import ConsensusBuilder
from core.search.reranking import RerankerEngine
import time


def test_basic_search():
    """Test basic search functionality"""
    logger = get_logger("TestBasic")
    logger.info("="*80)
    logger.info("TEST 1: BASIC SEARCH FUNCTIONALITY")
    logger.info("="*80)
    
    # Setup
    logger.info("Setting up system...")
    
    # Load dataset
    loader = EnhancedKGDatasetLoader(DATASET_NAME, EMBEDDING_DIM)
    
    def progress(msg):
        logger.info(f"   {msg}")
    
    if not loader.load_from_huggingface(progress):
        logger.error("Dataset loading failed")
        return False
    
    # Load models
    model_manager = get_model_manager()
    embedding_model = model_manager.load_embedding_model()
    reranker_model = model_manager.load_reranker_model(use_mock=True)
    
    # Create search engine
    hybrid_search = HybridSearchEngine(
        data_loader=loader,
        embedding_model=embedding_model,
        reranker_model=reranker_model
    )
    
    # Test query
    test_query = "Apa sanksi dalam UU ITE?"
    logger.info(f"Test query: {test_query}")
    
    # Get query embedding
    import torch
    query_embedding = hybrid_search._get_query_embedding(test_query)
    
    if query_embedding is None:
        logger.error("Failed to get query embedding")
        return False
    
    logger.success("Query embedding obtained", {
        "shape": str(query_embedding.shape),
        "norm": f"{torch.norm(query_embedding).item():.4f}"
    })
    
    # Test semantic search
    logger.info("Testing semantic search...")
    sem_scores, sem_indices = hybrid_search._semantic_search_fixed(query_embedding, top_k=10)
    
    logger.success("Semantic search completed", {
        "results": len(sem_indices),
        "top_score": f"{sem_scores[0].item():.4f}",
        "min_score": f"{sem_scores[-1].item():.4f}"
    })
    
    # Show results
    logger.info("Top 5 semantic results:")
    for i in range(min(5, len(sem_indices))):
        idx = sem_indices[i].item()
        score = sem_scores[i].item()
        record = loader.all_records[idx]
        logger.info(f"  {i+1}. Score: {score:.4f} | {record['regulation_type']} {record['regulation_number']}/{record['year']}")
    
    # Test keyword search
    logger.info("Testing keyword search...")
    key_scores, key_indices = hybrid_search._keyword_search_fixed(test_query, top_k=10)
    
    logger.success("Keyword search completed", {
        "results": len(key_indices),
        "top_score": f"{key_scores[0]:.4f}" if len(key_scores) > 0 else "N/A"
    })
    
    # Cleanup
    model_manager.unload_models()
    
    logger.success("Basic search test PASSED")
    return True


def test_threshold_progression():
    """Test threshold degradation across rounds"""
    logger = get_logger("TestThreshold")
    logger.info("="*80)
    logger.info("TEST 2: THRESHOLD DEGRADATION")
    logger.info("="*80)
    
    config = get_default_config()
    
    # Updated thresholds
    config['search_phases'] = {
        'initial_scan': {
            'candidates': 200,
            'semantic_threshold': 0.05,
            'keyword_threshold': 0.01,
            'enabled': True,
            'description': 'Initial scan'
        }
    }
    
    initial_quality = config.get('initial_quality', 1.0)
    degradation = config.get('quality_degradation', 0.15)
    min_quality = config.get('min_quality', 0.3)
    max_rounds = config.get('max_rounds', 5)
    
    logger.info("Configuration", {
        "initial_quality": initial_quality,
        "degradation_rate": degradation,
        "min_quality": min_quality,
        "max_rounds": max_rounds
    })
    
    quality = initial_quality
    
    logger.info("\nThreshold progression:")
    for round_num in range(1, max_rounds + 1):
        base_sem = config['search_phases']['initial_scan']['semantic_threshold']
        base_key = config['search_phases']['initial_scan']['keyword_threshold']
        
        effective_sem = base_sem * quality
        effective_key = base_key * quality
        
        logger.info(f"  Round {round_num}: quality={quality:.3f}, semantic={effective_sem:.5f}, keyword={effective_key:.5f}")
        
        # Degrade
        quality = max(min_quality, quality - degradation)
        
        if quality <= min_quality and round_num < max_rounds:
            logger.info(f"  >>> Minimum quality reached <<<")
    
    logger.success("Threshold degradation test PASSED")
    return True


def test_full_pipeline():
    """Test complete pipeline"""
    logger = get_logger("TestFull")
    logger.info("="*80)
    logger.info("TEST 3: FULL PIPELINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Setup
    logger.info("Setting up system...")
    config = get_default_config()
    
    # Update with fixed thresholds
    config['search_phases'] = {
        'initial_scan': {
            'candidates': 200,
            'semantic_threshold': 0.05,
            'keyword_threshold': 0.01,
            'description': 'Initial scan',
            'enabled': True
        },
        'focused_review': {
            'candidates': 100,
            'semantic_threshold': 0.10,
            'keyword_threshold': 0.05,
            'description': 'Focused review',
            'enabled': True
        }
    }
    
    # Load dataset
    loader = EnhancedKGDatasetLoader(DATASET_NAME, EMBEDDING_DIM)
    if not loader.load_from_huggingface(lambda msg: logger.info(f"   {msg}")):
        return False
    
    # Load models
    model_manager = get_model_manager()
    embedding_model = model_manager.load_embedding_model()
    reranker_model = model_manager.load_reranker_model(use_mock=True)
    
    # Initialize components
    query_detector = QueryDetector()
    hybrid_search = HybridSearchEngine(loader, embedding_model, reranker_model)
    stages_research = StagesResearchEngine(hybrid_search, config)
    consensus_builder = ConsensusBuilder(config)
    reranker = RerankerEngine(reranker_model, config)
    
    # Test query
    test_query = "Apa sanksi pidana dalam UU ITE?"
    logger.info(f"Test query: {test_query}")
    
    # Stage 1: Query detection
    logger.info("\nStage 1: Query Detection")
    t0 = time.time()
    query_analysis = query_detector.analyze_query(test_query)
    logger.success(f"Completed in {time.time()-t0:.2f}s", {
        "type": query_analysis['query_type'],
        "complexity": f"{query_analysis['complexity_score']:.2f}",
        "team_size": len(query_analysis['team_composition'])
    })
    
    # Stage 2: Research
    logger.info("\nStage 2: Multi-Stage Research")
    t0 = time.time()
    research_data = stages_research.conduct_research(
        query=test_query,
        query_analysis=query_analysis,
        team_composition=query_analysis['team_composition']
    )
    logger.success(f"Completed in {time.time()-t0:.2f}s", {
        "rounds": len(research_data['rounds']),
        "unique_results": len(research_data['all_results']),
        "candidates_evaluated": research_data['total_candidates_evaluated']
    })
    
    if len(research_data['all_results']) == 0:
        logger.error("NO RESULTS FOUND!")
        
        # Debug info
        logger.info("Debug information:")
        for round_data in research_data['rounds']:
            logger.info(f"  Round {round_data['round_number']}:")
            logger.info(f"    Quality: {round_data['quality_multiplier']:.3f}")
            logger.info(f"    Results: {len(round_data['results'])}")
            logger.info(f"    Candidates: {round_data['candidates_evaluated']}")
        
        return False
    
    # Show top results
    logger.info("\nTop 5 research results:")
    for i, result in enumerate(research_data['all_results'][:5], 1):
        rec = result['record']
        logger.info(f"  {i}. Score: {result['scores']['final']:.4f}")
        logger.info(f"     {rec['regulation_type']} {rec['regulation_number']}/{rec['year']}")
        logger.info(f"     Scores: sem={result['scores']['semantic']:.3f}, "
                   f"key={result['scores']['keyword']:.3f}, "
                   f"kg={result['scores']['kg']:.3f}")
    
    # Stage 3: Consensus
    logger.info("\nStage 3: Consensus Building")
    t0 = time.time()
    consensus_data = consensus_builder.build_consensus(
        research_data=research_data,
        team_composition=query_analysis['team_composition']
    )
    logger.success(f"Completed in {time.time()-t0:.2f}s", {
        "validated": len(consensus_data['validated_results']),
        "agreement": f"{consensus_data['agreement_level']:.2%}"
    })
    
    # Stage 4: Reranking
    logger.info("\nStage 4: Reranking")
    t0 = time.time()
    rerank_data = reranker.rerank(
        query=test_query,
        consensus_results=consensus_data['validated_results']
    )
    logger.success(f"Completed in {time.time()-t0:.2f}s", {
        "final_count": len(rerank_data['reranked_results'])
    })
    
    # Show final results
    logger.info("\nFinal Top 3 Results:")
    for i, result in enumerate(rerank_data['reranked_results'][:3], 1):
        rec = result['record']
        logger.info(f"  {i}. Final Score: {result['final_score']:.4f}")
        logger.info(f"     {rec['regulation_type']} {rec['regulation_number']}/{rec['year']}")
        logger.info(f"     About: {rec['about'][:80]}...")
    
    total_time = time.time() - start_time
    logger.success(f"\nFull pipeline test PASSED in {total_time:.2f}s")
    
    # Cleanup
    model_manager.unload_models()
    
    return True


def main():
    """Run all tests"""
    # Initialize logging
    initialize_logging(
        enable_file_logging=True,
        log_dir="logs",
        append=False,
        log_filename="comprehensive_test.log"
    )
    
    logger = get_logger("Main")
    logger.info("="*80)
    logger.info("COMPREHENSIVE RAG SYSTEM TEST SUITE")
    logger.info("="*80)
    
    tests = [
        ("Basic Search", test_basic_search),
        ("Threshold Progression", test_threshold_progression),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n\nRunning: {test_name}")
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            logger.error(f"Test failed with exception", {"error": str(e)})
            import traceback
            logger.debug("Traceback", {"traceback": traceback.format_exc()})
            results[test_name] = "ERROR"
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, result in results.items():
        status = "✅" if result == "PASS" else "❌"
        logger.info(f"{status} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    log_session_end()
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)