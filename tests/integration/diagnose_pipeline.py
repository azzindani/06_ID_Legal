"""
Diagnostic Pipeline - Trace where results disappear
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.logger_utils import initialize_logging, get_logger
from config import DATASET_NAME, EMBEDDING_DIM, get_default_config
from loader.dataloader import EnhancedKGDatasetLoader
from core.model_manager import get_model_manager
from core.search.query_detection import QueryDetector
from core.search.hybrid_search import HybridSearchEngine
from core.search.stages_research import StagesResearchEngine
from core.search.consensus import ConsensusBuilder
from core.search.reranking import RerankerEngine

initialize_logging(enable_file_logging=True, log_filename="diagnostic.log")
logger = get_logger("Diagnostic")

print("\n" + "="*80)
print("DIAGNOSTIC: Tracing Result Loss")
print("="*80)

# Setup
print("\n1. Loading system...")
config = get_default_config()

# CRITICAL: Use very permissive thresholds
config['search_phases'] = {
    'initial_scan': {
        'candidates': 200,
        'semantic_threshold': 0.01,  # Very permissive
        'keyword_threshold': 0.001,   # Very permissive
        'description': 'Initial scan',
        'enabled': True,
        'time_limit': 30,
        'focus_areas': ['regulation_type']
    }
}

# CRITICAL: Lower consensus threshold
config['consensus_threshold'] = 0.3  # Allow results with 30% team agreement

loader = EnhancedKGDatasetLoader(DATASET_NAME, EMBEDDING_DIM)
if not loader.load_from_huggingface(lambda msg: print(f"   {msg}")):
    sys.exit(1)

model_manager = get_model_manager()
embedding_model = model_manager.load_embedding_model()
reranker_model = model_manager.load_reranker_model(use_mock=True)

query_detector = QueryDetector()
hybrid_search = HybridSearchEngine(loader, embedding_model, reranker_model)
stages_research = StagesResearchEngine(hybrid_search, config)
consensus_builder = ConsensusBuilder(config)
reranker = RerankerEngine(reranker_model, config)

# Test query
test_query = "Apa sanksi dalam UU ITE?"
print(f"\n2. Test Query: {test_query}")

# Stage 1: Query Detection
print("\n3. STAGE 1: Query Detection")
query_analysis = query_detector.analyze_query(test_query)
print(f"   Type: {query_analysis['query_type']}")
print(f"   Team: {query_analysis['team_composition']}")
print(f"   Complexity: {query_analysis['complexity_score']:.2f}")

# Stage 2: Research
print("\n4. STAGE 2: Multi-Stage Research")
research_data = stages_research.conduct_research(
    query=test_query,
    query_analysis=query_analysis,
    team_composition=query_analysis['team_composition']
)

print(f"   Rounds executed: {research_data.get('rounds_executed', 0)}")
print(f"   Total unique results: {len(research_data.get('all_results', []))}")
print(f"   Candidates evaluated: {research_data.get('total_candidates_evaluated', 0)}")

# CHECKPOINT 1: Results after research
if len(research_data.get('all_results', [])) == 0:
    print("\n   ❌ PROBLEM: No results from research!")
    print("\n   Debugging research rounds:")
    for i, round_data in enumerate(research_data.get('rounds', []), 1):
        print(f"\n   Round {i}:")
        print(f"     Quality: {round_data.get('quality_multiplier', 0):.3f}")
        print(f"     Results: {len(round_data.get('results', []))}")
        print(f"     Phase breakdown: {round_data.get('phase_breakdown', {})}")
        print(f"     Persona breakdown: {round_data.get('persona_breakdown', {})}")
    
    print("\n   SOLUTION: Thresholds are still too high or search is broken")
    sys.exit(1)
else:
    print(f"\n   ✓ Research found {len(research_data['all_results'])} results")
    
    # Show top 3
    print("\n   Top 3 research results:")
    for i, result in enumerate(research_data['all_results'][:3], 1):
        rec = result['record']
        print(f"     {i}. Score: {result['scores']['final']:.4f}")
        print(f"        {rec['regulation_type']} {rec['regulation_number']}/{rec['year']}")
        print(f"        Persona: {result['metadata'].get('persona', 'N/A')}")

# Stage 3: Consensus
print("\n5. STAGE 3: Consensus Building")
consensus_data = consensus_builder.build_consensus(
    research_data=research_data,
    team_composition=query_analysis['team_composition']
)

print(f"   Validated results: {len(consensus_data.get('validated_results', []))}")
print(f"   Agreement level: {consensus_data.get('agreement_level', 0):.2%}")
print(f"   Consensus threshold: {config['consensus_threshold']}")

# CHECKPOINT 2: Results after consensus
if len(consensus_data.get('validated_results', [])) == 0:
    print("\n   ❌ PROBLEM: No results after consensus!")
    print("\n   Analyzing why results were filtered:")
    
    # Group results by document
    from collections import defaultdict
    results_by_doc = defaultdict(list)
    for result in research_data['all_results']:
        global_id = result['record']['global_id']
        results_by_doc[global_id].append(result)
    
    print(f"\n   Total unique documents: {len(results_by_doc)}")
    print(f"   Team size: {len(query_analysis['team_composition'])}")
    print(f"   Required consensus: {config['consensus_threshold']:.0%}")
    
    # Check voting ratios
    print("\n   Voting analysis:")
    for global_id, doc_results in list(results_by_doc.items())[:5]:
        personas = set(r['metadata'].get('persona', 'unknown') for r in doc_results)
        voting_ratio = len(personas) / len(query_analysis['team_composition'])
        passed = "✓" if voting_ratio >= config['consensus_threshold'] else "✗"
        
        print(f"     {passed} Doc {global_id}: {len(personas)}/{len(query_analysis['team_composition'])} "
              f"personas ({voting_ratio:.0%})")
    
    print("\n   SOLUTION: Lower consensus_threshold or ensure all personas find results")
    sys.exit(1)
else:
    print(f"\n   ✓ Consensus validated {len(consensus_data['validated_results'])} results")
    
    # Show top 3
    print("\n   Top 3 consensus results:")
    for i, result in enumerate(consensus_data['validated_results'][:3], 1):
        rec = result['record']
        print(f"     {i}. Consensus Score: {result['consensus_score']:.4f}")
        print(f"        Voting Ratio: {result['voting_ratio']:.0%}")
        print(f"        {rec['regulation_type']} {rec['regulation_number']}/{rec['year']}")
        print(f"        Personas: {', '.join(result['personas_agreed'])}")

# Stage 4: Reranking
print("\n6. STAGE 4: Reranking")
rerank_data = reranker.rerank(
    query=test_query,
    consensus_results=consensus_data['validated_results']
)

print(f"   Final results: {len(rerank_data.get('reranked_results', []))}")

# CHECKPOINT 3: Final results
if len(rerank_data.get('reranked_results', [])) == 0:
    print("\n   ❌ PROBLEM: No results after reranking!")
    print("\n   This shouldn't happen if consensus had results")
    sys.exit(1)
else:
    print(f"\n   ✓ Reranking produced {len(rerank_data['reranked_results'])} final results")
    
    # Show all final results
    print("\n   Final Results:")
    for i, result in enumerate(rerank_data['reranked_results'], 1):
        rec = result['record']
        print(f"\n     {i}. Final Score: {result.get('final_score', 0):.4f}")
        print(f"        Rerank Score: {result.get('rerank_score', 0):.4f}")
        print(f"        Consensus Score: {result.get('consensus_score', 0):.4f}")
        print(f"        {rec['regulation_type']} {rec['regulation_number']}/{rec['year']}")
        print(f"        About: {rec['about'][:100]}...")

print("\n" + "="*80)
print("✓ DIAGNOSTIC COMPLETE - Pipeline working correctly!")
print("="*80)

model_manager.unload_models()