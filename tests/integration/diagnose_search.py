"""
Quick diagnostic script to identify why search returns 0 results
"""

import sys
import os
# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.logger_utils import initialize_logging, get_logger
from config import DATASET_NAME, EMBEDDING_DIM, get_default_config
from loader.dataloader import EnhancedKGDatasetLoader
from core.model_manager import get_model_manager
import numpy as np

initialize_logging(enable_file_logging=True, log_filename="diagnose.log")
logger = get_logger("Diagnose")

logger.info("="*80)
logger.info("DIAGNOSTIC SCRIPT - SEARCH DEBUGGING")
logger.info("="*80)

# Load dataset
logger.info("Loading dataset...")
loader = EnhancedKGDatasetLoader(DATASET_NAME, EMBEDDING_DIM)

def progress(msg):
    logger.info(f"   {msg}")

if not loader.load_from_huggingface(progress):
    logger.error("Failed to load dataset")
    sys.exit(1)

stats = loader.get_statistics()
logger.info("Dataset loaded", {
    "records": f"{stats['total_records']:,}",
    "kg_enhanced": f"{stats['kg_enhanced']:,}",
    "avg_authority": f"{stats['avg_authority_score']:.3f}",
    "avg_temporal": f"{stats['avg_temporal_score']:.3f}"
})

# Load models
logger.info("Loading models...")
model_manager = get_model_manager()

try:
    embedding_model = model_manager.load_embedding_model()
    logger.success("Embedding model loaded")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    sys.exit(1)

# Test query embedding
logger.info("Testing query embedding...")
test_query = "Apa sanksi dalam UU ITE?"

try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        inputs = embedding_model.tokenize([test_query], padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info("Tokenization successful", {
            "input_ids_shape": str(inputs['input_ids'].shape),
            "attention_mask_shape": str(inputs['attention_mask'].shape)
        })
        
        outputs = embedding_model(**inputs)
        
        logger.info("Forward pass successful", {
            "last_hidden_state_shape": str(outputs.last_hidden_state.shape)
        })
        
        # Mean pooling
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        query_embedding = sum_embeddings / sum_mask
        
        logger.success("Query embedding extracted", {
            "shape": str(query_embedding.shape),
            "norm": f"{torch.norm(query_embedding).item():.4f}"
        })
        
except Exception as e:
    logger.error(f"Query embedding failed: {e}")
    import traceback
    logger.debug("Traceback", {"traceback": traceback.format_exc()})
    sys.exit(1)

# Test semantic search
logger.info("Testing semantic search...")

try:
    # Normalize embeddings
    query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    doc_embeddings_norm = torch.nn.functional.normalize(loader.embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    similarities = torch.mm(query_norm, doc_embeddings_norm.t()).squeeze(0)
    
    # Get top-10
    scores, indices = torch.topk(similarities, min(10, len(similarities)))
    
    logger.success("Semantic search successful", {
        "total_docs": len(similarities),
        "top_score": f"{scores[0].item():.4f}",
        "min_score": f"{scores[-1].item():.4f}"
    })
    
    # Show top results
    logger.info("Top 10 semantic results:")
    for i, (score, idx) in enumerate(zip(scores, indices), 1):
        record = loader.all_records[idx.item()]
        logger.info(f"   {i}. Score: {score.item():.4f} | {record['regulation_type']} {record['regulation_number']}/{record['year']}")
        logger.info(f"      About: {record['about'][:80]}...")
        
except Exception as e:
    logger.error(f"Semantic search failed: {e}")
    import traceback
    logger.debug("Traceback", {"traceback": traceback.format_exc()})
    sys.exit(1)

# Test with actual search config
logger.info("Testing with search phase config...")

from config import DEFAULT_SEARCH_PHASES

phase_config = DEFAULT_SEARCH_PHASES['initial_scan']
logger.info("Phase config", {
    "candidates": phase_config['candidates'],
    "semantic_threshold": phase_config['semantic_threshold'],
    "keyword_threshold": phase_config['keyword_threshold']
})

# Check how many would pass threshold
passed = (scores >= phase_config['semantic_threshold']).sum().item()
logger.info(f"Results passing semantic threshold: {passed}/{len(scores)}")

if passed == 0:
    logger.error("NO RESULTS PASS THRESHOLD!")
    logger.warning("Issue: Semantic threshold too high")
    logger.info("Suggestions:")
    logger.info(f"   1. Lower semantic_threshold from {phase_config['semantic_threshold']} to 0.1")
    logger.info(f"   2. Lower keyword_threshold from {phase_config['keyword_threshold']} to 0.01")
    logger.info(f"   3. Check if embeddings are correctly loaded")
else:
    logger.success(f"{passed} results pass threshold")

# Test TF-IDF
logger.info("Testing TF-IDF search...")

if loader.tfidf_matrix is not None:
    try:
        query_tfidf = loader.tfidf_vectorizer.transform([test_query])
        similarities_tfidf = (loader.tfidf_matrix * query_tfidf.T).toarray().flatten()
        
        top_indices_tfidf = np.argsort(similarities_tfidf)[-10:][::-1]
        top_scores_tfidf = similarities_tfidf[top_indices_tfidf]
        
        logger.success("TF-IDF search successful", {
            "top_score": f"{top_scores_tfidf[0]:.4f}",
            "min_score": f"{top_scores_tfidf[-1]:.4f}"
        })
        
        passed_tfidf = (top_scores_tfidf >= phase_config['keyword_threshold']).sum()
        logger.info(f"TF-IDF results passing threshold: {passed_tfidf}/10")
        
    except Exception as e:
        logger.error(f"TF-IDF search failed: {e}")
else:
    logger.warning("TF-IDF matrix not available")

logger.info("="*80)
logger.info("DIAGNOSIS COMPLETE")
logger.info("="*80)

logger.info("Summary:")
logger.info(f"   - Dataset: {stats['total_records']:,} records loaded")
logger.info(f"   - Embeddings: Working correctly")
logger.info(f"   - Semantic search: {passed} results pass threshold")
logger.info(f"   - Issue: {'Threshold too high' if passed == 0 else 'Search working'}")

if passed == 0:
    logger.error("CRITICAL: Thresholds are too high for current embeddings")
    logger.info("SOLUTION: Update config.py search phases with lower thresholds:")
    logger.info("""
    'initial_scan': {
        'semantic_threshold': 0.10,  # Changed from 0.20
        'keyword_threshold': 0.01,   # Changed from 0.06
        ...
    }
    """)