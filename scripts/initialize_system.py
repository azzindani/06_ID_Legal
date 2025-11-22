#!/usr/bin/env python3
"""
System Initialization Script

Downloads models, validates configuration, and prepares the system for use.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL,
    DATASET_NAME, EMBEDDING_DEVICE, LLM_DEVICE
)
from logger_utils import get_logger, initialize_logging

logger = get_logger("Initialize")


def check_environment():
    """Check environment variables and configuration"""
    logger.info("Checking environment configuration...")

    issues = []

    # Check for API keys if using API providers
    llm_provider = os.getenv("LLM_PROVIDER", "local")
    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY not set")
    elif llm_provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        issues.append("ANTHROPIC_API_KEY not set")
    elif llm_provider == "google" and not os.getenv("GOOGLE_API_KEY"):
        issues.append("GOOGLE_API_KEY not set")

    if issues:
        for issue in issues:
            logger.warning(issue)
        return False

    logger.info("Environment configuration OK")
    return True


def download_models():
    """Download required models"""
    logger.info("Downloading models...")

    try:
        # Try importing to trigger download
        from sentence_transformers import SentenceTransformer

        logger.info(f"Downloading embedding model: {EMBEDDING_MODEL}")
        SentenceTransformer(EMBEDDING_MODEL)

        logger.info(f"Downloading reranker model: {RERANKER_MODEL}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        AutoTokenizer.from_pretrained(RERANKER_MODEL)
        AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)

        logger.info("Models downloaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        return False


def load_dataset():
    """Load and validate dataset"""
    logger.info(f"Loading dataset: {DATASET_NAME}")

    try:
        from loader import EnhancedKGDatasetLoader

        # Default embedding dimension for sentence-transformers models
        embedding_dim = 384
        loader = EnhancedKGDatasetLoader(DATASET_NAME, embedding_dim)
        data = loader.load()

        logger.info(f"Dataset loaded: {len(data.get('documents', []))} documents")
        return True

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False


def initialize_pipeline():
    """Initialize RAG pipeline"""
    logger.info("Initializing RAG pipeline...")

    try:
        from pipeline import RAGPipeline

        pipeline = RAGPipeline()
        success = pipeline.initialize()

        if success:
            logger.info("Pipeline initialized successfully")
            pipeline.shutdown()
        else:
            logger.error("Pipeline initialization failed")

        return success

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return False


def main():
    """Main initialization function"""
    initialize_logging()
    logger.info("=" * 60)
    logger.info("Indonesian Legal RAG System - Initialization")
    logger.info("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Reranker Model: {RERANKER_MODEL}")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Embedding Device: {EMBEDDING_DEVICE}")
    print(f"  LLM Device: {LLM_DEVICE}")
    print()

    steps = [
        ("Environment Check", check_environment),
        ("Download Models", download_models),
        ("Load Dataset", load_dataset),
        ("Initialize Pipeline", initialize_pipeline),
    ]

    results = []
    for name, func in steps:
        print(f"[...] {name}")
        success = func()
        status = "OK" if success else "FAILED"
        results.append((name, success))
        print(f"[{status}] {name}\n")

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = all(r[1] for r in results)
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")

    print("=" * 60)
    if all_passed:
        print("\nSystem initialization complete!")
        print("Run with: python ui/gradio_app.py")
    else:
        print("\nSome steps failed. Check logs for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
