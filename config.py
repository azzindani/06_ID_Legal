"""
Configuration Module for KG-Enhanced Indonesian Legal RAG System
Enhanced with environment variable support, validation, and auto-detection
"""

import os
from typing import Dict, Any, List
import warnings
from pathlib import Path

# Make dotenv optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Lazy logger initialization to avoid circular import
_logger = None

def _get_logger():
    """Lazy logger initialization to avoid circular imports"""
    global _logger
    if _logger is None:
        from utils.logger_utils import get_logger
        _logger = get_logger("Config")
    return _logger

# =============================================================================
# AUTO-DETECTION CONFIGURATION
# =============================================================================

# Enable auto-detection of hardware for optimal configuration
AUTO_DETECT_HARDWARE = os.getenv("AUTO_DETECT_HARDWARE", "true").lower() == "true"

def _get_auto_config():
    """Get auto-detected hardware configuration (no logging to avoid circular import)"""
    if not AUTO_DETECT_HARDWARE:
        return {}

    try:
        from core.hardware_detection import detect_hardware
        config = detect_hardware()
        # Note: No logging here to avoid circular import during module initialization
        return {
            'embedding_device': config.embedding_device,
            'reranker_device': config.reranker_device,
            'llm_device': config.llm_device,
            'llm_load_in_4bit': config.llm_quantization == '4bit',
            'llm_load_in_8bit': config.llm_quantization == '8bit',
            'recommended_model': config.recommended_model,
        }
    except Exception:
        # Silent fail to avoid circular import during module initialization
        return {}

# Get auto-detected settings (empty if disabled or unavailable)
_auto_config = _get_auto_config()

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASET_NAME = os.getenv("DATASET_NAME", "Azzindani/ID_REG_DB_2510")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
LLM_MODEL = os.getenv("LLM_MODEL", _auto_config.get('recommended_model', "Azzindani/Deepseek_ID_Legal_Preview"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "32768"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# =============================================================================
# LOCAL MODEL CONFIGURATION
# =============================================================================

# Enable loading models from local directory instead of HuggingFace
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"

# Base directory for local models
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "./models")

# Individual local model paths (override base directory)
LOCAL_EMBEDDING_PATH = os.getenv("LOCAL_EMBEDDING_PATH", "")
LOCAL_RERANKER_PATH = os.getenv("LOCAL_RERANKER_PATH", "")
LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", "")

def get_model_path(model_type: str) -> str:
    """
    Get the model path based on configuration.

    Args:
        model_type: 'embedding', 'reranker', or 'llm'

    Returns:
        Local path if USE_LOCAL_MODELS is True, otherwise HuggingFace model name
    """
    if not USE_LOCAL_MODELS:
        if model_type == 'embedding':
            return EMBEDDING_MODEL
        elif model_type == 'reranker':
            return RERANKER_MODEL
        elif model_type == 'llm':
            return LLM_MODEL
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # Check for individual path override first
    if model_type == 'embedding':
        if LOCAL_EMBEDDING_PATH:
            return LOCAL_EMBEDDING_PATH
        return os.path.join(LOCAL_MODEL_DIR, "embedding")
    elif model_type == 'reranker':
        if LOCAL_RERANKER_PATH:
            return LOCAL_RERANKER_PATH
        return os.path.join(LOCAL_MODEL_DIR, "reranker")
    elif model_type == 'llm':
        if LOCAL_LLM_PATH:
            return LOCAL_LLM_PATH
        return os.path.join(LOCAL_MODEL_DIR, "llm")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# =============================================================================
# DEVICE & INFERENCE CONFIGURATION
# =============================================================================

# Device settings - auto-detected or manual override
DEVICE = os.getenv("DEVICE", "cuda")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", _auto_config.get('embedding_device', "cpu"))
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", _auto_config.get('reranker_device', "cpu"))
LLM_DEVICE = os.getenv("LLM_DEVICE", _auto_config.get('llm_device', "cuda"))

# Quantization settings - auto-detected or manual override
LLM_QUANTIZATION = os.getenv("LLM_QUANTIZATION", "4bit")  # none, 4bit, 8bit
_default_4bit = "true" if _auto_config.get('llm_load_in_4bit', True) else "false"
_default_8bit = "true" if _auto_config.get('llm_load_in_8bit', False) else "false"
LLM_LOAD_IN_4BIT = os.getenv("LLM_LOAD_IN_4BIT", _default_4bit).lower() == "true"
LLM_LOAD_IN_8BIT = os.getenv("LLM_LOAD_IN_8BIT", _default_8bit).lower() == "true"
EMBEDDING_DTYPE = os.getenv("EMBEDDING_DTYPE", "float32")  # float32, float16, bfloat16

# =============================================================================
# LLM PROVIDER CONFIGURATION
# =============================================================================

# Provider: local, openai, anthropic, google, openrouter
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")

# API Keys for cloud providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# API Model names (when using cloud providers)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")

# API Configuration
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))

# =============================================================================
# CONTEXT CACHE CONFIGURATION
# =============================================================================

# Efficient context management (inspired by Claude Code)
ENABLE_CONTEXT_CACHE = os.getenv("ENABLE_CONTEXT_CACHE", "true").lower() == "true"
CONTEXT_CACHE_SIZE = int(os.getenv("CONTEXT_CACHE_SIZE", "100"))  # Max cached contexts
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "8192"))  # Max tokens per context
CONTEXT_COMPRESSION = os.getenv("CONTEXT_COMPRESSION", "true").lower() == "true"
CONTEXT_SUMMARY_THRESHOLD = int(os.getenv("CONTEXT_SUMMARY_THRESHOLD", "4096"))

# =============================================================================
# ENHANCED MEMORY MANAGER CONFIGURATION
# =============================================================================

# Legal-optimized memory settings for conversational RAG
# These defaults are 3x more context-aware than standard chatbots
MEMORY_MAX_HISTORY_TURNS = int(os.getenv("MEMORY_MAX_HISTORY_TURNS", "100"))  # Total turns stored
MEMORY_MAX_CONTEXT_TURNS = int(os.getenv("MEMORY_MAX_CONTEXT_TURNS", "30"))    # Turns passed to LLM
MEMORY_MIN_CONTEXT_TURNS = int(os.getenv("MEMORY_MIN_CONTEXT_TURNS", "10"))    # Minimum context
MEMORY_MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "128000"))               # Max tokens for context

# Intelligent summarization settings
MEMORY_ENABLE_SUMMARIZATION = os.getenv("MEMORY_ENABLE_SUMMARIZATION", "true").lower() == "true"
MEMORY_SUMMARIZATION_THRESHOLD = int(os.getenv("MEMORY_SUMMARIZATION_THRESHOLD", "20"))

# Key facts extraction for legal consultations
MEMORY_ENABLE_KEY_FACTS = os.getenv("MEMORY_ENABLE_KEY_FACTS", "true").lower() == "true"

# LRU cache for conversation contexts
MEMORY_ENABLE_CACHE = os.getenv("MEMORY_ENABLE_CACHE", "true").lower() == "true"
MEMORY_CACHE_SIZE = int(os.getenv("MEMORY_CACHE_SIZE", "100"))

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
LOG_DIR = os.getenv("LOG_DIR", "logs")
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "15000"))

# =============================================================================
# LOGGING VERBOSITY CONFIGURATION
# =============================================================================

# Logging verbosity mode:
# - 'minimal': Only critical messages (ERROR, WARNING, SUCCESS, key INFO) - DEFAULT
# - 'normal': Standard logging (all INFO + above)
# - 'verbose': Full debug logging (all DEBUG + INFO + above)
LOG_VERBOSITY = os.getenv("LOG_VERBOSITY", "minimal")

# Determines which logs are printed to console (file always gets everything)
VERBOSE_CONSOLE_LOGGING = {
    'minimal': False,   # Only critical messages to console
    'normal': True,     # Standard logging to console
    'verbose': True     # All logs to console including DEBUG
}.get(LOG_VERBOSITY, False)

# Create necessary directories
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# DOCUMENT PARSER CONFIGURATION
# =============================================================================

# Document upload limits
DOCUMENT_MAX_FILES_PER_SESSION = int(os.getenv("DOCUMENT_MAX_FILES_PER_SESSION", "5"))
DOCUMENT_MAX_FILE_SIZE_MB = int(os.getenv("DOCUMENT_MAX_FILE_SIZE_MB", "5"))
DOCUMENT_MAX_CHARS_PER_FILE = int(os.getenv("DOCUMENT_MAX_CHARS_PER_FILE", "50000"))
DOCUMENT_MAX_CHARS_TOTAL = int(os.getenv("DOCUMENT_MAX_CHARS_TOTAL", "100000"))

# Storage settings
DOCUMENT_TTL_HOURS = int(os.getenv("DOCUMENT_TTL_HOURS", "24"))
DOCUMENT_TEMP_DIR = os.getenv("DOCUMENT_TEMP_DIR", "uploads/temp")

# OCR settings
DOCUMENT_OCR_PROVIDER = os.getenv("DOCUMENT_OCR_PROVIDER", "tesseract")  # tesseract, easyocr
DOCUMENT_OCR_LANGUAGES = os.getenv("DOCUMENT_OCR_LANGUAGES", "ind,eng").split(",")

# Supported formats (can be restricted via environment)
DOCUMENT_ALLOWED_EXTENSIONS = os.getenv(
    "DOCUMENT_ALLOWED_EXTENSIONS",
    ".pdf,.docx,.doc,.txt,.md,.html,.htm,.json,.csv,.xml,.rtf,.png,.jpg,.jpeg,.tiff,.bmp"
).split(",")

# Document parser configuration dict
DOCUMENT_PARSER_CONFIG = {
    'max_documents_per_session': DOCUMENT_MAX_FILES_PER_SESSION,
    'max_file_size_mb': DOCUMENT_MAX_FILE_SIZE_MB,
    'max_chars_per_document': DOCUMENT_MAX_CHARS_PER_FILE,
    'max_chars_total': DOCUMENT_MAX_CHARS_TOTAL,
    'document_ttl_hours': DOCUMENT_TTL_HOURS,
    'temp_upload_dir': DOCUMENT_TEMP_DIR,
    'ocr_provider': DOCUMENT_OCR_PROVIDER,
    'ocr_languages': DOCUMENT_OCR_LANGUAGES,
    'allowed_extensions': DOCUMENT_ALLOWED_EXTENSIONS
}

# =============================================================================
# URL EXTRACTION CONFIGURATION
# =============================================================================

# URL extraction settings
URL_EXTRACTION_ENABLED = os.getenv("URL_EXTRACTION_ENABLED", "true").lower() == "true"
URL_EXTRACTION_TIMEOUT = int(os.getenv("URL_EXTRACTION_TIMEOUT", "10"))
URL_EXTRACTION_MAX_SIZE_MB = int(os.getenv("URL_EXTRACTION_MAX_SIZE_MB", "5"))

# Optional domain whitelist (empty = allow all public URLs)
# Example: "go.id,kemenkeu.go.id,hukumonline.com"
URL_ALLOWED_DOMAINS = os.getenv("URL_ALLOWED_DOMAINS", "").split(",") if os.getenv("URL_ALLOWED_DOMAINS") else None

# URL extraction configuration dict
URL_EXTRACTION_CONFIG = {
    'enabled': URL_EXTRACTION_ENABLED,
    'timeout': URL_EXTRACTION_TIMEOUT,
    'max_size_bytes': URL_EXTRACTION_MAX_SIZE_MB * 1024 * 1024,
    'allowed_domains': URL_ALLOWED_DOMAINS,
    'user_agent': 'LegalRAG-Bot/1.0 (+https://github.com/azzindani/06_ID_Legal)'
}


# =============================================================================
# THINKING MODE CONFIGURATION
# =============================================================================

# Default thinking mode for legal analysis
# Options: 'low', 'medium', 'high'
DEFAULT_THINKING_MODE = os.getenv("DEFAULT_THINKING_MODE", "low")

# Thinking mode token budgets
THINKING_MODE_CONFIG = {
    'low': {
        'min_tokens': 2048,
        'max_tokens': 4096,
        'description': 'Basic analysis for straightforward queries'
    },
    'medium': {
        'min_tokens': 4096,
        'max_tokens': 8192,
        'description': 'Deep thinking for moderate complexity'
    },
    'high': {
        'min_tokens': 8192,
        'max_tokens': 16384,
        'description': 'Iterative & recursive thinking for complex analysis'
    }
}

# Enable thinking mode in pipeline
ENABLE_THINKING_PIPELINE = os.getenv("ENABLE_THINKING_PIPELINE", "true").lower() == "true"

# =============================================================================
# ITERATIVE EXPANSION CONFIGURATION (Phases 1-4)
# =============================================================================
#
# Detective-style document expansion beyond initial scoring
# Phase 1: Metadata expansion (same regulation context)
# Phase 2: KG & Citation expansion (entity networks, citation traversal)
# Phase 3: Semantic clustering (embedding space neighbors)
# Phase 4: Hybrid adaptive (query-type-specific strategy selection)
#

DEFAULT_EXPANSION_CONFIG = {
    # Master switch
    'enable_expansion': True,  # ✅ ENABLED by default for testing and production

    # Expansion limits
    'max_expansion_rounds': 2,        # Number of expansion iterations
    'max_pool_size': 10000,            # Stop if pool exceeds this
    'min_docs_per_round': 5,          # Stop if round adds fewer than this

    # Seed selection
    'seeds_per_round': 10,            # Top-K seeds for expansion per round
    'seed_score_threshold': 0.50,     # Only expand from high-scoring docs

    # Strategy 1: Metadata Expansion (Phase 1)
    'metadata_expansion': {
        'enabled': True,
        'max_docs_per_regulation': 50,  # Limit docs from same regulation
        'include_preamble': True,        # Include regulation preambles
        'include_attachments': True       # Include regulation attachments
    },

    # Strategy 2: KG Expansion (Phase 2) - Entity co-occurrence & citation following
    'kg_expansion': {
        'enabled': True,                 # ✅ ENABLED by default
        'max_entity_docs': 20,
        'entity_score_threshold': 0.3,
        'follow_citations': True,
        'citation_max_hops': 2
    },

    # Strategy 3: Citation Network Traversal (Phase 2) - Multi-hop citation expansion
    'citation_expansion': {
        'enabled': True,                 # ✅ ENABLED by default
        'max_hops': 2,
        'bidirectional': True
    },

    # Strategy 4: Semantic Clustering (Phase 3) - Embedding space neighbors
    'semantic_expansion': {
        'enabled': True,                 # ✅ ENABLED by default
        'cluster_radius': 0.15,          # Distance threshold for clustering
        'min_cluster_size': 3,           # Minimum docs in cluster
        'max_neighbors': 30,             # Max similar docs per seed
        'similarity_threshold': 0.70     # Cosine similarity threshold
    },

    # Strategy 5: Hybrid Adaptive (Phase 4) - Query-type-specific strategy selection
    'hybrid_expansion': {
        'enabled': True,                 # ✅ ENABLED by default
        'adaptive_strategy': True,       # Enable adaptive strategy selection
        'query_type_detection': True,    # Auto-detect query type
        'strategy_weights': {
            'metadata': 0.4,
            'kg': 0.3,
            'citation': 0.2,
            'semantic': 0.1
        }
    },

    # Strategy 6: Temporal Expansion (Phase 5) - Legal amendments/versions
    'temporal_expansion': {
        'enabled': True,                 # ✅ ENABLED - Critical for Indonesian law
        'max_years_range': 30,           # Look back 30 years for amendments
        'prioritize_recent': True,       # Newest versions rank higher
        'include_superseded': True       # Include old versions for context
    },

    # Strategy 7: Hierarchical Expansion (Phase 6) - Legal hierarchy (UU → PP → Perpres)
    'hierarchical_expansion': {
        'enabled': True,                 # ✅ ENABLED - Critical for legal hierarchy
        'expand_up': True,               # Find parent regulations (PP → UU)
        'expand_down': True,             # Find implementing regulations (UU → PP)
        'max_hierarchy_distance': 1,     # Max levels up/down (1=direct parent/child only)
        'max_docs_per_level': 15,        # Maximum documents per hierarchy level
        'year_range': 5,                 # Only include regulations within ±5 years
        'conservative_in_conversation': True  # Use stricter limits in conversational mode
    },

    # Strategy 8: Topical Expansion (Phase 7) - Legal domain/topic clustering
    'topical_expansion': {
        'enabled': True,                 # ✅ ENABLED - Important for legal topic clustering
        'max_docs_per_topic': 20,        # Limit docs from same legal domain
        'domain_threshold': 0.7          # Minimum domain confidence (high confidence only)
    },

    # Smart Filtering - Reduce noise after expansion
    'smart_filtering': {
        'enabled': True,                 # ✅ ENABLED - Filter expanded pool before reranking
        'semantic_threshold': 0.60,      # Min similarity to top-10 initial docs
        'max_pool_size': 500,            # Maximum docs after filtering (hard limit)
        'diversity_weight': 0.3,         # Balance between relevance (0.7) and diversity (0.3)
        'timeout_seconds': 60            # Max time for filtering (prevent hangs)
    },

    # Conversational Mode Detection - Conservative expansion for multi-turn conversations
    'conversational_mode': {
        'enabled': True,                 # ✅ ENABLED - Detect and adapt to conversations
        'conservative_expansion': True,  # Use stricter limits in conversations
        'max_expansion_rounds': 1,       # Reduce rounds in conversations (vs 2 for single)
        'max_pool_multiplier': 0.5       # Halve pool sizes in conversations
    }
}

# =============================================================================
# DEFAULT SYSTEM CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'final_top_k': int(os.getenv("FINAL_TOP_K", "3")),
    'max_rounds': int(os.getenv("MAX_ROUNDS", "5")),
    'initial_quality': float(os.getenv("INITIAL_QUALITY", "0.95")),
    'quality_degradation': float(os.getenv("QUALITY_DEGRADATION", "0.1")),
    'min_quality': float(os.getenv("MIN_QUALITY", "0.5")),
    'parallel_research': os.getenv("PARALLEL_RESEARCH", "true").lower() == "true",
    'research_team_size': int(os.getenv("RESEARCH_TEAM_SIZE", "4")),
    'temperature': float(os.getenv("TEMPERATURE", "0.7")),
    'max_new_tokens': int(os.getenv("MAX_NEW_TOKENS", "2048")),
    'top_p': float(os.getenv("TOP_P", "1.0")),
    'top_k': int(os.getenv("TOP_K", "20")),
    'min_p': float(os.getenv("MIN_P", "0.1")),
    'enable_cross_validation': os.getenv("ENABLE_CROSS_VALIDATION", "true").lower() == "true",
    'enable_devil_advocate': os.getenv("ENABLE_DEVIL_ADVOCATE", "true").lower() == "true",
    'consensus_threshold': float(os.getenv("CONSENSUS_THRESHOLD", "0.4")),
    'thinking_mode': DEFAULT_THINKING_MODE,
    'enable_thinking_pipeline': ENABLE_THINKING_PIPELINE,
    'batch_size': BATCH_SIZE,
    'cache_dir': CACHE_DIR,

    # Expansion configuration (can be overridden)
    'expansion_config': DEFAULT_EXPANSION_CONFIG.copy()
}

# =============================================================================
# SEARCH PHASES CONFIGURATION
# =============================================================================

DEFAULT_SEARCH_PHASES = {
    'initial_scan': {
        'candidates': 400,
        'semantic_threshold': 0.15,  # ↓ from 0.25 - more permissive to capture relevant docs
        'keyword_threshold': 0.05,   # ↓ from 0.10 - more permissive for keyword matching
        'description': 'Quick broad scan like human initial reading',
        'time_limit': 30,
        'focus_areas': ['regulation_type', 'enacting_body'],
        'enabled': True
    },
    'focused_review': {
        'candidates': 150,
        'semantic_threshold': 0.35,  # = (unchanged, already good)
        'keyword_threshold': 0.12,   # = (unchanged, already good)
        'description': 'Focused review of promising candidates',
        'time_limit': 45,
        'focus_areas': ['content', 'chapter', 'article'],
        'enabled': True
    },
    'deep_analysis': {
        'candidates': 60,
        'semantic_threshold': 0.45,  # = (unchanged, already strict)
        'keyword_threshold': 0.18,   # = (unchanged, already strict)
        'description': 'Deep contextual analysis like careful reading',
        'time_limit': 60,
        'focus_areas': ['kg_entities', 'cross_references'],
        'enabled': True
    },
    'verification': {
        'candidates': 30,
        'semantic_threshold': 0.55,  # = (unchanged, very strict)
        'keyword_threshold': 0.22,   # = (unchanged, very strict)
        'description': 'Final verification and cross-checking',
        'time_limit': 30,
        'focus_areas': ['authority_score', 'temporal_score'],
        'enabled': True
    },
    'expert_review': {
        'candidates': 45,
        'semantic_threshold': 0.50,  # = (unchanged, strict)
        'keyword_threshold': 0.20,   # = (unchanged, strict)
        'description': 'Expert specialist review for complex cases',
        'time_limit': 40,
        'focus_areas': ['legal_richness', 'completeness_score'],
        'enabled': False
    }
}

# =============================================================================
# RESEARCH TEAM PERSONAS
# =============================================================================

RESEARCH_TEAM_PERSONAS = {
    'senior_legal_researcher': {
        'name': '👨‍⚖️ Senior Legal Researcher',
        'experience_years': 15,
        'specialties': ['constitutional_law', 'procedural_law', 'precedent_analysis'],
        'approach': 'systematic_thorough',
        'strengths': ['authority_analysis', 'hierarchy_understanding', 'precedent_matching'],
        'weaknesses': ['modern_technology', 'informal_language'],
        'bias_towards': 'established_precedents',
        'search_style': {
            'semantic_weight': 0.25,
            'authority_weight': 0.35,
            'kg_weight': 0.25,
            'temporal_weight': 0.15
        },
        'phases_preference': ['verification', 'deep_analysis'],
        'speed_multiplier': 0.8,
        'accuracy_bonus': 0.15
    },
    'junior_legal_researcher': {
        'name': '👩‍⚖️ Junior Legal Researcher',
        'experience_years': 3,
        'specialties': ['research_methodology', 'digital_search', 'comprehensive_coverage'],
        'approach': 'broad_comprehensive',
        'strengths': ['semantic_search', 'keyword_matching', 'broad_coverage'],
        'weaknesses': ['authority_evaluation', 'precedent_weighting'],
        'bias_towards': 'comprehensive_results',
        'search_style': {
            'semantic_weight': 0.45,
            'authority_weight': 0.15,
            'kg_weight': 0.25,
            'temporal_weight': 0.15
        },
        'phases_preference': ['initial_scan', 'focused_review'],
        'speed_multiplier': 1.2,
        'accuracy_bonus': 0.0
    },
    'specialist_researcher': {
        'name': '📚 Knowledge Graph Specialist',
        'experience_years': 8,
        'specialties': ['knowledge_graphs', 'semantic_analysis', 'entity_relationships'],
        'approach': 'relationship_focused',
        'strengths': ['kg_analysis', 'entity_extraction', 'relationship_mapping'],
        'weaknesses': ['traditional_legal_hierarchy', 'formal_procedures'],
        'bias_towards': 'interconnected_concepts',
        'search_style': {
            'semantic_weight': 0.20,
            'authority_weight': 0.15,
            'kg_weight': 0.50,
            'temporal_weight': 0.15
        },
        'phases_preference': ['deep_analysis', 'expert_review'],
        'speed_multiplier': 0.9,
        'accuracy_bonus': 0.1
    },
    'procedural_expert': {
        'name': '⚖️ Procedural Law Expert',
        'experience_years': 12,
        'specialties': ['procedural_law', 'administrative_law', 'process_analysis'],
        'approach': 'step_by_step_methodical',
        'strengths': ['procedure_analysis', 'step_identification', 'requirement_mapping'],
        'weaknesses': ['abstract_concepts', 'philosophical_law'],
        'bias_towards': 'clear_procedures',
        'search_style': {
            'semantic_weight': 0.30,
            'authority_weight': 0.25,
            'kg_weight': 0.30,
            'temporal_weight': 0.15
        },
        'phases_preference': ['focused_review', 'verification'],
        'speed_multiplier': 1.0,
        'accuracy_bonus': 0.08
    },
    'devils_advocate': {
        'name': "🔍 Devil's Advocate Reviewer",
        'experience_years': 10,
        'specialties': ['critical_analysis', 'alternative_interpretations', 'edge_cases'],
        'approach': 'critical_challenging',
        'strengths': ['weakness_identification', 'alternative_perspectives', 'critical_thinking'],
        'weaknesses': ['positive_reinforcement', 'consensus_building'],
        'bias_towards': 'challenging_assumptions',
        'search_style': {
            'semantic_weight': 0.35,
            'authority_weight': 0.20,
            'kg_weight': 0.30,
            'temporal_weight': 0.15
        },
        'phases_preference': ['verification', 'expert_review'],
        'speed_multiplier': 0.7,
        'accuracy_bonus': 0.12
    }
}

# =============================================================================
# QUERY TEAM COMPOSITIONS
# =============================================================================

QUERY_TEAM_COMPOSITIONS = {
    'specific_article': ['senior_legal_researcher', 'specialist_researcher', 'devils_advocate'],
    'procedural': ['procedural_expert', 'junior_legal_researcher', 'senior_legal_researcher'],
    'definitional': ['senior_legal_researcher', 'specialist_researcher', 'junior_legal_researcher'],
    'sanctions': ['senior_legal_researcher', 'procedural_expert', 'devils_advocate'],
    'general': ['senior_legal_researcher', 'junior_legal_researcher', 'specialist_researcher', 'procedural_expert']
}

# =============================================================================
# HUMAN PRIORITIES
# =============================================================================
#
# UPDATED 2025-12-19 (Iteration 2): Further increased relevance priority
#
# Iteration 1 (65% relevance) improved results but relevant docs still ranked #4-17
# Root cause: Score differences too small (0.727 vs 0.722 = 0.005 gap)
# Solution: Increase relevance to 80%, reduce metadata to 20%
#
# Previous: Relevance 65%, Metadata 35% → Cooperatives law ranked #1 for tax query
# New: Relevance 80%, Metadata 20% → Tax laws should dominate
#
DEFAULT_HUMAN_PRIORITIES = {
    # RELEVANCE SCORES (PRIMARY) - 80%
    'semantic_match': 0.50,       # ↑ from 0.40 (+25%) - embedding similarity is KING
    'keyword_precision': 0.30,    # ↑ from 0.25 (+20%) - exact term matching critical

    # METADATA SCORES (SECONDARY) - 20%
    'knowledge_graph': 0.10,      # ↓ from 0.15 (-33%) - tie-breaker only
    'authority_hierarchy': 0.05,  # ↓ from 0.10 (-50%) - minimal weight
    'temporal_relevance': 0.03,   # ↓ from 0.05 (-40%) - rarely decisive
    'legal_completeness': 0.02,   # ↓ from 0.05 (-60%) - rarely decisive
}

# =============================================================================
# QUERY PATTERNS
# =============================================================================

QUERY_PATTERNS = {
    'specific_article': {
        'indicators': ['pasal', 'ayat', 'huruf', 'angka', 'butir'],
        'priority_weights': {
            'semantic_match': 0.45,       # ↑ Relevance dominant (80% total)
            'keyword_precision': 0.35,    # ↑ Keywords critical for article search
            'knowledge_graph': 0.10,      # ↓ Entity matching helps minimally
            'authority_hierarchy': 0.07,  # ↓ Less weight for authority
            'temporal_relevance': 0.03    # ↓ Minimal
        }
    },
    'procedural': {
        'indicators': ['prosedur', 'tata cara', 'persyaratan', 'cara', 'langkah'],
        'priority_weights': {
            'semantic_match': 0.50,       # ↑ Relevance dominant (80% total)
            'keyword_precision': 0.30,    # ↑ Keywords important
            'knowledge_graph': 0.10,      # ↓ Procedure steps in KG
            'legal_completeness': 0.05,   # ↓ Want complete procedures
            'temporal_relevance': 0.03,   # ↓ Prefer recent
            'authority_hierarchy': 0.02   # ↓ Minimal
        }
    },
    'definitional': {
        'indicators': ['definisi', 'pengertian', 'dimaksud dengan', 'adalah'],
        'priority_weights': {
            'semantic_match': 0.50,       # ↑ Relevance dominant (80% total)
            'keyword_precision': 0.30,    # ↑ Exact term matching critical
            'authority_hierarchy': 0.10,  # ↓ Official definitions matter somewhat
            'knowledge_graph': 0.07,      # ↓ Concept relationships
            'temporal_relevance': 0.03    # ↓ Definitions rarely change
        }
    },
    'sanctions': {
        'indicators': ['sanksi', 'pidana', 'denda', 'hukuman', 'larangan'],
        'priority_weights': {
            'semantic_match': 0.50,       # ↑ Relevance dominant (80% total)
            'keyword_precision': 0.30,    # ↑ Sanction keywords critical
            'knowledge_graph': 0.10,      # ↓ Violation-sanction relationships
            'authority_hierarchy': 0.05,  # ↓ Official sources
            'temporal_relevance': 0.05    # ↓ Recent sanctions may differ
        }
    },
    'general': {
        'indicators': [],
        'priority_weights': DEFAULT_HUMAN_PRIORITIES
    }
}

# =============================================================================
# KG WEIGHTS
# =============================================================================

KG_WEIGHTS = {
    'direct_match': 1.0,
    'one_hop': 0.8,
    'two_hop': 0.6,
    'concept_cluster': 0.7,
    'hierarchy_boost': 0.5,
    'temporal_relevance': 0.4,
    'cross_reference': 0.6,
    'domain_match': 0.5,
    'legal_action_match': 0.7,
    'sanction_relevance': 0.8,
    'citation_impact': 0.4,
    'connectivity_boost': 0.3
}

# =============================================================================
# INDONESIAN STOPWORDS
# =============================================================================

INDONESIAN_STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan', 'adalah',
    'ini', 'itu', 'atau', 'jika', 'maka', 'akan', 'telah', 'dapat', 'harus', 'tidak',
    'ada', 'oleh', 'sebagai', 'karena', 'sehingga', 'bahwa', 'tentang', 'antara',
    'seperti', 'setelah', 'sebelum', 'sampai', 'hingga', 'namun', 'tetapi', 'juga'
}

# =============================================================================
# REGULATION PATTERNS
# =============================================================================

REGULATION_TYPE_PATTERNS = {
    'undang-undang': ['undang-undang', 'uu', 'undang undang'],
    'peraturan_pemerintah': ['peraturan pemerintah', 'pp', 'perpem'],
    'peraturan_presiden': ['peraturan presiden', 'perpres', 'pres'],
    'peraturan_menteri': ['peraturan menteri', 'permen', 'permenkeu', 'permendikbud'],
    'peraturan_daerah': ['peraturan daerah', 'perda', 'peraturan daerah provinsi', 'peraturan daerah kabupaten'],
    'keputusan_presiden': ['keputusan presiden', 'keppres', 'kepres'],
    'peraturan_gubernur': ['peraturan gubernur', 'pergub'],
    'peraturan_bupati': ['peraturan bupati', 'perbup'],
    'peraturan_walikota': ['peraturan walikota', 'perwali']
}

YEAR_SEPARATORS = ['tahun', 'th', 'th.', '/', '-']

REGULATION_PRONOUNS = [
    'peraturan tersebut', 'peraturan ini', 'pp tersebut', 'pp ini',
    'uu tersebut', 'uu ini', 'regulasi tersebut', 'regulasi ini',
    'ketentuan tersebut', 'ketentuan ini', 'undang-undang tersebut',
    'undang-undang ini', 'perda tersebut', 'perda ini'
]

FOLLOWUP_INDICATORS = [
    'apa yang diatur', 'mengatur apa', 'isi dari', 'membahas apa',
    'tentang apa', 'mengenai apa', 'berisi apa', 'materi apa',
    'ketentuan apa', 'pasal apa', 'bagaimana dengan', 'lalu bagaimana',
    'terus', 'kemudian', 'selanjutnya', 'dan', 'serta'
]

CLARIFICATION_INDICATORS = [
    'tidak melihat', 'tidak ada', 'tidak menemukan', 'bukan tentang',
    'seharusnya', 'maksud saya', 'yang saya maksud', 'saya kira',
    'tetapi', 'namun', 'tapi', 'kok', 'kenapa', 'mengapa'
]

CONTENT_QUERY_KEYWORDS = [
    'mengatur', 'diatur', 'pengaturan', 'ketentuan', 'isi', 'materi',
    'membahas', 'berisi', 'tentang', 'mengenai', 'menyangkut', 'terkait'
]

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = '''Anda adalah asisten AI yang ahli di bidang hukum Indonesia. Anda dapat membantu konsultasi hukum, menjawab pertanyaan, dan memberikan analisis berdasarkan peraturan perundang-undangan yang relevan.

Untuk setiap respons, Anda HARUS mengikuti format ini:

<think>
[Mode-specific thinking instructions are provided based on thinking mode]
</think>

[Setelah tag </think>, tuliskan jawaban akhir Anda secara jelas, ringkas, profesional, dan berempati jika diperlukan]

Pedoman untuk jawaban akhir:
- Gunakan bahasa hukum yang mudah dipahami
- Sertakan referensi hukum Indonesia yang relevan dengan format [Dokumen X]
- Berikan penjelasan yang terstruktur dan sistematis
- Selalu rekomendasikan konsultasi dengan ahli hukum untuk keputusan final
- Manfaatkan hubungan semantik antar konsep hukum untuk memberikan konteks yang lebih kaya'''

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration before use"""
    _get_logger().info("Starting configuration validation")
    
    issues = []
    warnings_list = []
    
    try:
        # Basic settings validation
        if config.get('final_top_k', 0) < 1:
            issues.append("final_top_k must be >= 1")
            _get_logger().error("Invalid final_top_k", {"value": config.get('final_top_k')})
        
        if config.get('temperature', 0) < 0 or config.get('temperature', 2) > 2:
            issues.append("temperature must be between 0 and 2")
            _get_logger().error("Invalid temperature", {"value": config.get('temperature')})
        
        if config.get('max_new_tokens', 0) < 128:
            issues.append("max_new_tokens must be >= 128")
            _get_logger().error("Invalid max_new_tokens", {"value": config.get('max_new_tokens')})
        
        # Team settings validation
        if config.get('research_team_size', 0) < 1 or config.get('research_team_size', 0) > 5:
            issues.append("research_team_size must be between 1 and 5")
            _get_logger().error("Invalid research_team_size", {"value": config.get('research_team_size')})
        
        if config.get('consensus_threshold', 0) < 0.3 or config.get('consensus_threshold', 0) > 0.9:
            warnings_list.append("consensus_threshold outside recommended range (0.3-0.9)")
            _get_logger().warning("Consensus threshold outside range", {"value": config.get('consensus_threshold')})
        
        # Search phases validation
        search_phases = config.get('search_phases', {})
        if not search_phases:
            issues.append("search_phases configuration missing")
            _get_logger().error("Search phases missing")
        else:
            enabled_phases = 0
            for phase_name, phase_config in search_phases.items():
                if phase_config.get('enabled', False):
                    enabled_phases += 1
                    
                    candidates = phase_config.get('candidates', 0)
                    if candidates < 10:
                        issues.append(f"{phase_name}: candidates must be >= 10")
                        _get_logger().error(f"Invalid candidates in {phase_name}", {"candidates": candidates})
                    elif candidates > 1000:
                        warnings_list.append(f"{phase_name}: high candidate count ({candidates}) may impact performance")
                        _get_logger().warning(f"High candidates in {phase_name}", {"candidates": candidates})
                    
                    sem_threshold = phase_config.get('semantic_threshold', 0)
                    if sem_threshold < 0.1 or sem_threshold > 0.9:
                        warnings_list.append(f"{phase_name}: semantic_threshold outside normal range (0.1-0.9)")
                        _get_logger().warning(f"Semantic threshold outside range in {phase_name}", {"threshold": sem_threshold})
                    
                    key_threshold = phase_config.get('keyword_threshold', 0)
                    if key_threshold < 0.02 or key_threshold > 0.5:
                        warnings_list.append(f"{phase_name}: keyword_threshold outside normal range (0.02-0.5)")
                        _get_logger().warning(f"Keyword threshold outside range in {phase_name}", {"threshold": key_threshold})
            
            if enabled_phases == 0:
                issues.append("At least one search phase must be enabled")
                _get_logger().error("No search phases enabled")
            else:
                _get_logger().info("Search phases validated", {"enabled_phases": enabled_phases})
        
        # LLM generation parameters validation
        if config.get('top_p', 1.0) < 0.1 or config.get('top_p', 1.0) > 1.0:
            issues.append("top_p must be between 0.1 and 1.0")
            _get_logger().error("Invalid top_p", {"value": config.get('top_p')})
        
        if config.get('top_k', 20) < 1 or config.get('top_k', 20) > 100:
            warnings_list.append("top_k outside recommended range (1-100)")
            _get_logger().warning("top_k outside range", {"value": config.get('top_k')})
        
        if config.get('min_p', 0.1) < 0.01 or config.get('min_p', 0.1) > 0.5:
            warnings_list.append("min_p outside recommended range (0.01-0.5)")
            _get_logger().warning("min_p outside range", {"value": config.get('min_p')})
        
        # Quality degradation parameters
        if config.get('initial_quality', 0.8) < 0.5 or config.get('initial_quality', 0.8) > 1.0:
            warnings_list.append("initial_quality outside recommended range (0.5-1.0)")
            _get_logger().warning("initial_quality outside range", {"value": config.get('initial_quality')})
        
        if config.get('quality_degradation', 0.15) < 0.05 or config.get('quality_degradation', 0.15) > 0.3:
            warnings_list.append("quality_degradation outside recommended range (0.05-0.3)")
            _get_logger().warning("quality_degradation outside range", {"value": config.get('quality_degradation')})
        
        if config.get('min_quality', 0.3) < 0.2 or config.get('min_quality', 0.3) > 0.5:
            warnings_list.append("min_quality outside recommended range (0.2-0.5)")
            _get_logger().warning("min_quality outside range", {"value": config.get('min_quality')})
        
        # Log final result
        if len(issues) == 0:
            _get_logger().success("Configuration validation passed", {
                "warnings": len(warnings_list)
            })
        else:
            _get_logger().error("Configuration validation failed", {
                "issues": len(issues),
                "warnings": len(warnings_list)
            })
        
    except Exception as e:
        issues.append(f"Configuration validation error: {str(e)}")
        _get_logger().error("Validation exception", {
            "error": str(e),
            "error_type": type(e).__name__
        })
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings_list
    }


def apply_validated_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply configuration after validation"""
    _get_logger().info("Applying validated configuration")
    
    validation_result = validate_config(config)
    
    if not validation_result['valid']:
        error_msg = "Configuration validation failed:\n"
        error_msg += "\n".join([f"X {issue}" for issue in validation_result['issues']])
        if validation_result['warnings']:
            error_msg += "\n\nWarnings:\n"
            error_msg += "\n".join([f"! {warning}" for warning in validation_result['warnings']])
        
        _get_logger().error("Config application failed due to validation errors")
        raise ValueError(error_msg)
    
    if validation_result['warnings']:
        for warning in validation_result['warnings']:
            warnings.warn(f"! {warning}")
            _get_logger().warning(warning)
    
    _get_logger().success("Configuration applied successfully")
    return config


def get_default_config() -> Dict[str, Any]:
    """Get a copy of the default configuration"""
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['search_phases'] = copy.deepcopy(DEFAULT_SEARCH_PHASES)
    return config


def save_config(config: Dict[str, Any], filepath: str = "config_runtime.json"):
    """Save configuration to JSON file"""
    import json
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        _get_logger().success(f"Configuration saved to {filepath}")
    except Exception as e:
        _get_logger().error(f"Failed to save configuration: {e}")


def load_config_from_file(filepath: str = "config_runtime.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    import json
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        _get_logger().success(f"Configuration loaded from {filepath}")
        return apply_validated_config(config)
    except Exception as e:
        _get_logger().error(f"Failed to load configuration: {e}")
        return get_default_config()

def print_threshold_progression():
    """
    Helper function to visualize threshold degradation across rounds
    """
    config = DEFAULT_CONFIG
    phases = DEFAULT_SEARCH_PHASES
    
    print("\n" + "="*80)
    print("THRESHOLD DEGRADATION VISUALIZATION")
    print("="*80)
    
    quality = config['initial_quality']
    
    for round_num in range(1, config['max_rounds'] + 1):
        print(f"\nRound {round_num}:")
        print(f"  Quality Multiplier: {quality:.3f}")
        print(f"  Effective Thresholds:")
        
        for phase_name, phase_config in phases.items():
            if not phase_config.get('enabled', True):
                continue
            
            effective_sem = phase_config['semantic_threshold'] * quality
            effective_key = phase_config['keyword_threshold'] * quality
            
            print(f"    {phase_name:20s}: semantic={effective_sem:.4f}, keyword={effective_key:.4f}")
        
        # Degrade for next round
        quality = max(config['min_quality'], quality - config['quality_degradation'])
        
        if quality <= config['min_quality']:
            print(f"\n  >>> Minimum quality reached ({config['min_quality']}) <<<")
            break
    
    print("\n" + "="*80)


def get_adaptive_thresholds(query_complexity: float) -> dict:
    """
    Get adaptive thresholds based on query complexity
    
    Args:
        query_complexity: 0-1 score from query analysis
        
    Returns:
        Adjusted phase configuration
    """
    phases = DEFAULT_SEARCH_PHASES.copy()
    
    # Lower thresholds for complex queries (they need more results)
    # Higher thresholds for simple queries (they can be more selective)
    complexity_factor = 1.0 - (query_complexity * 0.3)  # Max 30% reduction
    
    for phase_name in phases:
        if phase_name in phases:
            phases[phase_name] = phases[phase_name].copy()
            phases[phase_name]['semantic_threshold'] *= complexity_factor
            phases[phase_name]['keyword_threshold'] *= complexity_factor
    

    return phases

