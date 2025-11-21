"""
Configuration Module for KG-Enhanced Indonesian Legal RAG System
Enhanced with environment variable support and better validation
"""

import os
from typing import Dict, Any, List
import warnings
from pathlib import Path
from dotenv import load_dotenv
from logger_utils import get_logger

# Load environment variables
load_dotenv()

# Initialize logger for config module
logger = get_logger("Config")

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
LLM_MODEL = os.getenv("LLM_MODEL", "Azzindani/Deepseek_ID_Legal_Preview")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "32768"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

DEVICE = os.getenv("DEVICE", "cuda")
LOG_DIR = os.getenv("LOG_DIR", "logs")
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "15000"))

# Create necessary directories
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

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
    'consensus_threshold': float(os.getenv("CONSENSUS_THRESHOLD", "0.6")),
    'batch_size': BATCH_SIZE,
    'cache_dir': CACHE_DIR
}

# =============================================================================
# SEARCH PHASES CONFIGURATION
# =============================================================================

DEFAULT_SEARCH_PHASES = {
    'initial_scan': {
        'candidates': 400,
        'semantic_threshold': 0.20,
        'keyword_threshold': 0.06,
        'description': 'Quick broad scan like human initial reading',
        'time_limit': 30,
        'focus_areas': ['regulation_type', 'enacting_body'],
        'enabled': True
    },
    'focused_review': {
        'candidates': 150,
        'semantic_threshold': 0.35,
        'keyword_threshold': 0.12,
        'description': 'Focused review of promising candidates',
        'time_limit': 45,
        'focus_areas': ['content', 'chapter', 'article'],
        'enabled': True
    },
    'deep_analysis': {
        'candidates': 60,
        'semantic_threshold': 0.45,
        'keyword_threshold': 0.18,
        'description': 'Deep contextual analysis like careful reading',
        'time_limit': 60,
        'focus_areas': ['kg_entities', 'cross_references'],
        'enabled': True
    },
    'verification': {
        'candidates': 30,
        'semantic_threshold': 0.55,
        'keyword_threshold': 0.22,
        'description': 'Final verification and cross-checking',
        'time_limit': 30,
        'focus_areas': ['authority_score', 'temporal_score'],
        'enabled': True
    },
    'expert_review': {
        'candidates': 45,
        'semantic_threshold': 0.50,
        'keyword_threshold': 0.20,
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

DEFAULT_HUMAN_PRIORITIES = {
    'authority_hierarchy': 0.20,
    'temporal_relevance': 0.18,
    'semantic_match': 0.18,
    'knowledge_graph': 0.15,
    'keyword_precision': 0.12,
    'legal_completeness': 0.09,
    'cross_validation': 0.08
}

# =============================================================================
# QUERY PATTERNS
# =============================================================================

QUERY_PATTERNS = {
    'specific_article': {
        'indicators': ['pasal', 'ayat', 'huruf', 'angka', 'butir'],
        'priority_weights': {
            'authority_hierarchy': 0.30,
            'semantic_match': 0.25,
            'knowledge_graph': 0.20,
            'keyword_precision': 0.15,
            'temporal_relevance': 0.10
        }
    },
    'procedural': {
        'indicators': ['prosedur', 'tata cara', 'persyaratan', 'cara', 'langkah'],
        'priority_weights': {
            'semantic_match': 0.25,
            'knowledge_graph': 0.20,
            'legal_completeness': 0.20,
            'temporal_relevance': 0.20,
            'authority_hierarchy': 0.15
        }
    },
    'definitional': {
        'indicators': ['definisi', 'pengertian', 'dimaksud dengan', 'adalah'],
        'priority_weights': {
            'authority_hierarchy': 0.35,
            'semantic_match': 0.25,
            'knowledge_graph': 0.15,
            'keyword_precision': 0.15,
            'temporal_relevance': 0.10
        }
    },
    'sanctions': {
        'indicators': ['sanksi', 'pidana', 'denda', 'hukuman', 'larangan'],
        'priority_weights': {
            'authority_hierarchy': 0.30,
            'knowledge_graph': 0.25,
            'keyword_precision': 0.20,
            'temporal_relevance': 0.15,
            'semantic_match': 0.10
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

SYSTEM_PROMPT = '''Anda adalah asisten AI yang ahli di bidang hukum Indonesia. Anda dapat membantu konsultasi hukum, menjawab pertanyaan, dan memberikan analisis berdasarkan peraturan perundang-undangan yang relevan. Untuk setiap respons, Anda harus berfikir dan menjawab dengan Bahasa Indonesia, serta gunakan format: <think> ... </think> Tuliskan jawaban akhir secara jelas, ringkas, profesional, dan berempati jika diperlukan. Gunakan bahasa hukum yang mudah dipahami. Sertakan referensi hukum Indonesia yang relevan. Selalu rekomendasikan konsultasi dengan ahli hukum untuk keputusan final. Manfaatkan hubungan semantik antar konsep hukum untuk memberikan konteks yang lebih kaya.'''

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration before use"""
    logger.info("Starting configuration validation")
    
    issues = []
    warnings_list = []
    
    try:
        # Basic settings validation
        if config.get('final_top_k', 0) < 1:
            issues.append("final_top_k must be >= 1")
            logger.error("Invalid final_top_k", {"value": config.get('final_top_k')})
        
        if config.get('temperature', 0) < 0 or config.get('temperature', 2) > 2:
            issues.append("temperature must be between 0 and 2")
            logger.error("Invalid temperature", {"value": config.get('temperature')})
        
        if config.get('max_new_tokens', 0) < 128:
            issues.append("max_new_tokens must be >= 128")
            logger.error("Invalid max_new_tokens", {"value": config.get('max_new_tokens')})
        
        # Team settings validation
        if config.get('research_team_size', 0) < 1 or config.get('research_team_size', 0) > 5:
            issues.append("research_team_size must be between 1 and 5")
            logger.error("Invalid research_team_size", {"value": config.get('research_team_size')})
        
        if config.get('consensus_threshold', 0) < 0.3 or config.get('consensus_threshold', 0) > 0.9:
            warnings_list.append("consensus_threshold outside recommended range (0.3-0.9)")
            logger.warning("Consensus threshold outside range", {"value": config.get('consensus_threshold')})
        
        # Search phases validation
        search_phases = config.get('search_phases', {})
        if not search_phases:
            issues.append("search_phases configuration missing")
            logger.error("Search phases missing")
        else:
            enabled_phases = 0
            for phase_name, phase_config in search_phases.items():
                if phase_config.get('enabled', False):
                    enabled_phases += 1
                    
                    candidates = phase_config.get('candidates', 0)
                    if candidates < 10:
                        issues.append(f"{phase_name}: candidates must be >= 10")
                        logger.error(f"Invalid candidates in {phase_name}", {"candidates": candidates})
                    elif candidates > 1000:
                        warnings_list.append(f"{phase_name}: high candidate count ({candidates}) may impact performance")
                        logger.warning(f"High candidates in {phase_name}", {"candidates": candidates})
                    
                    sem_threshold = phase_config.get('semantic_threshold', 0)
                    if sem_threshold < 0.1 or sem_threshold > 0.9:
                        warnings_list.append(f"{phase_name}: semantic_threshold outside normal range (0.1-0.9)")
                        logger.warning(f"Semantic threshold outside range in {phase_name}", {"threshold": sem_threshold})
                    
                    key_threshold = phase_config.get('keyword_threshold', 0)
                    if key_threshold < 0.02 or key_threshold > 0.5:
                        warnings_list.append(f"{phase_name}: keyword_threshold outside normal range (0.02-0.5)")
                        logger.warning(f"Keyword threshold outside range in {phase_name}", {"threshold": key_threshold})
            
            if enabled_phases == 0:
                issues.append("At least one search phase must be enabled")
                logger.error("No search phases enabled")
            else:
                logger.info("Search phases validated", {"enabled_phases": enabled_phases})
        
        # LLM generation parameters validation
        if config.get('top_p', 1.0) < 0.1 or config.get('top_p', 1.0) > 1.0:
            issues.append("top_p must be between 0.1 and 1.0")
            logger.error("Invalid top_p", {"value": config.get('top_p')})
        
        if config.get('top_k', 20) < 1 or config.get('top_k', 20) > 100:
            warnings_list.append("top_k outside recommended range (1-100)")
            logger.warning("top_k outside range", {"value": config.get('top_k')})
        
        if config.get('min_p', 0.1) < 0.01 or config.get('min_p', 0.1) > 0.5:
            warnings_list.append("min_p outside recommended range (0.01-0.5)")
            logger.warning("min_p outside range", {"value": config.get('min_p')})
        
        # Quality degradation parameters
        if config.get('initial_quality', 0.8) < 0.5 or config.get('initial_quality', 0.8) > 1.0:
            warnings_list.append("initial_quality outside recommended range (0.5-1.0)")
            logger.warning("initial_quality outside range", {"value": config.get('initial_quality')})
        
        if config.get('quality_degradation', 0.15) < 0.05 or config.get('quality_degradation', 0.15) > 0.3:
            warnings_list.append("quality_degradation outside recommended range (0.05-0.3)")
            logger.warning("quality_degradation outside range", {"value": config.get('quality_degradation')})
        
        if config.get('min_quality', 0.3) < 0.2 or config.get('min_quality', 0.3) > 0.5:
            warnings_list.append("min_quality outside recommended range (0.2-0.5)")
            logger.warning("min_quality outside range", {"value": config.get('min_quality')})
        
        # Log final result
        if len(issues) == 0:
            logger.success("Configuration validation passed", {
                "warnings": len(warnings_list)
            })
        else:
            logger.error("Configuration validation failed", {
                "issues": len(issues),
                "warnings": len(warnings_list)
            })
        
    except Exception as e:
        issues.append(f"Configuration validation error: {str(e)}")
        logger.error("Validation exception", {
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
    logger.info("Applying validated configuration")
    
    validation_result = validate_config(config)
    
    if not validation_result['valid']:
        error_msg = "Configuration validation failed:\n"
        error_msg += "\n".join([f"X {issue}" for issue in validation_result['issues']])
        if validation_result['warnings']:
            error_msg += "\n\nWarnings:\n"
            error_msg += "\n".join([f"! {warning}" for warning in validation_result['warnings']])
        
        logger.error("Config application failed due to validation errors")
        raise ValueError(error_msg)
    
    if validation_result['warnings']:
        for warning in validation_result['warnings']:
            warnings.warn(f"! {warning}")
            logger.warning(warning)
    
    logger.success("Configuration applied successfully")
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
        logger.success(f"Configuration saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")


def load_config_from_file(filepath: str = "config_runtime.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    import json
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.success(f"Configuration loaded from {filepath}")
        return apply_validated_config(config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
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