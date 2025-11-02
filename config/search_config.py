# config/search_config.py
"""
Search configuration including phases, personas, and patterns.
Extracted from the original monolithic configuration.
"""

# Regulation type patterns for flexible matching
REGULATION_TYPE_PATTERNS = {
    'undang-undang': ['undang-undang', 'uu', 'undang undang'],
    'peraturan_pemerintah': ['peraturan pemerintah', 'pp', 'perpem'],
    'peraturan_presiden': ['peraturan presiden', 'perpres', 'pres'],
    'peraturan_menteri': ['peraturan menteri', 'permen', 'permenkeu', 'permendikbud'],
    'peraturan_daerah': ['peraturan daerah', 'perda', 'peraturan daerah provinsi'],
    'keputusan_presiden': ['keputusan presiden', 'keppres', 'kepres'],
    'peraturan_gubernur': ['peraturan gubernur', 'pergub'],
    'peraturan_bupati': ['peraturan bupati', 'perbup'],
    'peraturan_walikota': ['peraturan walikota', 'perwali']
}

# Research team personas with expertise profiles
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
        'name': '🔍 Devil\'s Advocate Reviewer',
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

# Query-specific team compositions
QUERY_TEAM_COMPOSITIONS = {
    'specific_article': ['senior_legal_researcher', 'specialist_researcher', 'devils_advocate'],
    'procedural': ['procedural_expert', 'junior_legal_researcher', 'senior_legal_researcher'],
    'definitional': ['senior_legal_researcher', 'specialist_researcher', 'junior_legal_researcher'],
    'sanctions': ['senior_legal_researcher', 'procedural_expert', 'devils_advocate'],
    'general': ['senior_legal_researcher', 'junior_legal_researcher', 'specialist_researcher', 'procedural_expert']
}

# Default search phases configuration
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

# Default RAG configuration
DEFAULT_RAG_CONFIG = {
    'final_top_k': 3,
    'max_rounds': 5,
    'initial_quality': 0.95,
    'quality_degradation': 0.1,
    'min_quality': 0.5,
    'parallel_research': True,
    'research_team_size': 4,
    'enable_cross_validation': True,
    'enable_devil_advocate': True,
    'consensus_threshold': 0.6,
    'search_phases': DEFAULT_SEARCH_PHASES
}