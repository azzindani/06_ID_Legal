"""
Utils Package - Data Processing and Export Utilities
"""

from .export_helpers import (
    format_complete_search_metadata,
    export_conversation_to_markdown,
    export_conversation_to_json,
    export_conversation_to_html
)

from .formatting import (
    format_sources_info,
    _extract_all_documents_from_metadata,
    format_all_documents,
    format_retrieved_metadata,
    final_selection_with_kg
)

from .research_transparency import (
    format_detailed_research_process,
    format_researcher_summary
)

from .text_utils import (
    parse_think_tags,
    truncate_text,
    clean_whitespace
)

from .health import (
    system_health_check,
    format_health_report
)

from .system_info import (
    format_system_info,
    get_dataset_statistics
)

__all__ = [
    # Export helpers
    'format_complete_search_metadata',
    'export_conversation_to_markdown',
    'export_conversation_to_json',
    'export_conversation_to_html',
    # Formatting functions
    'format_sources_info',
    '_extract_all_documents_from_metadata',
    'format_all_documents',
    'format_retrieved_metadata',
    'final_selection_with_kg',
    # Research transparency
    'format_detailed_research_process',
    'format_researcher_summary',
    # Text utilities
    'parse_think_tags',
    'truncate_text',
    'clean_whitespace',
    # Health monitoring
    'system_health_check',
    'format_health_report',
    # System information
    'format_system_info',
    'get_dataset_statistics'
]
