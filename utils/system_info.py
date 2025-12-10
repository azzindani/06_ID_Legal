"""
System Information Utilities

This module provides functions for gathering and formatting system information,
including model details, device configuration, and dataset statistics.

File: utils/system_info.py
"""

from typing import Dict, Any, Optional


def format_system_info(
    embedding_model: str,
    reranker_model: str,
    llm_model: str,
    embedding_device: str,
    llm_device: str,
    current_provider: str,
    dataset_stats: Optional[Dict[str, Any]] = None,
    initialization_complete: bool = True
) -> str:
    """
    Format system information for display

    Args:
        embedding_model: Name of embedding model
        reranker_model: Name of reranker model
        llm_model: Name of LLM model
        embedding_device: Device for embedding (e.g., "cuda:1")
        llm_device: Device for LLM (e.g., "cuda:0")
        current_provider: Current LLM provider (e.g., "local")
        dataset_stats: Optional dataset statistics dictionary
        initialization_complete: Whether system is initialized

    Returns:
        Formatted markdown string with system information
    """
    if not initialization_complete:
        return "Sistem belum selesai inisialisasi."

    info = f"""## 📊 Enhanced KG Legal RAG System Information

**Models:**
- **Embedding:** {embedding_model}
- **Reranker:** {reranker_model}
- **LLM:** {llm_model}

**Device Configuration:**
- **Embedding Device:** {embedding_device}
- **LLM Device:** {llm_device}
- **Provider:** {current_provider}
"""

    # Add dataset statistics if available
    if dataset_stats:
        info += f"""
**Dataset Statistics:**
- **Total Documents:** {dataset_stats.get('total_records', 0):,}
- **KG-Enhanced:** {dataset_stats.get('kg_enhanced', 0):,} ({dataset_stats.get('kg_enhancement_rate', 0):.1%})
- **Avg Entities/Doc:** {dataset_stats.get('avg_entities_per_doc', 0):.1f}
- **Avg Authority Score:** {dataset_stats.get('avg_authority_score', 0):.3f}
- **Avg KG Connectivity:** {dataset_stats.get('avg_connectivity_score', dataset_stats.get('avg_kg_connectivity', 0)):.3f}

**Performance Metrics:**
- **Authority Tiers:** {dataset_stats.get('authority_tiers', 0)}
- **Temporal Tiers:** {dataset_stats.get('temporal_tiers', 0)}
- **KG Connectivity Tiers:** {dataset_stats.get('kg_connectivity_tiers', 0)}
- **Unique Domains:** {dataset_stats.get('unique_domains', 0)}
- **Memory Optimized:** {dataset_stats.get('memory_optimized', False)}
"""

    return info


def get_dataset_statistics(dataset_loader: Any) -> Optional[Dict[str, Any]]:
    """
    Extract statistics from dataset loader

    Args:
        dataset_loader: Dataset loader instance

    Returns:
        Dictionary of dataset statistics, or None if unavailable
    """
    if not dataset_loader:
        return None

    try:
        if hasattr(dataset_loader, 'get_statistics'):
            return dataset_loader.get_statistics()
        return None
    except Exception as e:
        print(f"Error getting dataset statistics: {e}")
        return None
