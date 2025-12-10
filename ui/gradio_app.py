"""
Gradio Interface - Indonesian Legal RAG System

Web-based chat interface for legal consultation with provider selection,
document upload, and advanced features.

This implementation replicates the Kaggle_Demo architecture with:
- Generator-based streaming progress
- Advanced query analysis display
- Research team settings
- Community cluster display
- Phase configuration
- TextIteratorStreamer for live token streaming
"""

import gradio as gr
import sys
import os
import time
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from threading import Thread

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RAGPipeline
from conversation import ConversationManager, MarkdownExporter, JSONExporter, HTMLExporter, get_context_cache
from providers import get_provider, switch_provider, list_providers
from config import (
    LLM_PROVIDER, EMBEDDING_DEVICE, LLM_DEVICE,
    DEFAULT_CONFIG, DEFAULT_SEARCH_PHASES, RESEARCH_TEAM_PERSONAS,
    EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL
)
from logger_utils import get_logger

# Import TextIteratorStreamer for live streaming
try:
    from transformers import TextIteratorStreamer
    HAS_STREAMER = True
except ImportError:
    HAS_STREAMER = False

logger = get_logger(__name__)

# Global instances
pipeline: Optional[RAGPipeline] = None
manager: Optional[ConversationManager] = None
current_session: Optional[str] = None
current_provider: str = LLM_PROVIDER
initialization_complete: bool = False

# Direct component references (populated during initialization)
search_engine = None
knowledge_graph = None
reranker = None
llm_generator = None
llm_model = None
llm_tokenizer = None
conversation_manager = None
dataset_loader = None


def initialize_system(provider_type: str = None):
    """Initialize the RAG system with specified provider"""
    global pipeline, manager, current_session, current_provider, initialization_complete
    global search_engine, knowledge_graph, reranker, llm_generator, llm_model, llm_tokenizer
    global conversation_manager, dataset_loader

    if provider_type:
        current_provider = provider_type

    if pipeline is None:
        logger.info(f"Initializing RAG system with provider: {current_provider}")
        pipeline = RAGPipeline({'llm_provider': current_provider})
        if not pipeline.initialize():
            return "Failed to initialize pipeline"
        logger.info("Pipeline initialized")

        # Extract component references for direct access
        if hasattr(pipeline, 'search_orchestrator'):
            search_engine = pipeline.search_orchestrator
        if hasattr(pipeline, 'knowledge_graph'):
            knowledge_graph = pipeline.knowledge_graph
        if hasattr(pipeline, 'reranker'):
            reranker = pipeline.reranker
        if hasattr(pipeline, 'generator'):
            llm_generator = pipeline.generator
        if hasattr(pipeline, 'llm_model'):
            llm_model = pipeline.llm_model
        if hasattr(pipeline, 'llm_tokenizer'):
            llm_tokenizer = pipeline.llm_tokenizer
        if hasattr(pipeline, 'data_loader'):
            dataset_loader = pipeline.data_loader

    if manager is None:
        manager = ConversationManager()
        conversation_manager = manager

    if current_session is None:
        current_session = manager.start_session()

    initialization_complete = True
    device_info = f"Embedding: {EMBEDDING_DEVICE}, LLM: {LLM_DEVICE}"
    return f"Initialized with {current_provider} provider. {device_info}"


def change_provider(provider_type: str):
    """Switch to a different LLM provider"""
    global pipeline, current_provider

    try:
        if pipeline:
            pipeline.shutdown()
            pipeline = None

        current_provider = provider_type
        return initialize_system(provider_type)
    except Exception as e:
        return f"Failed to switch provider: {e}"


def parse_think_tags(text: str) -> Tuple[str, str]:
    """Extract content from <think> tags and return (thinking, answer)"""
    import re
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL)
    thinking = '\n\n'.join(matches) if matches else ''
    answer = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
    return thinking, answer


def format_sources_info(results: List[Dict], config_dict: Dict) -> str:
    """Format LEGAL REFERENCES (Top K Documents Used in LLM Prompt) with FULL details"""
    if not results:
        return "Tidak ada sumber yang ditemukan."

    try:
        # No duplicate header - header is in collapsible summary
        output = [f"**Documents Used in Prompt: {len(results)}**"]
        output.append("")
        output.append("These are the final selected documents sent to the LLM for answer generation.")
        output.append("")

        for i, result in enumerate(results, 1):
            try:
                record = result.get('record', result)

                reg_type = record.get('regulation_type', 'N/A')
                reg_num = record.get('regulation_number', 'N/A')
                year = record.get('year', 'N/A')
                about = record.get('about', 'N/A')
                enacting_body = record.get('enacting_body', 'N/A')
                global_id = record.get('global_id', 'N/A')

                output.append(f"### {i}. {reg_type} No. {reg_num}/{year}")
                output.append(f"- **Global ID:** {global_id}")
                output.append(f"- **Tentang:** {about}")
                output.append(f"- **Ditetapkan oleh:** {enacting_body}")

                # Article/Chapter location
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', ''))
                section = record.get('section', record.get('bagian', ''))
                paragraph = record.get('paragraph', record.get('ayat', ''))

                location_parts = []
                if chapter:
                    location_parts.append(f"Bab {chapter}")
                if section:
                    location_parts.append(f"Bagian {section}")
                if article:
                    location_parts.append(f"Pasal {article}")
                if paragraph:
                    location_parts.append(f"Ayat {paragraph}")

                if location_parts:
                    output.append(f"- **Lokasi:** {' > '.join(location_parts)}")
                else:
                    output.append(f"- **Lokasi:** (Dokumen Lengkap)")

                # All scores
                scores_parts = []
                if 'final_score' in result:
                    scores_parts.append(f"Final: {result['final_score']:.4f}")
                if result.get('semantic_score', 0) > 0:
                    scores_parts.append(f"Semantic: {result.get('semantic_score', 0):.4f}")
                if result.get('keyword_score', 0) > 0:
                    scores_parts.append(f"Keyword: {result.get('keyword_score', 0):.4f}")
                if result.get('rerank_score', 0) > 0:
                    scores_parts.append(f"Rerank: {result['rerank_score']:.4f}")
                if result.get('composite_score', 0) > 0:
                    scores_parts.append(f"Search: {result['composite_score']:.4f}")
                if result.get('kg_score', 0) > 0:
                    scores_parts.append(f"KG: {result['kg_score']:.4f}")
                if result.get('authority_score', 0) > 0:
                    scores_parts.append(f"Authority: {result.get('authority_score', 0):.4f}")
                if result.get('temporal_score', 0) > 0:
                    scores_parts.append(f"Temporal: {result.get('temporal_score', 0):.4f}")
                if result.get('completeness_score', 0) > 0:
                    scores_parts.append(f"Completeness: {result.get('completeness_score', 0):.4f}")

                if scores_parts:
                    output.append(f"- **Skor:** {' | '.join(scores_parts)}")

                # KG metadata
                kg_parts = []
                if record.get('kg_primary_domain'):
                    kg_parts.append(f"Domain: {record['kg_primary_domain']}")
                if record.get('kg_hierarchy_level', 0) > 0:
                    kg_parts.append(f"Hierarchy: Level {record['kg_hierarchy_level']}")
                if record.get('kg_cross_ref_count', 0) > 0:
                    kg_parts.append(f"Cross-refs: {record['kg_cross_ref_count']}")
                if record.get('kg_pagerank', 0) > 0:
                    kg_parts.append(f"PageRank: {record['kg_pagerank']:.4f}")

                if kg_parts:
                    output.append(f"- **Knowledge Graph:** {' | '.join(kg_parts)}")

                # Team consensus info
                if result.get('team_consensus', False):
                    consensus_info = f"Team Consensus: Yes"
                    if 'researcher_agreement' in result:
                        consensus_info += f" (Agreement: {result['researcher_agreement']})"
                    if 'supporting_researchers' in result:
                        researchers = result['supporting_researchers']
                        researcher_names = [RESEARCH_TEAM_PERSONAS.get(r, {}).get('name', r) for r in researchers[:3]]
                        consensus_info += f" | Researchers: {', '.join(researcher_names)}"
                    output.append(f"- **{consensus_info}**")

                # Content preview (800 chars)
                content = record.get('content', '')
                if content:
                    content_preview = content[:800].replace('\n', ' ')
                    output.append(f"- **Isi (preview):** {content_preview}...")

                output.append("")

            except Exception as e:
                output.append(f"Error formatting source {i}: {e}")
                continue

        return "\n".join(output)
    except Exception as e:
        return f"Error formatting sources: {e}"


def _extract_all_documents_from_metadata(metadata: Dict) -> List[Dict]:
    """Extract all retrieved documents from various metadata locations - matching test pattern"""
    all_docs = []
    seen_ids = set()

    # Try phase_metadata first (most complete)
    phase_metadata = metadata.get('phase_metadata', {})
    for phase_name, phase_data in phase_metadata.items():
        if isinstance(phase_data, dict):
            candidates = phase_data.get('candidates', phase_data.get('results', []))
            for doc in candidates:
                doc_copy = dict(doc) if isinstance(doc, dict) else {'record': doc}
                record = doc_copy.get('record', doc_copy)
                doc_id = record.get('global_id', str(hash(str(doc))))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    doc_copy['_phase'] = phase_data.get('phase', phase_name)
                    doc_copy['_researcher'] = phase_data.get('researcher_name', phase_data.get('researcher', ''))
                    all_docs.append(doc_copy)

    # Try research_data
    if not all_docs:
        research_data = metadata.get('research_data', {})
        all_results = research_data.get('all_results', [])
        for doc in all_results:
            doc_copy = dict(doc) if isinstance(doc, dict) else {'record': doc}
            record = doc_copy.get('record', doc_copy)
            doc_id = record.get('global_id', str(hash(str(doc))))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc_copy)

    # Try research_log
    if not all_docs:
        research_log = metadata.get('research_log', {})
        phase_results = research_log.get('phase_results', {})
        for phase_name, phase_data in phase_results.items():
            if isinstance(phase_data, dict):
                candidates = phase_data.get('candidates', phase_data.get('results', []))
                for doc in candidates:
                    doc_copy = dict(doc) if isinstance(doc, dict) else {'record': doc}
                    record = doc_copy.get('record', doc_copy)
                    doc_id = record.get('global_id', str(hash(str(doc))))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        doc_copy['_phase'] = phase_name
                        all_docs.append(doc_copy)

    # Try consensus_data
    if not all_docs:
        consensus_data = metadata.get('consensus_data', {})
        final_results = consensus_data.get('final_results', [])
        for doc in final_results:
            doc_copy = dict(doc) if isinstance(doc, dict) else {'record': doc}
            record = doc_copy.get('record', doc_copy)
            doc_id = record.get('global_id', str(hash(str(doc))))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc_copy)

    # Try sources/citations at top level (fallback)
    if not all_docs:
        sources = metadata.get('sources', metadata.get('citations', []))
        for doc in sources:
            doc_id = doc.get('global_id', doc.get('regulation_number', str(hash(str(doc)))))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # Convert source format to document format
                all_docs.append({
                    'record': doc,
                    'scores': {
                        'final': doc.get('score', 0),
                        'semantic': doc.get('semantic_score', 0),
                        'keyword': doc.get('keyword_score', 0),
                        'kg': doc.get('kg_score', 0),
                        'authority': doc.get('authority_score', 0),
                        'temporal': doc.get('temporal_score', 0)
                    }
                })

    return all_docs


def format_all_documents(metadata: Dict, max_docs: int = 50) -> str:
    """Format ALL Retrieved Documents (Article-Level Details) - Top 50 using existing extraction pattern"""
    if not metadata:
        return ""

    try:
        # Use existing extraction function matching test_complete_output.py pattern
        all_docs = _extract_all_documents_from_metadata(metadata)

        if not all_docs:
            return "No documents retrieved during research process."

        # No duplicate header - header is in collapsible summary
        output = [f"**Showing {min(len(all_docs), max_docs)} documents with detailed metadata**"]
        output.append("")

        for i, doc in enumerate(all_docs[:max_docs], 1):
            record = doc.get('record', doc)
            scores = doc.get('scores', {})

            # Basic info
            reg_type = record.get('regulation_type', 'N/A')
            reg_num = record.get('regulation_number', 'N/A')
            year = record.get('year', 'N/A')
            about = record.get('about', 'N/A')
            enacting_body = record.get('enacting_body', 'N/A')
            global_id = record.get('global_id', 'N/A')

            # Scores
            final_score = scores.get('final', doc.get('final_score', doc.get('composite_score', record.get('score', 0))))
            semantic = scores.get('semantic', doc.get('semantic_score', 0))
            keyword = scores.get('keyword', doc.get('keyword_score', 0))
            kg = scores.get('kg', doc.get('kg_score', 0))
            authority = scores.get('authority', doc.get('authority_score', 0))
            temporal = scores.get('temporal', doc.get('temporal_score', 0))
            completeness = scores.get('completeness', doc.get('completeness_score', 0))

            # Article-level location
            chapter = record.get('chapter', record.get('bab', ''))
            article = record.get('article', record.get('pasal', ''))
            section = record.get('section', record.get('bagian', ''))
            paragraph = record.get('paragraph', record.get('ayat', ''))

            # KG metadata
            kg_domain = record.get('kg_primary_domain', record.get('primary_domain', ''))
            kg_hierarchy = record.get('kg_hierarchy_level', record.get('hierarchy_level', 0))
            kg_cross_refs = record.get('kg_cross_ref_count', record.get('cross_ref_count', 0))

            # Phase info
            phase = doc.get('_phase', '')
            researcher = doc.get('_researcher', '')

            output.append(f"**[{i}] {reg_type} No. {reg_num}/{year}**")
            output.append(f"- Global ID: {global_id}")
            output.append(f"- About: {about}")
            output.append(f"- Enacting Body: {enacting_body}")

            # Article-level location
            location_parts = []
            if chapter:
                location_parts.append(f"Bab {chapter}")
            if section:
                location_parts.append(f"Bagian {section}")
            if article:
                location_parts.append(f"Pasal {article}")
            if paragraph:
                location_parts.append(f"Ayat {paragraph}")

            if location_parts:
                output.append(f"- Location: {' > '.join(location_parts)}")
            else:
                output.append(f"- Location: (Full Document)")

            # All scores
            output.append(f"- Scores: Final={final_score:.4f} | Semantic={semantic:.4f} | Keyword={keyword:.4f}")
            output.append(f"- Scores: KG={kg:.4f} | Authority={authority:.4f} | Temporal={temporal:.4f} | Completeness={completeness:.4f}")

            # KG metadata
            if kg_domain or kg_hierarchy:
                output.append(f"- Knowledge Graph: Domain={kg_domain or 'N/A'} | Hierarchy={kg_hierarchy} | CrossRefs={kg_cross_refs}")

            # Research info
            if phase or researcher:
                output.append(f"- Discovery: Phase={phase} | Researcher={researcher}")

            # Content (truncated - 300 chars)
            content = record.get('content', '')
            if content:
                content_truncated = content[:300].replace('\n', ' ').strip()
                output.append(f"- Content (truncated): {content_truncated}...")

            output.append("")
            output.append("---")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"Error formatting all documents: {e}"


def format_retrieved_metadata(phase_metadata: Dict, config_dict: Dict) -> str:
    """Format all retrieved documents metadata with detailed research process info"""
    if not phase_metadata:
        return ""

    try:
        phase_order = ['initial_scan', 'focused_review', 'deep_analysis', 'verification', 'expert_review']

        phase_groups = {}
        total_kg_enhanced = 0
        unique_researchers = set()

        for phase_key, phase_data in phase_metadata.items():
            if not isinstance(phase_data, dict):
                continue

            phase_name = phase_data.get('phase', phase_key)
            researcher = phase_data.get('researcher', 'unknown')
            unique_researchers.add(researcher)

            if phase_name not in phase_groups:
                phase_groups[phase_name] = {}
            if researcher not in phase_groups[phase_name]:
                phase_groups[phase_name][researcher] = {'candidates': [], 'confidence': 100.0}

            candidates = phase_data.get('candidates', phase_data.get('results', []))
            kg_candidates = [c for c in candidates if c.get('kg_score', 0) > 0.3]
            total_kg_enhanced += len(kg_candidates)

            confidence = phase_data.get('confidence', 100.0)
            phase_groups[phase_name][researcher]['candidates'].extend(candidates)
            phase_groups[phase_name][researcher]['confidence'] = confidence

        # Calculate totals
        total_retrieved = 0
        for phase_name in phase_order:
            if phase_name not in phase_groups:
                continue
            for researcher_data in phase_groups[phase_name].values():
                total_retrieved += len(researcher_data['candidates'])

        # Build output - no duplicate header (header is in collapsible summary)
        output = [f"**Team Members:** {len(unique_researchers)} | **Total Documents:** {total_retrieved:,} | **Phases:** {len(phase_groups)}"]
        output.append("")

        # Detailed phase breakdown
        for phase_name in phase_order:
            if phase_name not in phase_groups:
                continue

            researchers = phase_groups[phase_name]
            output.append(f"#### 📂 PHASE: {phase_name.upper()}")
            output.append("")

            for researcher, researcher_data in researchers.items():
                candidates = researcher_data['candidates']
                confidence = researcher_data['confidence']

                # Get researcher display name with emoji
                if researcher in RESEARCH_TEAM_PERSONAS:
                    persona = RESEARCH_TEAM_PERSONAS[researcher]
                    researcher_name = persona.get('name', researcher)
                    emoji = persona.get('emoji', '👤')
                    full_name = f"**{emoji} {researcher_name}:**"
                else:
                    full_name = f"**👤 {researcher}:**"

                output.append(f"{full_name} {len(candidates)} documents (Confidence: {confidence:.2f}%)")

                # List top documents with scores
                for i, candidate in enumerate(candidates[:5], 1):
                    try:
                        record = candidate.get('record', candidate)
                        score = candidate.get('composite_score', candidate.get('score', 0))
                        kg_score = candidate.get('kg_score', 0)

                        reg_type = record.get('regulation_type', 'N/A')
                        reg_num = record.get('regulation_number', 'N/A')
                        year = record.get('year', 'N/A')

                        output.append(f"   {i}. {reg_type} No. {reg_num}/{year} (Score: {score:.3f}, KG: {kg_score:.3f})")
                    except Exception:
                        continue

                if len(candidates) > 5:
                    output.append(f"   ... and {len(candidates) - 5} more documents")
                output.append("")

        # Summary section - removed KG Enhancement Stats as per user request
        output.append(f"**Summary:** {total_retrieved:,} total documents retrieved across {len(phase_groups)} phases")

        return "\n".join(output)
    except Exception as e:
        return f"Error formatting metadata: {e}"


def system_health_check() -> Dict[str, Any]:
    """Run system health check and return results"""
    import psutil

    health = {
        'status': 'healthy',
        'components': {},
        'memory': {},
        'gpu': {},
        'issues': []
    }

    # Check initialization
    health['components']['pipeline'] = pipeline is not None
    health['components']['manager'] = manager is not None
    health['components']['initialization'] = initialization_complete

    # Memory check
    mem = psutil.virtual_memory()
    health['memory']['used_gb'] = mem.used / 1024**3
    health['memory']['total_gb'] = mem.total / 1024**3
    health['memory']['percent'] = mem.percent

    if mem.percent > 90:
        health['issues'].append("Critical: Memory usage above 90%")
        health['status'] = 'critical'
    elif mem.percent > 80:
        health['issues'].append("Warning: Memory usage above 80%")
        if health['status'] == 'healthy':
            health['status'] = 'warning'

    # GPU check
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            health['gpu'][f'gpu_{i}'] = {
                'used_gb': mem_used,
                'total_gb': mem_total,
                'percent': (mem_used / mem_total) * 100 if mem_total > 0 else 0
            }
    else:
        health['gpu']['available'] = False

    return health


def format_health_report(health: Dict[str, Any]) -> str:
    """Format health check results for display"""
    status_emoji = {
        'healthy': '✅',
        'warning': '⚠️',
        'critical': '❌'
    }

    report = f"## {status_emoji.get(health['status'], '❓')} System Health: {health['status'].upper()}\n\n"

    # Components
    report += "### Components\n"
    for comp, status in health['components'].items():
        emoji = '✅' if status else '❌'
        report += f"- {comp}: {emoji}\n"

    # Memory
    report += f"\n### Memory\n"
    report += f"- Used: {health['memory']['used_gb']:.1f} / {health['memory']['total_gb']:.1f} GB ({health['memory']['percent']:.1f}%)\n"

    # GPU
    if health['gpu']:
        report += f"\n### GPU\n"
        if 'available' in health['gpu'] and not health['gpu']['available']:
            report += "- No GPU available\n"
        else:
            for gpu_id, gpu_info in health['gpu'].items():
                if isinstance(gpu_info, dict):
                    report += f"- {gpu_id}: {gpu_info['used_gb']:.1f} / {gpu_info['total_gb']:.1f} GB ({gpu_info['percent']:.1f}%)\n"

    # Issues
    if health['issues']:
        report += f"\n### Issues\n"
        for issue in health['issues']:
            report += f"- {issue}\n"

    return report


def final_selection_with_kg(candidates: List[Dict], query_type: str, config_dict: Dict) -> List[Dict]:
    """Final selection of results with KG enhancement"""
    if not candidates:
        return []

    top_k = config_dict.get('final_top_k', 3)

    # Sort by final score
    sorted_candidates = sorted(
        candidates,
        key=lambda x: x.get('final_score', x.get('rerank_score', x.get('consensus_score', 0))),
        reverse=True
    )

    return sorted_candidates[:top_k]


# =============================================================================
# MAIN CHAT FUNCTION - Kaggle_Demo Style with FULL Streaming Progress
# =============================================================================

def chat_with_legal_rag(message, history, config_dict, show_thinking=True, show_sources=True, show_metadata=True):
    """Main chat function with ADVANCED QUERY ANALYSIS, HYBRID SEARCH, and LIVE STREAMING"""
    if not message.strip():
        return history, ""

    try:
        # Ensure system is initialized
        if pipeline is None:
            initialize_system()

        current_progress = []

        def add_progress(msg):
            """Helper to add progress updates in Gradio 6.x message format"""
            current_progress.append(msg)
            progress_display = "\n".join([f"🔄 {m}" for m in current_progress])
            # Gradio 6.x format: list of message dicts with 'role' and 'content'
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"**Mencari dan menganalisis...**\n\n{progress_display}"}
            ]

        yield add_progress("🚀 Memulai analisis query..."), ""

        # *** ADVANCED QUERY ANALYSIS DISPLAY - MATCHING ORIGINAL ***
        try:
            if hasattr(pipeline, 'search_orchestrator') and hasattr(pipeline.search_orchestrator, 'query_analyzer'):
                query_analysis = pipeline.search_orchestrator.query_analyzer.analyze_query(message)

                yield add_progress(f"🧠 Strategy: {query_analysis['search_strategy']} ({query_analysis['confidence']:.0%})"), ""

                if query_analysis.get('reasoning'):
                    yield add_progress(f"💡 {query_analysis['reasoning']}"), ""

                if query_analysis.get('key_phrases'):
                    phrases = [p['phrase'] for p in query_analysis['key_phrases']]
                    yield add_progress(f"🎯 Key phrases: {', '.join(phrases)}"), ""

                if query_analysis.get('law_name_detected'):
                    law_name = query_analysis['specific_entities'][0]['name']
                    yield add_progress(f"📜 Law name detected: {law_name}"), ""
            else:
                query_analysis = None
        except Exception as e:
            logger.debug(f"Query analysis display skipped: {e}")
            query_analysis = None

        # Get conversation context
        context = manager.get_context_for_query(current_session) if current_session else None

        # *** RESEARCHER PROGRESS - MATCHING ORIGINAL ***
        yield add_progress("🔍 Conducting intelligent search..."), ""

        # Show team assembly
        try:
            team_size = config_dict.get('research_team_size', 4)
            yield add_progress(f"👥 Assembling research team ({team_size} members)..."), ""
        except Exception:
            pass

        try:
            # Use REAL streaming for LLM output
            result = None
            all_phase_metadata = {}

            # Build progress header for display during streaming
            progress_header = f'<details open><summary>📋 <b>Proses Penelitian</b></summary>\n\n'
            progress_header += "\n".join([f"🔄 {m}" for m in current_progress])
            progress_header += '\n</details>\n\n'

            # Use streaming for local provider (pipeline handles streaming internally)
            use_real_streaming = (current_provider == 'local')
            logger.info(f"Streaming mode: {use_real_streaming} (provider: {current_provider})")

            if use_real_streaming:
                streamed_answer = ""
                chunk_count = 0
                result = None

                # Variables for <think> tag processing (like original)
                thinking_content = []
                final_answer = []
                live_output = []
                in_thinking_block = False
                saw_think_tag = False
                thinking_header_shown = False
                accumulated_text = ''

                try:
                    stream_response = pipeline.query(message, conversation_history=context, stream=True)

                    # Check if we got a generator or a dict (dict means no streaming/error)
                    if isinstance(stream_response, dict):
                        # Not a generator - use as result directly
                        result = stream_response
                        logger.info("Pipeline returned dict instead of stream")
                    else:
                        # Iterate over the stream (like original with TextIteratorStreamer)
                        for chunk in stream_response:
                            chunk_type = chunk.get('type', '')

                            if chunk_type == 'token':
                                new_text = chunk.get('token', '')
                                streamed_answer += new_text
                                accumulated_text += new_text
                                chunk_count += 1

                                # Process <think> tags DURING streaming (like original)
                                if '<think>' in new_text:
                                    in_thinking_block = True
                                    saw_think_tag = True
                                    new_text = new_text.replace('<think>', '')
                                    if not thinking_header_shown and show_thinking:
                                        live_output = [f"**Proses Penelitian:**\n\n{final_progress}\n\n---\n\n🧠 **Sedang berfikir...**\n"]
                                        thinking_header_shown = True

                                if '</think>' in new_text:
                                    in_thinking_block = False
                                    new_text = new_text.replace('</think>', '')
                                    if show_thinking:
                                        live_output.append('\n\n-----\n✅ **Jawaban:**\n')

                                # Append to appropriate section (like original)
                                if saw_think_tag:
                                    if in_thinking_block:
                                        thinking_content.append(new_text)
                                        if show_thinking:
                                            live_output.append(new_text)
                                    else:
                                        final_answer.append(new_text)
                                        live_output.append(new_text)
                                else:
                                    # No <think> tag seen yet
                                    if len(accumulated_text) > 20 and not saw_think_tag:
                                        if not thinking_header_shown:
                                            live_output = [f"**Proses Penelitian:**\n\n{final_progress}\n\n---\n\n⭐ **Jawaban langsung:**\n\n"]
                                            thinking_header_shown = True
                                        final_answer.append(new_text)
                                        live_output.append(new_text)
                                    else:
                                        if not thinking_header_shown:
                                            progress_with_generation = final_progress + f"\n\n🤖 Generating response..."
                                            live_output = [f"**Proses Penelitian:**\n\n{progress_with_generation}\n\n{new_text}"]
                                        else:
                                            live_output.append(new_text)

                                # Yield with live output (like original)
                                yield history + [
                                    {"role": "user", "content": message},
                                    {"role": "assistant", "content": ''.join(live_output)}
                                ], ""

                                # Log first few tokens for debugging
                                if chunk_count <= 3:
                                    logger.debug(f"Yielded token #{chunk_count}: {new_text[:20]}...")

                            elif chunk_type == 'complete':
                                # Extract response text from live answer (like original)
                                response_text = ''.join(final_answer).strip()

                                # Build final output with collapsible sections (like original)
                                final_output = f'<details><summary>📋 <b>Proses Penelitian Selesai (klik untuk melihat)</b></summary>\n\n{final_progress}\n</details>\n\n'

                                if thinking_content and show_thinking:
                                    thinking_text = ''.join(thinking_content).strip()
                                    final_output += (
                                        '<details><summary>🧠 <b>Proses berfikir (klik untuk melihat)</b></summary>\n\n'
                                        + thinking_text +
                                        '\n</details>\n\n'
                                        + '-----\n✅ **Jawaban:**\n'
                                        + response_text
                                    )
                                else:
                                    final_output += f"✅ **Jawaban:**\n{response_text}"

                                result = {
                                    'answer': chunk.get('answer', response_text),
                                    'sources': chunk.get('sources', []),
                                    'citations': chunk.get('citations', []),
                                    'metadata': chunk.get('metadata', {}),
                                    'phase_metadata': chunk.get('phase_metadata', {}),
                                    'thinking': ''.join(thinking_content).strip(),
                                    'consensus_data': chunk.get('consensus_data', {}),
                                    'research_data': chunk.get('research_data', {}),
                                    'communities': chunk.get('communities', [])
                                }
                                all_phase_metadata = result.get('phase_metadata', {})

                            elif chunk_type == 'error':
                                error_msg = chunk.get('error', 'Unknown error')
                                yield history + [
                                    {"role": "user", "content": message},
                                    {"role": "assistant", "content": progress_header + f"❌ Error: {error_msg}"}
                                ], ""
                                result = {'answer': '', 'sources': [], 'metadata': {}}
                                break

                        if result is None and streamed_answer:
                            result = {
                                'answer': streamed_answer,
                                'sources': [],
                                'metadata': {},
                                'phase_metadata': {}
                            }

                        logger.info(f"Streaming completed: {chunk_count} tokens")

                except Exception as stream_error:
                    logger.warning(f"Streaming failed, falling back to non-streaming: {stream_error}")
                    # Fall back to non-streaming
                    result = pipeline.query(message, conversation_history=context, stream=False)
                    all_phase_metadata = result.get('phase_metadata', result.get('all_retrieved_metadata', {}))

            else:
                # Non-streaming: show progress while waiting
                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": progress_header + "⏳ **Menghasilkan jawaban...**"}
                ], ""

                result = pipeline.query(message, conversation_history=context, stream=False)
                all_phase_metadata = result.get('phase_metadata', result.get('all_retrieved_metadata', {}))

        except Exception as e:
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"❌ Error: {str(e)}"}
            ], ""
            import traceback
            traceback.print_exc()
            result = {'answer': '', 'sources': [], 'metadata': {}}

        # Build final progress for collapsible section
        final_progress = "\n".join([msg for msg in current_progress])

        if result and result.get('answer'):
            try:
                # Use thinking and answer extracted during streaming (no need to re-parse)
                thinking_content = result.get('thinking', '')
                response_text = result.get('answer', '')

                # Build final output with collapsible sections (matching original format)
                final_output = f'<details><summary>📋 <b>Proses Penelitian Selesai (klik untuk melihat)</b></summary>\n\n{final_progress}\n</details>\n\n'

                if thinking_content and show_thinking:
                    final_output += (
                        '<details><summary>🧠 <b>Proses berfikir (klik untuk melihat)</b></summary>\n\n'
                        + thinking_content +
                        '\n</details>\n\n'
                        + '-----\n✅ **Jawaban:**\n'
                        + response_text
                    )
                else:
                    final_output += f"✅ **Jawaban:**\n{response_text}"

                # *** COMMUNITY CLUSTERS DISPLAY - MATCHING ORIGINAL ***
                if result.get('communities') or result.get('clusters'):
                    communities = result.get('communities', result.get('clusters', []))
                    if communities:
                        final_output += "\n\n---\n\n### 🌐 Discovered Thematic Clusters\n\n"
                        final_output += "_The research team identified these interconnected legal themes:_\n\n"

                        for cluster_idx, cluster_data in enumerate(communities[:5], 1):
                            if isinstance(cluster_data, dict):
                                name = cluster_data.get('name', f'Cluster {cluster_idx}')
                                size = cluster_data.get('size', 0)
                                theme = cluster_data.get('dominant_theme', cluster_data.get('theme', ''))

                                final_output += f"**{name}** ({size} documents)\n"
                                if theme:
                                    final_output += f"- **Theme:** {theme}\n"
                                if cluster_data.get('top_keywords'):
                                    keywords = cluster_data['top_keywords']
                                    if isinstance(keywords, list):
                                        keywords_str = ", ".join([f"`{kw}`" for kw in keywords[:8]])
                                        final_output += f"- **Key Terms:** {keywords_str}\n"
                                if cluster_data.get('primary_domain'):
                                    final_output += f"- **Domain:** {cluster_data['primary_domain']}\n"
                                final_output += "\n"

                # Add sources in collapsible section
                collapsible_sections = []

                if show_sources and result.get('citations'):
                    citations = result['citations']
                    if citations:
                        sources_info = format_sources_info(citations, config_dict)
                        collapsible_sections.append(
                            f'<details><summary>📖 <b>LEGAL REFERENCES {len(citations)} Articles</b></summary>\n\n{sources_info}\n</details>'
                        )

                # *** COMPLETE SEARCH METADATA - DETAILED RESEARCH PROCESS ***
                if show_metadata and all_phase_metadata:
                    metadata_info = format_retrieved_metadata(all_phase_metadata, config_dict)
                    if metadata_info.strip():
                        collapsible_sections.append(
                            f'<details><summary>🔍 <b>RESEARCH PROCESS DETAILS</b></summary>\n\n{metadata_info}\n</details>'
                        )

                    # Add ALL Retrieved Documents (Article-Level Details) - Top 50
                    # Pass full metadata including phase_metadata, research_data, consensus_data, sources
                    full_metadata_for_docs = {
                        'phase_metadata': all_phase_metadata,
                        'research_data': result.get('research_data', {}),
                        'research_log': result.get('metadata', {}).get('research_log', {}),
                        'consensus_data': result.get('consensus_data', {}),
                        'sources': result.get('sources', []),
                        'citations': result.get('citations', [])
                    }
                    all_docs_info = format_all_documents(full_metadata_for_docs, max_docs=50)
                    if all_docs_info.strip() and all_docs_info != "No documents retrieved during research process.":
                        collapsible_sections.append(
                            f'<details><summary>📄 <b>ALL Retrieved Documents (Article-Level Details) - TOP 50</b></summary>\n\n{all_docs_info}\n</details>'
                        )

                # Add research team info - more transparent
                if result.get('consensus_data') or result.get('research_data'):
                    consensus = result.get('consensus_data', {})
                    research = result.get('research_data', {})

                    team_content = "**Ringkasan Tim Peneliti:**\n\n"

                    # Team members info
                    team_members = research.get('team_members', [])
                    if team_members:
                        team_content += f"- **Anggota Tim ({len(team_members)}):**\n"
                        for member in team_members[:5]:
                            if isinstance(member, dict):
                                name = member.get('name', member.get('persona', 'Unknown'))
                                team_content += f"  - {name}\n"
                            elif member in RESEARCH_TEAM_PERSONAS:
                                persona = RESEARCH_TEAM_PERSONAS[member]
                                team_content += f"  - {persona.get('emoji', '👤')} {persona.get('name', member)}\n"
                            else:
                                team_content += f"  - {member}\n"

                    # Consensus details
                    if consensus.get('agreement_level'):
                        team_content += f"\n**Konsensus:**\n"
                        team_content += f"- Tingkat Kesepakatan: **{consensus['agreement_level']:.0%}**\n"
                    if consensus.get('final_results'):
                        team_content += f"- Dokumen Akhir Terpilih: {len(consensus['final_results'])}\n"
                    if consensus.get('voting_details'):
                        team_content += f"- Detail Voting: {consensus['voting_details']}\n"

                    # Research process details
                    if research.get('rounds_executed'):
                        team_content += f"\n**Proses Penelitian:**\n"
                        team_content += f"- Ronde Dieksekusi: {research['rounds_executed']}/5\n"
                    if research.get('total_candidates_evaluated'):
                        team_content += f"- Total Dokumen Dievaluasi: {research['total_candidates_evaluated']:,}\n"
                    if research.get('cross_validation_enabled'):
                        team_content += f"- Cross-Validation: Aktif\n"
                    if research.get('devil_advocate_enabled'):
                        team_content += f"- Devil's Advocate: Aktif\n"

                    if team_content.strip() != "**Ringkasan Tim Peneliti:**":
                        collapsible_sections.append(
                            f'<details><summary>👥 <b>Tim Peneliti & Konsensus</b></summary>\n\n{team_content}\n</details>'
                        )

                # Add query metadata - combined with query analysis
                meta = result.get('metadata', {})
                meta_content = ""

                # Add query analysis info inside Metadata Pencarian
                if query_analysis:
                    meta_content += "**Analisis Query:**\n"
                    if query_analysis.get('search_strategy'):
                        meta_content += f"- Strategi Pencarian: **{query_analysis['search_strategy']}**\n"
                    if query_analysis.get('confidence'):
                        meta_content += f"- Confidence: {query_analysis['confidence']:.0%}\n"
                    if query_analysis.get('reasoning'):
                        meta_content += f"- Reasoning: {query_analysis['reasoning']}\n"
                    if query_analysis.get('key_phrases'):
                        phrases = [p['phrase'] for p in query_analysis['key_phrases'][:5]]
                        meta_content += f"- Key Phrases: {', '.join(phrases)}\n"
                    if query_analysis.get('law_name_detected') and query_analysis.get('specific_entities'):
                        law_name = query_analysis['specific_entities'][0]['name']
                        meta_content += f"- Law Name Detected: {law_name}\n"
                    meta_content += "\n"

                # Add other metadata
                meta_content += "**Metadata Hasil:**\n"
                if meta.get('query_type'):
                    meta_content += f"- Tipe Query: {meta['query_type']}\n"
                if meta.get('processing_time'):
                    meta_content += f"- Waktu Proses: {meta['processing_time']:.2f}s\n"
                if meta.get('total_results'):
                    meta_content += f"- Total Hasil: {meta['total_results']}\n"
                if meta.get('strategy'):
                    meta_content += f"- Strategy: {meta['strategy']}\n"
                if meta.get('retrieval_time'):
                    meta_content += f"- Retrieval Time: {meta['retrieval_time']:.2f}s\n"
                if meta.get('generation_time'):
                    meta_content += f"- Generation Time: {meta['generation_time']:.2f}s\n"

                if meta_content.strip():
                    collapsible_sections.append(
                        f'<details><summary>📊 <b>Metadata Pencarian</b></summary>\n\n{meta_content}\n</details>'
                    )

                if collapsible_sections:
                    final_output += f"\n\n---\n\n" + "\n\n".join(collapsible_sections)

                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": final_output}
                ], ""

                # *** SAVE FULL RESEARCH LOG - MATCHING ORIGINAL ***
                if current_session:
                    # Build comprehensive metadata for export
                    full_metadata = result.get('metadata', {})
                    full_metadata['research_log'] = {
                        'phase_results': all_phase_metadata,
                        'team_members': list(RESEARCH_TEAM_PERSONAS.keys())[:config_dict.get('research_team_size', 4)],
                        'total_documents_retrieved': sum(
                            len(phase.get('candidates', phase.get('results', [])))
                            for phase in all_phase_metadata.values()
                            if isinstance(phase, dict)
                        ) if all_phase_metadata else 0
                    }

                    manager.add_turn(
                        session_id=current_session,
                        query=message,
                        answer=final_output,
                        metadata=full_metadata
                    )

            except Exception as e:
                error_output = f'<details><summary>📋 <b>Proses Penelitian Selesai (klik untuk melihat)</b></summary>\n\n{final_progress}\n</details>\n\n'
                error_output += f"❌ **Error generating response:** {str(e)}\n\n"
                error_output += "Maaf, terjadi kesalahan saat membuat respons. Silakan coba lagi."

                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": error_output}
                ], ""

                import traceback
                traceback.print_exc()

        else:
            final_output = f'<details><summary>📋 <b>Proses Penelitian Selesai (klik untuk melihat)</b></summary>\n\n{final_progress}\n</details>\n\n'
            final_output += "❌ **Tidak ada hasil ditemukan**\n\n"
            final_output += "Maaf, tidak ditemukan dokumen hukum yang relevan dengan pertanyaan Anda. Silakan coba:\n"
            final_output += "- Menggunakan kata kunci yang berbeda\n"
            final_output += "- Memperjelas pertanyaan Anda\n"
            final_output += "- Menggunakan istilah hukum yang lebih spesifik"

            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": final_output}
            ], ""

    except Exception as e:
        error_msg = f"❌ **Terjadi kesalahan sistem:**\n\n{str(e)}\n\n"
        error_msg += "Silakan coba lagi atau hubungi administrator jika masalah berlanjut."
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ], ""

        import traceback
        traceback.print_exc()


def clear_conversation():
    """Clear conversation history"""
    global manager, current_session

    try:
        if manager:
            current_session = manager.start_session()
        return [], ""
    except Exception as e:
        print(f"Error clearing conversation: {e}")
        return [], ""


def get_system_info():
    """Get system information with DATASET STATISTICS - MATCHING ORIGINAL"""
    if not initialization_complete:
        return "Sistem belum selesai inisialisasi."

    try:
        # Get dataset statistics
        stats = {}
        if dataset_loader and hasattr(dataset_loader, 'get_statistics'):
            stats = dataset_loader.get_statistics()

        info = f"""## 📊 Enhanced KG Legal RAG System Information

**Enhanced Features:**
- **Realistic Research Team**: 5 distinct researcher personas with unique expertise
- **Query-Specific Assembly**: Optimal team selection based on query type
- **Multi-Stage Process**: Individual → Cross-validation → Devil's Advocate → Consensus
- **Advanced Customization**: Granular control over all search phases

**Research Team Personas:**
- **👨‍⚖️ Senior Legal Researcher**: 15 years exp, authority-focused
- **👩‍⚖️ Junior Legal Researcher**: 3 years exp, comprehensive coverage
- **📚 Knowledge Graph Specialist**: 8 years exp, relationship-focused
- **⚖️ Procedural Law Expert**: 12 years exp, methodical analysis
- **🔍 Devil's Advocate**: 10 years exp, critical challenges

**Models:**
- **Embedding:** {EMBEDDING_MODEL}
- **Reranker:** {RERANKER_MODEL}
- **LLM:** {LLM_MODEL}

**Device Configuration:**
- **Embedding Device:** {EMBEDDING_DEVICE}
- **LLM Device:** {LLM_DEVICE}
- **Provider:** {current_provider}
"""

        # Add dataset statistics if available - MATCHING ORIGINAL
        if stats:
            info += f"""
**Dataset Statistics:**
- **Total Documents:** {stats.get('total_records', 0):,}
- **KG-Enhanced:** {stats.get('kg_enhanced', 0):,} ({stats.get('kg_enhancement_rate', 0):.1%})
- **Avg Entities/Doc:** {stats.get('avg_entities_per_doc', 0):.1f}
- **Avg Authority Score:** {stats.get('avg_authority_score', 0):.3f}
- **Avg KG Connectivity:** {stats.get('avg_connectivity_score', stats.get('avg_kg_connectivity', 0)):.3f}

**Performance Metrics:**
- **Authority Tiers:** {stats.get('authority_tiers', 0)}
- **Temporal Tiers:** {stats.get('temporal_tiers', 0)}
- **KG Connectivity Tiers:** {stats.get('kg_connectivity_tiers', 0)}
- **Unique Domains:** {stats.get('unique_domains', 0)}
- **Memory Optimized:** {stats.get('memory_optimized', False)}
"""

        return info
    except Exception as e:
        return f"Error getting system info: {e}"


# =============================================================================
# ENHANCED GRADIO INTERFACE - Kaggle_Demo Style
# =============================================================================

def create_gradio_interface():
    """Create enhanced Gradio interface with full customization"""

    custom_css = """
    /* Base container - responsive to zoom */
    .gradio-container {
        max-width: 100%;
        width: 100%;
        margin: 0 auto;
        padding: 0;
        overflow-x: hidden;
    }

    /* Main chat area - scalable dimensions */
    .main-chat-area {
        width: 100%;
        max-width: 75em;
        margin: 0 auto;
        padding: 1.25em;
        box-sizing: border-box;
    }

    /* Chatbot container - responsive sizing */
    .chat-container {
        height: 75vh;
        min-height: 25em;
        max-height: none;
        width: 100%;
        overflow-y: auto;
        border: 0.0625em solid #e0e0e0;
        border-radius: 0.75em;
        background: white;
        box-sizing: border-box;
        resize: vertical;
    }

    /* Prevent width changes from content expansion */
    .chatbot {
        width: 100%;
        max-width: none;
        min-width: 0;
    }

    /* Chat messages - scalable overflow handling */
    .message-wrap {
        max-width: 100%;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    /* Center the chatbot placeholder */
    .chatbot .wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .chatbot .placeholder {
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        width: 100%;
    }

    .chatbot .empty {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        width: 100%;
        text-align: center;
        color: #666;
        font-size: 1em;
    }

    /* Input area styling */
    .input-row {
        margin-top: 0.9375em;
        width: 100%;
    }

    .input-row .form {
        width: 100%;
    }

    /* Settings panels - scalable */
    .settings-panel {
        background-color: #f8f9fa;
        padding: 1.25em;
        border-radius: 0.75em;
        margin-bottom: 0.9375em;
        box-shadow: 0 0.125em 0.25em rgba(0,0,0,0.1);
        width: 100%;
        box-sizing: border-box;
    }

    .status-panel {
        background-color: #e8f4fd;
        padding: 0.9375em;
        border-radius: 0.5em;
        border-left: 0.25em solid #2196F3;
        margin-bottom: 0.625em;
    }

    /* Responsive breakpoints */
    @media (max-width: 87.5em) {
        .main-chat-area {
            max-width: 95%;
            padding: 0.9375em;
        }
    }

    @media (max-width: 64em) {
        .chat-container {
            height: 70vh;
            min-height: 20em;
        }

        .main-chat-area {
            padding: 0.9375em;
        }
    }

    @media (max-width: 48em) {
        .chat-container {
            height: 65vh;
            min-height: 18em;
        }

        .main-chat-area {
            padding: 0.625em;
        }

        .settings-panel {
            padding: 0.9375em;
        }
    }

    @media (max-width: 30em) {
        .chat-container {
            height: 60vh;
            min-height: 15em;
        }

        .main-chat-area {
            padding: 0.5em;
        }

        .settings-panel {
            padding: 0.75em;
            margin-bottom: 0.625em;
        }
    }

    /* Prevent layout shifts from dynamic content */
    .block {
        min-width: 0;
    }

    /* Tab content - centered tabs */
    .tab-nav {
        margin-bottom: 1.25em;
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }

    /* Center the tab navigation */
    .tabs {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }

    /* Style the tab buttons - scalable */
    .tab-nav button {
        margin: 0 0.5em;
        padding: 0.75em 1.5em;
        border-radius: 0.5em;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    /* Center tab container */
    .tabitem {
        width: 100%;
        max-width: 75em;
        margin: 0 auto;
    }

    /* Examples styling */
    .examples {
        margin-top: 0.9375em;
    }

    /* Button styling */
    .clear-btn {
        margin-left: auto;
    }

    /* Ensure consistent column widths in settings */
    .settings-columns {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.25em;
        width: 100%;
    }

    @media (max-width: 48em) {
        .settings-columns {
            grid-template-columns: 1fr;
        }
    }

    /* Fix for expandable content not affecting layout */
    .prose {
        max-width: 100%;
    }

    /* Prevent horizontal scroll */
    * {
        box-sizing: border-box;
    }

    /* Enhanced zoom support */
    html {
        -webkit-text-size-adjust: 100%;
        -ms-text-size-adjust: 100%;
    }

    /* Ensure text scales properly with browser zoom */
    body, .gradio-container, .chatbot {
        font-size: 1em;
    }
    """

    # Note: Gradio 6.x removed theme and css parameters from gr.Blocks()
    # Custom styling would need to be applied differently in Gradio 6.x
    with gr.Blocks(
        title="Enhanced Indonesian Legal Assistant"
    ) as interface:

        with gr.Tabs():
            # Main Chat Tab
            with gr.TabItem("💬 Konsultasi Hukum", id="chat"):
                with gr.Column(elem_classes="main-chat-area"):
                    # Note: Gradio 6.x removed many Chatbot parameters
                    # Keeping only essential ones that work in 6.x
                    chatbot = gr.Chatbot(
                        height="75vh",
                        show_label=False,
                        autoscroll=True
                    )

                    with gr.Row(elem_classes="input-row"):
                        msg_input = gr.Textbox(
                            placeholder="Tanyakan tentang hukum Indonesia...",
                            show_label=False,
                            container=False,
                            scale=10,
                            submit_btn=True,
                            lines=1,
                            max_lines=3,
                            interactive=True
                        )

                    with gr.Row():
                        with gr.Column():
                            gr.Examples(
                                examples=[
                                    "Apakah ada pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?",
                                    "Apakah terdapat mekanisme pengawasan terhadap penyimpanan uang negara agar terhindar dari penyalahgunaan atau kebocoran keuangan?",
                                    "Bagaimana mekanisme hukum untuk memperoleh izin resmi bagi pihak yang menjalankan usaha sebagai pengusaha pabrik, penyimpanan, importir, penyalur, maupun penjual eceran barang kena cukai?",
                                    "Apakah terdapat kewajiban pemerintah untuk menyediakan dana khusus bagi penyuluhan, atau dapat melibatkan sumber pendanaan alternatif seperti swasta dan masyarakat?",
                                    "Bagaimana prosedur hukum yang harus ditempuh sebelum sanksi denda administrasi di bidang cukai dapat dikenakan kepada pelaku usaha?",
                                    "Bagaimana sistem perencanaan kas disusun agar mampu mengantisipasi kebutuhan mendesak negara/daerah tanpa mengganggu stabilitas fiskal?",
                                    "syarat dan prosedur perceraian menurut hukum Indonesia",
                                    "hak dan kewajiban pekerja dalam UU Ketenagakerjaan"
                                ],
                                inputs=msg_input,
                                examples_per_page=2,
                                label=""
                            )

            # Enhanced Settings Tab
            with gr.TabItem("⚙️ Pengaturan Sistem", id="settings"):
                with gr.Row():
                    with gr.Column():
                        # Basic Settings
                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 🎯 Basic Settings")
                            final_top_k = gr.Slider(1, 10, value=3, step=1, label="Final Top K Results")
                            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="LLM Temperature")
                            max_new_tokens = gr.Slider(512, 4096, value=2048, step=256, label="Max New Tokens")

                        # Research Team Settings
                        with gr.Group(elem_classes="settings-panel researcher-settings"):
                            gr.Markdown("#### 👥 Research Team Configuration")
                            research_team_size = gr.Slider(1, 5, value=4, step=1, label="Team Size")
                            enable_cross_validation = gr.Checkbox(label="Enable Cross-Validation", value=True)
                            enable_devil_advocate = gr.Checkbox(label="Enable Devil's Advocate", value=True)
                            consensus_threshold = gr.Slider(0.3, 0.9, value=0.6, step=0.05, label="Consensus Threshold")

                        # Display Settings
                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 💬 Display Settings")
                            show_thinking = gr.Checkbox(label="Show Thinking Process", value=True)
                            show_sources = gr.Checkbox(label="Show Legal Sources", value=True)
                            show_metadata = gr.Checkbox(label="Show All Retrieved Metadata", value=True)

                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 🧠 LLM Generation Settings")
                            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="Top P")
                            top_k = gr.Slider(1, 100, value=20, step=1, label="Top K")
                            min_p = gr.Slider(0.01, 0.3, value=0.1, step=0.01, label="Min P")

                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 📊 System Information")
                            system_info_btn = gr.Button("📈 View System Stats", variant="primary")
                            reset_defaults_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
                            system_info_output = gr.Markdown("")

                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 🏥 System Health")
                            health_check_btn = gr.Button("🔍 Run Health Check", variant="secondary")
                            health_report_output = gr.Markdown("")

                        # Connect health check button
                        health_check_btn.click(
                            lambda: format_health_report(system_health_check()),
                            outputs=health_report_output
                        )

                    with gr.Column():
                        # Enhanced Search Phase Configuration
                        with gr.Group(elem_classes="settings-panel phase-settings"):
                            gr.Markdown("#### 🔍 Search Phase Configuration")

                            gr.Markdown("**Initial Scan Phase**")
                            initial_scan_enabled = gr.Checkbox(label="Enable Initial Scan", value=True)
                            initial_scan_candidates = gr.Slider(100, 800, value=400, step=50, label="Candidates")
                            initial_scan_semantic = gr.Slider(0.1, 0.5, value=0.20, step=0.05, label="Semantic Threshold")
                            initial_scan_keyword = gr.Slider(0.02, 0.15, value=0.06, step=0.01, label="Keyword Threshold")

                            gr.Markdown("**Focused Review Phase**")
                            focused_review_enabled = gr.Checkbox(label="Enable Focused Review", value=True)
                            focused_review_candidates = gr.Slider(50, 300, value=150, step=25, label="Candidates")
                            focused_review_semantic = gr.Slider(0.2, 0.6, value=0.35, step=0.05, label="Semantic Threshold")
                            focused_review_keyword = gr.Slider(0.05, 0.2, value=0.12, step=0.01, label="Keyword Threshold")

                            gr.Markdown("**Deep Analysis Phase**")
                            deep_analysis_enabled = gr.Checkbox(label="Enable Deep Analysis", value=True)
                            deep_analysis_candidates = gr.Slider(20, 120, value=60, step=10, label="Candidates")
                            deep_analysis_semantic = gr.Slider(0.3, 0.7, value=0.45, step=0.05, label="Semantic Threshold")
                            deep_analysis_keyword = gr.Slider(0.1, 0.3, value=0.18, step=0.01, label="Keyword Threshold")

                            gr.Markdown("**Verification Phase**")
                            verification_enabled = gr.Checkbox(label="Enable Verification", value=True)
                            verification_candidates = gr.Slider(10, 60, value=30, step=5, label="Candidates")
                            verification_semantic = gr.Slider(0.4, 0.8, value=0.55, step=0.05, label="Semantic Threshold")
                            verification_keyword = gr.Slider(0.15, 0.35, value=0.22, step=0.01, label="Keyword Threshold")

                            gr.Markdown("**Expert Review Phase (Optional)**")
                            expert_review_enabled = gr.Checkbox(label="Enable Expert Review", value=True)
                            expert_review_candidates = gr.Slider(15, 80, value=45, step=5, label="Candidates")
                            expert_review_semantic = gr.Slider(0.35, 0.75, value=0.50, step=0.05, label="Semantic Threshold")
                            expert_review_keyword = gr.Slider(0.12, 0.3, value=0.20, step=0.01, label="Keyword Threshold")


            with gr.TabItem("📥 Export Conversation", id="export"):
                with gr.Column(elem_classes="main-chat-area"):
                    gr.Markdown("""
                    ## Export Your Conversation

                    Download your complete consultation history including:
                    - All questions and answers
                    - Research team process details
                    - Legal sources consulted
                    - Metadata and analysis
                    """)

                    with gr.Row():
                        export_format = gr.Radio(
                            choices=["Markdown", "JSON", "HTML"],
                            value="Markdown",
                            label="Export Format"
                        )

                    with gr.Row():
                        include_metadata_export = gr.Checkbox(
                            label="Include Technical Metadata",
                            value=True
                        )
                        include_research_process_export = gr.Checkbox(
                            label="Include Research Team Process",
                            value=True
                        )
                        include_full_content_export = gr.Checkbox(
                            label="Include Full Document Content (JSON only)",
                            value=False
                        )

                    with gr.Row():
                        export_button = gr.Button("📥 Generate Export", variant="primary", size="lg")

                    export_output = gr.Textbox(
                        label="Export Output",
                        lines=20,
                        max_lines=30
                    )

                    download_file = gr.File(
                        label="Download Export File",
                        visible=True
                    )

                    gr.Markdown("""
                    ### Export Format Guide

                    - **Markdown**: Human-readable format, great for reading and sharing
                    - **JSON**: Structured data, perfect for processing or archiving
                    - **HTML**: Styled webpage, best for printing or presentation
                    """)

            # Export function
            def export_conversation_handler(export_format, include_metadata, include_research, include_full):
                """Handle export button click"""
                try:
                    if not manager or not current_session:
                        return "No conversation to export.", None

                    session_data = manager.get_session(current_session)
                    if not session_data:
                        return "Session not found.", None

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    exporters = {
                        'Markdown': MarkdownExporter,
                        'JSON': JSONExporter,
                        'HTML': HTMLExporter
                    }

                    exporter_class = exporters.get(export_format, MarkdownExporter)
                    exporter = exporter_class()

                    content = exporter.export(session_data)

                    ext_map = {'Markdown': 'md', 'JSON': 'json', 'HTML': 'html'}
                    extension = ext_map.get(export_format, 'md')
                    filename = f"legal_consultation_{timestamp}.{extension}"

                    # Save to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{extension}', delete=False, encoding='utf-8') as f:
                        f.write(content)
                        temp_path = f.name

                    return content, temp_path

                except Exception as e:
                    return f"Export failed: {str(e)}", None

            # Connect export button
            export_button.click(
                export_conversation_handler,
                inputs=[export_format, include_metadata_export, include_research_process_export, include_full_content_export],
                outputs=[export_output, download_file]
            )

            # About Tab
            with gr.TabItem("ℹ️ About Enhanced System", id="about"):
                with gr.Column(elem_classes="main-chat-area"):
                    gr.Markdown("""
                    # 🏛️ Enhanced KG-Indonesian Legal RAG System

                    ## 🆕 Enhanced Features

                    ### 👥 Realistic Research Team Simulation
                    The system now features **5 distinct researcher personas** with unique characteristics:

                    - **👨‍⚖️ Senior Legal Researcher**: 15 years experience, authority-focused, systematic approach
                    - **👩‍⚖️ Junior Legal Researcher**: 3 years experience, broad comprehensive coverage
                    - **📚 Knowledge Graph Specialist**: 8 years experience, relationship and semantic focus
                    - **⚖️ Procedural Law Expert**: 12 years experience, methodical step-by-step analysis
                    - **🔍 Devil's Advocate**: 10 years experience, critical analysis and challenges

                    ### 🎯 Query-Specific Team Assembly
                    Teams are automatically assembled based on query type:
                    - **Specific Article**: Senior + Specialist + Devil's Advocate
                    - **Procedural**: Procedural Expert + Junior + Senior
                    - **Definitional**: Senior + Specialist + Junior
                    - **Sanctions**: Senior + Procedural Expert + Devil's Advocate
                    - **General**: All researchers (customizable team size)

                    ### 🔄 Multi-Stage Research Process
                    1. **Individual Research**: Each researcher conducts research based on their expertise
                    2. **Cross-Validation**: Team members validate each other's findings
                    3. **Devil's Advocate Review**: Critical challenges to prevent groupthink
                    4. **Consensus Building**: Weighted consensus based on experience and accuracy

                    ### ⚙️ Advanced Customization
                    - **Granular Phase Control**: Enable/disable and adjust each search phase individually
                    - **Team Configuration**: Control team size, cross-validation, devil's advocate
                    - **Consensus Thresholds**: Adjust agreement requirements for final results
                    - **Real-time Updates**: All settings apply immediately to the research process

                    ## 🔧 Configuration Guide

                    ### Recommended Settings
                    - **Team Size**: 3-4 for optimal balance between coverage and efficiency
                    - **Consensus Threshold**: 0.6 for balanced precision/recall
                    - **Cross-Validation**: Enable for complex queries requiring validation
                    - **Devil's Advocate**: Enable for critical decisions and sanctions queries

                    ### Search Phase Optimization
                    - **Initial Scan**: High candidate count, low thresholds for broad coverage
                    - **Focused Review**: Moderate filtering for promising candidates
                    - **Deep Analysis**: Strict thresholds for quality documents
                    - **Verification**: Highest standards for final validation
                    - **Expert Review**: Optional phase for complex specialized queries

                    ### Performance Tuning
                    - **Lower thresholds**: Increase recall but may reduce precision
                    - **Higher candidate counts**: More comprehensive but slower processing
                    - **Team size optimization**: Larger teams for complex queries, smaller for simple ones

                    ## 📊 Research Analytics

                    The enhanced system provides detailed insights into the research process:
                    - **Per-Researcher Metrics**: Success rates and specialization effectiveness
                    - **Phase Analysis**: Which phases contribute most to final results
                    - **Consensus Tracking**: Team agreement patterns and conflict resolution
                    - **Query Success Patterns**: Learning from successful query-answer pairs

                    ## 🚀 Technical Improvements

                    - **Memory Optimization**: Efficient handling of large legal document collections
                    - **Parallel Processing**: Multiple researchers work simultaneously
                    - **Smart Caching**: Researchers build on each other's work
                    - **Error Handling**: Robust fallback mechanisms for edge cases
                    - **Streaming Responses**: Real-time progress updates during research

                    **Note**: This enhanced system combines human-like legal research methodology with AI efficiency, providing transparency into the research process while maintaining high accuracy and comprehensive coverage.
                    """)

        # Hidden state for enhanced configuration
        config_state = gr.State(DEFAULT_CONFIG)

        def update_enhanced_config(*args):
            """Update configuration with all enhanced settings"""
            try:
                search_phases = {
                    'initial_scan': {
                        'candidates': int(args[5]),
                        'semantic_threshold': float(args[6]),
                        'keyword_threshold': float(args[7]),
                        'description': 'Quick broad scan like human initial reading',
                        'time_limit': 30,
                        'focus_areas': ['regulation_type', 'enacting_body'],
                        'enabled': bool(args[4])
                    },
                    'focused_review': {
                        'candidates': int(args[9]),
                        'semantic_threshold': float(args[10]),
                        'keyword_threshold': float(args[11]),
                        'description': 'Focused review of promising candidates',
                        'time_limit': 45,
                        'focus_areas': ['content', 'chapter', 'article'],
                        'enabled': bool(args[8])
                    },
                    'deep_analysis': {
                        'candidates': int(args[13]),
                        'semantic_threshold': float(args[14]),
                        'keyword_threshold': float(args[15]),
                        'description': 'Deep contextual analysis like careful reading',
                        'time_limit': 60,
                        'focus_areas': ['kg_entities', 'cross_references'],
                        'enabled': bool(args[12])
                    },
                    'verification': {
                        'candidates': int(args[17]),
                        'semantic_threshold': float(args[18]),
                        'keyword_threshold': float(args[19]),
                        'description': 'Final verification and cross-checking',
                        'time_limit': 30,
                        'focus_areas': ['authority_score', 'temporal_score'],
                        'enabled': bool(args[16])
                    },
                    'expert_review': {
                        'candidates': int(args[21]),
                        'semantic_threshold': float(args[22]),
                        'keyword_threshold': float(args[23]),
                        'description': 'Expert specialist review for complex cases',
                        'time_limit': 40,
                        'focus_areas': ['legal_richness', 'completeness_score'],
                        'enabled': bool(args[20])
                    }
                }

                new_config = {
                    'final_top_k': int(args[0]),
                    'temperature': float(args[1]),
                    'max_new_tokens': int(args[2]),
                    'research_team_size': int(args[3]),
                    'enable_cross_validation': bool(args[24]),
                    'enable_devil_advocate': bool(args[25]),
                    'consensus_threshold': float(args[26]),
                    'top_p': float(args[27]),
                    'top_k': int(args[28]),
                    'min_p': float(args[29]),
                    'search_phases': search_phases,
                    'max_rounds': 5,
                    'initial_quality': 0.8,
                    'quality_degradation': 0.15,
                    'min_quality': 0.3,
                    'parallel_research': True
                }

                return new_config

            except Exception as e:
                print(f"Error updating enhanced config: {e}")
                return DEFAULT_CONFIG

        def reset_to_enhanced_defaults():
            """Reset to enhanced default values"""
            try:
                return (
                    DEFAULT_CONFIG['final_top_k'],  # 0
                    DEFAULT_CONFIG['temperature'],  # 1
                    DEFAULT_CONFIG['max_new_tokens'],  # 2
                    DEFAULT_CONFIG['research_team_size'],  # 3
                    DEFAULT_SEARCH_PHASES['initial_scan']['enabled'],  # 4
                    DEFAULT_SEARCH_PHASES['initial_scan']['candidates'],  # 5
                    DEFAULT_SEARCH_PHASES['initial_scan']['semantic_threshold'],  # 6
                    DEFAULT_SEARCH_PHASES['initial_scan']['keyword_threshold'],  # 7
                    DEFAULT_SEARCH_PHASES['focused_review']['enabled'],  # 8
                    DEFAULT_SEARCH_PHASES['focused_review']['candidates'],  # 9
                    DEFAULT_SEARCH_PHASES['focused_review']['semantic_threshold'],  # 10
                    DEFAULT_SEARCH_PHASES['focused_review']['keyword_threshold'],  # 11
                    DEFAULT_SEARCH_PHASES['deep_analysis']['enabled'],  # 12
                    DEFAULT_SEARCH_PHASES['deep_analysis']['candidates'],  # 13
                    DEFAULT_SEARCH_PHASES['deep_analysis']['semantic_threshold'],  # 14
                    DEFAULT_SEARCH_PHASES['deep_analysis']['keyword_threshold'],  # 15
                    DEFAULT_SEARCH_PHASES['verification']['enabled'],  # 16
                    DEFAULT_SEARCH_PHASES['verification']['candidates'],  # 17
                    DEFAULT_SEARCH_PHASES['verification']['semantic_threshold'],  # 18
                    DEFAULT_SEARCH_PHASES['verification']['keyword_threshold'],  # 19
                    DEFAULT_SEARCH_PHASES['expert_review']['enabled'],  # 20
                    DEFAULT_SEARCH_PHASES['expert_review']['candidates'],  # 21
                    DEFAULT_SEARCH_PHASES['expert_review']['semantic_threshold'],  # 22
                    DEFAULT_SEARCH_PHASES['expert_review']['keyword_threshold'],  # 23
                    DEFAULT_CONFIG['enable_cross_validation'],  # 24
                    DEFAULT_CONFIG['enable_devil_advocate'],  # 25
                    DEFAULT_CONFIG['consensus_threshold'],  # 26
                    DEFAULT_CONFIG['top_p'],  # 27
                    DEFAULT_CONFIG['top_k'],  # 28
                    DEFAULT_CONFIG['min_p']   # 29
                )
            except Exception as e:
                print(f"Error resetting to defaults: {e}")
                return tuple([0.5] * 30)  # Fallback

        # All configuration inputs
        config_inputs = [
            final_top_k, temperature, max_new_tokens, research_team_size,  # 0-3
            initial_scan_enabled, initial_scan_candidates, initial_scan_semantic, initial_scan_keyword,  # 4-7
            focused_review_enabled, focused_review_candidates, focused_review_semantic, focused_review_keyword,  # 8-11
            deep_analysis_enabled, deep_analysis_candidates, deep_analysis_semantic, deep_analysis_keyword,  # 12-15
            verification_enabled, verification_candidates, verification_semantic, verification_keyword,  # 16-19
            expert_review_enabled, expert_review_candidates, expert_review_semantic, expert_review_keyword,  # 20-23
            enable_cross_validation, enable_devil_advocate, consensus_threshold,  # 24-26
            top_p, top_k, min_p  # 27-29
        ]

        # Connect all inputs to config update
        for input_component in config_inputs:
            try:
                input_component.change(
                    update_enhanced_config,
                    inputs=config_inputs,
                    outputs=config_state
                )
            except Exception as e:
                print(f"Error connecting config input: {e}")

        # Reset button
        try:
            reset_defaults_btn.click(
                reset_to_enhanced_defaults,
                outputs=config_inputs
            )
        except Exception as e:
            print(f"Error setting up reset button: {e}")

        # Chat functionality with streaming support
        # Note: Gradio 6.x handles streaming automatically for generator functions
        try:
            msg_input.submit(
                chat_with_legal_rag,
                inputs=[msg_input, chatbot, config_state, show_thinking, show_sources, show_metadata],
                outputs=[chatbot, msg_input]
            )
        except Exception as e:
            print(f"Error setting up chat: {e}")

        # System info
        try:
            system_info_btn.click(
                get_system_info,
                outputs=system_info_output
            )
        except Exception as e:
            print(f"Error setting up system info: {e}")

    # Enable queue for streaming support
    # Note: Gradio 6.x has different queue parameters
    interface.queue()

    return interface


def launch_app(share: bool = False, server_port: int = 7860):
    """Launch Gradio app with pre-initialization"""
    global pipeline, manager, current_session

    # Initialize system BEFORE launching UI
    logger.info("Pre-initializing system before UI launch...")

    result = initialize_system()
    logger.info(f"Initialization result: {result}")

    logger.info("System fully initialized, launching UI...")

    # Now create and launch demo
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    launch_app()

