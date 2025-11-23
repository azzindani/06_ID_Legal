"""
Gradio Interface - Indonesian Legal RAG System

Web-based chat interface for legal consultation with provider selection,
document upload, and advanced features.
"""

import gradio as gr
import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RAGPipeline
from conversation import ConversationManager, MarkdownExporter, JSONExporter, HTMLExporter, get_context_cache
from providers import get_provider, switch_provider, list_providers
from config import LLM_PROVIDER, EMBEDDING_DEVICE, LLM_DEVICE
from logger_utils import get_logger
from core.document_parser import parse_document
from core.form_generator import get_form_generator
from core.analytics import get_analytics

logger = get_logger(__name__)

# Global instances
pipeline: Optional[RAGPipeline] = None
manager: Optional[ConversationManager] = None
current_session: Optional[str] = None
current_provider: str = LLM_PROVIDER


def initialize_system(provider_type: str = None):
    """Initialize the RAG system with specified provider"""
    global pipeline, manager, current_session, current_provider

    if provider_type:
        current_provider = provider_type

    if pipeline is None:
        logger.info(f"Initializing RAG system with provider: {current_provider}")
        pipeline = RAGPipeline({'llm_provider': current_provider})
        if not pipeline.initialize():
            return "Failed to initialize pipeline"
        logger.info("Pipeline initialized")

    if manager is None:
        manager = ConversationManager()

    if current_session is None:
        current_session = manager.start_session()

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


def chat(message: str, history: List[Tuple[str, str]],
         show_thinking: bool = True, show_sources: bool = True,
         show_metadata: bool = True, show_analysis: bool = False) -> Tuple[str, List[Tuple[str, str]]]:
    """Process chat message and return response with display options"""
    global pipeline, manager, current_session

    if not message.strip():
        return "", history

    try:
        if pipeline is None:
            initialize_system()

        # Get conversation context
        context = manager.get_context_for_query(current_session) if current_session else None

        # Generate response
        result = pipeline.query(message, conversation_history=context, stream=False)

        # Build response based on display options with collapsible sections
        response_parts = []

        # Add query analysis in collapsible section
        if show_analysis and result.get('metadata'):
            meta = result['metadata']
            analysis_content = ""
            if meta.get('query_type'):
                analysis_content += f"- **Tipe Query:** {meta['query_type']}\n"
            if meta.get('strategy'):
                analysis_content += f"- **Strategi:** {meta['strategy']}\n"
            if meta.get('complexity'):
                analysis_content += f"- **Kompleksitas:** {meta['complexity']}\n"
            if meta.get('keywords'):
                keywords = meta['keywords'][:10] if isinstance(meta['keywords'], list) else meta['keywords']
                analysis_content += f"- **Keywords:** {', '.join(keywords) if isinstance(keywords, list) else keywords}\n"
            if meta.get('legal_terms'):
                terms = meta['legal_terms'][:5] if isinstance(meta['legal_terms'], list) else meta['legal_terms']
                analysis_content += f"- **Legal Terms:** {', '.join(terms) if isinstance(terms, list) else terms}\n"
            if meta.get('entities'):
                analysis_content += f"- **Entities:** {meta['entities']}\n"

            # Search phase details
            if meta.get('search_phases'):
                phases = meta['search_phases']
                analysis_content += f"\n**Search Phases:**\n"
                for i, phase in enumerate(phases, 1):
                    if isinstance(phase, dict):
                        phase_name = phase.get('name', f'Phase {i}')
                        phase_time = phase.get('time', 0)
                        phase_results = phase.get('results', 0)
                        analysis_content += f"- {phase_name}: {phase_results} results ({phase_time:.2f}s)\n"

            if analysis_content:
                analysis_html = f"""<details><summary>🔍 Analisis Query (klik untuk expand)</summary>

{analysis_content}

</details>

---

"""
                response_parts.append(analysis_html)

        # Parse <think> tags from answer if present
        answer_text = result.get('answer', '')
        thinking_from_tags, clean_answer = parse_think_tags(answer_text)

        # Combine thinking sources
        thinking_content = result.get('thinking', '')
        if thinking_from_tags:
            thinking_content = thinking_from_tags if not thinking_content else f"{thinking_content}\n\n{thinking_from_tags}"

        # Add thinking process in collapsible section
        if show_thinking and thinking_content:
            thinking_html = f"""<details><summary>🧠 Proses Berpikir (klik untuk expand)</summary>

{thinking_content}

</details>

---

"""
            response_parts.append(thinking_html)

        # Main answer (use cleaned answer if think tags were parsed)
        final_answer = clean_answer if thinking_from_tags else answer_text
        response_parts.append(f"✅ **Jawaban:**\n\n{final_answer}")

        # Add sources in collapsible section with enhanced metadata
        if show_sources and result.get('sources'):
            sources = result['sources']
            if sources:
                source_content = ""
                for i, src in enumerate(sources[:5], 1):
                    if isinstance(src, dict):
                        title = src.get('title', src.get('regulation', f'Source {i}'))
                        score = src.get('score', 0)
                        doc_type = src.get('type', src.get('jenis_peraturan', ''))

                        source_content += f"**{i}. {title}**\n"
                        source_content += f"- Skor: {score:.3f}\n"

                        # Enhanced regulation metadata
                        if src.get('regulation_number') or src.get('nomor_peraturan'):
                            reg_num = src.get('regulation_number', src.get('nomor_peraturan', ''))
                            year = src.get('year', src.get('tahun', ''))
                            if reg_num and year:
                                source_content += f"- Nomor: {reg_num} Tahun {year}\n"
                            elif reg_num:
                                source_content += f"- Nomor: {reg_num}\n"

                        if doc_type:
                            source_content += f"- Jenis: {doc_type}\n"

                        # Legal domain
                        domain = src.get('kg_primary_domain', src.get('domain', ''))
                        if domain:
                            source_content += f"- Domain: {domain}\n"

                        # Authority and complexity scores
                        authority = src.get('kg_authority_score', src.get('authority_score', 0))
                        if authority:
                            auth_level = "Tinggi" if authority > 0.7 else "Sedang" if authority > 0.4 else "Rendah"
                            source_content += f"- Otoritas: {auth_level} ({authority:.2f})\n"

                        complexity = src.get('kg_legal_complexity', 0)
                        if complexity:
                            source_content += f"- Kompleksitas: {complexity:.2f}\n"

                        # Content indicators
                        indicators = []
                        if src.get('kg_has_obligations'):
                            indicators.append("Kewajiban")
                        if src.get('kg_has_prohibitions'):
                            indicators.append("Larangan")
                        if src.get('kg_has_permissions'):
                            indicators.append("Izin")
                        if indicators:
                            source_content += f"- Mengandung: {', '.join(indicators)}\n"

                        # Network importance
                        pagerank = src.get('kg_pagerank', 0)
                        if pagerank and pagerank > 0.01:
                            source_content += f"- Konektivitas: {pagerank:.3f}\n"

                        # Research team consensus info
                        if src.get('voting_ratio'):
                            voting = src['voting_ratio']
                            source_content += f"- Konsensus Tim: {voting:.0%}\n"
                        if src.get('personas_agreed'):
                            personas = src['personas_agreed']
                            if len(personas) > 1:
                                source_content += f"- Peneliti: {', '.join(personas[:3])}\n"

                        source_content += "\n"
                    else:
                        source_content += f"- {src}\n"

                source_html = f"""

---

<details><summary>📖 Sumber Hukum ({len(sources[:5])} dokumen)</summary>

{source_content}
</details>"""
                response_parts.append(source_html)

        # Add research team consensus display
        if result.get('consensus_data') or result.get('research_data'):
            consensus = result.get('consensus_data', {})
            research = result.get('research_data', {})

            team_content = ""

            # Agreement level
            if consensus.get('agreement_level'):
                team_content += f"- **Tingkat Kesepakatan:** {consensus['agreement_level']:.0%}\n"

            # Rounds executed
            if research.get('rounds_executed'):
                team_content += f"- **Ronde Penelitian:** {research['rounds_executed']}/5\n"

            # Candidates evaluated
            if research.get('total_candidates_evaluated'):
                team_content += f"- **Dokumen Dievaluasi:** {research['total_candidates_evaluated']:,}\n"

            # Results by persona
            if research.get('persona_results'):
                team_content += "\n**Hasil per Peneliti:**\n"
                for persona, results in research['persona_results'].items():
                    team_content += f"- {persona}: {len(results)} dokumen\n"

            # Devil's advocate flags
            if consensus.get('devil_advocate_flags'):
                flags = consensus['devil_advocate_flags']
                if flags:
                    team_content += f"\n**⚠️ Catatan Devil's Advocate:** {len(flags)} peringatan\n"

            if team_content:
                team_html = f"""

<details><summary>👥 Tim Peneliti & Konsensus</summary>

{team_content}
</details>"""
                response_parts.append(team_html)

        # Add community clusters display
        if result.get('communities') or result.get('clusters'):
            communities = result.get('communities', result.get('clusters', []))
            if communities:
                cluster_content = ""
                for i, community in enumerate(communities[:5], 1):
                    if isinstance(community, dict):
                        name = community.get('name', f'Cluster {i}')
                        size = community.get('size', 0)
                        theme = community.get('dominant_theme', community.get('theme', ''))
                        cluster_content += f"**{i}. {name}** ({size} dokumen)\n"
                        if theme:
                            cluster_content += f"- Tema: {theme}\n"
                        if community.get('members'):
                            members = community['members'][:3]
                            cluster_content += f"- Anggota: {', '.join(str(m) for m in members)}...\n"
                        cluster_content += "\n"

                if cluster_content:
                    cluster_html = f"""

<details><summary>🔗 Cluster Tematik</summary>

{cluster_content}
</details>"""
                    response_parts.append(cluster_html)

        # Add metadata in collapsible section
        if show_metadata and result.get('metadata'):
            meta = result['metadata']
            meta_content = ""
            if meta.get('query_type'):
                meta_content += f"- **Tipe Query:** {meta['query_type']}\n"
            if meta.get('processing_time'):
                meta_content += f"- **Waktu Proses:** {meta['processing_time']:.2f}s\n"
            if meta.get('total_results'):
                meta_content += f"- **Total Hasil:** {meta['total_results']}\n"
            if meta.get('search_phases'):
                phases = meta['search_phases']
                meta_content += f"- **Search Phases:** {len(phases)}\n"

            if meta_content:
                meta_html = f"""

<details><summary>📊 Metadata Pencarian</summary>

{meta_content}
</details>"""
                response_parts.append(meta_html)

        response = "".join(response_parts)

        # Save to conversation history
        if current_session:
            manager.add_turn(
                session_id=current_session,
                query=message,
                answer=response,
                metadata=result.get('metadata')
            )

        history.append((message, response))
        return "", history

    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_msg = f"Error: {str(e)}"
        history.append((message, error_msg))
        return "", history


def chat_streaming(message: str, history: List[Tuple[str, str]],
                   show_thinking: bool = True, show_sources: bool = True,
                   show_metadata: bool = True, show_analysis: bool = False):
    """
    Streaming chat function that yields partial responses with progress tracking.

    This provides real-time feedback as the response is generated,
    matching the original Kaggle_Demo.ipynb streaming behavior with
    collapsible sections and timestamps.
    """
    import time
    global pipeline, manager, current_session

    if not message.strip():
        yield "", history
        return

    try:
        if pipeline is None:
            initialize_system()

        start_time = time.time()

        # Get conversation context
        context = manager.get_context_for_query(current_session) if current_session else None

        # Progress tracking with timestamps
        progress_lines = []

        def add_progress(msg):
            elapsed = time.time() - start_time
            progress_lines.append(f"🔄 [{elapsed:.1f}s] {msg}")
            return "\n".join(progress_lines)

        # Phase 1: Show initial progress
        progress = add_progress("Memulai analisis query...")
        progress_html = f"""<details open><summary>📋 Proses Penelitian</summary>

{progress}

</details>"""
        history_with_status = history + [(message, progress_html)]
        yield "", history_with_status

        # Generate response
        result = pipeline.query(message, conversation_history=context, stream=True)

        # Update progress
        progress = add_progress("Query dianalisis")
        if result.get('metadata', {}).get('query_type'):
            progress = add_progress(f"Tipe: {result['metadata']['query_type']}")
        if result.get('metadata', {}).get('strategy'):
            progress = add_progress(f"Strategi: {result['metadata']['strategy']}")

        progress = add_progress("Pencarian dokumen selesai")
        if result.get('metadata', {}).get('total_documents_retrieved'):
            total = result['metadata']['total_documents_retrieved']
            progress = add_progress(f"Ditemukan {total} dokumen")

        # Build response progressively
        response_parts = []

        # Add progress section (collapsible, closed by default in final)
        progress_html = f"""<details><summary>📋 Proses Penelitian ({time.time() - start_time:.1f}s)</summary>

{progress}

</details>

---

"""
        response_parts.append(progress_html)

        # Add query analysis in collapsible section
        if show_analysis and result.get('metadata'):
            meta = result['metadata']
            analysis_content = ""
            if meta.get('query_type'):
                analysis_content += f"- **Tipe Query:** {meta['query_type']}\n"
            if meta.get('strategy'):
                analysis_content += f"- **Strategi:** {meta['strategy']}\n"
            if meta.get('complexity'):
                analysis_content += f"- **Kompleksitas:** {meta['complexity']}\n"
            if meta.get('keywords'):
                keywords = meta['keywords'][:10] if isinstance(meta['keywords'], list) else meta['keywords']
                analysis_content += f"- **Keywords:** {', '.join(keywords) if isinstance(keywords, list) else keywords}\n"
            if meta.get('legal_terms'):
                terms = meta['legal_terms'][:5] if isinstance(meta['legal_terms'], list) else meta['legal_terms']
                analysis_content += f"- **Legal Terms:** {', '.join(terms) if isinstance(terms, list) else terms}\n"

            # Search phase details
            if meta.get('search_phases'):
                phases = meta['search_phases']
                analysis_content += f"\n**Search Phases:**\n"
                for i, phase in enumerate(phases, 1):
                    if isinstance(phase, dict):
                        phase_name = phase.get('name', f'Phase {i}')
                        phase_time = phase.get('time', 0)
                        phase_results = phase.get('results', 0)
                        analysis_content += f"- {phase_name}: {phase_results} results ({phase_time:.2f}s)\n"

            if analysis_content:
                analysis_html = f"""<details><summary>🔍 Analisis Query (klik untuk expand)</summary>

{analysis_content}

</details>

---

"""
                response_parts.append(analysis_html)

        # Parse <think> tags from answer if present
        answer_text = result.get('answer', '')
        thinking_from_tags, clean_answer = parse_think_tags(answer_text)

        # Combine thinking sources
        thinking_content = result.get('thinking', '')
        if thinking_from_tags:
            thinking_content = thinking_from_tags if not thinking_content else f"{thinking_content}\n\n{thinking_from_tags}"

        # Phase 2: Show thinking process
        if show_thinking and thinking_content:
            thinking_html = f"""<details><summary>🧠 Proses Berpikir (klik untuk expand)</summary>

{thinking_content}

</details>

---

"""
            response_parts.append(thinking_html)
            current_display = "".join(response_parts) + "⏳ Generating answer..."
            history_with_progress = history + [(message, current_display)]
            yield "", history_with_progress

        # Phase 3: Stream the main answer
        answer = clean_answer if thinking_from_tags else answer_text

        # If streaming iterator is available, use it
        if hasattr(result, '__iter__') and not isinstance(result, (dict, str)):
            partial_answer = ""
            for chunk in result:
                if isinstance(chunk, str):
                    partial_answer += chunk
                elif isinstance(chunk, dict) and 'text' in chunk:
                    partial_answer += chunk['text']

                current_display = "".join(response_parts) + f"✅ **Jawaban:**\n\n{partial_answer}"
                history_with_progress = history + [(message, current_display)]
                yield "", history_with_progress

            response_parts.append(f"✅ **Jawaban:**\n\n{partial_answer}")
        else:
            # Non-streaming fallback
            response_parts.append(f"✅ **Jawaban:**\n\n{answer}")

        # Phase 4: Add sources in collapsible section with enhanced metadata
        if show_sources and result.get('sources'):
            sources = result['sources']
            if sources:
                source_content = ""
                for i, src in enumerate(sources[:5], 1):
                    if isinstance(src, dict):
                        title = src.get('title', src.get('regulation', f'Source {i}'))
                        score = src.get('score', 0)
                        doc_type = src.get('type', src.get('jenis_peraturan', ''))

                        source_content += f"**{i}. {title}**\n"
                        source_content += f"- Skor: {score:.3f}\n"

                        # Enhanced regulation metadata
                        if src.get('regulation_number') or src.get('nomor_peraturan'):
                            reg_num = src.get('regulation_number', src.get('nomor_peraturan', ''))
                            year = src.get('year', src.get('tahun', ''))
                            if reg_num and year:
                                source_content += f"- Nomor: {reg_num} Tahun {year}\n"
                            elif reg_num:
                                source_content += f"- Nomor: {reg_num}\n"

                        if doc_type:
                            source_content += f"- Jenis: {doc_type}\n"

                        # Legal domain
                        domain = src.get('kg_primary_domain', src.get('domain', ''))
                        if domain:
                            source_content += f"- Domain: {domain}\n"

                        # Authority and complexity scores
                        authority = src.get('kg_authority_score', src.get('authority_score', 0))
                        if authority:
                            auth_level = "Tinggi" if authority > 0.7 else "Sedang" if authority > 0.4 else "Rendah"
                            source_content += f"- Otoritas: {auth_level} ({authority:.2f})\n"

                        complexity = src.get('kg_legal_complexity', 0)
                        if complexity:
                            source_content += f"- Kompleksitas: {complexity:.2f}\n"

                        # Content indicators
                        indicators = []
                        if src.get('kg_has_obligations'):
                            indicators.append("Kewajiban")
                        if src.get('kg_has_prohibitions'):
                            indicators.append("Larangan")
                        if src.get('kg_has_permissions'):
                            indicators.append("Izin")
                        if indicators:
                            source_content += f"- Mengandung: {', '.join(indicators)}\n"

                        # Network importance
                        pagerank = src.get('kg_pagerank', 0)
                        if pagerank and pagerank > 0.01:
                            source_content += f"- Konektivitas: {pagerank:.3f}\n"

                        # Research team consensus info
                        if src.get('voting_ratio'):
                            voting = src['voting_ratio']
                            source_content += f"- Konsensus Tim: {voting:.0%}\n"
                        if src.get('personas_agreed'):
                            personas = src['personas_agreed']
                            if len(personas) > 1:
                                source_content += f"- Peneliti: {', '.join(personas[:3])}\n"

                        source_content += "\n"
                    else:
                        source_content += f"- {src}\n"

                source_html = f"""

---

<details><summary>📖 Sumber Hukum ({len(sources[:5])} dokumen)</summary>

{source_content}
</details>"""
                response_parts.append(source_html)

        # Add research team consensus display
        if result.get('consensus_data') or result.get('research_data'):
            consensus = result.get('consensus_data', {})
            research = result.get('research_data', {})

            team_content = ""

            # Agreement level
            if consensus.get('agreement_level'):
                team_content += f"- **Tingkat Kesepakatan:** {consensus['agreement_level']:.0%}\n"

            # Rounds executed
            if research.get('rounds_executed'):
                team_content += f"- **Ronde Penelitian:** {research['rounds_executed']}/5\n"

            # Candidates evaluated
            if research.get('total_candidates_evaluated'):
                team_content += f"- **Dokumen Dievaluasi:** {research['total_candidates_evaluated']:,}\n"

            # Results by persona
            if research.get('persona_results'):
                team_content += "\n**Hasil per Peneliti:**\n"
                for persona, results in research['persona_results'].items():
                    team_content += f"- {persona}: {len(results)} dokumen\n"

            # Devil's advocate flags
            if consensus.get('devil_advocate_flags'):
                flags = consensus['devil_advocate_flags']
                if flags:
                    team_content += f"\n**⚠️ Catatan Devil's Advocate:** {len(flags)} peringatan\n"

            if team_content:
                team_html = f"""

<details><summary>👥 Tim Peneliti & Konsensus</summary>

{team_content}
</details>"""
                response_parts.append(team_html)

        # Add community clusters display
        if result.get('communities') or result.get('clusters'):
            communities = result.get('communities', result.get('clusters', []))
            if communities:
                cluster_content = ""
                for i, community in enumerate(communities[:5], 1):
                    if isinstance(community, dict):
                        name = community.get('name', f'Cluster {i}')
                        size = community.get('size', 0)
                        theme = community.get('dominant_theme', community.get('theme', ''))
                        cluster_content += f"**{i}. {name}** ({size} dokumen)\n"
                        if theme:
                            cluster_content += f"- Tema: {theme}\n"
                        if community.get('members'):
                            members = community['members'][:3]
                            cluster_content += f"- Anggota: {', '.join(str(m) for m in members)}...\n"
                        cluster_content += "\n"

                if cluster_content:
                    cluster_html = f"""

<details><summary>🔗 Cluster Tematik</summary>

{cluster_content}
</details>"""
                    response_parts.append(cluster_html)

        # Phase 5: Add metadata in collapsible section
        if show_metadata and result.get('metadata'):
            meta = result['metadata']
            meta_content = ""
            if meta.get('query_type'):
                meta_content += f"- **Tipe Query:** {meta['query_type']}\n"
            if meta.get('processing_time'):
                meta_content += f"- **Waktu Proses:** {meta['processing_time']:.2f}s\n"
            if meta.get('total_results'):
                meta_content += f"- **Total Hasil:** {meta['total_results']}\n"

            if meta_content:
                meta_html = f"""

<details><summary>📊 Metadata Pencarian</summary>

{meta_content}
</details>"""
                response_parts.append(meta_html)

        # Final response
        final_response = "".join(response_parts)

        # Save to conversation history
        if current_session:
            manager.add_turn(
                session_id=current_session,
                query=message,
                answer=final_response,
                metadata=result.get('metadata')
            )

        history.append((message, final_response))
        yield "", history

    except Exception as e:
        logger.error(f"Chat streaming error: {e}")
        error_msg = f"Error: {str(e)}"
        history.append((message, error_msg))
        yield "", history


def clear_chat():
    """Clear chat and start new session"""
    global manager, current_session

    if manager:
        current_session = manager.start_session()

    return [], f"New session: {current_session}"


def export_conversation(format_type: str) -> str:
    """Export current conversation"""
    global manager, current_session

    if not manager or not current_session:
        return "No active session"

    session_data = manager.get_session(current_session)
    if not session_data:
        return "Session not found"

    exporters = {
        'Markdown': MarkdownExporter,
        'JSON': JSONExporter,
        'HTML': HTMLExporter
    }

    exporter_class = exporters.get(format_type, MarkdownExporter)
    exporter = exporter_class()

    try:
        path = exporter.export_and_save(session_data, directory='exports')
        return f"Exported to: {path}"
    except Exception as e:
        return f"Export error: {e}"


def get_session_info() -> str:
    """Get current session information"""
    global manager, current_session, current_provider

    if not manager or not current_session:
        return "No active session"

    summary = manager.get_session_summary(current_session)
    if not summary:
        return "Session not found"

    cache = get_context_cache()
    cache_stats = cache.get_stats()

    return f"""Session ID: {summary['session_id']}
Provider: {current_provider}
Total Turns: {summary['total_turns']}
Total Tokens: {summary['total_tokens']}
Total Time: {summary['total_time']:.2f}s
Cache Size: {cache_stats['size']}/{cache_stats['max_size']}"""


def upload_document(file) -> str:
    """Handle document upload and parsing"""
    if file is None:
        return "No file uploaded"

    try:
        file_path = file.name
        result = parse_document(file_path)

        if not result['success']:
            return f"Parse error: {result['error']}"

        meta = result['metadata']
        content_preview = result['content'][:500] + "..." if len(result['content']) > 500 else result['content']

        # Build status message
        status = f"**Document Parsed Successfully**\n\n"
        status += f"**File:** {meta.get('filename', 'Unknown')}\n"
        status += f"**Format:** {meta.get('format', 'Unknown')}\n"

        if 'pages' in meta:
            status += f"**Pages:** {meta['pages']}\n"
        if 'paragraphs' in meta:
            status += f"**Paragraphs:** {meta['paragraphs']}\n"

        status += f"**Words:** {meta.get('word_count', 0)}\n"
        status += f"**Characters:** {meta.get('char_count', 0)}\n"

        if meta.get('title'):
            status += f"**Title:** {meta['title']}\n"
        if meta.get('author'):
            status += f"**Author:** {meta['author']}\n"

        status += f"\n**Content Preview:**\n```\n{content_preview}\n```"

        # Store parsed content for potential use in chat
        global _uploaded_document
        _uploaded_document = result

        return status

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return f"Upload error: {e}"


# Store uploaded document for use in chat
_uploaded_document = None


def generate_legal_form(template_id: str, field_values: str) -> str:
    """Generate a legal form from template"""
    try:
        generator = get_form_generator()

        # Parse field values (simple key=value format)
        data = {}
        for line in field_values.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                data[key.strip()] = value.strip()

        result = generator.generate_form(template_id, data)

        if result['success']:
            return f"**{result['template_name']}**\n\n```\n{result['content']}\n```"
        else:
            return f"Error: {result['error']}"
    except Exception as e:
        return f"Error generating form: {e}"


def get_form_templates() -> str:
    """Get list of available form templates"""
    generator = get_form_generator()
    templates = generator.list_templates()

    result = "**Available Templates:**\n\n"
    for t in templates:
        result += f"- **{t['id']}**: {t['name']} ({t['description']})\n"

        # Get template details
        template = generator.get_template(t['id'])
        if template:
            result += f"  Fields: {', '.join(f['name'] for f in template['fields'])}\n\n"

    return result


def get_analytics_summary() -> str:
    """Get analytics dashboard summary"""
    analytics = get_analytics()
    summary = analytics.get_summary()

    result = f"""**Analytics Summary**

**Session**
- Duration: {summary['session']['duration_formatted']}

**Queries**
- Total: {summary['queries']['total']}
- Successful: {summary['queries']['successful']}
- Failed: {summary['queries']['failed']}
- By Type: {dict(summary['queries']['by_type'])}

**Performance**
- Avg Response: {summary['performance']['avg_response_time']:.3f}s
- Min Response: {summary['performance']['min_response_time']:.3f}s
- Max Response: {summary['performance']['max_response_time']:.3f}s

**Providers Used**: {dict(summary['providers'])}

**Errors**: {summary['errors']['total']}
"""
    return result


def get_performance_report() -> str:
    """Get detailed performance report"""
    analytics = get_analytics()
    report = analytics.get_performance_report()

    if isinstance(report, dict) and 'message' in report:
        return report['message']

    result = "**Performance Report**\n\n"
    for component, ops in report.items():
        result += f"**{component}**\n"
        for op, stats in ops.items():
            result += f"  - {op}: avg={stats['avg']:.3f}s, count={stats['count']}\n"
        result += "\n"

    return result


def get_system_health() -> str:
    """Get comprehensive system health status"""
    global pipeline, manager, current_session
    import torch
    import psutil

    report = ["## 🏥 System Health Report\n"]

    # Pipeline status
    report.append("### Pipeline Status")
    if pipeline and pipeline._initialized:
        report.append("✅ **RAG Pipeline:** Initialized and ready")
        info = pipeline.get_pipeline_info()
        report.append(f"- Init time: {info.get('initialization_time', 0):.2f}s")
        if 'dataset_stats' in info:
            stats = info['dataset_stats']
            report.append(f"- Total records: {stats.get('total_records', 0)}")
            report.append(f"- KG enhanced: {stats.get('kg_enhanced_records', 0)}")
    else:
        report.append("❌ **RAG Pipeline:** Not initialized")

    # Session status
    report.append("\n### Session Status")
    if manager:
        report.append("✅ **Conversation Manager:** Active")
        if current_session:
            report.append(f"- Current session: `{current_session[:8]}...`")
    else:
        report.append("❌ **Conversation Manager:** Not initialized")

    # Device status
    report.append("\n### Device Status")
    report.append(f"- **Embedding Device:** {EMBEDDING_DEVICE}")
    report.append(f"- **LLM Device:** {LLM_DEVICE}")

    if torch.cuda.is_available():
        report.append(f"- **CUDA Available:** Yes")
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            report.append(f"- GPU {i}: {mem_used:.1f}/{mem_total:.1f} GB used")
    else:
        report.append("- **CUDA Available:** No (using CPU)")

    # System resources
    report.append("\n### System Resources")
    mem = psutil.virtual_memory()
    report.append(f"- **RAM:** {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)")
    cpu_percent = psutil.cpu_percent(interval=0.1)
    report.append(f"- **CPU:** {cpu_percent}%")

    # Provider status
    report.append("\n### LLM Provider")
    report.append(f"- **Current Provider:** {current_provider}")
    report.append(f"- **Available:** {', '.join(list_providers())}")

    return "\n".join(report)


def get_about_info() -> str:
    """Get system documentation and feature information"""
    return """## 📘 Indonesian Legal RAG System

### Overview
Advanced Retrieval-Augmented Generation system for Indonesian legal document consultation.

### Key Features

**🔍 Multi-Stage Search**
- Hybrid search combining semantic + keyword matching
- Metadata-first search with PERFECT SCORE OVERRIDE for exact regulation matches
- Citation chain traversal for related documents
- Dynamic community detection for thematic clustering

**👥 Research Team Approach**
- 5 specialized research personas (Analis Konstitusi, Spesialis Regulasi, Pakar Prosedural, Spesialis Ketenagakerjaan, Pencari Preseden)
- Adaptive learning based on performance history
- Consensus-based result aggregation

**🧠 Advanced Query Analysis**
- Automatic strategy selection (keyword_first, semantic_first, hybrid_balanced)
- Query type detection (specific_regulation, procedural, sanctions, etc.)
- Context-dependent query handling

**📊 Knowledge Graph Integration**
- Cross-reference extraction and traversal
- Legal hierarchy understanding
- Domain classification

**💬 Conversation Features**
- Multi-turn context awareness
- Session management with export (Markdown, JSON, HTML)
- Document upload and parsing

### Configuration

**Advanced Settings:**
- **Research Team Size:** Number of personas (1-5)
- **Consensus Threshold:** Agreement level required (0.3-0.9)
- **Quality Threshold:** Minimum result quality (0.3-0.9)
- **Temperature:** LLM creativity (0.0-1.0)
- **Max Tokens:** Response length limit

### Data Source
Legal documents from Indonesian government regulations indexed with knowledge graph enhancements.

### Version Info
- **System:** Indonesian Legal RAG v2.0
- **Architecture:** LangGraph-based orchestration
- **Models:** Sentence-transformers for embeddings, BGE reranker
"""


def get_export_preview(format_type: str) -> Tuple[str, str]:
    """Get export preview and prepare file for download"""
    global manager, current_session

    if not manager or not current_session:
        return "No active session", ""

    session_data = manager.get_session(current_session)
    if not session_data:
        return "Session not found", ""

    exporters = {
        'Markdown': MarkdownExporter,
        'JSON': JSONExporter,
        'HTML': HTMLExporter
    }

    exporter_class = exporters.get(format_type, MarkdownExporter)
    exporter = exporter_class()

    try:
        # Generate content
        content = exporter.export(session_data)

        # Preview (truncated for display)
        preview = content[:2000] + "..." if len(content) > 2000 else content

        return preview, content
    except Exception as e:
        return f"Export error: {e}", ""


def create_demo() -> gr.Blocks:
    """Create Gradio demo interface with all features"""

    # CSS for clean chat interface - responsive to zoom
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
        max-width: 75em; /* Changed from rem to em for zoom scaling */
        margin: 0 auto;
        padding: 1.25em; /* Changed from rem to em */
        box-sizing: border-box;
    }

    /* Chatbot container - responsive sizing */
    .chat-container {
        height: 75vh; /* Viewport height scales with zoom */
        min-height: 25em; /* Reduced and changed to em */
        max-height: none; /* Remove max-height to allow scaling */
        width: 100%;
        overflow-y: auto;
        border: 0.0625em solid #e0e0e0; /* Changed to em */
        border-radius: 0.75em; /* Changed to em */
        background: white;
        box-sizing: border-box;
        resize: vertical; /* Allow manual resizing if needed */
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
        font-size: 1em; /* Changed to em for scaling */
    }

    /* Input area styling */
    .input-row {
        margin-top: 0.9375em; /* Changed to em */
        width: 100%;
    }

    .input-row .form {
        width: 100%;
    }

    /* Settings panels - scalable */
    .settings-panel {
        background-color: #f8f9fa;
        padding: 1.25em; /* Changed to em */
        border-radius: 0.75em; /* Changed to em */
        margin-bottom: 0.9375em; /* Changed to em */
        box-shadow: 0 0.125em 0.25em rgba(0,0,0,0.1); /* Changed to em */
        width: 100%;
        box-sizing: border-box;
    }

    .status-panel {
        background-color: #e8f4fd;
        padding: 0.9375em; /* Changed to em */
        border-radius: 0.5em; /* Changed to em */
        border-left: 0.25em solid #2196F3; /* Changed to em */
        margin-bottom: 0.625em; /* Changed to em */
    }

    /* Responsive breakpoints - using em for zoom-friendly breakpoints */
    @media (max-width: 87.5em) {
        .main-chat-area {
            max-width: 95%;
            padding: 0.9375em;
        }
    }

    @media (max-width: 64em) {
        .chat-container {
            height: 70vh;
            min-height: 20em; /* Reduced for better mobile experience */
        }

        .main-chat-area {
            padding: 0.9375em;
        }
    }

    @media (max-width: 48em) {
        .chat-container {
            height: 65vh;
            min-height: 18em; /* Further reduced */
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
            min-height: 15em; /* Minimum for usability */
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
        margin-bottom: 1.25em; /* Changed to em */
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
        margin: 0 0.5em; /* Changed to em */
        padding: 0.75em 1.5em; /* Changed to em */
        border-radius: 0.5em; /* Changed to em */
        font-weight: 500;
        transition: all 0.2s ease;
    }

    /* Center tab container */
    .tabitem {
        width: 100%;
        max-width: 75em; /* Changed to em */
        margin: 0 auto;
    }

    /* Examples styling */
    .examples {
        margin-top: 0.9375em; /* Changed to em */
    }

    /* Button styling */
    .clear-btn {
        margin-left: auto;
    }

    /* Ensure consistent column widths in settings */
    .settings-columns {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.25em; /* Changed to em */
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
        font-size: 1em; /* Base font size that scales with zoom */
    }

    .export-preview {
        font-family: monospace;
        white-space: pre-wrap;
        background: #f5f5f5;
        padding: 1em;
        border-radius: 0.5em;
        max-height: 400px;
        overflow-y: auto;
    }
    """

    with gr.Blocks(
        title="Enhanced Indonesian Legal Assistant",
        theme=gr.themes.Default(),
        css=custom_css
    ) as demo:

        # State variables for display options
        show_thinking = gr.State(True)
        show_sources = gr.State(True)
        show_metadata = gr.State(False)
        show_analysis = gr.State(False)

        with gr.Tabs():
            # Main Chat Tab
            with gr.TabItem("💬 Konsultasi Hukum", id="chat"):
                with gr.Column(elem_classes="main-chat-area"):
                    chatbot = gr.Chatbot(
                        height="75vh",
                        show_label=False,
                        container=True,
                        bubble_full_width=True,
                        elem_classes="chat-container",
                        show_copy_button=True,
                        sanitize_html=True,
                        render_markdown=True,
                    )

                    with gr.Row(elem_classes="input-row"):
                        msg = gr.Textbox(
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
                                inputs=msg,
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
                            max_tokens = gr.Slider(512, 4096, value=2048, step=256, label="Max New Tokens")

                        # Research Team Settings
                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 👥 Research Team Configuration")
                            team_size = gr.Slider(1, 5, value=4, step=1, label="Team Size")
                            enable_cross_validation = gr.Checkbox(label="Enable Cross-Validation", value=True)
                            enable_devils_advocate = gr.Checkbox(label="Enable Devil's Advocate", value=True)
                            consensus_threshold = gr.Slider(0.3, 0.9, value=0.6, step=0.05, label="Consensus Threshold")

                        # Display Settings
                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 💬 Display Settings")
                            show_thinking_cb = gr.Checkbox(label="Show Thinking Process", value=True)
                            show_sources_cb = gr.Checkbox(label="Show Legal Sources", value=True)
                            show_metadata_cb = gr.Checkbox(label="Show All Retrieved Metadata", value=True)
                            show_analysis_cb = gr.Checkbox(label="Show Query Analysis", value=False)

                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 🧠 LLM Generation Settings")
                            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="Top P")
                            top_k_gen = gr.Slider(1, 100, value=20, step=1, label="Top K")
                            min_p = gr.Slider(0.01, 0.3, value=0.1, step=0.01, label="Min P")

                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 📊 System Information")
                            system_info_btn = gr.Button("📈 View System Stats", variant="primary")
                            reset_defaults_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
                            system_info_output = gr.Markdown("")

                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 🏥 System Health")
                            health_btn = gr.Button("🔍 Run Health Check", variant="secondary")
                            health_output = gr.Markdown("")

                        with gr.Group(elem_classes="settings-panel"):
                            gr.Markdown("#### 🔌 Provider Settings")
                            provider_dropdown = gr.Dropdown(
                                choices=['local', 'openai', 'anthropic', 'google', 'openrouter'],
                                value='local',
                                label="LLM Provider"
                            )
                            provider_btn = gr.Button("Switch Provider")
                            provider_status = gr.Textbox(label="Provider Status", interactive=False)

                    with gr.Column():
                        # Enhanced Search Phase Configuration
                        with gr.Group(elem_classes="settings-panel"):
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

            # Export Tab
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

                    export_preview = gr.Textbox(
                        label="Export Output",
                        lines=20,
                        max_lines=30,
                        show_copy_button=True,
                        elem_classes="export-preview"
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

        # Update state from checkboxes
        show_thinking_cb.change(lambda x: x, inputs=[show_thinking_cb], outputs=[show_thinking])
        show_sources_cb.change(lambda x: x, inputs=[show_sources_cb], outputs=[show_sources])
        show_metadata_cb.change(lambda x: x, inputs=[show_metadata_cb], outputs=[show_metadata])
        show_analysis_cb.change(lambda x: x, inputs=[show_analysis_cb], outputs=[show_analysis])

        # Chat with streaming - shows message immediately then generates
        def chat_with_streaming(message, history, thinking, sources, metadata, analysis):
            """Chat function that shows user message immediately then streams response"""
            if not message.strip():
                return "", history

            # Add user message immediately
            history = history + [(message, None)]
            yield "", history

            # Now generate response
            try:
                if pipeline is None:
                    initialize_system()

                context = manager.get_context_for_query(current_session) if current_session else None
                result = pipeline.query(message, conversation_history=context, stream=False)

                # Build response
                response_parts = []

                # Parse think tags
                answer_text = result.get('answer', '')
                thinking_content_from_tags, clean_answer = parse_think_tags(answer_text)

                # Combine thinking
                thinking_content = result.get('thinking', '')
                if thinking_content_from_tags:
                    thinking_content = thinking_content_from_tags if not thinking_content else f"{thinking_content}\n\n{thinking_content_from_tags}"

                # Add thinking
                if thinking and thinking_content:
                    response_parts.append(f"""<details><summary>🧠 Proses Berpikir</summary>\n\n{thinking_content}\n\n</details>\n\n---\n\n""")

                # Main answer
                final_answer = clean_answer if thinking_content_from_tags else answer_text
                response_parts.append(f"✅ **Jawaban:**\n\n{final_answer}")

                # Add sources
                if sources and result.get('sources'):
                    src_list = result['sources']
                    if src_list:
                        source_content = ""
                        for i, src in enumerate(src_list[:5], 1):
                            if isinstance(src, dict):
                                title = src.get('title', src.get('regulation', f'Source {i}'))
                                score = src.get('score', 0)
                                doc_type = src.get('type', src.get('jenis_peraturan', ''))
                                source_content += f"**{i}. {title}**\n- Skor: {score:.3f}\n"
                                if doc_type:
                                    source_content += f"- Jenis: {doc_type}\n"
                                source_content += "\n"
                        response_parts.append(f"""\n\n---\n\n<details><summary>📖 Sumber Hukum ({len(src_list[:5])} dokumen)</summary>\n\n{source_content}</details>""")

                # Add metadata
                if metadata and result.get('metadata'):
                    meta = result['metadata']
                    meta_content = ""
                    if meta.get('query_type'):
                        meta_content += f"- **Tipe Query:** {meta['query_type']}\n"
                    if meta.get('processing_time'):
                        meta_content += f"- **Waktu Proses:** {meta['processing_time']:.2f}s\n"
                    if meta_content:
                        response_parts.append(f"""\n\n<details><summary>📊 Metadata</summary>\n\n{meta_content}</details>""")

                response = "".join(response_parts)

                # Save to history
                if current_session:
                    manager.add_turn(current_session, message, response, result.get('metadata'))

                history[-1] = (message, response)
                yield "", history

            except Exception as e:
                logger.error(f"Chat error: {e}")
                history[-1] = (message, f"Error: {str(e)}")
                yield "", history

        # Export handler function
        def export_conversation_handler(format_type, include_metadata, include_research, include_full):
            """Handle export button click"""
            import tempfile
            from datetime import datetime

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

                exporter_class = exporters.get(format_type, MarkdownExporter)
                exporter = exporter_class()

                content = exporter.export(session_data)

                ext_map = {'Markdown': 'md', 'JSON': 'json', 'HTML': 'html'}
                extension = ext_map.get(format_type, 'md')
                filename = f"legal_consultation_{timestamp}.{extension}"

                # Save to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{extension}', delete=False, encoding='utf-8') as f:
                    f.write(content)
                    temp_path = f.name

                return content, temp_path

            except Exception as e:
                return f"Export failed: {str(e)}", None

        # Event handlers
        msg.submit(
            chat_with_streaming,
            inputs=[msg, chatbot, show_thinking, show_sources, show_metadata, show_analysis],
            outputs=[msg, chatbot]
        )

        # Export button handler
        export_button.click(
            export_conversation_handler,
            inputs=[export_format, include_metadata_export, include_research_process_export, include_full_content_export],
            outputs=[export_preview, download_file]
        )

        # Provider handler
        provider_btn.click(change_provider, inputs=[provider_dropdown], outputs=[provider_status])

        # System info and health handlers
        system_info_btn.click(get_about_info, outputs=[system_info_output])
        health_btn.click(get_system_health, outputs=[health_output])

        # Reset defaults function
        def reset_to_defaults():
            return (
                3,      # final_top_k
                0.7,    # temperature
                2048,   # max_tokens
                4,      # team_size
                True,   # enable_cross_validation
                True,   # enable_devils_advocate
                0.6,    # consensus_threshold
                True,   # show_thinking_cb
                True,   # show_sources_cb
                True,   # show_metadata_cb
                False,  # show_analysis_cb
                1.0,    # top_p
                20,     # top_k_gen
                0.1,    # min_p
            )

        reset_defaults_btn.click(
            reset_to_defaults,
            outputs=[
                final_top_k, temperature, max_tokens, team_size,
                enable_cross_validation, enable_devils_advocate, consensus_threshold,
                show_thinking_cb, show_sources_cb, show_metadata_cb, show_analysis_cb,
                top_p, top_k_gen, min_p
            ]
        )

    return demo


def launch_app(share: bool = False, server_port: int = 7860):
    """Launch Gradio app with pre-initialization"""
    global pipeline, manager, current_session

    # Initialize system BEFORE launching UI
    logger.info("Pre-initializing system before UI launch...")

    # Initialize pipeline
    if pipeline is None:
        logger.info(f"Initializing RAG pipeline with provider: {current_provider}")
        pipeline = RAGPipeline({'llm_provider': current_provider})
        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            raise RuntimeError("Pipeline initialization failed")
        logger.info("Pipeline initialized successfully")

    # Initialize conversation manager
    if manager is None:
        manager = ConversationManager()
        logger.info("Conversation manager initialized")

    # Start session
    if current_session is None:
        current_session = manager.start_session()
        logger.info(f"Session started: {current_session}")

    logger.info("System fully initialized, launching UI...")

    # Now create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    launch_app()
