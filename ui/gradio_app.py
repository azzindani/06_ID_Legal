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


def chat(message: str, history: List[Tuple[str, str]],
         show_thinking: bool = True, show_sources: bool = True,
         show_metadata: bool = True) -> Tuple[str, List[Tuple[str, str]]]:
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

        # Add thinking process in collapsible section
        if show_thinking and result.get('thinking'):
            thinking_html = f"""<details><summary>🧠 Proses Berpikir (klik untuk expand)</summary>

{result['thinking']}

</details>

---

"""
            response_parts.append(thinking_html)

        # Main answer
        response_parts.append(f"✅ **Jawaban:**\n\n{result['answer']}")

        # Add sources in collapsible section
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
                        if doc_type:
                            source_content += f"- Jenis: {doc_type}\n"
                        source_content += "\n"
                    else:
                        source_content += f"- {src}\n"

                source_html = f"""

---

<details><summary>📖 Sumber Hukum ({len(sources[:5])} dokumen)</summary>

{source_content}
</details>"""
                response_parts.append(source_html)

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
                   show_metadata: bool = True):
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

        # Phase 2: Show thinking process
        if show_thinking and result.get('thinking'):
            thinking_html = f"""<details><summary>🧠 Proses Berpikir (klik untuk expand)</summary>

{result['thinking']}

</details>

---

"""
            response_parts.append(thinking_html)
            current_display = "".join(response_parts) + "⏳ Generating answer..."
            history_with_progress = history + [(message, current_display)]
            yield "", history_with_progress

        # Phase 3: Stream the main answer
        answer = result.get('answer', '')

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

        # Phase 4: Add sources in collapsible section
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
                        if doc_type:
                            source_content += f"- Jenis: {doc_type}\n"
                        source_content += "\n"
                    else:
                        source_content += f"- {src}\n"

                source_html = f"""

---

<details><summary>📖 Sumber Hukum ({len(sources[:5])} dokumen)</summary>

{source_content}
</details>"""
                response_parts.append(source_html)

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


def create_demo() -> gr.Blocks:
    """Create Gradio demo interface with all features"""

    # Original CSS from Kaggle_Demo.ipynb - zoom-friendly responsive design
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

    with gr.Blocks(
        title="Enhanced Indonesian Legal Assistant",
        theme=gr.themes.Default(),
        css=custom_css
    ) as demo:

        gr.Markdown(
            """
            # Indonesian Legal RAG System
            ### Sistem Konsultasi Hukum Indonesia
            """
        )

        with gr.Tabs():
            # Chat Tab
            with gr.TabItem("Konsultasi Hukum", id="chat"):
                with gr.Column(elem_classes="main-chat-area"):
                    chatbot = gr.Chatbot(
                        height="75vh",
                        show_label=False,
                        container=True,
                        bubble_full_width=True,
                        elem_classes="chat-container",
                        show_copy_button=True,
                        render_markdown=True,
                    )

                    with gr.Row(elem_classes="input-row"):
                        msg = gr.Textbox(
                            placeholder="Tanyakan tentang hukum Indonesia...",
                            show_label=False,
                            container=False,
                            scale=10,
                            lines=1,
                            max_lines=3,
                            interactive=True
                        )
                        submit_btn = gr.Button("Kirim", variant="primary", scale=1)

                    # Example questions - comprehensive like original
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
                                examples_per_page=4,
                                label="Contoh Pertanyaan"
                            )

                # Settings panel in separate row
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("Display Options", open=True):
                            show_thinking = gr.Checkbox(label="Show Thinking Process", value=True)
                            show_sources = gr.Checkbox(label="Show Sources", value=True)
                            show_metadata = gr.Checkbox(label="Show Metadata", value=False)

                    with gr.Column(scale=1):
                        with gr.Accordion("Provider Settings", open=False):
                            provider_dropdown = gr.Dropdown(
                                choices=['local', 'openai', 'anthropic', 'google', 'openrouter'],
                                value='local',
                                label="LLM Provider"
                            )
                            provider_btn = gr.Button("Switch Provider")
                            provider_status = gr.Textbox(label="Provider Status", interactive=False)

                    with gr.Column(scale=1):
                        with gr.Accordion("Session & Export", open=False):
                            clear_btn = gr.Button("New Session", variant="secondary")
                            status = gr.Textbox(label="Status", interactive=False)
                            export_format = gr.Radio(
                                choices=["Markdown", "JSON", "HTML"],
                                value="Markdown",
                                label="Export Format"
                            )
                            export_btn = gr.Button("Export")
                            export_status = gr.Textbox(label="Export Result", interactive=False)

                    with gr.Column(scale=1):
                        with gr.Accordion("Document Upload", open=False):
                            file_upload = gr.File(
                                label="Upload Document",
                                file_types=[".pdf", ".docx", ".doc", ".txt"]
                            )
                            upload_status = gr.Textbox(label="Upload Status", interactive=False)

                        with gr.Accordion("Info", open=False):
                            info_btn = gr.Button("Refresh Info")
                            session_info = gr.Textbox(label="Session Info", interactive=False, lines=7)

            # Form Generator Tab
            with gr.TabItem("Form Generator"):
                gr.Markdown("### Legal Document Generator")
                with gr.Row():
                    with gr.Column():
                        templates_display = gr.Markdown()
                        templates_btn = gr.Button("Show Templates")

                        template_id = gr.Dropdown(
                            choices=['surat_kuasa', 'surat_pernyataan', 'perjanjian_kerja', 'pengaduan', 'somasi'],
                            value='surat_kuasa',
                            label="Template"
                        )
                        field_values = gr.Textbox(
                            label="Field Values (key=value per line)",
                            placeholder="pemberi_kuasa=John Doe\npenerima_kuasa=Jane Smith\nkeperluan=Mengurus dokumen\ntanggal=22 November 2025\ntempat=Jakarta",
                            lines=8
                        )
                        generate_btn = gr.Button("Generate Form", variant="primary")

                    with gr.Column():
                        form_output = gr.Markdown(label="Generated Form")

            # Analytics Tab
            with gr.TabItem("Analytics"):
                gr.Markdown("### System Analytics")
                with gr.Row():
                    with gr.Column():
                        analytics_btn = gr.Button("Refresh Analytics", variant="primary")
                        analytics_output = gr.Markdown()

                    with gr.Column():
                        performance_btn = gr.Button("Performance Report")
                        performance_output = gr.Markdown()

        # Event handlers
        submit_btn.click(
            chat,
            inputs=[msg, chatbot, show_thinking, show_sources, show_metadata],
            outputs=[msg, chatbot]
        )

        msg.submit(
            chat,
            inputs=[msg, chatbot, show_thinking, show_sources, show_metadata],
            outputs=[msg, chatbot]
        )

        provider_btn.click(
            change_provider,
            inputs=[provider_dropdown],
            outputs=[provider_status]
        )

        clear_btn.click(
            clear_chat,
            outputs=[chatbot, status]
        )

        export_btn.click(
            export_conversation,
            inputs=[export_format],
            outputs=[export_status]
        )

        file_upload.change(
            upload_document,
            inputs=[file_upload],
            outputs=[upload_status]
        )

        info_btn.click(
            get_session_info,
            outputs=[session_info]
        )

        # Form Generator handlers
        templates_btn.click(
            get_form_templates,
            outputs=[templates_display]
        )

        generate_btn.click(
            generate_legal_form,
            inputs=[template_id, field_values],
            outputs=[form_output]
        )

        # Analytics handlers
        analytics_btn.click(
            get_analytics_summary,
            outputs=[analytics_output]
        )

        performance_btn.click(
            get_performance_report,
            outputs=[performance_output]
        )

        # Initialize on load
        demo.load(
            initialize_system,
            outputs=[status]
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
