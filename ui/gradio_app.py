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
from conversation import (
    ConversationManager, MarkdownExporter, JSONExporter, HTMLExporter,
    get_context_cache, create_conversational_service
)
from config import (
    LLM_PROVIDER, EMBEDDING_DEVICE, LLM_DEVICE,
    DEFAULT_CONFIG, DEFAULT_SEARCH_PHASES, RESEARCH_TEAM_PERSONAS,
    EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL
)
from logger_utils import get_logger
from utils.formatting import (
    format_sources_info,
    _extract_all_documents_from_metadata,
    format_all_documents,
    format_retrieved_metadata,
    final_selection_with_kg
)
from utils.text_utils import parse_think_tags
from utils.health import system_health_check, format_health_report
from utils.system_info import format_system_info, get_dataset_statistics
from ui.services.system_service import (
    initialize_rag_system,
    change_llm_provider,
    clear_conversation_session
)

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

    if pipeline is None:
        pipeline, manager, current_session, current_provider, components = initialize_rag_system(
            RAGPipeline,
            ConversationManager,
            provider_type,
            current_provider
        )

        # Extract component references
        search_engine = components.get('search_engine')
        knowledge_graph = components.get('knowledge_graph')
        reranker = components.get('reranker')
        llm_generator = components.get('llm_generator')
        llm_model = components.get('llm_model')
        llm_tokenizer = components.get('llm_tokenizer')
        dataset_loader = components.get('dataset_loader')
        conversation_manager = manager

        initialization_complete = True

    device_info = f"Embedding: {EMBEDDING_DEVICE}, LLM: {LLM_DEVICE}"
    return f"Initialized with {current_provider} provider. {device_info}"


def change_provider(provider_type: str):
    """Switch to a different LLM provider"""
    global pipeline, current_provider

    pipeline, current_provider, message = change_llm_provider(
        pipeline,
        provider_type,
        current_provider
    )
    return message

# parse_think_tags is now imported from utils.text_utils


# =============================================================================
# UTILITY FUNCTIONS - Now imported from utils modules
# - parse_think_tags from utils.text_utils
# - system_health_check, format_health_report from utils.health
# - format_sources_info, _extract_all_documents_from_metadata, etc. from utils.formatting
# =============================================================================


# =============================================================================
# MAIN CHAT FUNCTION - Kaggle_Demo Style with FULL Streaming Progress
# =============================================================================

def clear_conversation():
    """Clear conversation history"""
    global manager, current_session

    current_session, history, text = clear_conversation_session(manager)
    return history, text
def chat_with_legal_rag(message, history, config_dict, show_thinking=True, show_sources=True, show_metadata=True):
    """
    Main chat function with conversational RAG - Now uses ConversationalRAGService

    This is a thin wrapper around ConversationalRAGService that handles Gradio-specific
    display formatting and progress updates.
    """
    if not message.strip():
        return history, ""

    try:
        # Ensure system is initialized
        if pipeline is None:
            initialize_system()

        # Create conversational service
        service = create_conversational_service(pipeline, manager, current_provider)

        # Track progress for Gradio display
        current_progress = []
        all_phase_metadata = {}
        streamed_answer = ""
        result = None

        def add_progress(msg):
            """Helper to add progress updates in Gradio 6.x message format"""
            current_progress.append(msg)
            progress_display = "\n".join([f"🔄 {m}" for m in current_progress])
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"**Mencari dan menganalisis...**\n\n{progress_display}"}
            ]

        # Process query with service
        for event in service.process_query(
            message,
            current_session,
            config_dict
        ):
            event_type = event.get('type')
            data = event.get('data', {})

            if event_type == 'progress':
                # Show progress update
                yield add_progress(data['message']), ""

            elif event_type == 'query_analysis':
                # Query analysis completed - progress already added by callback
                yield add_progress("Query analysis complete"), ""

            elif event_type == 'streaming_chunk':
                # Accumulate streamed text
                streamed_answer = data['accumulated']
                chunk_count = data['chunk_count']

                # Build streaming display
                progress_header = '<details open><summary>📋 <b>Proses Penelitian</b></summary>\n\n'
                progress_header += "\n".join([f"🔄 {m}" for m in current_progress])
                progress_header += '\n</details>\n\n'

                display_text = progress_header + f"**✍️ Generating answer...**\n\n{streamed_answer}"

                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": display_text}
                ], ""

            elif event_type == 'final_result':
                # Final result received
                result = data
                # Extract phase metadata from result
                all_phase_metadata = result.get('phase_metadata', result.get('all_retrieved_metadata', {}))
                break

            elif event_type == 'error':
                # Error occurred
                error_msg = data.get('error', 'Unknown error')
                error_display = f"❌ **Error:**\n\n{error_msg}"
                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": error_display}
                ], ""
                return

        # Format final output
        if result:
            final_output = ""
            response_text = result.get('answer', streamed_answer)

            # Get thinking content from result field first, then try parsing from answer
            thinking_content = result.get('thinking', '')
            if not thinking_content:
                thinking_content, response_text = parse_think_tags(response_text)

            # Add research process summary (what was done)
            if current_progress:
                final_output += '<details open><summary>📋 <b>Proses yang Sudah Dilakukan</b></summary>\n\n'
                for msg in current_progress:
                    final_output += f"✅ {msg}\n"
                final_output += '\n</details>\n\n---\n\n'

            # Add thinking section if available
            if show_thinking and thinking_content:
                final_output += (
                    '<details><summary>🧠 <b>Proses Berpikir (klik untuk melihat)</b></summary>\n\n'
                    + thinking_content +
                    '\n</details>\n\n'
                    + '---\n\n✅ **Jawaban:**\n\n'
                    + response_text
                )
            else:
                final_output += f"✅ **Jawaban:**\n\n{response_text}"

            # Add community clusters if available
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

            # Add collapsible sections
            collapsible_sections = []

            # Add sources
            if show_sources and result.get('citations'):
                citations = result['citations']
                if citations:
                    sources_info = format_sources_info(citations, config_dict)
                    collapsible_sections.append(
                        f'<details><summary>📖 <b>Sumber Hukum Utama ({len(citations)})</b></summary>\n\n{sources_info}\n</details>'
                    )

            # Add metadata
            if show_metadata and all_phase_metadata:
                metadata_info = format_retrieved_metadata(all_phase_metadata, config_dict)
                if metadata_info.strip():
                    collapsible_sections.append(
                        f'<details><summary>🔬 <b>Detail Proses Penelitian</b></summary>\n\n{metadata_info}\n</details>'
                    )

            # Add all retrieved documents
            if show_metadata:
                all_docs_formatted = format_all_documents(result, max_docs=50)
                if all_docs_formatted:
                    collapsible_sections.append(
                        f'<details><summary>📚 <b>Semua Dokumen yang Ditemukan</b></summary>\n\n{all_docs_formatted}\n</details>'
                    )

            # Combine all sections
            if collapsible_sections:
                final_output += "\n\n---\n\n" + "\n\n".join(collapsible_sections)

            # Update conversation history with full formatted output (so export includes everything)
            service.update_conversation(
                current_session,
                message,
                final_output,  # Save the complete formatted output, not just plain answer
                result
            )

            # Return final result
            final_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": final_output}
            ]

            yield final_history, ""

        else:
            # No result received
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "❌ No response received"}
            ], ""

    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback
        traceback.print_exc()

        error_display = f"❌ **Error:**\n\n{str(e)}"
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_display}
        ], ""
def clear_conversation():
    """Clear conversation history"""
    global manager, current_session

    current_session, history, text = clear_conversation_session(manager)
    return history, text


def get_system_info():
    """Get system information with dataset statistics"""
    try:
        # Get dataset statistics
        stats = get_dataset_statistics(dataset_loader)

        # Format system information
        return format_system_info(
            EMBEDDING_MODEL,
            RERANKER_MODEL,
            LLM_MODEL,
            EMBEDDING_DEVICE,
            LLM_DEVICE,
            current_provider,
            stats,
            initialization_complete
        )
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


