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

        # Build response based on display options
        response_parts = []

        # Add thinking process if available and enabled
        if show_thinking and result.get('thinking'):
            response_parts.append(f"🧠 **Proses Berpikir:**\n{result['thinking']}\n\n---\n")

        # Main answer
        response_parts.append(result['answer'])

        # Add sources if available and enabled
        if show_sources and result.get('sources'):
            sources = result['sources']
            if sources:
                source_text = "\n\n---\n📚 **Sumber:**\n"
                for i, src in enumerate(sources[:5], 1):
                    if isinstance(src, dict):
                        title = src.get('title', src.get('regulation', f'Source {i}'))
                        score = src.get('score', 0)
                        source_text += f"- {title} (skor: {score:.2f})\n"
                    else:
                        source_text += f"- {src}\n"
                response_parts.append(source_text)

        # Add metadata if enabled
        if show_metadata and result.get('metadata'):
            meta = result['metadata']
            meta_text = "\n\n---\n📊 **Metadata:**\n"
            if meta.get('query_type'):
                meta_text += f"- Tipe: {meta['query_type']}\n"
            if meta.get('processing_time'):
                meta_text += f"- Waktu: {meta['processing_time']:.2f}s\n"
            if meta.get('total_results'):
                meta_text += f"- Hasil: {meta['total_results']}\n"
            response_parts.append(meta_text)

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


def create_demo() -> gr.Blocks:
    """Create Gradio demo interface with all features"""

    with gr.Blocks(
        title="Indonesian Legal RAG System",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; padding: 20px; }
        """
    ) as demo:

        gr.Markdown(
            """
            # Indonesian Legal RAG System
            ### Sistem Konsultasi Hukum Indonesia

            Tanyakan pertanyaan tentang hukum dan peraturan Indonesia.
            Supports multiple LLM providers and local inference.
            """
        )

        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Percakapan",
                    height=450,
                    show_copy_button=True
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Pertanyaan Anda",
                        placeholder="Ketik pertanyaan hukum Anda di sini...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Kirim", variant="primary", scale=1)

            # Settings panel
            with gr.Column(scale=1):
                with gr.Accordion("Display Options", open=True):
                    show_thinking = gr.Checkbox(label="Show Thinking Process", value=True)
                    show_sources = gr.Checkbox(label="Show Sources", value=True)
                    show_metadata = gr.Checkbox(label="Show Metadata", value=False)

                with gr.Accordion("Provider Settings", open=True):
                    provider_dropdown = gr.Dropdown(
                        choices=['local', 'openai', 'anthropic', 'google', 'openrouter'],
                        value='local',
                        label="LLM Provider"
                    )
                    provider_btn = gr.Button("Switch Provider")
                    provider_status = gr.Textbox(label="Provider Status", interactive=False)

                with gr.Accordion("Session", open=True):
                    clear_btn = gr.Button("New Session", variant="secondary")
                    status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("Export", open=False):
                    export_format = gr.Radio(
                        choices=["Markdown", "JSON", "HTML"],
                        value="Markdown",
                        label="Format"
                    )
                    export_btn = gr.Button("Export")
                    export_status = gr.Textbox(label="Export Result", interactive=False)

                with gr.Accordion("Document Upload", open=False):
                    file_upload = gr.File(
                        label="Upload Document",
                        file_types=[".pdf", ".docx", ".doc", ".txt"]
                    )
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)

                with gr.Accordion("Info", open=False):
                    info_btn = gr.Button("Refresh Info")
                    session_info = gr.Textbox(label="Session Info", interactive=False, lines=7)

        # Example questions
        gr.Markdown("### Contoh Pertanyaan")
        examples = gr.Examples(
            examples=[
                "Apa itu UU Ketenagakerjaan?",
                "Apa sanksi pelanggaran UU Perlindungan Konsumen?",
                "Bagaimana prosedur PHK menurut hukum?",
                "Apa hak karyawan kontrak?",
                "Apa definisi konsumen menurut UU?"
            ],
            inputs=msg
        )

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

        # Initialize on load
        demo.load(
            initialize_system,
            outputs=[status]
        )

    return demo


def launch_app(share: bool = False, server_port: int = 7860):
    """Launch Gradio app"""
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share
    )


if __name__ == "__main__":
    launch_app()
