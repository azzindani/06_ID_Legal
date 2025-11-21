"""
Gradio Interface - Indonesian Legal RAG System

Web-based chat interface for legal consultation.
"""

import gradio as gr
import sys
import os
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RAGPipeline
from conversation import ConversationManager, MarkdownExporter, JSONExporter, HTMLExporter
from logger_utils import get_logger

logger = get_logger(__name__)

# Global instances
pipeline: Optional[RAGPipeline] = None
manager: Optional[ConversationManager] = None
current_session: Optional[str] = None


def initialize_system():
    """Initialize the RAG system"""
    global pipeline, manager, current_session

    if pipeline is None:
        logger.info("Initializing RAG system...")
        pipeline = RAGPipeline()
        if not pipeline.initialize():
            raise RuntimeError("Failed to initialize pipeline")
        logger.info("Pipeline initialized")

    if manager is None:
        manager = ConversationManager()

    if current_session is None:
        current_session = manager.start_session()

    return "System initialized successfully!"


def chat(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Process chat message and return response

    Args:
        message: User message
        history: Chat history

    Returns:
        Response and updated history
    """
    global pipeline, manager, current_session

    if not message.strip():
        return "", history

    try:
        # Initialize if needed
        if pipeline is None:
            initialize_system()

        # Get conversation context
        context = manager.get_context_for_query(current_session) if current_session else None

        # Generate response
        result = pipeline.query(message, conversation_history=context, stream=False)
        response = result['answer']

        # Save to conversation history
        if current_session:
            manager.add_turn(
                session_id=current_session,
                query=message,
                answer=response,
                metadata=result.get('metadata')
            )

        # Update history
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
    global manager, current_session

    if not manager or not current_session:
        return "No active session"

    summary = manager.get_session_summary(current_session)
    if not summary:
        return "Session not found"

    return f"""Session ID: {summary['session_id']}
Total Turns: {summary['total_turns']}
Total Tokens: {summary['total_tokens']}
Total Time: {summary['total_time']:.2f}s"""


def create_demo() -> gr.Blocks:
    """Create Gradio demo interface"""

    with gr.Blocks(
        title="Indonesian Legal RAG System",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 900px; margin: auto; }
        .header { text-align: center; padding: 20px; }
        """
    ) as demo:

        gr.Markdown(
            """
            # Indonesian Legal RAG System
            ### Sistem Konsultasi Hukum Indonesia

            Tanyakan pertanyaan tentang hukum dan peraturan Indonesia.
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Percakapan",
                    height=500,
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

            with gr.Column(scale=1):
                gr.Markdown("### Aksi")

                clear_btn = gr.Button("Sesi Baru", variant="secondary")
                status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Ekspor")
                export_format = gr.Radio(
                    choices=["Markdown", "JSON", "HTML"],
                    value="Markdown",
                    label="Format"
                )
                export_btn = gr.Button("Ekspor")
                export_status = gr.Textbox(label="Hasil Ekspor", interactive=False)

                gr.Markdown("### Info Sesi")
                info_btn = gr.Button("Refresh Info")
                session_info = gr.Textbox(label="Info", interactive=False, lines=5)

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
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )

        msg.submit(
            chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
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
