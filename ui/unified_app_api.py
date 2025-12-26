"""
Unified API-Based UI - Indonesian Legal RAG System

Production UI that connects to the FastAPI backend for all operations.
Exact replica of gradio_app.py interface but using API calls.

This is an API-based version of the original gradio_app.py that:
- Uses HTTP API calls instead of direct pipeline imports
- Maintains the exact same UI layout and styling
- Works in production Docker/Kaggle environments
"""

import gradio as gr
import os
import sys
import time
import json
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.services.api_client import LegalRAGAPIClient, Document, HealthStatus

# =============================================================================
# GLOBAL STATE
# =============================================================================

api_client: Optional[LegalRAGAPIClient] = None
current_session: Optional[str] = None
authenticated_user: Optional[str] = None

# Demo users for dummy auth
DEMO_USERS = {
    "demo": "demo123",
    "admin": "admin123"
}

# Default configuration matching gradio_app.py
DEFAULT_CONFIG = {
    'api_url': os.environ.get('LEGAL_API_URL', 'http://127.0.0.1:8000/api/v1'),
    'api_key': os.environ.get('LEGAL_API_KEY', 'your-api-key-here'),
    'final_top_k': 3,
    'temperature': 0.7,
    'max_new_tokens': 2048,
    'research_team_size': 4,
    'enable_cross_validation': True,
    'enable_devil_advocate': True,
    'consensus_threshold': 0.6,
    'top_p': 1.0,
    'top_k': 20,
    'min_p': 0.1,
    'thinking_mode': 'low'
}

# Example queries from gradio_app.py
EXAMPLE_QUERIES = [
    "Apakah ada pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?",
    "Apakah terdapat mekanisme pengawasan terhadap penyimpanan uang negara agar terhindar dari penyalahgunaan atau kebocoran keuangan?",
    "Bagaimana mekanisme hukum untuk memperoleh izin resmi bagi pihak yang menjalankan usaha sebagai pengusaha pabrik, penyimpanan, importir, penyalur, maupun penjual eceran barang kena cukai?",
    "Apakah terdapat kewajiban pemerintah untuk menyediakan dana khusus bagi penyuluhan, atau dapat melibatkan sumber pendanaan alternatif seperti swasta dan masyarakat?",
    "Bagaimana prosedur hukum yang harus ditempuh sebelum sanksi denda administrasi di bidang cukai dapat dikenakan kepada pelaku usaha?",
    "Bagaimana sistem perencanaan kas disusun agar mampu mengantisipasi kebutuhan mendesak negara/daerah tanpa mengganggu stabilitas fiskal?",
    "syarat dan prosedur perceraian menurut hukum Indonesia",
    "hak dan kewajiban pekerja dalam UU Ketenagakerjaan"
]

# Test questions for conversational test
TEST_QUESTIONS = [
    "Apakah terdapat pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?",
    "Berdasarkan PP No. 41 Tahun 2009, sebutkan jenis-jenis tunjangan yang diatur di dalamnya.",
    "Masih merujuk pada PP No. 41 Tahun 2009, jelaskan perbedaan kriteria penerima, besaran, dan sumber pendanaan antara Tunjangan Khusus dan Tunjangan Kehormatan Profesor",
    "Ganti topik. Jelaskan secara singkat pengertian kawasan pabean menurut Undang-Undang Kepabeanan.",
    "Berdasarkan Undang-Undang Kepabeanan tersebut, jelaskan sanksi pidana bagi pihak yang dengan sengaja salah memberitahukan jenis dan jumlah barang impor sehingga merugikan negara.",
    "Sekarang beralih ke UU No. 13 Tahun 2003. Jelaskan secara umum ruang lingkup dan pokok bahasan undang-undang tersebut.",
    "Apa yang diatur dalam Pasal 1 UU No. 13 Tahun 2003?",
    "Terakhir, jelaskan secara ringkas PP No. 8 Tahun 2007, termasuk fokus pengaturannya."
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def initialize_api():
    """Initialize API client"""
    global api_client
    
    if api_client is None:
        api_client = LegalRAGAPIClient(
            base_url=DEFAULT_CONFIG['api_url'],
            api_key=DEFAULT_CONFIG['api_key']
        )
    
    try:
        health = api_client.health_check()
        if health.ready:
            return f"✅ Initialized with API. Pipeline ready."
        elif health.healthy:
            return f"⚠️ Connected. Pipeline loading: {health.message}"
        else:
            return f"❌ Connection failed: {health.message}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def dummy_login(username: str, password: str):
    """Dummy authentication"""
    global authenticated_user
    
    if not username or not password:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "⚠️ Masukkan username dan password"
        )
    
    if username in DEMO_USERS and DEMO_USERS[username] == password:
        authenticated_user = username
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            ""
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "❌ Username atau password salah"
        )


def dummy_logout():
    """Logout"""
    global authenticated_user
    authenticated_user = None
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        []
    )


def clear_conversation():
    """Clear conversation history"""
    global current_session
    current_session = None
    return [], ""


def get_system_info():
    """Get system information via API"""
    if api_client is None:
        return "❌ API not connected"
    
    try:
        health = api_client.health_check()
        return f"""
## 📊 System Information

| Component | Status |
|-----------|--------|
| **API Connection** | {"✅ Connected" if health.healthy else "❌ Disconnected"} |
| **Pipeline Ready** | {"✅ Yes" if health.ready else "⏳ Loading"} |
| **Provider** | API-based |
| **Message** | {health.message} |

---

### 🔧 Current Configuration

- **API URL**: {DEFAULT_CONFIG['api_url']}
- **Final Top K**: {DEFAULT_CONFIG['final_top_k']}
- **Temperature**: {DEFAULT_CONFIG['temperature']}
- **Max Tokens**: {DEFAULT_CONFIG['max_new_tokens']}
- **Thinking Mode**: {DEFAULT_CONFIG['thinking_mode']}
"""
    except Exception as e:
        return f"❌ Error: {str(e)}"


def format_health_report():
    """Format health check report"""
    if api_client is None:
        return "❌ API not connected"
    
    try:
        health = api_client.health_check()
        status_icon = "✅" if health.ready else ("⚠️" if health.healthy else "❌")
        
        return f"""
## 🏥 Health Check Report

**Status**: {status_icon} {health.message}

| Check | Result |
|-------|--------|
| API Healthy | {"✅ Pass" if health.healthy else "❌ Fail"} |
| Pipeline Ready | {"✅ Pass" if health.ready else "⏳ Loading"} |
"""
    except Exception as e:
        return f"❌ Health check failed: {str(e)}"


# =============================================================================
# MAIN CHAT FUNCTION - API Based
# =============================================================================

def chat_with_legal_rag(message, history, config_dict, show_thinking=True, show_sources=True, show_metadata=True):
    """Main chat function - uses API for processing"""
    if not message.strip():
        return history, ""
    
    global api_client, current_session
    
    # Initialize if needed
    if api_client is None:
        initialize_api()
    
    if api_client is None:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "❌ API not connected. Please refresh the page."}
        ]
        yield history, ""
        return
    
    # Add user message
    history = history + [{"role": "user", "content": message}]
    
    # Show progress
    current_progress = []
    def add_progress(msg):
        current_progress.append(msg)
        progress_display = "\n".join([f"🔄 {m}" for m in current_progress])
        return history + [
            {"role": "assistant", "content": f"**Mencari dan menganalisis...**\n\n{progress_display}"}
        ]
    
    yield add_progress("Menghubungkan ke server..."), ""
    
    try:
        # Get thinking mode from config
        thinking_mode = config_dict.get('thinking_mode', 'low')
        if isinstance(thinking_mode, str):
            thinking_mode = thinking_mode.lower()
        
        yield add_progress("Menganalisis query..."), ""
        
        # Stream response from API
        thinking_buffer = ""
        answer_buffer = ""
        sources = []
        legal_refs = ""
        
        yield add_progress("Melakukan penelitian..."), ""
        
        for chunk in api_client.chat_stream(
            query=message,
            session_id=current_session,
            thinking_level=thinking_mode
        ):
            chunk_type = chunk.get('type', '')
            content = chunk.get('content', '')
            
            if chunk_type == 'progress':
                # Progress update
                msg = chunk.get('message', content)
                yield add_progress(msg), ""
            
            elif chunk_type == 'thinking':
                # Thinking process
                thinking_buffer += content
                if show_thinking:
                    display = f"💭 **Proses Berpikir:**\n\n{thinking_buffer}"
                    if answer_buffer:
                        display += f"\n\n---\n\n**Jawaban:**\n\n{answer_buffer}"
                    history_copy = history.copy()
                    history_copy.append({"role": "assistant", "content": display})
                    yield history_copy, ""
            
            elif chunk_type == 'chunk':
                # Content chunk (API sends 'chunk', not 'content')
                answer_buffer += content
                if show_thinking and thinking_buffer:
                    display = f"💭 **Proses Berpikir:**\n\n{thinking_buffer}\n\n---\n\n**Jawaban:**\n\n{answer_buffer}"
                else:
                    display = answer_buffer
                history_copy = history.copy()
                history_copy.append({"role": "assistant", "content": display})
                yield history_copy, ""
            
            elif chunk_type == 'done':
                # Final message (API sends 'done', not 'complete')
                answer_buffer = chunk.get('answer', answer_buffer)
                legal_refs = chunk.get('legal_references', '')
        
        # Build final response
        final_content = ""
        
        if show_thinking and thinking_buffer:
            final_content += f"💭 **Proses Berpikir:**\n\n{thinking_buffer}\n\n---\n\n"
        
        final_content += f"**Jawaban:**\n\n{answer_buffer}"
        
        # Add legal references from API
        if show_sources and legal_refs:
            final_content += f"\n\n---\n\n### 📚 Referensi Hukum\n\n{legal_refs}"
        
        history.append({"role": "assistant", "content": final_content})
        yield history, ""
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_display = f"❌ **Error:**\n\n{str(e)}"
        history.append({"role": "assistant", "content": error_display})
        yield history, ""


def run_conversational_test(history, config_dict, show_thinking, show_sources, show_metadata):
    """Run conversational test - auto-feed questions"""
    
    # Initial message
    initial_msg = {
        "role": "assistant",
        "content": f"🧪 **Starting Conversational Test (8 Questions)**\n\nAuto-feeding questions through API...\n\n**Questions to test:**\n" +
                   "\n".join([f"{i+1}. {q[:80]}..." for i, q in enumerate(TEST_QUESTIONS)])
    }
    history = history + [initial_msg]
    yield history, ""
    
    # Process each question
    for i, question in enumerate(TEST_QUESTIONS, 1):
        # Add progress
        progress_msg = {
            "role": "assistant",
            "content": f"🔄 **Question {i}/8**\n\nProcessing: _{question[:100]}..._"
        }
        history = history + [progress_msg]
        yield history, ""
        
        # Remove progress
        history = history[:-1]
        
        # Call chat
        for updated_history, cleared_input in chat_with_legal_rag(
            question, history, config_dict, show_thinking, show_sources, show_metadata
        ):
            yield updated_history, cleared_input
        
        history = updated_history
    
    # Completion
    completion_msg = {
        "role": "assistant",
        "content": f"✅ **Conversational Test Complete**\n\nSuccessfully processed all {len(TEST_QUESTIONS)} questions through the API."
    }
    history = history + [completion_msg]
    yield history, ""


def update_config(*args):
    """Update configuration from UI inputs"""
    try:
        new_config = {
            'final_top_k': int(args[0]),
            'temperature': float(args[1]),
            'max_new_tokens': int(args[2]),
            'thinking_mode': str(args[3]).lower(),
            'research_team_size': int(args[4]),
            'enable_cross_validation': bool(args[5]),
            'enable_devil_advocate': bool(args[6]),
            'consensus_threshold': float(args[7]),
            'top_p': float(args[8]),
            'top_k': int(args[9]),
            'min_p': float(args[10]),
        }
        return new_config
    except Exception as e:
        print(f"Error updating config: {e}")
        return DEFAULT_CONFIG


def reset_to_defaults():
    """Reset configuration to defaults"""
    return (
        DEFAULT_CONFIG['final_top_k'],
        DEFAULT_CONFIG['temperature'],
        DEFAULT_CONFIG['max_new_tokens'],
        DEFAULT_CONFIG['thinking_mode'].capitalize(),
        DEFAULT_CONFIG['research_team_size'],
        DEFAULT_CONFIG['enable_cross_validation'],
        DEFAULT_CONFIG['enable_devil_advocate'],
        DEFAULT_CONFIG['consensus_threshold'],
        DEFAULT_CONFIG['top_p'],
        DEFAULT_CONFIG['top_k'],
        DEFAULT_CONFIG['min_p'],
    )


def export_conversation_handler(export_format, history):
    """Handle export button click"""
    try:
        if not history:
            return "No conversation to export.", None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "JSON":
            content = json.dumps(history, ensure_ascii=False, indent=2)
            filename = f"legal_consultation_{timestamp}.json"
        elif export_format == "Markdown":
            lines = ["# Legal Consultation Export\n\n"]
            for msg in history:
                role = msg.get('role', 'unknown')
                content_text = msg.get('content', '')
                if role == 'user':
                    lines.append(f"## 👤 User\n\n{content_text}\n\n")
                else:
                    lines.append(f"## 🤖 Assistant\n\n{content_text}\n\n")
            content = "\n".join(lines)
            filename = f"legal_consultation_{timestamp}.md"
        else:  # HTML
            lines = ["<html><body><h1>Legal Consultation</h1>"]
            for msg in history:
                role = msg.get('role', 'unknown')
                content_text = msg.get('content', '').replace('\n', '<br>')
                if role == 'user':
                    lines.append(f"<h3>👤 User</h3><p>{content_text}</p>")
                else:
                    lines.append(f"<h3>🤖 Assistant</h3><p>{content_text}</p>")
            lines.append("</body></html>")
            content = "\n".join(lines)
            filename = f"legal_consultation_{timestamp}.html"
        
        # Save to temp file
        import tempfile
        ext = {'JSON': 'json', 'Markdown': 'md', 'HTML': 'html'}[export_format]
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        return content[:5000] + "..." if len(content) > 5000 else content, temp_path
        
    except Exception as e:
        return f"Export failed: {str(e)}", None


# =============================================================================
# GRADIO INTERFACE - Exact replica of gradio_app.py
# =============================================================================

def create_gradio_interface():
    """Create enhanced Gradio interface - exact match to gradio_app.py"""
    
    with gr.Blocks(title="Enhanced Indonesian Legal Assistant") as interface:
        
        # =====================================================================
        # LOGIN PANEL (Added for production)
        # =====================================================================
        with gr.Column(visible=True) as login_panel:
            gr.HTML("""
            <div style="text-align: center; padding: 50px 20px; max-width: 400px; margin: 0 auto;">
                <h1 style="color: #1e3a5f;">🏛️ Indonesian Legal Assistant</h1>
                <p style="color: #666;">Login to access the system</p>
            </div>
            """)
            with gr.Row():
                gr.Column(scale=1)
                with gr.Column(scale=2):
                    gr.Markdown("### 🔐 Login")
                    gr.Markdown("**Demo:** `demo` / `demo123` or `admin` / `admin123`")
                    username_input = gr.Textbox(label="Username")
                    password_input = gr.Textbox(label="Password", type="password")
                    login_btn = gr.Button("🔓 Login", variant="primary")
                    login_error = gr.Markdown("")
                gr.Column(scale=1)
        
        # =====================================================================
        # MAIN APP (Hidden until login)
        # =====================================================================
        with gr.Column(visible=False) as main_app:
            
            # Logout button at top
            with gr.Row():
                gr.Column(scale=9)
                logout_btn = gr.Button("🚪 Logout", size="sm", scale=1)
            
            with gr.Tabs():
                
                # =============================================================
                # CHAT TAB - Exactly like gradio_app.py
                # =============================================================
                with gr.TabItem("💬 Konsultasi Hukum", id="chat"):
                    with gr.Column(elem_classes="main-chat-area"):
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
                                    examples=EXAMPLE_QUERIES,
                                    inputs=msg_input,
                                    examples_per_page=2,
                                    label=""
                                )
                
                # =============================================================
                # SETTINGS TAB - Exactly like gradio_app.py
                # =============================================================
                with gr.TabItem("⚙️ Pengaturan Sistem", id="settings"):
                    with gr.Row():
                        with gr.Column():
                            # Basic Settings
                            with gr.Group(elem_classes="settings-panel"):
                                gr.Markdown("#### 🎯 Basic Settings")
                                final_top_k = gr.Slider(1, 10, value=3, step=1, label="Final Top K Results")
                                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="LLM Temperature")
                                max_new_tokens = gr.Slider(512, 4096, value=2048, step=256, label="Max New Tokens")
                                thinking_mode = gr.Radio(
                                    choices=["Low", "Medium", "High"],
                                    value="Low",
                                    label="Thinking Level",
                                    info="Higher levels provide deeper legal analysis but take more time."
                                )
                            
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
                            
                            # Connect health check
                            health_check_btn.click(format_health_report, outputs=health_report_output)
                            
                            with gr.Group(elem_classes="settings-panel"):
                                gr.Markdown("#### 🧪 Production Test Runners")
                                gr.Markdown("Run comprehensive integration tests and see results in the chat tab.")
                                test_conversational_btn = gr.Button("🔬 Run Conversational Test (8 Questions)", variant="primary")
                
                # =============================================================
                # EXPORT TAB - Exactly like gradio_app.py
                # =============================================================
                with gr.TabItem("📥 Export Conversation", id="export"):
                    with gr.Column(elem_classes="main-chat-area"):
                        with gr.Row():
                            export_format = gr.Radio(
                                choices=["Markdown", "JSON", "HTML"],
                                value="Markdown",
                                label="Export Format"
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
            
            # Hidden config state
            config_state = gr.State(DEFAULT_CONFIG)
            
            # Config inputs list
            config_inputs = [
                final_top_k, temperature, max_new_tokens, thinking_mode,
                research_team_size, enable_cross_validation, enable_devil_advocate, consensus_threshold,
                top_p, top_k, min_p
            ]
            
            # Connect config updates
            for input_component in config_inputs:
                try:
                    input_component.change(
                        update_config,
                        inputs=config_inputs,
                        outputs=config_state
                    )
                except Exception as e:
                    print(f"Error connecting config: {e}")
            
            # Reset button
            reset_defaults_btn.click(reset_to_defaults, outputs=config_inputs)
            
            # System info
            system_info_btn.click(get_system_info, outputs=system_info_output)
            
            # Chat functionality
            msg_input.submit(
                chat_with_legal_rag,
                inputs=[msg_input, chatbot, config_state, show_thinking, show_sources, show_metadata],
                outputs=[chatbot, msg_input]
            )
            
            # Test runner
            test_conversational_btn.click(
                run_conversational_test,
                inputs=[chatbot, config_state, show_thinking, show_sources, show_metadata],
                outputs=[chatbot, msg_input]
            )
            
            # Export
            export_button.click(
                export_conversation_handler,
                inputs=[export_format, chatbot],
                outputs=[export_output, download_file]
            )
        
        # =====================================================================
        # AUTH HANDLERS
        # =====================================================================
        login_btn.click(
            fn=dummy_login,
            inputs=[username_input, password_input],
            outputs=[login_panel, main_app, login_error]
        )
        
        logout_btn.click(
            fn=dummy_logout,
            outputs=[login_panel, main_app, chatbot]
        )
    
    interface.queue()
    return interface


# =============================================================================
# LAUNCHER
# =============================================================================

def launch_app(share: bool = False, server_port: int = 7860, server_name: str = "0.0.0.0"):
    """Launch Gradio app"""
    print("\n" + "=" * 60)
    print("🏛️ LEGAL RAG INDONESIA - UNIFIED API UI")
    print("=" * 60)
    
    # Initialize API
    result = initialize_api()
    print(f"API Status: {result}")
    
    print("=" * 60 + "\n")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share
    )


# Alias for backward compatibility
launch_unified_app = launch_app


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    launch_app(share=share, server_port=server_port)
