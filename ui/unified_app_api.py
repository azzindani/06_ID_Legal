"""
Unified Legal RAG UI - API-Based Gradio Interface

Production-ready unified interface with tabs for Search, Chat, Settings, and Status.
Communicates with the FastAPI backend via HTTP API.

File: ui/unified_app_api.py

Run:
    python -m ui.unified_app_api
"""

import os
import sys
import time
import gradio as gr
from typing import Optional, List, Dict, Any, Generator

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.services.api_client import LegalRAGAPIClient, create_api_client, Document

# =============================================================================
# GLOBAL STATE
# =============================================================================

api_client: Optional[LegalRAGAPIClient] = None
current_session_id: Optional[str] = None
authenticated_user: Optional[str] = None

# Demo users for dummy auth (in production, use proper auth system)
DEMO_USERS = {
    'demo': 'demo123',
    'admin': 'admin123',
    'user': 'password'
}

# Default configuration
DEFAULT_CONFIG = {
    'api_url': os.environ.get('LEGAL_API_URL', 'http://127.0.0.1:8000/api/v1'),
    'api_key': os.environ.get('LEGAL_API_KEY', 'test_key'),
    'thinking_level': 'low',
    'team_size': 3,
    'top_k': 5,
    'show_thinking': True,
    'show_sources': True
}


# =============================================================================
# CSS STYLING
# =============================================================================

UNIFIED_CSS = """
/* Base styling */
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
    color: white;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}

.app-header h1 {
    margin: 0;
    font-size: 1.8rem;
}

.app-header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
}

/* Status indicator */
.status-connected {
    color: #48bb78;
    font-weight: bold;
}

.status-disconnected {
    color: #f56565;
    font-weight: bold;
}

/* Document cards */
.doc-card {
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f7fafc;
}

.doc-card h4 {
    margin: 0 0 0.5rem 0;
    color: #1e3a5f;
}

/* Legal disclaimer */
.legal-disclaimer {
    background: #fff5f5;
    border: 1px solid #feb2b2;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin: 1rem 0;
    font-size: 0.85rem;
    color: #c53030;
}

/* Tab styling */
.tab-nav button {
    font-size: 1rem !important;
}
"""

# Legal disclaimer text
LEGAL_DISCLAIMER = """
⚠️ **Pemberitahuan Penting**: Informasi yang diberikan bersifat edukatif dan bukan merupakan nasihat hukum resmi. 
Untuk kepentingan hukum, silakan konsultasikan dengan advokat profesional.
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def initialize_client(api_url: str, api_key: str) -> str:
    """Initialize API client"""
    global api_client
    
    try:
        api_client = LegalRAGAPIClient(base_url=api_url, api_key=api_key)
        health = api_client.health_check()
        
        if health.ready:
            return f"✅ Connected | Pipeline ready"
        elif health.healthy:
            return f"⚠️ Connected | Pipeline loading: {health.message}"
        else:
            return f"❌ Connection failed: {health.message}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def dummy_login(username: str, password: str):
    """
    Dummy authentication (for demo purposes)
    In production, replace with proper OAuth/JWT auth
    """
    global authenticated_user
    
    if not username or not password:
        return (
            gr.update(visible=True),   # login panel stays visible
            gr.update(visible=False),  # main app hidden
            "⚠️ Masukkan username dan password"
        )
    
    if username in DEMO_USERS and DEMO_USERS[username] == password:
        authenticated_user = username
        return (
            gr.update(visible=False),  # hide login
            gr.update(visible=True),   # show main app
            ""
        )
    else:
        return (
            gr.update(visible=True),   # login stays visible
            gr.update(visible=False),  # main app hidden
            "❌ Username atau password salah"
        )


def dummy_logout():
    """Logout and return to login screen"""
    global authenticated_user
    authenticated_user = None
    return (
        gr.update(visible=True),   # show login
        gr.update(visible=False),  # hide main app
        []  # clear chat history
    )


def format_document_card(doc: Document, index: int) -> str:
    """Format a document as markdown card"""
    return f"""
### [{index}] {doc.regulation_type} No. {doc.regulation_number}/{doc.year}

**Tentang**: {doc.about}

**Skor Relevansi**: {doc.score:.4f}

{f"**Bab/Pasal**: {doc.chapter or ''} {doc.article or ''}" if doc.chapter or doc.article else ""}

{f"**Preview**: {doc.content_preview[:300]}..." if doc.content_preview else ""}

---
"""


def format_search_results(result: Dict) -> str:
    """Format search results as markdown"""
    docs = result.get('documents', [])
    
    if not docs:
        return "Tidak ditemukan dokumen yang relevan."
    
    output = f"## Hasil Pencarian\n\n"
    output += f"**Query**: {result.get('query', '')}\n\n"
    output += f"**Ditemukan**: {len(docs)} dokumen dalam {result.get('search_time', 0):.2f} detik\n\n"
    output += "---\n\n"
    
    for i, doc in enumerate(docs, 1):
        output += format_document_card(doc, i)
    
    return output


# =============================================================================
# SEARCH TAB FUNCTIONS
# =============================================================================

def search_documents(query: str, top_k: int, min_score: float) -> str:
    """Search for documents via API"""
    global api_client
    
    if not api_client:
        return "❌ API not connected. Please configure in Settings tab."
    
    if not query.strip():
        return "⚠️ Masukkan query pencarian."
    
    try:
        result = api_client.retrieve(query, top_k=int(top_k), min_score=min_score)
        return format_search_results(result)
    except Exception as e:
        return f"❌ Error: {str(e)}"


# =============================================================================
# CHAT TAB FUNCTIONS
# =============================================================================

def chat_with_api(
    message: str,
    history: List[Dict],
    thinking_level: str,
    show_thinking: bool,
    session_id: str
) -> Generator:
    """Stream chat response from API"""
    global api_client, current_session_id
    
    if not api_client:
        yield history + [{"role": "assistant", "content": "❌ API not connected. Configure in Settings."}]
        return
    
    if not message.strip():
        yield history + [{"role": "assistant", "content": "⚠️ Masukkan pertanyaan."}]
        return
    
    # Ensure session exists
    if not session_id:
        session_id = api_client.create_session()
        current_session_id = session_id
    
    # Add user message to history
    history = history + [{"role": "user", "content": message}]
    
    # Start with empty assistant message
    assistant_msg = ""
    thinking_content = ""
    in_thinking = False
    
    try:
        for event in api_client.chat_stream(
            query=message,
            session_id=session_id,
            thinking_level=thinking_level
        ):
            event_type = event.get('type', '')
            
            if event_type == 'progress':
                # Show progress in assistant message
                progress_msg = event.get('message', '')
                yield history + [{"role": "assistant", "content": f"⏳ {progress_msg}..."}]
                
            elif event_type == 'thinking':
                # Accumulate thinking content
                if show_thinking:
                    thinking_content += event.get('content', '')
                    display = f"🧠 **Proses Berpikir:**\n\n{thinking_content}\n\n---\n\n{assistant_msg}"
                    yield history + [{"role": "assistant", "content": display}]
                    
            elif event_type == 'chunk':
                # Accumulate answer content
                assistant_msg += event.get('content', '')
                if show_thinking and thinking_content:
                    display = f"🧠 **Proses Berpikir:**\n\n{thinking_content}\n\n---\n\n{assistant_msg}"
                else:
                    display = assistant_msg
                yield history + [{"role": "assistant", "content": display}]
                
            elif event_type == 'done':
                # Final response with metadata
                final_answer = event.get('answer', assistant_msg)
                references = event.get('legal_references', '')
                
                # Build final display
                final_display = ""
                if show_thinking and thinking_content:
                    final_display += f"🧠 **Proses Berpikir:**\n\n{thinking_content}\n\n---\n\n"
                
                final_display += final_answer
                
                if references:
                    final_display += f"\n\n---\n\n📚 **Referensi Hukum:**\n\n{references}"
                
                # Add legal disclaimer
                final_display += f"\n\n{LEGAL_DISCLAIMER}"
                
                yield history + [{"role": "assistant", "content": final_display}]
                return
                
    except Exception as e:
        yield history + [{"role": "assistant", "content": f"❌ Error: {str(e)}"}]


def clear_chat():
    """Clear chat history"""
    return []


def new_session():
    """Create new session"""
    global api_client, current_session_id
    
    if api_client:
        current_session_id = api_client.create_session()
        return current_session_id, []
    return "", []


# =============================================================================
# SETTINGS TAB FUNCTIONS
# =============================================================================

def apply_settings(api_url: str, api_key: str) -> str:
    """Apply API settings"""
    return initialize_client(api_url, api_key)


def get_sessions_list() -> str:
    """Get list of sessions"""
    global api_client
    
    if not api_client:
        return "API not connected"
    
    try:
        sessions = api_client.list_sessions()
        if not sessions:
            return "Tidak ada sesi aktif"
        
        output = "## Sesi Aktif\n\n"
        for s in sessions:
            output += f"- **{s.get('session_id', 'N/A')}** | {s.get('total_turns', 0)} turns | {s.get('total_time', 0):.1f}s\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def export_current_session(format: str) -> str:
    """Export current session"""
    global api_client, current_session_id
    
    if not api_client or not current_session_id:
        return "Tidak ada sesi aktif"
    
    try:
        return api_client.export_session(current_session_id, format)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# STATUS TAB FUNCTIONS
# =============================================================================

def check_status() -> str:
    """Check API status"""
    global api_client
    
    if not api_client:
        return "❌ API not configured"
    
    try:
        health = api_client.health_check()
        
        output = "## Status API\n\n"
        output += f"**Health**: {'✅ Healthy' if health.healthy else '❌ Unhealthy'}\n\n"
        output += f"**Ready**: {'✅ Ready' if health.ready else '⏳ Loading'}\n\n"
        output += f"**Message**: {health.message}\n\n"
        
        if health.components:
            output += "### Komponen\n\n"
            for comp, status in health.components.items():
                output += f"- {comp}: {'✅' if status else '❌'}\n"
        
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_unified_interface():
    """Create unified Gradio interface with tabs and auth"""
    
    with gr.Blocks(css=UNIFIED_CSS, title="Legal RAG Indonesia") as app:
        
        # =====================================================================
        # LOGIN PANEL (visible by default)
        # =====================================================================
        with gr.Column(visible=True) as login_panel:
            gr.HTML("""
            <div class="app-header">
                <h1>🏛️ Asisten Hukum Indonesia</h1>
                <p>Silakan login untuk mengakses sistem</p>
            </div>
            """)
            
            gr.Markdown("### 🔐 Login")
            gr.Markdown("""
            **Demo Accounts:**
            - Username: `demo` / Password: `demo123`
            - Username: `admin` / Password: `admin123`
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    username_input = gr.Textbox(label="Username", placeholder="Masukkan username")
                    password_input = gr.Textbox(label="Password", type="password", placeholder="Masukkan password")
                    login_btn = gr.Button("🔓 Login", variant="primary")
                    login_error = gr.Markdown("")
                with gr.Column(scale=1):
                    pass
        
        # =====================================================================
        # MAIN APP (hidden until login)
        # =====================================================================
        with gr.Column(visible=False) as main_app:
            
            # Header with logout
            with gr.Row():
                with gr.Column(scale=4):
                    gr.HTML("""
                    <div class="app-header">
                        <h1>🏛️ Asisten Hukum Indonesia</h1>
                        <p>Pencarian dan Konsultasi Regulasi berbasis AI</p>
                    </div>
                    """)
                with gr.Column(scale=1):
                    logout_btn = gr.Button("🚪 Logout")
            
            # Connection status
            status_display = gr.Markdown("⏳ Connecting to API...", elem_classes=["status-display"])
            
            with gr.Tabs():
                
                # =================================================================
                # SEARCH TAB
                # =================================================================
                with gr.Tab("🔍 Pencarian", id="search"):
                    gr.Markdown("### Cari Dokumen Hukum")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            search_query = gr.Textbox(
                                label="Query Pencarian",
                                placeholder="Contoh: Syarat pendirian PT...",
                                lines=2
                            )
                        with gr.Column(scale=1):
                            search_top_k = gr.Slider(1, 20, value=5, step=1, label="Jumlah Hasil")
                            search_min_score = gr.Slider(0, 1, value=0.0, step=0.1, label="Skor Minimum")
                    
                    search_btn = gr.Button("🔍 Cari Dokumen", variant="primary")
                    search_results = gr.Markdown("", label="Hasil Pencarian")
                    
                    search_btn.click(
                        fn=search_documents,
                        inputs=[search_query, search_top_k, search_min_score],
                        outputs=search_results
                    )
            
                # =================================================================
                # CHAT TAB
                # =================================================================
                with gr.Tab("💬 Konsultasi", id="chat"):
                    gr.Markdown("### Konsultasi Hukum dengan AI")
                    
                    # Legal disclaimer
                    gr.HTML(f'<div class="legal-disclaimer">{LEGAL_DISCLAIMER}</div>')
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="Percakapan",
                                height=500,
                                type="messages"
                            )
                            
                            with gr.Row():
                                chat_input = gr.Textbox(
                                    label="Pertanyaan",
                                    placeholder="Ketik pertanyaan hukum Anda...",
                                    scale=4
                                )
                                send_btn = gr.Button("Kirim", variant="primary", scale=1)
                        
                        with gr.Column(scale=1):
                            session_id_display = gr.Textbox(
                                label="Session ID",
                                interactive=False
                            )
                            thinking_level = gr.Radio(
                                ["low", "medium", "high"],
                                value="low",
                                label="Tingkat Analisis"
                            )
                            show_thinking = gr.Checkbox(
                                value=True,
                                label="Tampilkan Proses Berpikir"
                            )
                            
                            new_session_btn = gr.Button("🆕 Sesi Baru")
                            clear_btn = gr.Button("🗑️ Hapus Chat")
                    
                    # Chat handlers
                    send_btn.click(
                        fn=chat_with_api,
                        inputs=[chat_input, chatbot, thinking_level, show_thinking, session_id_display],
                        outputs=chatbot
                    ).then(
                        fn=lambda: "",
                        outputs=chat_input
                    )
                    
                    chat_input.submit(
                        fn=chat_with_api,
                        inputs=[chat_input, chatbot, thinking_level, show_thinking, session_id_display],
                        outputs=chatbot
                    ).then(
                        fn=lambda: "",
                        outputs=chat_input
                    )
                    
                    clear_btn.click(fn=clear_chat, outputs=chatbot)
                    new_session_btn.click(fn=new_session, outputs=[session_id_display, chatbot])
                
                # =================================================================
                # SETTINGS TAB
                # =================================================================
                with gr.Tab("⚙️ Pengaturan", id="settings"):
                    gr.Markdown("### Konfigurasi API")
                    
                    with gr.Row():
                        with gr.Column():
                            api_url_input = gr.Textbox(
                                label="API URL",
                                value=DEFAULT_CONFIG['api_url'],
                                placeholder="http://localhost:8000/api/v1"
                            )
                            api_key_input = gr.Textbox(
                                label="API Key",
                                value=DEFAULT_CONFIG['api_key'],
                                type="password"
                            )
                            apply_btn = gr.Button("💾 Terapkan", variant="primary")
                            settings_status = gr.Markdown("")
                    
                    gr.Markdown("---")
                    gr.Markdown("### Manajemen Sesi")
                    
                    with gr.Row():
                        refresh_sessions_btn = gr.Button("🔄 Refresh Sesi")
                        sessions_list = gr.Markdown("")
                    
                    with gr.Row():
                        export_format = gr.Radio(["md", "json", "html"], value="md", label="Format Export")
                        export_btn = gr.Button("📥 Export Sesi Aktif")
                        export_output = gr.Textbox(label="Hasil Export", lines=10)
                    
                    apply_btn.click(
                        fn=apply_settings,
                        inputs=[api_url_input, api_key_input],
                        outputs=settings_status
                    )
                    
                    refresh_sessions_btn.click(fn=get_sessions_list, outputs=sessions_list)
                    export_btn.click(fn=export_current_session, inputs=export_format, outputs=export_output)
                
                # =================================================================
                # STATUS TAB
                # =================================================================
                with gr.Tab("📊 Status", id="status"):
                    gr.Markdown("### Status Sistem")
                    
                    check_btn = gr.Button("🔄 Periksa Status", variant="primary")
                    status_output = gr.Markdown("")
                    
                    check_btn.click(fn=check_status, outputs=status_output)
        
        # =====================================================================
        # AUTH EVENT HANDLERS
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
        
        # Initialize API on successful login
        main_app.change(
            fn=lambda: initialize_client(DEFAULT_CONFIG['api_url'], DEFAULT_CONFIG['api_key']),
            outputs=status_display
        )
    
    return app


# =============================================================================
# LAUNCHER
# =============================================================================

def launch_unified_app(
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1"
):
    """Launch unified app"""
    print("\n" + "=" * 60)
    print("🏛️ LEGAL RAG INDONESIA - UNIFIED UI")
    print("=" * 60)
    print(f"Starting on http://{server_name}:{server_port}")
    print("=" * 60 + "\n")
    
    app = create_unified_interface()
    app.launch(
        share=share,
        server_port=server_port,
        server_name=server_name
    )


if __name__ == "__main__":
    # Read share setting from environment (set by launch_dev.py)
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    
    launch_unified_app(share=share, server_name=server_name, server_port=server_port)
