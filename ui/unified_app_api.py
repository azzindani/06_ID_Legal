"""
Unified API-Based UI - Indonesian Legal RAG System

Production UI that connects to the FastAPI backend for all operations.
Style matches search_app.py and gradio_app.py.

Features:
- API client for HTTP communication with backend
- Tabbed interface: Search & Chat
- Example queries from existing apps
- Production-ready for Docker/cloud deployment
"""

import gradio as gr
import os
import sys
import time
import json
from typing import Optional, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.services.api_client import LegalRAGAPIClient, Document, HealthStatus

# =============================================================================
# GLOBAL STATE
# =============================================================================

api_client: Optional[LegalRAGAPIClient] = None
current_session_id: Optional[str] = None
authenticated_user: Optional[str] = None

# Demo users for dummy auth
DEMO_USERS = {
    "demo": "demo123",
    "admin": "admin123"
}

# Default configuration
DEFAULT_CONFIG = {
    'api_url': os.environ.get('LEGAL_API_URL', 'http://127.0.0.1:8000/api/v1'),
    'api_key': os.environ.get('LEGAL_API_KEY', 'your-api-key-here')
}

# CSS styling from search_app.py
CUSTOM_CSS = """
/* Base responsive sizing */
html { font-size: 16px; }

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header styling */
.header-title {
    text-align: center;
    font-size: 2em;
    font-weight: 700;
    color: #1e3a5f;
    margin-bottom: 0.5em;
}

.header-subtitle {
    text-align: center;
    font-size: 1em;
    color: #666;
    margin-bottom: 1.5em;
}

/* Search box styling */
.search-box textarea {
    font-size: 1.1em !important;
    padding: 1em !important;
    border-radius: 0.5em !important;
    border: 2px solid #e0e0e0 !important;
}

.search-box textarea:focus {
    border-color: #1e3a5f !important;
    box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.1) !important;
}

/* Results styling */
.results-container {
    margin-top: 1.5em;
    max-height: 70vh;
    overflow-y: auto;
}

.result-card {
    background: #f8f9fa;
    border-radius: 0.5em;
    padding: 1em;
    margin-bottom: 1em;
    border-left: 4px solid #1e3a5f;
}

/* Legal disclaimer */
.legal-disclaimer {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 0.9em;
}

/* Status display */
.status-display {
    background: #f0f4f8;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.9em;
    margin-bottom: 12px;
}

/* Button styling */
.search-btn {
    background: #1e3a5f !important;
    color: white !important;
    font-size: 1em !important;
    padding: 0.8em 2em !important;
    border-radius: 0.5em !important;
}

.search-btn:hover {
    background: #2c5282 !important;
}

/* Login panel */
.login-panel {
    max-width: 400px;
    margin: 100px auto;
    padding: 30px;
    background: #f8f9fa;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Responsive */
@media (max-width: 768px) {
    html { font-size: 14px; }
    .gradio-container { padding: 1em !important; }
}

@media (min-width: 1200px) {
    html { font-size: 18px; }
}
"""

# Legal disclaimer
LEGAL_DISCLAIMER = """
⚠️ **Pemberitahuan Penting**: Informasi yang diberikan bersifat edukatif dan bukan merupakan nasihat hukum resmi. 
Untuk kepentingan hukum, silakan konsultasikan dengan advokat profesional.
"""

# Example queries (from gradio_app.py test questions)
EXAMPLE_QUERIES = [
    "Apa syarat pendirian PT menurut UU No. 40 Tahun 2007?",
    "Jelaskan ketentuan pesangon dalam UU Ketenagakerjaan",
    "Bagaimana prosedur pengajuan izin lingkungan menurut PP No. 27 Tahun 2012?",
    "Apa sanksi pidana dalam UU ITE untuk penyebaran hoax?",
    "Jelaskan hak-hak konsumen menurut UU Perlindungan Konsumen",
]

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
    """Dummy authentication (for demo purposes)"""
    global authenticated_user
    
    if not username or not password:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "⚠️ Masukkan username dan password",
            ""
        )
    
    if username in DEMO_USERS and DEMO_USERS[username] == password:
        authenticated_user = username
        status_msg = initialize_client(DEFAULT_CONFIG['api_url'], DEFAULT_CONFIG['api_key'])
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            status_msg
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "❌ Username atau password salah",
            ""
        )


def dummy_logout():
    """Logout and return to login screen"""
    global authenticated_user
    authenticated_user = None
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        []
    )


def format_document_card(doc: Document, index: int) -> str:
    """Format a document as markdown card - matches search_app.py style"""
    score_pct = int(doc.score * 100)
    
    return f"""
### 📄 {index}. {doc.regulation_type} No. {doc.regulation_number}/{doc.year}

**Lokasi:** {doc.chapter or 'N/A'} | {doc.article or 'N/A'}
**Tentang:** {doc.about}
**Skor Relevansi:** {score_pct}%

---

#### 📝 Konten

{doc.content[:500]}{"..." if len(doc.content) > 500 else ""}

---
"""


def search_documents(query: str, num_results: int = 5, min_score: float = 0.0) -> str:
    """Search for documents via API"""
    if not query.strip():
        return "⚠️ Masukkan query pencarian."
    
    if api_client is None:
        return "❌ API client not initialized. Please refresh the page."
    
    try:
        docs = api_client.retrieve(query=query, top_k=int(num_results), min_score=min_score)
        
        if not docs:
            return "📭 Tidak ada dokumen yang ditemukan."
        
        output = [f"## 📚 Hasil Pencarian ({len(docs)} dokumen)\n"]
        output.append(f"**Query:** `{query}`\n\n---\n")
        
        for i, doc in enumerate(docs, 1):
            output.append(format_document_card(doc, i))
        
        return "\n".join(output)
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


def chat_with_api(message: str, history: list, thinking_level: str, show_thinking: bool, session_id: str):
    """Chat with the API using streaming"""
    if not message.strip():
        return history
    
    if api_client is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ API not connected. Please refresh."})
        return history
    
    # Add user message
    history.append({"role": "user", "content": message})
    
    # Add placeholder for assistant
    history.append({"role": "assistant", "content": "🔄 Menganalisis pertanyaan..."})
    
    try:
        accumulated = ""
        thinking_buffer = ""
        in_thinking = False
        
        for chunk in api_client.chat_stream(
            query=message,
            session_id=session_id or None,
            thinking_level=thinking_level
        ):
            chunk_type = chunk.get('type', '')
            content = chunk.get('content', '')
            
            if chunk_type == 'thinking':
                thinking_buffer += content
                if show_thinking:
                    display = f"💭 **Proses Berpikir:**\n\n{thinking_buffer}\n\n---\n\n{accumulated}"
                    history[-1] = {"role": "assistant", "content": display}
                    yield history
            
            elif chunk_type == 'content':
                accumulated += content
                if show_thinking and thinking_buffer:
                    display = f"💭 **Proses Berpikir:**\n\n{thinking_buffer}\n\n---\n\n**Jawaban:**\n\n{accumulated}"
                else:
                    display = accumulated
                history[-1] = {"role": "assistant", "content": display}
                yield history
            
            elif chunk_type == 'complete':
                # Final message with sources
                sources = chunk.get('sources', [])
                final_content = accumulated
                
                if sources:
                    final_content += "\n\n---\n\n### 📚 Referensi Hukum\n\n"
                    for i, src in enumerate(sources[:5], 1):
                        final_content += f"{i}. **{src.get('regulation_type', '')} No. {src.get('regulation_number', '')}/{src.get('year', '')}**\n"
                        final_content += f"   _{src.get('about', '')[:100]}_\n\n"
                
                if show_thinking and thinking_buffer:
                    final_content = f"💭 **Proses Berpikir:**\n\n{thinking_buffer}\n\n---\n\n**Jawaban:**\n\n{final_content}"
                
                history[-1] = {"role": "assistant", "content": final_content}
                yield history
                
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"❌ Error: {str(e)}"}
        yield history


def new_session():
    """Create a new session"""
    global current_session_id
    current_session_id = None
    if api_client:
        try:
            current_session_id = api_client.create_session()
        except:
            current_session_id = f"session_{int(time.time())}"
    return current_session_id or "", []


def clear_chat():
    """Clear chat history"""
    return []


def check_status() -> str:
    """Check API status"""
    if api_client is None:
        return "❌ Not connected"
    
    try:
        health = api_client.health_check()
        return f"""
## 📊 Status Sistem

| Komponen | Status |
|----------|--------|
| **API Connection** | {"✅ Connected" if health.healthy else "❌ Disconnected"} |
| **Pipeline** | {"✅ Ready" if health.ready else "⏳ Loading"} |
| **Message** | {health.message} |
"""
    except Exception as e:
        return f"❌ Error checking status: {str(e)}"


def use_example(example: str):
    """Fill example query into search box"""
    return example


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_unified_interface():
    """Create unified Gradio interface with tabs and auth"""
    
    with gr.Blocks(title="Legal RAG Indonesia") as app:
        
        # =====================================================================
        # LOGIN PANEL
        # =====================================================================
        with gr.Column(visible=True) as login_panel:
            gr.HTML("""
            <div style="text-align: center; padding: 50px 20px;">
                <h1 style="color: #1e3a5f; font-size: 2.5em;">🏛️ Asisten Hukum Indonesia</h1>
                <p style="color: #666; font-size: 1.1em;">Silakan login untuk mengakses sistem</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    gr.Markdown("### 🔐 Login")
                    gr.Markdown("""
                    **Demo Accounts:**
                    - Username: `demo` / Password: `demo123`
                    - Username: `admin` / Password: `admin123`
                    """)
                    username_input = gr.Textbox(label="Username", placeholder="Masukkan username")
                    password_input = gr.Textbox(label="Password", type="password", placeholder="Masukkan password")
                    login_btn = gr.Button("🔓 Login", variant="primary")
                    login_error = gr.Markdown("")
                with gr.Column(scale=1):
                    pass
        
        # =====================================================================
        # MAIN APP
        # =====================================================================
        with gr.Column(visible=False) as main_app:
            
            # Header
            with gr.Row():
                with gr.Column(scale=5):
                    gr.HTML("""
                    <div style="padding: 20px 0;">
                        <h1 style="color: #1e3a5f; margin: 0;">🏛️ Asisten Hukum Indonesia</h1>
                        <p style="color: #666; margin: 5px 0 0 0;">Pencarian dan Konsultasi Regulasi berbasis AI</p>
                    </div>
                    """)
                with gr.Column(scale=1):
                    logout_btn = gr.Button("🚪 Logout", size="sm")
            
            status_display = gr.Markdown("⏳ Connecting...", elem_classes=["status-display"])
            
            with gr.Tabs():
                
                # =============================================================
                # SEARCH TAB
                # =============================================================
                with gr.Tab("🔍 Pencarian Dokumen", id="search"):
                    gr.Markdown("### Cari Dokumen Regulasi Indonesia")
                    
                    # Example queries
                    gr.Markdown("**Contoh Query:**")
                    with gr.Row():
                        for i, example in enumerate(EXAMPLE_QUERIES[:3]):
                            gr.Button(example[:40] + "...", size="sm").click(
                                fn=lambda e=example: e,
                                outputs=[gr.Textbox(visible=False)]
                            )
                    
                    with gr.Row():
                        with gr.Column(scale=4):
                            search_query = gr.Textbox(
                                label="Query Pencarian",
                                placeholder="Contoh: Syarat pendirian PT, Sanksi UU ITE, Hak konsumen...",
                                lines=2,
                                elem_classes=["search-box"]
                            )
                        with gr.Column(scale=1):
                            search_top_k = gr.Slider(1, 20, value=5, step=1, label="Jumlah Hasil")
                            search_min_score = gr.Slider(0, 1, value=0.0, step=0.1, label="Skor Minimum")
                    
                    search_btn = gr.Button("🔍 Cari Dokumen", variant="primary", elem_classes=["search-btn"])
                    search_results = gr.Markdown("", elem_classes=["results-container"])
                    
                    search_btn.click(
                        fn=search_documents,
                        inputs=[search_query, search_top_k, search_min_score],
                        outputs=search_results
                    )
                
                # =============================================================
                # CHAT TAB  
                # =============================================================
                with gr.Tab("💬 Konsultasi Hukum", id="chat"):
                    gr.HTML(f'<div class="legal-disclaimer">{LEGAL_DISCLAIMER}</div>')
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="Percakapan",
                                height=500
                            )
                            
                            with gr.Row():
                                chat_input = gr.Textbox(
                                    label="Pertanyaan",
                                    placeholder="Tanyakan tentang hukum Indonesia...",
                                    scale=5
                                )
                                send_btn = gr.Button("Kirim", variant="primary", scale=1)
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### ⚙️ Pengaturan")
                            
                            session_id_display = gr.Textbox(
                                label="Session ID",
                                interactive=False,
                                value=""
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
                            
                            gr.Markdown("---")
                            
                            new_session_btn = gr.Button("🆕 Sesi Baru", size="sm")
                            clear_btn = gr.Button("🗑️ Hapus Chat", size="sm")
                            
                            gr.Markdown("---")
                            gr.Markdown("**Contoh Pertanyaan:**")
                            for example in EXAMPLE_QUERIES[:3]:
                                gr.Markdown(f"• _{example[:50]}..._")
                    
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
                
                # =============================================================
                # STATUS TAB
                # =============================================================
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
            outputs=[login_panel, main_app, login_error, status_display]
        )
        
        logout_btn.click(
            fn=dummy_logout,
            outputs=[login_panel, main_app, chatbot]
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
    if share:
        print("📡 Share mode enabled - public link will be generated")
    print("=" * 60 + "\n")
    
    app = create_unified_interface()
    app.launch(
        share=share,
        server_port=server_port,
        server_name=server_name
    )


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    
    launch_unified_app(share=share, server_name=server_name, server_port=server_port)
