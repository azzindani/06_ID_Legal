"""
Unified API-Based UI - Indonesian Legal RAG System

Production-ready UI that connects to the FastAPI backend.
Refactored version that uses existing modules for cleaner code.

File: ui/unified_app_api.py
"""

import gradio as gr
import os
import sys
import json
import time
from typing import Optional, Dict, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.services.api_client import LegalRAGAPIClient
from utils.text_utils import parse_think_tags
from conversation.export import (
    MarkdownExporter, JSONExporter, HTMLExporter,
    parse_gradio_content, history_to_session_data
)

# =============================================================================
# GLOBAL STATE & CONFIG
# =============================================================================

api_client: Optional[LegalRAGAPIClient] = None
current_session: Optional[str] = None
authenticated_user: Optional[str] = None

DEMO_USERS = {"demo": "demo123", "admin": "admin123"}

DEFAULT_CONFIG = {
    'api_url': os.environ.get('LEGAL_API_URL', 'http://127.0.0.1:8000/api/v1'),
    'api_key': os.environ.get('LEGAL_API_KEY', ''),
    'final_top_k': 3,
    'temperature': 0.7,
    'max_new_tokens': 2048,
    'research_team_size': 4,
    'thinking_mode': 'low'
}

EXAMPLE_QUERIES = [
    "Apakah ada pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?",
    "Bagaimana mekanisme hukum untuk memperoleh izin usaha sebagai pengusaha pabrik barang kena cukai?",
    "syarat dan prosedur perceraian menurut hukum Indonesia",
    "hak dan kewajiban pekerja dalam UU Ketenagakerjaan"
]

TEST_QUESTIONS = [
    "Apakah terdapat pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?",
    "Berdasarkan PP No. 41 Tahun 2009, sebutkan jenis-jenis tunjangan yang diatur di dalamnya.",
    "Ganti topik. Jelaskan secara singkat pengertian kawasan pabean menurut Undang-Undang Kepabeanan.",
    "Jelaskan secara umum ruang lingkup UU No. 13 Tahun 2003 tentang Ketenagakerjaan.",
]

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def initialize_api():
    """Initialize API client"""
    global api_client
    if api_client is None:
        api_client = LegalRAGAPIClient(
            base_url=DEFAULT_CONFIG['api_url'],
            api_key=DEFAULT_CONFIG['api_key'],
            timeout=900  # 15 minutes for long tests
        )
    try:
        health = api_client.health_check()
        return f"✅ Connected" if health.ready else f"⚠️ Loading: {health.message}"
    except Exception as e:
        return f"❌ Error: {e}"


def dummy_login(username: str, password: str):
    global authenticated_user
    if not username or not password:
        return gr.update(visible=True), gr.update(visible=False), "⚠️ Enter credentials"
    if username in DEMO_USERS and DEMO_USERS[username] == password:
        authenticated_user = username
        return gr.update(visible=False), gr.update(visible=True), ""
    return gr.update(visible=True), gr.update(visible=False), "❌ Invalid credentials"


def dummy_logout():
    global authenticated_user
    authenticated_user = None
    return gr.update(visible=True), gr.update(visible=False), []


def clear_conversation():
    global current_session
    current_session = None
    return [], ""


def get_system_info():
    if api_client is None:
        return "❌ API not connected"
    try:
        health = api_client.health_check()
        return f"**Status:** {'✅ Ready' if health.ready else '⏳ Loading'}\n**API:** {DEFAULT_CONFIG['api_url']}"
    except Exception as e:
        return f"❌ Error: {e}"


def format_health_report():
    if api_client is None:
        return "❌ Not connected"
    try:
        health = api_client.health_check()
        return f"**Health:** {'✅' if health.healthy else '❌'}\n**Ready:** {'✅' if health.ready else '⏳'}\n**Message:** {health.message}"
    except Exception as e:
        return f"❌ Error: {e}"


# =============================================================================
# CHAT FUNCTION - With proper <think> tag parsing
# =============================================================================

def chat_with_legal_rag(message, history, config_dict, show_thinking=True, show_sources=True, show_metadata=True):
    """Main chat function with streaming and think tag parsing"""
    if not message.strip():
        return history, ""
    
    global api_client, current_session
    if api_client is None:
        initialize_api()
    if api_client is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ API not connected"})
        yield history, ""
        return
    
    try:
        # State tracking
        accumulated_text = ""
        thinking_content = []
        final_answer = []
        live_output = []
        in_thinking = False
        saw_think = False
        header_shown = False
        result_data = None
        
        thinking_mode = str(config_dict.get('thinking_mode', 'low')).lower()
        
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "🔄 **Processing...**"}
        ], ""
        
        for chunk in api_client.chat_stream(query=message, session_id=current_session, thinking_level=thinking_mode):
            chunk_type = chunk.get('type', '')
            content = chunk.get('content', chunk.get('message', ''))
            
            if chunk_type == 'chunk':
                accumulated_text += content
                new_text = content
                
                # Parse <think> tags
                if '<think>' in new_text:
                    in_thinking = True
                    saw_think = True
                    new_text = new_text.replace('<think>', '')
                    if not header_shown and show_thinking:
                        live_output = ['🧠 **Sedang berfikir...**\n\n']
                        header_shown = True
                
                if '</think>' in new_text:
                    in_thinking = False
                    new_text = new_text.replace('</think>', '')
                    if show_thinking:
                        live_output.append('\n\n---\n\n✅ **Sedang menjawab...**\n\n')
                
                if saw_think:
                    if in_thinking:
                        thinking_content.append(new_text)
                        if show_thinking:
                            live_output.append(new_text)
                    else:
                        final_answer.append(new_text)
                        live_output.append(new_text)
                else:
                    if '<think>' in accumulated_text:
                        saw_think = True
                        in_thinking = '</think>' not in accumulated_text
                        if not header_shown and show_thinking:
                            live_output = ['🧠 **Sedang berfikir...**\n\n']
                            header_shown = True
                        think_start = accumulated_text.find('<think>') + 7
                        if '</think>' in accumulated_text:
                            think_end = accumulated_text.find('</think>')
                            thinking_content = [accumulated_text[think_start:think_end]]
                            final_answer.append(accumulated_text[think_end + 8:])
                            if show_thinking:
                                live_output.append(accumulated_text[think_start:think_end])
                                live_output.append('\n\n---\n\n✅ **Sedang menjawab...**\n\n')
                                live_output.append(accumulated_text[think_end + 8:])
                        else:
                            thinking_content = [accumulated_text[think_start:]]
                            if show_thinking:
                                live_output.append(accumulated_text[think_start:])
                    elif len(accumulated_text) > 100:
                        if not header_shown:
                            live_output = ['⭐ **Jawaban:**\n\n']
                            header_shown = True
                        final_answer.append(new_text)
                        live_output.append(new_text)
                    else:
                        live_output.append(new_text)
                
                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": ''.join(live_output)}
                ], ""
            
            elif chunk_type == 'done':
                result_data = chunk
                if result_data.get('session_id'):
                    current_session = result_data.get('session_id')
        
        # Build final output
        response_text = ''.join(final_answer).strip() if final_answer else (result_data.get('answer', accumulated_text) if result_data else accumulated_text)
        response_text = response_text.replace('<think>', '').replace('</think>', '').strip()
        
        thinking_text = ''.join(thinking_content).strip().replace('<think>', '').replace('</think>', '')
        
        final_output = ""
        if show_thinking and thinking_text:
            final_output += f'<details><summary>🧠 <strong>Proses Berpikir</strong></summary>\n\n{thinking_text}\n</details>\n\n---\n\n### ✅ Jawaban\n\n{response_text}'
        else:
            final_output += f"### ✅ Jawaban\n\n{response_text}"
        
        # Add sources
        if show_sources and result_data and result_data.get('legal_references'):
            final_output += f'\n\n---\n\n<details><summary>📖 <strong>Sumber Hukum</strong></summary>\n\n{result_data["legal_references"]}\n</details>'
        
        if show_metadata and result_data and result_data.get('research_process'):
            final_output += f'\n\n<details><summary>🔬 <strong>Detail Penelitian</strong></summary>\n\n{result_data["research_process"]}\n</details>'
        
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": final_output}
        ], ""
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ **Error:** {e}"}
        ], ""


# =============================================================================
# TEST RUNNERS
# =============================================================================

def run_conversational_test(history, config_dict, show_thinking, show_sources, show_metadata):
    """Run conversational test with reduced questions for reliability"""
    history = history + [{"role": "assistant", "content": f"🧪 **Starting Test ({len(TEST_QUESTIONS)} Questions)**"}]
    yield history, ""
    
    for i, q in enumerate(TEST_QUESTIONS, 1):
        history = history + [{"role": "assistant", "content": f"🔄 **Q{i}/{len(TEST_QUESTIONS)}:** _{q[:80]}..._"}]
        yield history, ""
        history = history[:-1]
        
        for h, _ in chat_with_legal_rag(q, history, config_dict, show_thinking, show_sources, show_metadata):
            yield h, ""
        history = h
    
    history = history + [{"role": "assistant", "content": "✅ **Test Complete**"}]
    yield history, ""


# =============================================================================
# SEARCH FUNCTION
# =============================================================================

def search_documents(query: str, num_results: int = 5):
    global api_client
    if not query.strip():
        return "⚠️ Enter a query", ""
    if api_client is None:
        initialize_api()
    if api_client is None:
        return "❌ API not connected", ""
    
    try:
        result = api_client.retrieve(query=query, top_k=int(num_results))
        if not result or not result.get('documents'):
            return "📭 No documents found", ""
        
        docs = result['documents']
        summary = f"## 📋 Results ({len(docs)} docs, {result.get('search_time', 0):.2f}s)\n\n"
        for i, d in enumerate(docs, 1):
            summary += f"{i}. **{d.regulation_type} No. {d.regulation_number}/{d.year}** (Score: {d.score:.4f})\n   _{d.about}_\n\n"
        
        return summary, ""
    except Exception as e:
        return f"❌ Error: {e}", ""


# =============================================================================
# EXPORT FUNCTION - Uses existing exporters
# =============================================================================

def export_conversation_handler(export_format, history):
    """Export using standard exporters with parsed content"""
    try:
        if not history:
            return "No conversation to export.", None
        
        session_data = history_to_session_data(history, current_session)
        
        exporters = {
            'Markdown': MarkdownExporter({'include_thinking': True}),
            'JSON': JSONExporter(),
            'HTML': HTMLExporter()
        }
        
        exporter = exporters.get(export_format, MarkdownExporter())
        content = exporter.export(session_data)
        
        ext_map = {'Markdown': 'md', 'JSON': 'json', 'HTML': 'html'}
        extension = ext_map.get(export_format, 'md')
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{extension}', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        preview = content[:5000] + "..." if len(content) > 5000 else content
        return preview, temp_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Export failed: {e}", None


# =============================================================================
# CONFIG FUNCTIONS
# =============================================================================

def update_config(*args):
    try:
        return {
            'final_top_k': int(args[0]), 'temperature': float(args[1]),
            'max_new_tokens': int(args[2]), 'thinking_mode': str(args[3]).lower(),
            'research_team_size': int(args[4])
        }
    except:
        return DEFAULT_CONFIG


def reset_to_defaults():
    return (3, 0.7, 2048, "Low", 4)


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# CSS for responsive login
LOGIN_CSS = """
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 80vh;
}
.login-box {
    max-width: 400px;
    width: 100%;
    padding: 40px;
    text-align: center;
}
.main-app {
    padding: 10px;
}
"""

def create_gradio_interface():
    with gr.Blocks(title="Indonesian Legal Assistant", css=LOGIN_CSS) as interface:
        
        # Login Panel - Centered
        with gr.Column(visible=True, elem_classes="login-container") as login_panel:
            with gr.Column(elem_classes="login-box"):
                gr.Markdown("# 🏛️ Indonesian Legal Assistant\n**Production API-Based UI**")
                gr.Markdown("Demo: `demo`/`demo123` or `admin`/`admin123`")
                username = gr.Textbox(label="Username")
                password = gr.Textbox(label="Password", type="password")
                login_btn = gr.Button("🔓 Login", variant="primary")
                login_error = gr.Markdown("")
        
        # Main App
        with gr.Column(visible=False, elem_classes="main-app") as main_app:
            with gr.Row():
                gr.Column(scale=9)
                logout_btn = gr.Button("🚪 Logout", size="sm", scale=1)
            
            with gr.Tabs():
                # Chat Tab
                with gr.TabItem("💬 Konsultasi Hukum"):
                    chatbot = gr.Chatbot(height="60vh", show_label=False)
                    with gr.Row():
                        msg = gr.Textbox(placeholder="Tanyakan tentang hukum Indonesia...", show_label=False, scale=10)
                        send_btn = gr.Button("📤", variant="primary", scale=1)
                    gr.Examples(examples=EXAMPLE_QUERIES, inputs=msg)
                
                # Search Tab
                with gr.TabItem("🔍 Pencarian"):
                    search_q = gr.Textbox(label="Query", placeholder="Cari dokumen...")
                    search_n = gr.Slider(1, 10, value=5, step=1, label="Jumlah Hasil")
                    search_btn = gr.Button("🔍 Cari", variant="primary")
                    search_result = gr.Markdown()
                
                # Settings Tab
                with gr.TabItem("⚙️ Pengaturan"):
                    with gr.Group():
                        gr.Markdown("#### Settings")
                        top_k = gr.Slider(1, 10, value=3, step=1, label="Top K")
                        temp = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                        max_tokens = gr.Slider(512, 8192, value=2048, step=256, label="Max Tokens")
                        thinking = gr.Radio(["Low", "Medium", "High"], value="Low", label="Thinking Level")
                        team_size = gr.Slider(1, 5, value=4, step=1, label="Team Size")
                    
                    with gr.Group():
                        gr.Markdown("#### Display")
                        show_thinking = gr.Checkbox(label="Show Thinking", value=True)
                        show_sources = gr.Checkbox(label="Show Sources", value=True)
                        show_metadata = gr.Checkbox(label="Show Metadata", value=True)
                    
                    with gr.Group():
                        gr.Markdown("#### System")
                        with gr.Row():
                            info_btn = gr.Button("📊 System Info")
                            health_btn = gr.Button("🏥 Health Check")
                            reset_btn = gr.Button("🔄 Reset Defaults")
                        system_output = gr.Markdown()
                    
                    with gr.Group():
                        gr.Markdown("#### Tests")
                        test_btn = gr.Button("🧪 Run Conversational Test", variant="primary")
                
                # Export Tab
                with gr.TabItem("📥 Export"):
                    export_fmt = gr.Radio(["Markdown", "JSON", "HTML"], value="Markdown", label="Format")
                    export_btn = gr.Button("📥 Export", variant="primary")
                    export_output = gr.Textbox(label="Preview", lines=15)
                    export_file = gr.File(label="Download")
            
            # State
            config_state = gr.State(DEFAULT_CONFIG)
            config_inputs = [top_k, temp, max_tokens, thinking, team_size]
            
            # Event handlers
            for inp in config_inputs:
                inp.change(update_config, inputs=config_inputs, outputs=config_state)
            
            reset_btn.click(reset_to_defaults, outputs=config_inputs)
            info_btn.click(get_system_info, outputs=system_output)
            health_btn.click(format_health_report, outputs=system_output)
            
            # Chat
            msg.submit(chat_with_legal_rag, [msg, chatbot, config_state, show_thinking, show_sources, show_metadata], [chatbot, msg])
            send_btn.click(chat_with_legal_rag, [msg, chatbot, config_state, show_thinking, show_sources, show_metadata], [chatbot, msg])
            
            # Search
            search_btn.click(search_documents, [search_q, search_n], [search_result])
            search_q.submit(search_documents, [search_q, search_n], [search_result])
            
            # Test
            test_btn.click(run_conversational_test, [chatbot, config_state, show_thinking, show_sources, show_metadata], [chatbot, msg])
            
            # Export
            export_btn.click(export_conversation_handler, [export_fmt, chatbot], [export_output, export_file])
        
        # Auth handlers
        login_btn.click(dummy_login, [username, password], [login_panel, main_app, login_error])
        username.submit(dummy_login, [username, password], [login_panel, main_app, login_error])
        password.submit(dummy_login, [username, password], [login_panel, main_app, login_error])
        logout_btn.click(dummy_logout, outputs=[login_panel, main_app, chatbot])
    
    interface.queue()
    return interface


def launch_app(share=False, server_port=7860, server_name="0.0.0.0"):
    print("\n" + "=" * 60)
    print("🏛️ LEGAL RAG INDONESIA - UNIFIED API UI")
    print("=" * 60)
    print(f"API Status: {initialize_api()}")
    print("=" * 60 + "\n")
    
    demo = create_gradio_interface()
    demo.launch(server_name=server_name, server_port=server_port, share=share)


launch_unified_app = launch_app

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    launch_app(share=share)
