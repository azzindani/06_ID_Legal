"""
Unified API-Based UI - Indonesian Legal RAG System

Production-ready UI that connects to the FastAPI backend.
Full-featured version matching gradio_app.py exactly.

File: ui/unified_app_api.py
"""

import gradio as gr
import os
import sys
import json
import time
from typing import Optional, Dict, List, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.services.api_client import LegalRAGAPIClient
from utils.text_utils import parse_think_tags
from conversation.export import (
    MarkdownExporter, JSONExporter, HTMLExporter,
    parse_gradio_content, history_to_session_data, extract_text_content
)

# =============================================================================
# GLOBAL STATE & CONFIG
# =============================================================================

api_client: Optional[LegalRAGAPIClient] = None
current_session: Optional[str] = None
authenticated_user: Optional[str] = None

DEMO_USERS = {"demo": "demo123", "admin": "admin123"}

# Default configuration matching gradio_app.py exactly
DEFAULT_CONFIG = {
    'api_url': os.environ.get('LEGAL_API_URL', 'http://127.0.0.1:8000/api/v1'),
    'api_key': os.environ.get('LEGAL_API_KEY', ''),
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

# Default search phases (same as gradio_app.py)
DEFAULT_SEARCH_PHASES = {
    'initial_scan': {'enabled': True, 'candidates': 400, 'semantic_threshold': 0.20, 'keyword_threshold': 0.06},
    'focused_review': {'enabled': True, 'candidates': 150, 'semantic_threshold': 0.35, 'keyword_threshold': 0.12},
    'deep_analysis': {'enabled': True, 'candidates': 60, 'semantic_threshold': 0.45, 'keyword_threshold': 0.18},
    'verification': {'enabled': True, 'candidates': 30, 'semantic_threshold': 0.55, 'keyword_threshold': 0.22},
    'expert_review': {'enabled': True, 'candidates': 45, 'semantic_threshold': 0.50, 'keyword_threshold': 0.20}
}

# 8 Example queries (same as gradio_app.py)
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

# 8 Test questions (same as gradio_app.py)
TEST_QUESTIONS = [
    # TOPIC 1: Tunjangan Guru/Dosen (3 Questions)
    "Apakah terdapat pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?",
    "Berdasarkan PP No. 41 Tahun 2009, sebutkan jenis-jenis tunjangan yang diatur di dalamnya.",
    "Masih merujuk pada PP No. 41 Tahun 2009, jelaskan perbedaan kriteria penerima, besaran, dan sumber pendanaan antara Tunjangan Khusus dan Tunjangan Kehormatan Profesor",
    # TOPIC 2: Kepabeanan (2 Questions)
    "Ganti topik. Jelaskan secara singkat pengertian kawasan pabean menurut Undang-Undang Kepabeanan.",
    "Berdasarkan Undang-Undang Kepabeanan tersebut, jelaskan sanksi pidana bagi pihak yang dengan sengaja salah memberitahukan jenis dan jumlah barang impor sehingga merugikan negara.",
    # TOPIC 3: Ketenagakerjaan (2 Questions)
    "Sekarang beralih ke UU No. 13 Tahun 2003. Jelaskan secara umum ruang lingkup dan pokok bahasan undang-undang tersebut.",
    "Apa yang diatur dalam Pasal 1 UU No. 13 Tahun 2003?",
    # TOPIC 4: PP No. 8 Tahun 2007 (1 Question)
    "Terakhir, jelaskan secara ringkas PP No. 8 Tahun 2007, termasuk fokus pengaturannya."
]

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def initialize_api():
    """Initialize API client with high timeout for long tests"""
    global api_client
    if api_client is None:
        api_client = LegalRAGAPIClient(
            base_url=DEFAULT_CONFIG['api_url'],
            api_key=DEFAULT_CONFIG['api_key'],
            timeout=1800  # 30 minutes for full test
        )
    try:
        health = api_client.health_check()
        return f"✅ Connected" if health.ready else f"⚠️ Loading: {health.message}"
    except Exception as e:
        return f"❌ Error: {e}"


def dummy_login(username: str, password: str):
    """Handle login"""
    global authenticated_user
    if not username or not password:
        return gr.update(visible=True), gr.update(visible=False), "⚠️ Enter credentials"
    if username in DEMO_USERS and DEMO_USERS[username] == password:
        authenticated_user = username
        status = initialize_api()
        return gr.update(visible=False), gr.update(visible=True), ""
    return gr.update(visible=True), gr.update(visible=False), "❌ Invalid credentials"


def dummy_logout():
    """Handle logout"""
    global authenticated_user
    authenticated_user = None
    return gr.update(visible=True), gr.update(visible=False), []


def clear_conversation():
    """Clear chat history"""
    global current_session
    current_session = None
    return [], ""


def get_system_info():
    """Get system information"""
    if api_client is None:
        return "❌ API not connected"
    try:
        health = api_client.health_check()
        info = f"""## 📊 System Information

**API Status:** {'✅ Ready' if health.ready else '⏳ Loading'}
**API URL:** `{DEFAULT_CONFIG['api_url']}`
**Session ID:** `{current_session or 'None'}`
**Authenticated User:** `{authenticated_user or 'None'}`

### Current Configuration
- **Top K:** {DEFAULT_CONFIG['final_top_k']}
- **Temperature:** {DEFAULT_CONFIG['temperature']}
- **Max Tokens:** {DEFAULT_CONFIG['max_new_tokens']}
- **Team Size:** {DEFAULT_CONFIG['research_team_size']}
- **Thinking Mode:** {DEFAULT_CONFIG['thinking_mode'].capitalize()}
"""
        return info
    except Exception as e:
        return f"❌ Error: {e}"


def format_health_report():
    """Format health check report"""
    if api_client is None:
        return "❌ Not connected"
    try:
        health = api_client.health_check()
        return f"""## 🏥 Health Check Report

**Overall Health:** {'✅ Healthy' if health.healthy else '❌ Unhealthy'}
**API Ready:** {'✅ Yes' if health.ready else '⏳ No'}
**Message:** {health.message}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    except Exception as e:
        return f"❌ Error: {e}"


# =============================================================================
# CHAT FUNCTION - With proper <think> tag parsing (same as gradio_app.py)
# =============================================================================

def chat_with_legal_rag(message, history, config_dict, show_thinking=True, show_sources=True, show_metadata=True):
    """
    Main chat function with streaming and think tag parsing.
    Matches gradio_app.py behavior exactly.
    """
    if not message.strip():
        return history, ""
    
    global api_client, current_session
    if api_client is None:
        initialize_api()
    if api_client is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ API not connected. Please refresh and try again."})
        yield history, ""
        return
    
    try:
        # State tracking (same as gradio_app.py)
        accumulated_text = ""
        thinking_content = []
        final_answer = []
        live_output = []
        in_thinking = False
        saw_think = False
        header_shown = False
        result_data = None
        
        thinking_mode = str(config_dict.get('thinking_mode', 'low')).lower()
        
        # Initial processing message
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "🔄 **Memproses permintaan...**"}
        ], ""
        
        # Stream response from API
        for chunk in api_client.chat_stream(
            query=message,
            session_id=current_session,
            thinking_level=thinking_mode
        ):
            chunk_type = chunk.get('type', '')
            content = chunk.get('content', chunk.get('message', ''))
            
            if chunk_type == 'chunk':
                accumulated_text += content
                new_text = content
                
                # Parse <think> tags (same logic as gradio_app.py)
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
                    # Check if <think> appeared anywhere in accumulated text
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
                        # No think tag, assume direct answer
                        if not header_shown:
                            live_output = ['✅ **Jawaban:**\n\n']
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
        
        # Build final output (same format as gradio_app.py)
        response_text = ''.join(final_answer).strip() if final_answer else (
            result_data.get('answer', accumulated_text) if result_data else accumulated_text
        )
        response_text = response_text.replace('<think>', '').replace('</think>', '').strip()
        
        thinking_text = ''.join(thinking_content).strip().replace('<think>', '').replace('</think>', '')
        
        final_output = ""
        
        # Add thinking section if available
        if show_thinking and thinking_text:
            final_output += f'<details><summary>🧠 <strong>Proses Berpikir</strong></summary>\n\n{thinking_text}\n</details>\n\n---\n\n### ✅ Jawaban\n\n{response_text}'
        else:
            final_output += f"### ✅ Jawaban\n\n{response_text}"
        
        # Add sources
        if show_sources and result_data and result_data.get('legal_references'):
            final_output += f'\n\n---\n\n<details><summary>📖 <strong>Sumber Hukum</strong></summary>\n\n{result_data["legal_references"]}\n</details>'
        
        # Add research process
        if show_metadata and result_data and result_data.get('research_process'):
            final_output += f'\n\n<details><summary>🔬 <strong>Detail Proses Penelitian</strong></summary>\n\n{result_data["research_process"]}\n</details>'
        
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
# SEARCH FUNCTION
# =============================================================================

def search_documents(query: str, num_results: int = 5):
    """Search documents without LLM generation"""
    global api_client
    if not query.strip():
        return "⚠️ Masukkan query pencarian", ""
    if api_client is None:
        initialize_api()
    if api_client is None:
        return "❌ API tidak terhubung", ""
    
    try:
        result = api_client.retrieve(query=query, top_k=int(num_results))
        if not result or not result.get('documents'):
            return "📭 Tidak ada dokumen ditemukan", ""
        
        docs = result['documents']
        
        # Format summary
        summary = f"""## 📋 Hasil Pencarian

**Query:** {query}
**Dokumen Ditemukan:** {len(docs)}
**Waktu Pencarian:** {result.get('search_time', 0):.2f}s

---

"""
        for i, d in enumerate(docs, 1):
            summary += f"""### {i}. {d.regulation_type} No. {d.regulation_number} Tahun {d.year}

**Tentang:** {d.about}
**Skor:** {d.score:.4f}
**Lokasi:** {getattr(d, 'location', 'N/A')}

---

"""
        
        return summary, ""
    except Exception as e:
        return f"❌ Error: {e}", ""


# =============================================================================
# TEST RUNNERS
# =============================================================================

def run_conversational_test(history, config_dict, show_thinking, show_sources, show_metadata):
    """Run full conversational test with 8 questions"""
    # Initial message
    history = history + [{
        "role": "assistant",
        "content": f"🧪 **Starting Conversational Test (8 Questions)**\n\nAuto-feeding questions through API...\n\n**Questions to test:**\n" +
                   "\n".join([f"{i+1}. {q[:80]}..." for i, q in enumerate(TEST_QUESTIONS)])
    }]
    yield history, ""
    
    # Process each question
    for i, question in enumerate(TEST_QUESTIONS, 1):
        # Add progress indicator
        history = history + [{
            "role": "assistant",
            "content": f"🔄 **Question {i}/8**\n\nProcessing: _{question[:100]}..._"
        }]
        yield history, ""
        
        # Remove progress and process question
        history = history[:-1]
        
        for updated_history, cleared_input in chat_with_legal_rag(
            question, history, config_dict, show_thinking, show_sources, show_metadata
        ):
            yield updated_history, cleared_input
        
        history = updated_history
    
    # Completion message
    history = history + [{
        "role": "assistant",
        "content": f"✅ **Conversational Test Complete**\n\nSuccessfully processed all {len(TEST_QUESTIONS)} questions through the API."
    }]
    yield history, ""


def run_stress_test(history, config_dict, show_thinking, show_sources, show_metadata):
    """Run stress test with maximum settings"""
    # Create stress config
    stress_config = {
        'final_top_k': 30,
        'research_team_size': 5,
        'max_new_tokens': 8192,
        'temperature': 0.7,
        'top_p': 1.0,
        'top_k': 80,
        'min_p': 0.05,
        'thinking_mode': 'high',
    }
    
    # Initial message
    history = history + [{
        "role": "assistant",
        "content": f"⚡ **Starting Stress Test (8 Questions)**\n\n**Configuration:** MAXIMUM SETTINGS\n- Team Size: 5 personas\n- Max Tokens: 8192\n- Thinking Mode: High\n\n**Auto-feeding questions...**"
    }]
    yield history, ""
    
    # Process each question with stress config
    for i, question in enumerate(TEST_QUESTIONS, 1):
        history = history + [{
            "role": "assistant",
            "content": f"⚡ **Stress Test - Question {i}/8**\n\nProcessing with maximum settings: _{question[:100]}..._"
        }]
        yield history, ""
        
        history = history[:-1]
        
        for updated_history, cleared_input in chat_with_legal_rag(
            question, history, stress_config, show_thinking, show_sources, show_metadata
        ):
            yield updated_history, cleared_input
        
        history = updated_history
    
    history = history + [{
        "role": "assistant",
        "content": f"✅ **Stress Test Complete**\n\nSuccessfully processed all {len(TEST_QUESTIONS)} questions with maximum settings."
    }]
    yield history, ""


# =============================================================================
# EXPORT FUNCTION - Uses existing exporters with full content
# =============================================================================

def export_conversation_handler(export_format, history):
    """
    Export conversation using standard exporters.
    Returns FULL content, not truncated.
    """
    try:
        if not history:
            return "No conversation to export.", None
        
        # Convert history to session data format
        session_data = history_to_session_data(history, current_session)
        
        # Use appropriate exporter with thinking enabled
        exporter_config = {'include_thinking': True, 'include_metadata': True, 'include_sources': True}
        exporters = {
            'Markdown': MarkdownExporter(exporter_config),
            'JSON': JSONExporter(exporter_config),
            'HTML': HTMLExporter(exporter_config)
        }
        
        exporter = exporters.get(export_format, MarkdownExporter(exporter_config))
        content = exporter.export(session_data)
        
        # Save to temp file
        ext_map = {'Markdown': 'md', 'JSON': 'json', 'HTML': 'html'}
        extension = ext_map.get(export_format, 'md')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{extension}', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        # Return FULL content, not truncated
        return content, temp_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Export failed: {e}", None


# =============================================================================
# CONFIG FUNCTIONS
# =============================================================================

def update_config(*args):
    """Update configuration from UI inputs"""
    try:
        return {
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


# =============================================================================
# GRADIO INTERFACE - Full featured, matching gradio_app.py
# =============================================================================

def create_gradio_interface():
    """Create enhanced Gradio interface - exact match to gradio_app.py"""
    
    with gr.Blocks(title="Indonesian Legal Assistant") as interface:
        
        # =====================================================================
        # LOGIN PANEL - Centered with CSS flexbox, no top banner
        # =====================================================================
        with gr.Column(visible=True, elem_id="login-panel") as login_panel:
            gr.HTML("""
            <style>
                #login-panel {
                    display: flex !important;
                    flex-direction: column !important;
                    justify-content: center !important;
                    align-items: center !important;
                    min-height: 80vh !important;
                }
                #login-panel > div {
                    max-width: 300px !important;
                    width: 100% !important;
                }
                #login-panel button {
                    max-width: 300px !important;
                    width: 100% !important;
                }
            </style>
            """)
            gr.Markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <span style="font-size: 2em;">🏛️</span><br>
                <strong style="color: #1e3a5f;">Indonesian Legal Assistant</strong>
            </div>
            """)
            username_input = gr.Textbox(label="Username", placeholder="demo")
            password_input = gr.Textbox(label="Password", type="password", placeholder="demo123")
            login_btn = gr.Button("🔓 Login", variant="primary")
            login_error = gr.Markdown("")
            gr.Markdown("<p style='text-align: center; color: #888; font-size: 12px; margin-top: 10px;'>Demo: demo/demo123 or admin/admin123</p>")
        
        # =====================================================================
        # MAIN APP
        # =====================================================================
        with gr.Column(visible=False) as main_app:
            # Logout button only (no header)
            with gr.Row():
                gr.Column(scale=9)
                logout_btn = gr.Button("🚪 Logout", size="sm")
            
            with gr.Tabs():
                # =====================================================================
                # CHAT TAB
                # =====================================================================
                with gr.TabItem("💬 Konsultasi Hukum"):
                    chatbot = gr.Chatbot(
                        height="75vh",
                        show_label=False,
                        autoscroll=True
                    )
                    
                    # Input row (same style as gradio_app.py)
                    with gr.Row():
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
                    
                    # 8 Examples with 2 per page
                    with gr.Row():
                        with gr.Column():
                            gr.Examples(
                                examples=EXAMPLE_QUERIES,
                                inputs=msg_input,
                                examples_per_page=2,
                                label=""
                            )
                
                # =====================================================================
                # SEARCH TAB
                # =====================================================================
                with gr.TabItem("🔍 Pencarian Dokumen"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            search_query = gr.Textbox(
                                label="Query Pencarian",
                                placeholder="Cari dokumen hukum...",
                                lines=2
                            )
                            search_num = gr.Slider(1, 20, value=5, step=1, label="Jumlah Hasil")
                            search_btn = gr.Button("🔍 Cari Dokumen", variant="primary")
                        
                        with gr.Column(scale=7):
                            search_result = gr.Markdown(label="Hasil Pencarian")
                
                # =====================================================================
                # SETTINGS TAB - Full settings matching gradio_app.py
                # =====================================================================
                with gr.TabItem("⚙️ Pengaturan Sistem"):
                    with gr.Row():
                        with gr.Column():
                            # Basic Settings
                            with gr.Group():
                                gr.Markdown("#### 🎯 Basic Settings")
                                final_top_k = gr.Slider(1, 10, value=3, step=1, label="Final Top K Results")
                                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="LLM Temperature")
                                max_new_tokens = gr.Slider(512, 8192, value=2048, step=256, label="Max New Tokens")
                                thinking_mode = gr.Radio(
                                    choices=["Low", "Medium", "High"],
                                    value="Low",
                                    label="Thinking Level",
                                    info="Higher levels provide deeper legal analysis but take more time."
                                )
                            
                            # Research Team Settings
                            with gr.Group():
                                gr.Markdown("#### 👥 Research Team Configuration")
                                research_team_size = gr.Slider(1, 5, value=4, step=1, label="Team Size")
                                enable_cross_validation = gr.Checkbox(label="Enable Cross-Validation", value=True)
                                enable_devil_advocate = gr.Checkbox(label="Enable Devil's Advocate", value=True)
                                consensus_threshold = gr.Slider(0.3, 0.9, value=0.6, step=0.05, label="Consensus Threshold")
                        
                        with gr.Column():
                            # Display Settings
                            with gr.Group():
                                gr.Markdown("#### 💬 Display Settings")
                                show_thinking = gr.Checkbox(label="Show Thinking Process", value=True)
                                show_sources = gr.Checkbox(label="Show Legal Sources", value=True)
                                show_metadata = gr.Checkbox(label="Show Research Metadata", value=True)
                            
                            # LLM Generation Settings
                            with gr.Group():
                                gr.Markdown("#### 🤖 LLM Generation Settings")
                                top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Top P (Nucleus Sampling)")
                                top_k_gen = gr.Slider(1, 100, value=20, step=1, label="Top K (Generation)")
                                min_p = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="Min P")
                            
                            # System Info Group
                            with gr.Group():
                                gr.Markdown("#### 📊 System Information")
                                with gr.Row():
                                    info_btn = gr.Button("📊 System Info")
                                    health_btn = gr.Button("🏥 Health Check")
                                    reset_btn = gr.Button("🔄 Reset Defaults")
                                system_output = gr.Markdown()
                            
                            # Test Runners
                            with gr.Group():
                                gr.Markdown("#### 🧪 Test Runners")
                                with gr.Row():
                                    test_btn = gr.Button("🧪 Run Conversational Test (8 Questions)", variant="primary")
                                    stress_btn = gr.Button("⚡ Run Stress Test")
                
                # =====================================================================
                # EXPORT TAB
                # =====================================================================
                with gr.TabItem("📥 Export Conversation"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            export_format = gr.Radio(
                                ["Markdown", "JSON", "HTML"],
                                value="Markdown",
                                label="Export Format"
                            )
                            export_btn = gr.Button("📥 Export", variant="primary")
                            export_file = gr.File(label="Download File")
                        
                        with gr.Column(scale=2):
                            export_output = gr.Textbox(
                                label="Export Preview",
                                lines=20,
                                max_lines=50
                            )
            
            # =====================================================================
            # STATE AND EVENT HANDLERS
            # =====================================================================
            config_state = gr.State(DEFAULT_CONFIG)
            
            config_inputs = [
                final_top_k, temperature, max_new_tokens, thinking_mode,
                research_team_size, enable_cross_validation, enable_devil_advocate,
                consensus_threshold, top_p, top_k_gen, min_p
            ]
            
            # Config updates
            for inp in config_inputs:
                inp.change(update_config, inputs=config_inputs, outputs=config_state)
            
            reset_btn.click(reset_to_defaults, outputs=config_inputs)
            info_btn.click(get_system_info, outputs=system_output)
            health_btn.click(format_health_report, outputs=system_output)
            
            # Chat handlers (submit_btn is built into Textbox)
            msg_input.submit(
                chat_with_legal_rag,
                [msg_input, chatbot, config_state, show_thinking, show_sources, show_metadata],
                [chatbot, msg_input]
            )
            
            # Search handlers
            search_btn.click(search_documents, [search_query, search_num], [search_result])
            search_query.submit(search_documents, [search_query, search_num], [search_result])
            
            # Test runners
            test_btn.click(
                run_conversational_test,
                [chatbot, config_state, show_thinking, show_sources, show_metadata],
                [chatbot, msg_input]
            )
            stress_btn.click(
                run_stress_test,
                [chatbot, config_state, show_thinking, show_sources, show_metadata],
                [chatbot, msg_input]
            )
            
            # Export handler
            export_btn.click(
                export_conversation_handler,
                [export_format, chatbot],
                [export_output, export_file]
            )
        
        # =====================================================================
        # AUTH HANDLERS
        # =====================================================================
        login_btn.click(
            dummy_login,
            [username_input, password_input],
            [login_panel, main_app, login_error]
        )
        username_input.submit(
            dummy_login,
            [username_input, password_input],
            [login_panel, main_app, login_error]
        )
        password_input.submit(
            dummy_login,
            [username_input, password_input],
            [login_panel, main_app, login_error]
        )
        logout_btn.click(
            dummy_logout,
            outputs=[login_panel, main_app, chatbot]
        )
    
    interface.queue()
    return interface


# =============================================================================
# LAUNCH FUNCTION
# =============================================================================

def launch_app(share=False, server_port=7860, server_name="0.0.0.0"):
    """Launch the Gradio application"""
    print("\n" + "=" * 60)
    print("🏛️ LEGAL RAG INDONESIA - UNIFIED API UI")
    print("=" * 60)
    print(f"API URL: {DEFAULT_CONFIG['api_url']}")
    print(f"API Status: {initialize_api()}")
    print("=" * 60 + "\n")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )


# Backward compatibility alias
launch_unified_app = launch_app


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    launch_app(share=share, server_name=server_name, server_port=server_port)
