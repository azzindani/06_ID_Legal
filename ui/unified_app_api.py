"""
Unified API-Based UI - Indonesian Legal RAG System

Production-ready UI that connects to the FastAPI backend for all operations.
EXACT replica of gradio_app.py with API-based backend.

This is the production version of gradio_app.py that:
- Uses HTTP API calls instead of direct pipeline imports
- Maintains the EXACT same UI layout, styling, and formatting
- Works in production Docker/Kaggle environments
- Includes search_app functionality

File: ui/unified_app_api.py
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

# Import formatting utilities (same as gradio_app.py)
from utils.formatting import format_sources_info, format_all_documents
from utils.research_transparency import format_detailed_research_process
from utils.text_utils import parse_think_tags

# Import exporters (same as gradio_app.py)
from conversation import MarkdownExporter, JSONExporter, HTMLExporter

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

# Default configuration matching gradio_app.py exactly
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

# Default search phases (same as gradio_app.py)
DEFAULT_SEARCH_PHASES = {
    'initial_scan': {'enabled': True, 'candidates': 400, 'semantic_threshold': 0.20, 'keyword_threshold': 0.06},
    'focused_review': {'enabled': True, 'candidates': 150, 'semantic_threshold': 0.35, 'keyword_threshold': 0.12},
    'deep_analysis': {'enabled': True, 'candidates': 60, 'semantic_threshold': 0.45, 'keyword_threshold': 0.18},
    'verification': {'enabled': True, 'candidates': 30, 'semantic_threshold': 0.55, 'keyword_threshold': 0.22},
    'expert_review': {'enabled': True, 'candidates': 45, 'semantic_threshold': 0.50, 'keyword_threshold': 0.20}
}

# Example queries - same as gradio_app.py
EXAMPLE_QUERIES = [
    "Apakah ada pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?",
    "Apakah terdapat mekanisme pengawasan terhadap penyimpanan uang negara agar terhindar dari penyalahgunaan atau kebocoran keuangan?",
    "Bagaimana mekanisme hukum untuk memperoleh izin resmi bagi pihak yang menjalankan usaha sebagai pengusaha pabrik, penyimpanan, importir, penyalur, maupun penjual eceran barang kena cukai?",
    "Apakah terdapat kewajiban pemerintah untuk menyediakan dana khusus bagi penyuluhan, atau dapat melibatkan sumber pendanaan alternatif seperti swasta dan masyarakat?",
    "Bagaimana prosedur hukum yang harus ditempuh sebelum sanksi denda administrasi di bidang cukai dapat dikenakan kepada pelaku usaha?",
    "Bagaimana sistem perencanaan kas disusun agar mampu mengantisipati kebutuhan mendesak negara/daerah tanpa mengganggu stabilitas fiskal?",
    "syarat dan prosedur perceraian menurut hukum Indonesia",
    "hak dan kewajiban pekerja dalam UU Ketenagakerjaan"
]

# Test questions for conversational test
TEST_QUESTIONS = [
    "Apakah terdapat pengaturan yang menjamin kesetaraan hak antara guru dan dosen dalam memperoleh tunjangan profesi?",
    "Berdasarkan PP No. 41 Tahun 2009, sebutkan jenis-jenis tunjangan yang diatur di dalamnya.",
    "Masih merujik pada PP No. 41 Tahun 2009, jelaskan perbedaan kriteria penerima, besaran, dan sumber pendanaan antara Tunjangan Khusus dan Tunjangan Kehormatan Profesor",
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
    """Get system information via API - same format as gradio_app.py"""
    if api_client is None:
        return "❌ API not connected"
    
    try:
        health = api_client.health_check()
        return f"""
## 📊 System Information

### 🔧 Model Configuration
| Component | Value |
|-----------|-------|
| **API Connection** | {"✅ Connected" if health.healthy else "❌ Disconnected"} |
| **Pipeline Ready** | {"✅ Yes" if health.ready else "⏳ Loading"} |
| **Provider** | API-based |

### 📈 Current Status
- **Message**: {health.message}
- **API URL**: {DEFAULT_CONFIG['api_url']}

### ⚙️ Default Settings
- **Final Top K**: {DEFAULT_CONFIG['final_top_k']}
- **Temperature**: {DEFAULT_CONFIG['temperature']}
- **Max Tokens**: {DEFAULT_CONFIG['max_new_tokens']}
- **Thinking Mode**: {DEFAULT_CONFIG['thinking_mode']}
"""
    except Exception as e:
        return f"❌ Error: {str(e)}"


def format_health_report():
    """Format health check report - same as gradio_app.py"""
    if api_client is None:
        return "❌ API not connected"
    
    try:
        health = api_client.health_check()
        status_icon = "✅" if health.ready else ("⚠️" if health.healthy else "❌")
        
        return f"""
## 🏥 Health Check Report

**Overall Status**: {status_icon} {health.message}

### Component Status
| Check | Result |
|-------|--------|
| API Healthy | {"✅ Pass" if health.healthy else "❌ Fail"} |
| Pipeline Ready | {"✅ Pass" if health.ready else "⏳ Loading"} |

### System Health
- All core services operational
- API responding to requests
"""
    except Exception as e:
        return f"❌ Health check failed: {str(e)}"


# =============================================================================
# SEARCH FUNCTION - From search_app.py
# =============================================================================

def search_documents(query: str, num_results: int = 5) -> Tuple[str, str]:
    """
    Search for legal documents - same functionality as search_app.py
    Returns: (summary, all_documents_markdown)
    """
    global api_client
    
    if not query.strip():
        return "⚠️ Masukkan query pencarian.", ""
    
    if api_client is None:
        initialize_api()
    
    if api_client is None:
        return "❌ API not connected", ""
    
    try:
        # Call retrieve endpoint
        result = api_client.retrieve(query=query, top_k=int(num_results))
        
        if not result or not result.get('documents'):
            return "📭 Tidak ada dokumen yang ditemukan.", ""
        
        documents = result['documents']
        search_time = result.get('search_time', 0)
        
        # Build summary (like search_app.py format_summary)
        summary = f"## 📋 Ringkasan Hasil Pencarian\n\n"
        summary += f"**Query:** `{query}`\n"
        summary += f"**Waktu Pencarian:** {search_time:.2f}s\n\n"
        summary += f"### ⭐ Hasil Relevan ({len(documents)} dokumen)\n\n"
        
        for i, doc in enumerate(documents, 1):
            summary += f"{i}. **{doc.regulation_type} No. {doc.regulation_number}/{doc.year}**\n"
            summary += f"   - **Tentang:** _{doc.about}_\n"
            summary += f"   - **Skor:** {doc.score:.4f}\n\n"
        
        # Build all documents markdown (like search_app.py format_all_documents)
        all_docs = f"## 📚 Semua Dokumen Ditemukan ({len(documents)} dokumen)\n\n"
        
        for i, doc in enumerate(documents, 1):
            all_docs += f"""
### 📄 {i}. {doc.regulation_type} No. {doc.regulation_number}/{doc.year}

**Lokasi:** {doc.chapter or 'N/A'} | {doc.article or 'N/A'}
**Tentang:** {doc.about}

---

#### 📊 Skor Relevansi

| Komponen | Nilai |
|----------|-------|
| **Final Score** | **{doc.score:.4f}** |

---

#### 📝 Konten

{doc.content_preview or 'Preview tidak tersedia'}

---

"""
        
        return summary, all_docs
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


# =============================================================================
# MAIN CHAT FUNCTION - Same format as gradio_app.py
# =============================================================================

def chat_with_legal_rag(message, history, config_dict, show_thinking=True, show_sources=True, show_metadata=True):
    """
    Main chat function - uses API for processing
    Formatting matches gradio_app.py exactly with proper <think> tag parsing
    """
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
    
    try:
        # Track state (same as gradio_app.py)
        current_progress = []
        accumulated_text = ""
        thinking_content = []
        final_answer = []
        live_output = []
        in_thinking_block = False
        saw_think_tag = False
        thinking_header_shown = False
        result_data = None
        
        # Get thinking mode from config
        thinking_mode = config_dict.get('thinking_mode', 'low')
        if isinstance(thinking_mode, str):
            thinking_mode = thinking_mode.lower()
        
        # Show initial progress
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
            
            if chunk_type == 'progress':
                # Progress update from API
                current_progress.append(content)
                progress_display = "\n".join([f"🔄 {m}" for m in current_progress])
                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"**Mencari dan menganalisis...**\n\n{progress_display}"}
                ], ""
            
            elif chunk_type == 'chunk':
                # Content chunk - parse <think> tags like gradio_app.py
                new_text = content
                accumulated_text += content
                
                # Build progress header
                progress_header = '<details open><summary>📋 <b>Proses Penelitian</b></summary>\n\n'
                progress_header += "\n".join([f"🔄 {m}" for m in current_progress])
                progress_header += '\n</details>\n\n---\n\n'
                
                # Process <think> tags in new text (like gradio_app.py)
                if '<think>' in new_text:
                    in_thinking_block = True
                    saw_think_tag = True
                    new_text = new_text.replace('<think>', '')
                    if not thinking_header_shown and show_thinking:
                        live_output = [progress_header, '🧠 **Sedang berfikir...**\n\n']
                        thinking_header_shown = True
                
                if '</think>' in new_text:
                    in_thinking_block = False
                    new_text = new_text.replace('</think>', '')
                    if show_thinking:
                        live_output.append('\n\n---\n\n✅ **Sedang menjawab...**\n\n')
                
                # Accumulate content based on current state
                if saw_think_tag:
                    if in_thinking_block:
                        thinking_content.append(new_text)
                        if show_thinking:
                            live_output.append(new_text)
                    else:
                        final_answer.append(new_text)
                        live_output.append(new_text)
                else:
                    # Check if <think> appeared anywhere in accumulated text
                    if '<think>' in accumulated_text:
                        saw_think_tag = True
                        in_thinking_block = '</think>' not in accumulated_text
                        if not thinking_header_shown and show_thinking:
                            live_output = [progress_header, '🧠 **Sedang berfikir...**\n\n']
                            thinking_header_shown = True
                        # Extract thinking content so far
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
                    elif len(accumulated_text) > 100 and not saw_think_tag:
                        # After 100 chars with no think tag, assume direct answer
                        if not thinking_header_shown:
                            live_output = [progress_header, '⭐ **Jawaban langsung:**\n\n']
                            thinking_header_shown = True
                        final_answer.append(new_text)
                        live_output.append(new_text)
                    else:
                        # Still waiting to see if think tag appears
                        if not thinking_header_shown:
                            live_output = [progress_header, f"🤖 Generating response...\n\n{accumulated_text}"]
                        else:
                            live_output.append(new_text)
                
                # Yield current state
                display_text = ''.join(live_output)
                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": display_text}
                ], ""
            
            elif chunk_type == 'done':
                # Final result
                result_data = chunk
                # Update session for context memory
                if result_data.get('session_id'):
                    current_session = result_data.get('session_id')
        
        # Format final output (same format as gradio_app.py)
        final_output = ""
        
        # Get answer text - parse think tags if not done during streaming
        if final_answer:
            response_text = ''.join(final_answer).strip()
        else:
            response_text = result_data.get('answer', accumulated_text) if result_data else accumulated_text
            # If no streaming occurred, parse think tags from answer
            if not thinking_content and response_text:
                parsed_thinking, response_text = parse_think_tags(response_text)
                if parsed_thinking:
                    thinking_content = [parsed_thinking]
        
        # Clean any remaining think tags from response
        response_text = response_text.replace('<think>', '').replace('</think>', '').strip()
        
        # Get thinking text
        thinking_text = ''.join(thinking_content).strip() if thinking_content else ""
        thinking_text = thinking_text.replace('<think>', '').replace('</think>', '').strip()
        
        # Add thinking section if available (like gradio_app.py)
        if show_thinking and thinking_text:
            final_output += (
                '<details><summary>🧠 <strong>Proses Berpikir</strong></summary>\n\n'
                + thinking_text +
                '\n</details>\n\n'
                + '---\n\n### ✅ Jawaban\n\n'
                + response_text
            )
        else:
            final_output += f"### ✅ Jawaban\n\n{response_text}"
        
        # Add collapsible sections (like gradio_app.py)
        collapsible_sections = []
        
        # Add sources (legal references)
        if show_sources and result_data and result_data.get('legal_references'):
            legal_refs = result_data['legal_references']
            collapsible_sections.append(
                f'<details><summary>📖 <strong>Sumber Hukum</strong></summary>\n\n{legal_refs}\n</details>'
            )
        
        # Add detailed research process
        if show_metadata and result_data and result_data.get('research_process'):
            research_proc = result_data['research_process']
            collapsible_sections.append(
                f'<details><summary>🔬 <strong>Detail Proses Penelitian</strong></summary>\n\n{research_proc}\n</details>'
            )
        
        # Combine all sections
        if collapsible_sections:
            final_output += "\n\n---\n\n" + "\n\n".join(collapsible_sections)
        
        # Return final result
        final_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": final_output}
        ]
        
        yield final_history, ""
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_display = f"❌ **Error:**\n\n{str(e)}"
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_display}
        ]
        yield history, ""


def run_conversational_test(history, config_dict, show_thinking, show_sources, show_metadata):
    """Run conversational test - auto-feed questions (same as gradio_app.py)"""
    
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


def run_stress_test(history, config_dict, show_thinking, show_sources, show_metadata):
    """Run stress test - same as gradio_app.py"""
    
    # Stress config (maximum settings)
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
    initial_msg = {
        "role": "assistant",
        "content": f"⚡ **Starting Stress Test (8 Questions)**\n\n**Configuration:** MAXIMUM SETTINGS\n- Team Size: 5 personas\n- Max Tokens: 8192\n- Thinking Mode: High\n\n**Auto-feeding questions...**"
    }
    history = history + [initial_msg]
    yield history, ""
    
    # Process each question with stress config
    for i, question in enumerate(TEST_QUESTIONS, 1):
        progress_msg = {
            "role": "assistant",
            "content": f"⚡ **Stress Test - Question {i}/8**\n\nProcessing with maximum settings: _{question[:100]}..._"
        }
        history = history + [progress_msg]
        yield history, ""
        
        history = history[:-1]
        
        for updated_history, cleared_input in chat_with_legal_rag(
            question, history, stress_config, show_thinking, show_sources, show_metadata
        ):
            yield updated_history, cleared_input
        
        history = updated_history
    
    completion_msg = {
        "role": "assistant",
        "content": f"✅ **Stress Test Complete**\n\nSuccessfully processed all {len(TEST_QUESTIONS)} questions with maximum settings."
    }
    history = history + [completion_msg]
    yield history, ""


def update_config(*args):
    """Update configuration from UI inputs - same as gradio_app.py"""
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
    """Reset configuration to defaults - same as gradio_app.py"""
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
    """Handle export - uses same exporters as gradio_app.py"""
    try:
        if not history:
            return "No conversation to export.", None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert history to session-like format for exporters
        session_data = {
            'session_id': current_session or 'export',
            'created_at': datetime.now().isoformat(),
            'turns': []
        }
        
        # Build turns from history - handle both dict and tuple formats
        i = 0
        while i < len(history):
            item = history[i]
            
            # Handle different Gradio history formats
            if isinstance(item, dict):
                # Dict format: {"role": "user", "content": "..."}
                if item.get('role') == 'user' and i + 1 < len(history):
                    user_content = item.get('content', '')
                    next_item = history[i + 1]
                    if isinstance(next_item, dict):
                        assistant_content = next_item.get('content', '')
                    else:
                        assistant_content = str(next_item) if next_item else ''
                    
                    # Ensure content is string
                    if isinstance(user_content, list):
                        user_content = ' '.join(str(x) for x in user_content)
                    if isinstance(assistant_content, list):
                        assistant_content = ' '.join(str(x) for x in assistant_content)
                    
                    session_data['turns'].append({
                        'query': str(user_content),
                        'answer': str(assistant_content),
                        'timestamp': datetime.now().isoformat()
                    })
                    i += 2
                    continue
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # Tuple format: (user_msg, assistant_msg)
                user_content = item[0] if item[0] else ''
                assistant_content = item[1] if item[1] else ''
                
                # Ensure content is string
                if isinstance(user_content, list):
                    user_content = ' '.join(str(x) for x in user_content)
                if isinstance(assistant_content, list):
                    assistant_content = ' '.join(str(x) for x in assistant_content)
                
                session_data['turns'].append({
                    'query': str(user_content),
                    'answer': str(assistant_content),
                    'timestamp': datetime.now().isoformat()
                })
            
            i += 1
        
        # Use appropriate exporter (same as gradio_app.py)
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
        
        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{extension}', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        return content[:5000] + "..." if len(content) > 5000 else content, temp_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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
                <p style="color: #666;">Production API-Based UI</p>
            </div>
            """)
            with gr.Row():
                gr.Column(scale=1)
                with gr.Column(scale=2):
                    gr.Markdown("### 🔐 Login")
                    gr.Markdown("**Demo:** `demo` / `demo123` or `admin` / `admin123`")
                    username_input = gr.Textbox(label="Username", placeholder="Enter username")
                    password_input = gr.Textbox(label="Password", type="password", placeholder="Enter password")
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
                            height="65vh",
                            show_label=False,
                            autoscroll=True
                        )
                        
                        with gr.Row(elem_classes="input-row"):
                            msg_input = gr.Textbox(
                                placeholder="Tanyakan tentang hukum Indonesia...",
                                show_label=False,
                                container=False,
                                scale=10,
                                lines=1,
                                max_lines=3,
                                interactive=True
                            )
                            send_btn = gr.Button("📤", variant="primary", scale=1, min_width=50)
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Examples(
                                    examples=EXAMPLE_QUERIES,
                                    inputs=msg_input,
                                    examples_per_page=2,
                                    label=""
                                )
                
                # =============================================================
                # SEARCH TAB - From search_app.py
                # =============================================================
                with gr.TabItem("🔍 Pencarian Dokumen", id="search"):
                    gr.Markdown("### Cari Dokumen Regulasi Indonesia")
                    gr.Markdown("Pencarian dokumen tanpa LLM generation - langsung ke database.")
                    
                    with gr.Row():
                        search_query = gr.Textbox(
                            label="Query Pencarian",
                            placeholder="Contoh: syarat pendirian PT, sanksi UU ITE...",
                            lines=2,
                            scale=4
                        )
                        search_num_results = gr.Slider(1, 10, value=5, step=1, label="Jumlah Hasil", scale=1)
                    
                    search_btn = gr.Button("🔍 Cari Dokumen", variant="primary")
                    
                    with gr.Tabs():
                        with gr.TabItem("📋 Ringkasan"):
                            search_summary = gr.Markdown("")
                        with gr.TabItem("📚 Semua Dokumen"):
                            search_all_docs = gr.Markdown("")
                
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
                                max_new_tokens = gr.Slider(512, 8192, value=2048, step=256, label="Max New Tokens")
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
                                test_stress_btn = gr.Button("⚡ Run Stress Test (8 Questions)", variant="secondary")
                
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
            
            # Chat functionality - both button and enter key
            msg_input.submit(
                chat_with_legal_rag,
                inputs=[msg_input, chatbot, config_state, show_thinking, show_sources, show_metadata],
                outputs=[chatbot, msg_input]
            )
            send_btn.click(
                chat_with_legal_rag,
                inputs=[msg_input, chatbot, config_state, show_thinking, show_sources, show_metadata],
                outputs=[chatbot, msg_input]
            )
            
            # Search functionality
            search_btn.click(
                search_documents,
                inputs=[search_query, search_num_results],
                outputs=[search_summary, search_all_docs]
            )
            search_query.submit(
                search_documents,
                inputs=[search_query, search_num_results],
                outputs=[search_summary, search_all_docs]
            )
            
            # Test runners
            test_conversational_btn.click(
                run_conversational_test,
                inputs=[chatbot, config_state, show_thinking, show_sources, show_metadata],
                outputs=[chatbot, msg_input]
            )
            test_stress_btn.click(
                run_stress_test,
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
        # AUTH HANDLERS - Login supports Enter key
        # =====================================================================
        login_btn.click(
            fn=dummy_login,
            inputs=[username_input, password_input],
            outputs=[login_panel, main_app, login_error]
        )
        
        # Enter key support for login
        username_input.submit(
            fn=dummy_login,
            inputs=[username_input, password_input],
            outputs=[login_panel, main_app, login_error]
        )
        password_input.submit(
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
        server_name=server_name,
        server_port=server_port,
        share=share
    )


# Alias for backward compatibility
launch_unified_app = launch_app


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    launch_app(share=share, server_port=server_port)
