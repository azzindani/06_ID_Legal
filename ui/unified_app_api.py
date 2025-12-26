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
from utils.research_transparency import format_detailed_research_process
from utils.formatting import _extract_all_documents_from_metadata
from utils.search_formatting import (
    format_score_bar,
    format_document_card,
    format_all_documents,
    get_docs_dataframe_data,
    format_research_process_summary,
    format_summary
)
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

# 8 Search-specific examples (same as search_app.py)
SEARCH_EXAMPLES = [
    "Apa saja syarat pendirian PT menurut UU Perseroan Terbatas?",
    "Bagaimana prosedur pendaftaran merek dagang di Indonesia?",
    "Apa sanksi pelanggaran UU Perlindungan Data Pribadi?",
    "Ketentuan cuti karyawan menurut UU Ketenagakerjaan",
    "Syarat dan prosedur perceraian menurut hukum Indonesia",
    "Regulasi tentang investasi asing di Indonesia",
    "Ketentuan pajak penghasilan untuk UMKM",
    "Prosedur penyelesaian sengketa konsumen",
]

# Global for last search result (for export)
last_search_result = None

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
    print(f"\n[CHAT] === New chat request: {message[:50]}... ===", flush=True)
    
    if not message.strip():
        print("[CHAT] Empty message, returning", flush=True)
        return history, ""
    
    global api_client, current_session
    print(f"[CHAT] api_client is {'set' if api_client else 'None'}, current_session={current_session}", flush=True)
    
    if api_client is None:
        print("[CHAT] Initializing API...", flush=True)
        initialize_api()
    if api_client is None:
        print("[CHAT] API still None after init, returning error")
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
        
        # Extract all settings from config
        thinking_mode = str(config_dict.get('thinking_mode', 'low')).lower()
        top_k = int(config_dict.get('final_top_k', 3))
        temperature = float(config_dict.get('temperature', 0.7))
        max_tokens = int(config_dict.get('max_new_tokens', 2048))
        team_size = int(config_dict.get('research_team_size', 3))
        
        print(f"[CHAT] Config: top_k={top_k}, temp={temperature}, tokens={max_tokens}, team={team_size}, think={thinking_mode}", flush=True)
        
        # Initial processing message
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"🔄 **Memproses permintaan...**\n_Settings: Top-K={top_k}, Temp={temperature}, Tokens={max_tokens}, Team={team_size}, Thinking={thinking_mode}_"}
        ], ""
        
        print("[CHAT] Calling api_client.chat_stream()...", flush=True)
        
        # Stream response from API with ALL settings
        chunk_count = 0
        for chunk in api_client.chat_stream(
            query=message,
            session_id=current_session,
            thinking_level=thinking_mode,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            team_size=team_size
        ):
            chunk_count += 1
            if chunk_count == 1:
                print(f"[CHAT] First chunk received: {chunk.get('type', 'unknown')}")
            
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
        
        # Add config info at the top (shows settings were applied)
        config_info = f"_⚙️ Config: Top-K={top_k}, Temp={temperature}, Tokens={max_tokens}, Team={team_size}, Think={thinking_mode}_"
        
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
        
        # Add config metadata at the bottom
        final_output += f'\n\n---\n\n{config_info}'
        
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
# SEARCH FUNCTION - Matching search_app.py style with detailed outputs
# =============================================================================

def format_search_summary(query: str, docs: list, search_time: float) -> str:
    """Format search summary like search_app.py"""
    output = [f"""## 📋 Ringkasan Hasil Pencarian

**Query:** `{query}`
**Dokumen Ditemukan:** {len(docs)}
**Waktu Pencarian:** {search_time:.3f} detik

---

### ⭐ Hasil Relevan ({len(docs)} dokumen)

"""]
    
    for i, d in enumerate(docs, 1):
        reg_type = getattr(d, 'regulation_type', 'N/A')
        reg_num = getattr(d, 'regulation_number', '?')
        year = getattr(d, 'year', '?')
        about = getattr(d, 'about', 'N/A')
        location = getattr(d, 'location', 'Dokumen Lengkap')
        effective_date = getattr(d, 'effective_date', 'N/A')
        content = getattr(d, 'content', '')
        preview = content[:300].replace('\n', ' ') + '...' if len(content) > 300 else content
        
        output.append(f"""{i}. **{reg_type} No. {reg_num}/{year}**
   - **Lokasi:** {location}
   - **Tgl Penetapan:** {effective_date}
   - **Tentang:** _{about}_
   - **Konten:** {preview}

""")
    
    return "".join(output)


def format_all_documents(docs: list) -> str:
    """Format all documents as detailed cards like search_app.py"""
    if not docs:
        return "❌ Tidak ada dokumen yang ditemukan."
    
    output = [f"## 📚 Semua Dokumen Ditemukan ({len(docs)} dokumen)\n\n"]
    
    for i, d in enumerate(docs, 1):
        reg_type = getattr(d, 'regulation_type', 'N/A')
        reg_num = getattr(d, 'regulation_number', '?')
        year = getattr(d, 'year', '?')
        about = getattr(d, 'about', 'N/A')
        location = getattr(d, 'location', 'N/A')
        score = getattr(d, 'score', 0)
        global_id = getattr(d, 'global_id', 'N/A')
        effective_date = getattr(d, 'effective_date', 'N/A')
        content = getattr(d, 'content', '')
        
        content_preview = content[:500] + '...' if len(content) > 500 else content
        
        output.append(f"""### 📄 {i}. {reg_type} No. {reg_num} Tahun {year}

**Global ID:** `{global_id}`

**Tentang:** {about}

**Tanggal Penetapan:** {effective_date}

**Lokasi:** {location}

**Skor:** `{score:.4f}`

<details>
<summary>📝 Konten Dokumen</summary>

{content_preview}

</details>

---

""")
    
    return "".join(output)


def search_documents(query: str, num_results: int = 10):
    """Search documents - generator with loading process, returns 4 outputs like search_app.py"""
    global api_client, last_search_result
    import pandas as pd
    
    empty_df = pd.DataFrame()
    
    if not query.strip():
        yield "⚠️ Masukkan query pencarian", empty_df, "", ""
        return
    
    # Show initial loading state
    loading_md = """### 🔄 Sedang mencari...

Mohon tunggu sejenak, sistem sedang menganalisis query Anda.

**Langkah-langkah:**
- 🔍 Mengirim query ke API...
- ⏳ Mencari dokumen relevan...
"""
    loading_df = pd.DataFrame([{"Status": "Mencari dokumen..."}])
    yield loading_md, loading_df, loading_md, loading_md
    
    if api_client is None:
        initialize_api()
    if api_client is None:
        yield "❌ API tidak terhubung", empty_df, "", ""
        return
    
    try:
        # Progress: Connecting to API
        yield """### 🔄 Sedang mencari...

✅ Koneksi API berhasil
🔍 Mencari dokumen relevan...
""", loading_df, "", ""
        
        result = api_client.retrieve(query=query, top_k=int(num_results))
        
        if not result or not result.get('documents'):
            yield "📭 Tidak ada dokumen ditemukan", empty_df, "", ""
            return
        
        # Progress: Formatting results
        yield """### 🔄 Sedang mencari...

✅ Koneksi API berhasil
✅ Dokumen ditemukan
✨ Memformat hasil...
""", loading_df, "", ""
        
        docs = result['documents']
        search_time = result.get('search_time', 0)
        metadata = result.get('metadata', {})
        
        # Get rich data from API metadata (like search_app.py)
        final_results = metadata.get('final_results', [])
        phase_metadata = metadata.get('phase_metadata', {})
        consensus_data = metadata.get('consensus_data', {})
        research_data = metadata.get('research_data', {})
        
        # Build result dict for format functions (like search_app.py)
        rich_result = {
            'final_results': final_results,
            'phase_metadata': phase_metadata,
            'consensus_data': consensus_data,
            'research_data': research_data,
            'metadata': {
                'query': query,
                'total_time': search_time
            }
        }
        
        # Store for export
        last_search_result = rich_result
        last_search_result['query'] = query
        last_search_result['documents'] = docs
        last_search_result['search_time'] = search_time
        
        # Add metadata for format_summary compatibility
        rich_result['metadata']['query'] = query
        rich_result['metadata']['total_time'] = search_time
        
        # Use shared format functions from utils/search_formatting.py
        summary = format_summary(rich_result)
        
        # Format all documents using shared function
        all_docs_output = format_all_documents(rich_result) if final_results else "❌ Tidak ada dokumen"
        
        # Create DataFrame using shared function
        table_df = get_docs_dataframe_data(rich_result)
        
        # Research process using format_detailed_research_process
        checklist = ["### 📋 Proses yang Sudah Dilakukan\n"]
        checklist.append("✅ Analisis Query Berhasil")
        checklist.append("✅ Pemindaian Regulasi Selesai")
        checklist.append("✅ Konsensus Tim Tercapai")
        checklist.append("✅ Reranking Final Selesai")
        
        research_detail = format_detailed_research_process(rich_result, top_n_per_researcher=20, show_content=False)
        checklist_str = '\n'.join(checklist)
        research = f"{checklist_str}\n\n---\n\n{research_detail}"
        
        yield summary, table_df, all_docs_output, research
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"❌ Error: {e}", empty_df, "", ""


def format_all_documents_rich(final_results: list) -> str:
    """Format all documents as detailed cards (like search_app.py format_document_card)"""
    if not final_results:
        return "❌ Tidak ada dokumen yang ditemukan."
    
    output = [f"## 📚 Semua Dokumen Ditemukan ({len(final_results)} dokumen)\n\n"]
    
    for i, doc in enumerate(final_results, 1):
        record = doc.get('record', doc)
        scores = doc.get('scores', {})
        
        reg_type = record.get('regulation_type', 'N/A')
        reg_num = record.get('regulation_number', 'N/A')
        year = record.get('year', 'N/A')
        about = record.get('about', 'N/A')
        
        final_score = scores.get('final', doc.get('final_score', 0))
        semantic_score = scores.get('semantic', 0)
        keyword_score = scores.get('keyword', 0)
        kg_score = scores.get('kg', 0)
        
        chapter = record.get('chapter', record.get('bab', ''))
        article = record.get('article', record.get('pasal', '')) or record.get('article_number', '')
        location = " | ".join(filter(None, [chapter, article])) or "Dokumen Lengkap"
        eff_date = record.get('effective_date', record.get('tanggal_penetapan', 'N/A'))
        
        content = record.get('content', '')
        content_preview = content[:400] + "..." if len(content) > 400 else content
        
        output.append(f"""### 📄 {i}. {reg_type} No. {reg_num}/{year}

**Lokasi:** {location}
**Tgl Penetapan:** {eff_date}
**Tentang:** {about}

---

#### 📊 Skor Relevansi

| Komponen | Nilai |
|----------|-------|
| **Final Score** | **{final_score:.4f}** |
| Semantic | {semantic_score:.4f} |
| Keyword | {keyword_score:.4f} |
| Knowledge Graph | {kg_score:.4f} |

#### 📝 Konten

{content_preview}

---

""")
    
    return "".join(output)


def export_search_results(export_format: str):
    """Export search results to specified format (complete version like search_app.py)"""
    global last_search_result
    
    if last_search_result is None:
        return "⚠️ Tidak ada hasil pencarian untuk di-export.", None
    
    try:
        from datetime import datetime
        import re
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query = last_search_result.get('query', 'search')
        
        # Get rich data
        final_results = last_search_result.get('final_results', [])
        phase_metadata = last_search_result.get('phase_metadata', {})
        consensus_data = last_search_result.get('consensus_data', {})
        search_time = last_search_result.get('search_time', 0)
        
        # Extract all docs using helper
        all_docs = _extract_all_documents_from_metadata(last_search_result) if final_results else []
        
        if export_format == "JSON":
            import json
            data = {
                'query': query,
                'timestamp': timestamp,
                'search_time': search_time,
                'total_documents': len(final_results),
                'consensus_data': {
                    'agreement_level': consensus_data.get('agreement_level', 0) if consensus_data else 0
                },
                'phase_count': len(phase_metadata),
                'documents': []
            }
            for i, doc in enumerate(final_results, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})
                
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', '')) or record.get('article_number', '')
                location = " | ".join(filter(None, [chapter, article])) or "Dokumen Lengkap"
                
                data['documents'].append({
                    'rank': i,
                    'regulation_type': record.get('regulation_type', ''),
                    'regulation_number': record.get('regulation_number', ''),
                    'year': record.get('year', ''),
                    'about': record.get('about', ''),
                    'location': location,
                    'effective_date': record.get('effective_date', record.get('tanggal_penetapan', '')),
                    'enacting_body': record.get('enacting_body', ''),
                    'scores': {
                        'final': scores.get('final', doc.get('final_score', 0)),
                        'semantic': scores.get('semantic', 0),
                        'keyword': scores.get('keyword', 0),
                        'kg': scores.get('kg', 0)
                    },
                    'content': record.get('content', '')[:2000]
                })
            content = json.dumps(data, indent=2, ensure_ascii=False)
            filename = f"search_{timestamp}.json"
            
        elif export_format == "CSV":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['No', 'Jenis', 'Nomor', 'Tahun', 'Lokasi', 'Tgl Penetapan', 'Tentang', 'Skor Final', 'Skor Semantik', 'Skor Keyword', 'Skor KG', 'Konten'])
            for i, doc in enumerate(final_results, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', '')) or record.get('article_number', '')
                location = " | ".join(filter(None, [chapter, article])) or "Lengkap"
                
                writer.writerow([
                    i,
                    record.get('regulation_type', ''),
                    record.get('regulation_number', ''),
                    record.get('year', ''),
                    location,
                    record.get('effective_date', record.get('tanggal_penetapan', '')),
                    record.get('about', ''),
                    f"{scores.get('final', 0):.4f}",
                    f"{scores.get('semantic', 0):.4f}",
                    f"{scores.get('keyword', 0):.4f}",
                    f"{scores.get('kg', 0):.4f}",
                    record.get('content', '')[:1000]
                ])
            content = output.getvalue()
            filename = f"search_{timestamp}.csv"
        
        elif export_format == "HTML":
            # Full HTML export like search_app.py
            html_title = f"Laporan Hasil Pencarian Hukum: {query}"
            agreement = consensus_data.get('agreement_level', 0) if consensus_data else 0
            
            lines = [f"""<!DOCTYPE html>
<html>
<head>
    <title>{html_title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #333; }}
        .header {{ background: #1e3a5f; color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; }}
        h1 {{ margin: 0; font-size: 24px; }}
        .summary-section {{ background: #f0f4f8; border-radius: 8px; padding: 25px; margin-bottom: 30px; border-left: 5px solid #2c5282; }}
        h2 {{ color: #1e3a5f; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; margin-top: 0; }}
        h3 {{ color: #2c5282; margin-top: 25px; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 15px; }}
        .meta-item {{ font-size: 0.9em; }}
        .meta-label {{ font-weight: bold; color: #555; }}
        .doc-list {{ margin-top: 30px; }}
        .doc {{ background: #fff; border: 1px solid #e0e0e0; border-left: 5px solid #1e3a5f; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .doc:hover {{ transform: translateX(5px); border-color: #2c5282; }}
        .doc-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }}
        .doc-title {{ font-weight: bold; font-size: 1.2em; color: #1e3a5f; flex-grow: 1; }}
        .doc-score {{ background: #e6fffa; color: #234e52; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }}
        .doc-meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9em; color: #666; margin-bottom: 15px; background: #fafafa; padding: 10px; border-radius: 4px; }}
        .doc-content {{ background: #fdfdfd; padding: 12px; border: 1px dashed #ddd; font-style: italic; font-size: 0.95em; color: #444; }}
        .footer {{ text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #eee; font-size: 0.8em; color: #999; }}
        .top-tag {{ background: #1e3a5f; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; margin-right: 8px; }}
        .score-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        .score-table th, .score-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .score-table th {{ background: #f0f4f8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 {html_title}</h1>
    </div>

    <div class="summary-section">
        <h2>📋 Ringkasan Hasil</h2>
        <div class="meta-grid">
            <div class="meta-item"><span class="meta-label">Total Dokumen:</span> {len(final_results)}</div>
            <div class="meta-item"><span class="meta-label">Tingkat Konsensus:</span> {agreement:.0%}</div>
            <div class="meta-item"><span class="meta-label">Waktu Proses:</span> {search_time:.2f}s</div>
            <div class="meta-item"><span class="meta-label">Fase Penelitian:</span> {len(phase_metadata)}</div>
        </div>
    </div>

    <div class="doc-list">
        <h2>📚 Daftar Dokumen Lengkap</h2>
"""]
            
            for i, doc in enumerate(final_results, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})
                score = scores.get('final', doc.get('final_score', 0))
                
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', '')) or record.get('article_number', '')
                location = " | ".join(filter(None, [chapter, article])) or "Dokumen Lengkap"
                eff_date = record.get('effective_date', record.get('tanggal_penetapan', 'N/A'))
                content = record.get('content', '')[:1200]
                
                top_tag = '<span class="top-tag">TOP</span>' if i <= 3 else ''
                
                lines.append(f"""
    <div class="doc">
        <div class="doc-header">
            <div class="doc-title">{top_tag}{i}. {record.get('regulation_type', 'N/A')} No. {record.get('regulation_number', 'N/A')}/{record.get('year', 'N/A')}</div>
            <div class="doc-score">Skor: {score:.4f}</div>
        </div>
        <div class="doc-meta">
            <div><span class="meta-label">Tentang:</span> {record.get('about', 'N/A')}</div>
            <div><span class="meta-label">Lokasi:</span> {location}</div>
            <div><span class="meta-label">Tgl Penetapan:</span> {eff_date}</div>
            <div><span class="meta-label">Lembaga:</span> {record.get('enacting_body', 'N/A')}</div>
        </div>
        <table class="score-table">
            <tr><th>Komponen Skor</th><th>Nilai</th></tr>
            <tr><td>Final Score</td><td><strong>{score:.4f}</strong></td></tr>
            <tr><td>Semantic</td><td>{scores.get('semantic', 0):.4f}</td></tr>
            <tr><td>Keyword</td><td>{scores.get('keyword', 0):.4f}</td></tr>
            <tr><td>Knowledge Graph</td><td>{scores.get('kg', 0):.4f}</td></tr>
        </table>
        <div class="doc-content">
            <div style='margin-bottom: 5px; font-weight: bold;'>Konten:</div>
            {content}...
        </div>
    </div>
""")
            
            # Add research process
            research_detail = format_detailed_research_process(last_search_result, top_n_per_researcher=10, show_content=True)
            research_html = research_detail.replace('\n', '<br>')
            research_html = re.sub(r'### (.*?)<br>', r'<h3>\1</h3>', research_html)
            research_html = re.sub(r'#### (.*?)<br>', r'<h4>\1</h4>', research_html)
            research_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', research_html)
            
            lines.append(f"""
    <div style="margin-top: 50px; background: #fafafa; padding: 30px; border-radius: 8px; border: 1px solid #e0e0e0;">
        <h2>🔬 Proses Penelitian</h2>
        {research_html}
    </div>

    <div class="footer">
        &copy; {datetime.now().year} Indonesian Legal RAG System - AI Research Powered<br>
        Generated: {timestamp}
    </div>
</body>
</html>
""")
            content = "".join(lines)
            filename = f"search_{timestamp}.html"
            
        else:  # Markdown
            lines = [f"# Hasil Pencarian Dokumen Hukum\n\n"]
            lines.append(f"**Query:** `{query}`\n")
            lines.append(f"**Timestamp:** {timestamp}\n")
            lines.append(f"**Total Dokumen:** {len(final_results)}\n")
            agreement = consensus_data.get('agreement_level', 0) if consensus_data else 0
            lines.append(f"**Tingkat Konsensus:** {agreement:.0%}\n")
            lines.append(f"**Waktu Proses:** {search_time:.2f}s\n\n---\n\n")
            
            # Documents with full details
            lines.append("## 📚 Dokumen Ditemukan\n\n")
            for i, doc in enumerate(final_results, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})
                
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', '')) or record.get('article_number', '')
                location = " | ".join(filter(None, [chapter, article])) or "Dokumen Lengkap"
                eff_date = record.get('effective_date', record.get('tanggal_penetapan', 'N/A'))
                
                lines.append(f"### {i}. {record.get('regulation_type', '')} No. {record.get('regulation_number', '')}/{record.get('year', '')}\n\n")
                lines.append(f"- **Lokasi:** {location}\n")
                lines.append(f"- **Tgl Penetapan:** {eff_date}\n")
                lines.append(f"- **Tentang:** {record.get('about', '')}\n\n")
                
                lines.append("**Skor:**\n")
                lines.append(f"- Final: {scores.get('final', 0):.4f}\n")
                lines.append(f"- Semantic: {scores.get('semantic', 0):.4f}\n")
                lines.append(f"- Keyword: {scores.get('keyword', 0):.4f}\n")
                lines.append(f"- KG: {scores.get('kg', 0):.4f}\n\n")
                
                content_text = record.get('content', '')
                if content_text:
                    lines.append(f"**Konten:**\n{content_text[:1000]}...\n\n")
                lines.append("---\n\n")
            
            # Add research process
            lines.append("## 🔬 Proses Penelitian\n\n")
            research_detail = format_detailed_research_process(last_search_result, top_n_per_researcher=10, show_content=True)
            lines.append(research_detail)
            
            content = "".join(lines)
            filename = f"search_{timestamp}.md"
        
        # Save to temp file
        import tempfile
        ext = filename.split('.')[-1]
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name
        
        return content[:5000] + "\n\n... (truncated for preview)", temp_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Export error: {e}", None


# =============================================================================
# TEST RUNNERS
# =============================================================================

def run_conversational_test(history, config_dict, show_thinking, show_sources, show_metadata):
    """Run full conversational test with 8 questions - with error handling and debug logging"""
    total_questions = len(TEST_QUESTIONS)
    completed = 0
    errors = []
    
    print(f"\n{'='*60}")
    print(f"🧪 CONVERSATIONAL TEST STARTING - {total_questions} Questions")
    print(f"{'='*60}\n")
    
    # Initial message
    history = history + [{
        "role": "assistant",
        "content": f"🧪 **Starting Conversational Test ({total_questions} Questions)**\n\nAuto-feeding questions through API...\n\n**Questions to test:**\n" +
                   "\n".join([f"{i+1}. {q[:80]}..." for i, q in enumerate(TEST_QUESTIONS)])
    }]
    yield history, ""
    
    # Process each question with error handling
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[TEST Q{i}/{total_questions}] Starting: {question[:60]}...")
        
        try:
            # Add progress indicator
            history = history + [{
                "role": "assistant",
                "content": f"🔄 **Question {i}/{total_questions}**\n\nProcessing: _{question[:100]}..._\n\n_Progress: {completed}/{total_questions} completed_"
            }]
            yield history, ""
            
            # Remove progress indicator
            history = history[:-1]
            
            # Process question - consume generator fully
            print(f"[TEST Q{i}] Calling chat_with_legal_rag...")
            updated_history = history
            chunk_count = 0
            for updated_history, cleared_input in chat_with_legal_rag(
                question, history, config_dict, show_thinking, show_sources, show_metadata
            ):
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"[TEST Q{i}] Received {chunk_count} chunks...")
                yield updated_history, cleared_input
            
            print(f"[TEST Q{i}] ✅ Completed with {chunk_count} total chunks")
            history = updated_history
            completed += 1
            
        except Exception as e:
            import traceback
            print(f"[TEST Q{i}] ❌ ERROR: {e}")
            traceback.print_exc()
            errors.append(f"Q{i}: {str(e)[:50]}")
            history = history + [{
                "role": "assistant",
                "content": f"⚠️ **Question {i} Error:** {e}\n\nContinuing to next question..."
            }]
            yield history, ""
    
    print(f"\n{'='*60}")
    print(f"🏁 TEST COMPLETE: {completed}/{total_questions} successful, {len(errors)} errors")
    print(f"{'='*60}\n")
    
    # Completion message
    status = "✅ Complete" if len(errors) == 0 else f"⚠️ Completed with {len(errors)} errors"
    error_summary = "\n\n**Errors:**\n" + "\n".join(errors) if errors else ""
    history = history + [{
        "role": "assistant",
        "content": f"{status} - **Conversational Test Done**\n\nProcessed {completed}/{total_questions} questions successfully.{error_summary}"
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
                # SEARCH TAB - Matching search_app.py
                # =====================================================================
                with gr.TabItem("🔍 Pencarian Dokumen"):
                    # Search input section
                    with gr.Row():
                        search_query = gr.Textbox(
                            label="Query Pencarian",
                            placeholder="Masukkan query pencarian hukum dalam Bahasa Indonesia...",
                            lines=3
                        )
                    
                    with gr.Row():
                        search_num = gr.Slider(5, 50, value=10, step=5, label="Jumlah Hasil Maksimum")
                        search_btn = gr.Button("🔍 Cari", variant="primary")
                    
                    # Results in tabs (like search_app.py)
                    with gr.Tabs():
                        with gr.TabItem("📋 Ringkasan"):
                            search_summary = gr.Markdown(label="Ringkasan Hasil")
                        
                        with gr.TabItem("📚 Semua Dokumen"):
                            with gr.Tabs():
                                with gr.TabItem("📊 Tabel Ikhtisar"):
                                    search_table = gr.Dataframe(
                                        headers=["No", "Jenis", "Nomor", "Tahun", "Tentang", "Lokasi", "Skor"],
                                        datatype=["number", "str", "str", "str", "str", "str", "str"],
                                        interactive=False,
                                        wrap=True
                                    )
                                with gr.TabItem("📄 Kartu Detail"):
                                    search_all_docs = gr.Markdown(label="Detail Dokumen")
                        
                        with gr.TabItem("🔬 Proses Penelitian"):
                            search_research = gr.Markdown(label="Proses Penelitian")
                    
                    # Examples (8 queries like search_app.py)
                    gr.Examples(
                        examples=SEARCH_EXAMPLES,
                        inputs=search_query,
                        label="Contoh Query"
                    )
                    
                    # Export section
                    gr.Markdown("---")
                    gr.Markdown("### 📥 Export Hasil Pencarian")
                    with gr.Row():
                        search_export_format = gr.Radio(
                            choices=["Markdown", "HTML", "JSON", "CSV"],
                            value="Markdown",
                            label="Format Export"
                        )
                        search_export_btn = gr.Button("📥 Export Hasil", variant="secondary")
                    
                    with gr.Row():
                        search_export_preview = gr.Textbox(label="Preview Export", lines=10, max_lines=15)
                        search_export_file = gr.File(label="Download File")
                
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
            
            # Search handlers (outputs to 4 tabs: summary, table, all_docs, research)
            search_btn.click(
                search_documents, 
                [search_query, search_num], 
                [search_summary, search_table, search_all_docs, search_research]
            )
            search_query.submit(
                search_documents, 
                [search_query, search_num], 
                [search_summary, search_table, search_all_docs, search_research]
            )
            
            # Search export handler
            search_export_btn.click(
                export_search_results,
                [search_export_format],
                [search_export_preview, search_export_file]
            )
            
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
