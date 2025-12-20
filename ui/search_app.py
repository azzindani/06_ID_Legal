"""
Search Engine UI - Indonesian Legal RAG System

Web-based search interface for legal document search using RAG.
This UI focuses on document retrieval with COMPLETE metadata display
showing ALL retrieved documents with full scoring breakdown.

Features:
- Full scoring display (semantic, keyword, KG, authority, temporal)
- ALL retrieved documents from every search phase
- Research process details with researcher personas
- Export functionality (Markdown, JSON, CSV)
"""

import gradio as gr
import sys
import os
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RAGPipeline
from config import (
    LLM_PROVIDER, EMBEDDING_DEVICE, LLM_DEVICE,
    RESEARCH_TEAM_PERSONAS
)
from utils.logger_utils import get_logger
from utils.formatting import _extract_all_documents_from_metadata

logger = get_logger(__name__)

# Global instances
pipeline: Optional[RAGPipeline] = None
current_provider: str = LLM_PROVIDER
last_search_result: Optional[Dict] = None

# Custom CSS for search interface
# NOTE: This CSS is currently used via gr.Blocks(css=SEARCH_CSS)
# If upgrading to Gradio 6.x, the css parameter is removed and this will need refactoring
SEARCH_CSS = """
/* Base responsive sizing */
html { font-size: 16px; }

.gradio-container {
    max-width: 1600px !important;
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

/* Score bars */
.score-bar {
    height: 8px;
    background: #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
    margin: 2px 0;
}

.score-fill {
    height: 100%;
    border-radius: 4px;
}

/* Metadata styling */
.metadata-box {
    background: #f0f4f8;
    border-radius: 0.5em;
    padding: 1em;
    font-family: 'Fira Code', monospace;
    font-size: 0.85em;
    max-height: 500px;
    overflow-y: auto;
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

.export-btn {
    background: #28a745 !important;
}

/* Responsive breakpoints */
@media (max-width: 768px) {
    html { font-size: 14px; }
    .gradio-container { padding: 1em !important; }
}

@media (min-width: 1200px) {
    html { font-size: 18px; }
}

/* Tabs styling */
.tab-nav {
    margin-bottom: 1em;
}
"""


def initialize_search_system(provider_type: str = None):
    """Initialize the RAG system for search"""
    global pipeline, current_provider

    if provider_type:
        current_provider = provider_type

    if pipeline is None:
        logger.info(f"Initializing search system with provider: {current_provider}")
        pipeline = RAGPipeline({'llm_provider': current_provider})
        if not pipeline.initialize():
            return "❌ Failed to initialize pipeline"
        logger.info("Search pipeline initialized")

    device_info = f"Embedding: {EMBEDDING_DEVICE}, LLM: {LLM_DEVICE}"
    return f"✅ Initialized with {current_provider}. {device_info}"


def format_score_bar(score: float, label: str, color: str = "#1e3a5f") -> str:
    """Create a visual score bar"""
    percentage = score * 100
    return f"""
**{label}:** {score:.4f}
<div style="background:#e0e0e0;height:8px;border-radius:4px;margin:2px 0;">
<div style="width:{percentage}%;height:100%;background:{color};border-radius:4px;"></div>
</div>
"""


# extract_all_documents is now imported from utils.formatting as _extract_all_documents_from_metadata


def format_document_card(doc: Dict, index: int) -> str:
    """Format a single document with full scoring details"""
    record = doc.get('record', doc)
    scores = doc.get('scores', {})

    # Basic info
    reg_type = record.get('regulation_type', 'N/A')
    reg_num = record.get('regulation_number', 'N/A')
    year = record.get('year', 'N/A')
    about = record.get('about', 'N/A')
    enacting_body = record.get('enacting_body', 'N/A')

    # Scores
    final_score = scores.get('final', doc.get('final_score', doc.get('composite_score', 0)))
    semantic_score = scores.get('semantic', doc.get('semantic_score', 0))
    keyword_score = scores.get('keyword', doc.get('keyword_score', 0))
    kg_score = scores.get('kg', doc.get('kg_score', 0))
    authority_score = scores.get('authority', doc.get('authority_score', 0))
    temporal_score = scores.get('temporal', doc.get('temporal_score', 0))
    completeness_score = scores.get('completeness', doc.get('completeness_score', 0))

    # KG Metadata
    domain = record.get('kg_primary_domain', record.get('primary_domain', ''))
    hierarchy = record.get('kg_hierarchy_level', record.get('hierarchy_level', 0))
    cross_refs = record.get('kg_cross_ref_count', 0)

    # Phase info
    phase = doc.get('_phase', '')
    researcher = doc.get('_researcher', '')

    # Team consensus
    consensus = doc.get('team_consensus', False)
    agreement = doc.get('researcher_agreement', 0)

    # Content
    content = record.get('content', '')
    content_preview = content[:400] + "..." if len(content) > 400 else content

    # Build card
    card = f"""
### 📄 {index}. {reg_type} No. {reg_num}/{year}

**Tentang:** {about}

**Ditetapkan oleh:** {enacting_body}

---

#### 📊 Skor Relevansi

| Komponen | Nilai |
|----------|-------|
| **Final Score** | **{final_score:.4f}** |
| Semantic | {semantic_score:.4f} |
| Keyword | {keyword_score:.4f} |
| KG Score | {kg_score:.4f} |
| Authority | {authority_score:.4f} |
| Temporal | {temporal_score:.4f} |
| Completeness | {completeness_score:.4f} |

"""

    if domain or hierarchy:
        card += f"""
#### 🔗 Knowledge Graph

- **Domain:** {domain or 'N/A'}
- **Hierarchy Level:** {hierarchy}
- **Cross References:** {cross_refs}

"""

    if phase or researcher:
        card += f"""
#### 🔍 Research Info

- **Phase:** {phase}
- **Researcher:** {researcher}

"""

    if consensus:
        card += f"""
#### ✅ Team Consensus

- **Agreement:** {agreement} researchers agreed
- **Status:** Validated by team

"""

    card += f"""
#### 📝 Konten

{content_preview}

---

"""
    return card


def format_all_documents(result: Dict) -> str:
    """Format ALL retrieved documents with complete metadata"""
    all_docs = _extract_all_documents_from_metadata(result)

    if not all_docs:
        return "❌ Tidak ada dokumen yang ditemukan."

    # Sort by final score
    all_docs.sort(
        key=lambda x: x.get('scores', {}).get('final',
            x.get('final_score', x.get('composite_score', 0))),
        reverse=True
    )

    output = [f"## 📚 Semua Dokumen Ditemukan ({len(all_docs)} dokumen)\n\n"]

    for i, doc in enumerate(all_docs, 1):
        output.append(format_document_card(doc, i))

    return "".join(output)


def format_research_process(result: Dict) -> str:
    """Format research process details"""
    output = ["## 🔬 Proses Penelitian\n\n"]

    # Phase metadata
    phase_metadata = result.get('phase_metadata', {})
    if phase_metadata:
        phase_order = ['initial_scan', 'focused_review', 'deep_analysis', 'verification', 'expert_review']

        output.append("### 📋 Fase Penelitian\n\n")

        total_docs = 0
        unique_researchers = set()

        for phase_name in phase_order:
            if phase_name not in phase_metadata:
                continue

            phase_data = phase_metadata[phase_name]
            if not isinstance(phase_data, dict):
                continue

            researcher = phase_data.get('researcher', 'unknown')
            unique_researchers.add(researcher)
            candidates = phase_data.get('candidates', phase_data.get('results', []))
            confidence = phase_data.get('confidence', 100.0)
            doc_count = len(candidates)
            total_docs += doc_count

            # Get researcher display name
            if researcher in RESEARCH_TEAM_PERSONAS:
                persona = RESEARCH_TEAM_PERSONAS[researcher]
                researcher_name = persona.get('name', researcher)
                emoji = persona.get('emoji', '👤')
            else:
                researcher_name = researcher
                emoji = '👤'

            output.append(f"#### {emoji} {phase_name.replace('_', ' ').title()}\n\n")
            output.append(f"- **Peneliti:** {researcher_name}\n")
            output.append(f"- **Dokumen:** {doc_count}\n")
            output.append(f"- **Confidence:** {confidence:.1f}%\n\n")

        output.append(f"### 📈 Ringkasan\n\n")
        output.append(f"- **Total Fase:** {len(phase_metadata)}\n")
        output.append(f"- **Total Peneliti:** {len(unique_researchers)}\n")
        output.append(f"- **Total Dokumen:** {total_docs}\n\n")

    # Consensus data
    consensus_data = result.get('consensus_data', {})
    if consensus_data:
        output.append("### 🤝 Konsensus Tim\n\n")
        agreement = consensus_data.get('agreement_level', 0)
        output.append(f"- **Tingkat Kesepakatan:** {agreement:.0%}\n")

        if consensus_data.get('final_results'):
            output.append(f"- **Hasil Final:** {len(consensus_data['final_results'])} dokumen\n")

    # Timing
    metadata = result.get('metadata', {})
    if metadata:
        output.append("\n### ⏱️ Waktu Eksekusi\n\n")
        total_time = metadata.get('total_time', metadata.get('processing_time', 0))
        retrieval_time = metadata.get('retrieval_time', 0)
        generation_time = metadata.get('generation_time', 0)

        output.append(f"- **Total:** {total_time:.2f}s\n")
        if retrieval_time:
            output.append(f"- **Retrieval:** {retrieval_time:.2f}s\n")
        if generation_time:
            output.append(f"- **Generation:** {generation_time:.2f}s\n")

        query_type = metadata.get('query_type', '')
        if query_type:
            output.append(f"- **Tipe Query:** {query_type}\n")

    return "".join(output)


def format_summary(result: Dict) -> str:
    """Format search summary with answer"""
    output = []

    # Query info
    metadata = result.get('metadata', {})
    query_type = metadata.get('query_type', 'general')
    output.append(f"**Tipe Query:** {query_type}\n\n")

    # Thinking process
    thinking = result.get('thinking', '')
    if thinking:
        output.append("<details><summary>🧠 **Proses Berpikir** (klik untuk melihat)</summary>\n\n")
        output.append(f"{thinking}\n\n")
        output.append("</details>\n\n")

    # Answer
    answer = result.get('answer', '')
    if answer:
        output.append("## 💡 Ringkasan Jawaban\n\n")
        output.append(f"{answer}\n\n")

    # Quick stats
    all_docs = _extract_all_documents_from_metadata(result)
    sources = result.get('sources', result.get('citations', []))

    output.append("---\n\n")
    output.append("### 📊 Statistik Pencarian\n\n")
    output.append(f"- **Total Dokumen Ditemukan:** {len(all_docs)}\n")
    output.append(f"- **Sumber Dikutip:** {len(sources)}\n")

    total_time = metadata.get('total_time', metadata.get('processing_time', 0))
    if total_time:
        output.append(f"- **Waktu Proses:** {total_time:.2f}s\n")

    return "".join(output)


def search_documents(query: str, num_results: int = 10) -> Tuple[str, str, str]:
    """
    Search for legal documents based on query.
    Returns: (summary, all_documents, research_process)
    """
    global pipeline, last_search_result

    if not query.strip():
        return "⚠️ Masukkan query pencarian.", "", ""

    try:
        if pipeline is None:
            initialize_search_system()

        logger.info(f"Searching for: {query}")
        start_time = time.time()

        # Perform search (retrieval only, no LLM generation)
        # Use pipeline.search_only() to skip LLM generation
        result = pipeline.search_only(query, top_k=num_results) if hasattr(pipeline, 'search_only') else pipeline.retrieve_documents(query, top_k=num_results)
        
        # If pipeline doesn't have search_only, manually call retrieval
        if result is None or not isinstance(result, dict):
            # Fallback: use retrieval components directly
            result = {
                'sources': pipeline.hybrid_search.search_with_persona(
                    query=query,
                    persona_name='generalist',
                    phase_config={'candidates': num_results},
                    priority_weights={},
                    top_k=num_results
                ),
                'metadata': {'query': query, 'total_time': time.time() - start_time}
            }
        
        last_search_result = result

        # Format outputs
        summary = format_summary(result)
        all_docs = format_all_documents(result)
        research = format_research_process(result)

        return summary, all_docs, research

    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", "", ""


def export_results(export_format: str) -> Tuple[str, Optional[str]]:
    """Export last search results to specified format"""
    global last_search_result

    if last_search_result is None:
        return "⚠️ Tidak ada hasil pencarian untuk di-export.", None

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_docs = _extract_all_documents_from_metadata(last_search_result)

        if export_format == "JSON":
            # Full JSON export
            export_data = {
                'export_timestamp': timestamp,
                'query': last_search_result.get('metadata', {}).get('query', ''),
                'total_documents': len(all_docs),
                'documents': [],
                'metadata': last_search_result.get('metadata', {}),
                'research_process': {
                    'phase_metadata': last_search_result.get('phase_metadata', {}),
                    'consensus_data': last_search_result.get('consensus_data', {})
                }
            }

            for doc in all_docs:
                record = doc.get('record', doc)
                scores = doc.get('scores', {})
                export_data['documents'].append({
                    'regulation_type': record.get('regulation_type', ''),
                    'regulation_number': record.get('regulation_number', ''),
                    'year': record.get('year', ''),
                    'about': record.get('about', ''),
                    'enacting_body': record.get('enacting_body', ''),
                    'content': record.get('content', ''),
                    'scores': {
                        'final': scores.get('final', doc.get('final_score', 0)),
                        'semantic': scores.get('semantic', doc.get('semantic_score', 0)),
                        'keyword': scores.get('keyword', doc.get('keyword_score', 0)),
                        'kg': scores.get('kg', doc.get('kg_score', 0)),
                        'authority': scores.get('authority', doc.get('authority_score', 0)),
                        'temporal': scores.get('temporal', doc.get('temporal_score', 0))
                    },
                    'kg_metadata': {
                        'domain': record.get('kg_primary_domain', ''),
                        'hierarchy_level': record.get('kg_hierarchy_level', 0),
                        'cross_refs': record.get('kg_cross_ref_count', 0)
                    },
                    'phase': doc.get('_phase', ''),
                    'researcher': doc.get('_researcher', ''),
                    'team_consensus': doc.get('team_consensus', False)
                })

            content = json.dumps(export_data, ensure_ascii=False, indent=2)
            filename = f"search_results_{timestamp}.json"

        elif export_format == "CSV":
            # CSV export for spreadsheet
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow([
                'No', 'Regulation_Type', 'Regulation_Number', 'Year', 'About',
                'Final_Score', 'Semantic', 'Keyword', 'KG', 'Authority', 'Temporal',
                'Domain', 'Hierarchy', 'Phase', 'Researcher', 'Consensus'
            ])

            # Data
            for i, doc in enumerate(all_docs, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})
                writer.writerow([
                    i,
                    record.get('regulation_type', ''),
                    record.get('regulation_number', ''),
                    record.get('year', ''),
                    record.get('about', '')[:100],
                    f"{scores.get('final', doc.get('final_score', 0)):.4f}",
                    f"{scores.get('semantic', doc.get('semantic_score', 0)):.4f}",
                    f"{scores.get('keyword', doc.get('keyword_score', 0)):.4f}",
                    f"{scores.get('kg', doc.get('kg_score', 0)):.4f}",
                    f"{scores.get('authority', doc.get('authority_score', 0)):.4f}",
                    f"{scores.get('temporal', doc.get('temporal_score', 0)):.4f}",
                    record.get('kg_primary_domain', ''),
                    record.get('kg_hierarchy_level', 0),
                    doc.get('_phase', ''),
                    doc.get('_researcher', ''),
                    'Yes' if doc.get('team_consensus') else 'No'
                ])

            content = output.getvalue()
            filename = f"search_results_{timestamp}.csv"

        else:  # Markdown
            lines = [f"# Hasil Pencarian Legal\n\n"]
            lines.append(f"**Timestamp:** {timestamp}\n\n")
            lines.append(f"**Total Dokumen:** {len(all_docs)}\n\n")
            lines.append("---\n\n")

            # Answer
            if last_search_result.get('answer'):
                lines.append("## Ringkasan\n\n")
                lines.append(f"{last_search_result['answer']}\n\n")

            # Documents
            lines.append("## Dokumen Ditemukan\n\n")
            for i, doc in enumerate(all_docs, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})

                lines.append(f"### {i}. {record.get('regulation_type', 'N/A')} No. {record.get('regulation_number', 'N/A')}/{record.get('year', 'N/A')}\n\n")
                lines.append(f"**Tentang:** {record.get('about', 'N/A')}\n\n")
                lines.append(f"**Skor:** Final={scores.get('final', doc.get('final_score', 0)):.4f}, ")
                lines.append(f"Semantic={scores.get('semantic', doc.get('semantic_score', 0)):.4f}, ")
                lines.append(f"KG={scores.get('kg', doc.get('kg_score', 0)):.4f}\n\n")

                content_preview = record.get('content', '')[:300]
                if content_preview:
                    lines.append(f"**Konten:** {content_preview}...\n\n")

                lines.append("---\n\n")

            content = "".join(lines)
            filename = f"search_results_{timestamp}.md"

        # Save to temp file
        import tempfile
        ext = filename.split('.')[-1]
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        return content[:5000] + "\n\n... (truncated for preview)", temp_path

    except Exception as e:
        logger.error(f"Export error: {e}")
        return f"❌ Export error: {str(e)}", None


def create_search_demo():
    """Create the enhanced search engine Gradio interface"""

    # Gradio 6: css moved to launch(), title remains here
    with gr.Blocks(title="Indonesian Legal Search Engine") as demo:
        # Header
        gr.HTML("""
            <div class="header-title">🔍 Indonesian Legal Search Engine</div>
            <div class="header-subtitle">Search Indonesian regulations with complete metadata and scoring</div>
        """)

        # Status
        status = gr.Textbox(
            label="System Status",
            value="Ready to initialize...",
            interactive=False
        )

        with gr.Tabs():
            # Search Tab
            with gr.TabItem("🔍 Pencarian", id="search"):
                # Search input
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Query Pencarian",
                        placeholder="Masukkan query pencarian hukum dalam Bahasa Indonesia...",
                        lines=3,
                        elem_classes=["search-box"]
                    )

                # Search options
                with gr.Row():
                    num_results = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                        label="Jumlah Hasil Maksimum"
                    )
                    search_btn = gr.Button(
                        "🔍 Cari",
                        variant="primary",
                        elem_classes=["search-btn"]
                    )

                # Results in tabs
                with gr.Tabs():
                    with gr.TabItem("📋 Ringkasan"):
                        summary_output = gr.Markdown(
                            label="Ringkasan Hasil",
                            elem_classes=["results-container"]
                        )

                    with gr.TabItem("📚 Semua Dokumen"):
                        docs_output = gr.Markdown(
                            label="Semua Dokumen",
                            elem_classes=["results-container"]
                        )

                    with gr.TabItem("🔬 Proses Penelitian"):
                        research_output = gr.Markdown(
                            label="Proses Penelitian",
                            elem_classes=["results-container"]
                        )

                # Example queries
                gr.Examples(
                    examples=[
                        ["Apa saja syarat pendirian PT menurut UU Perseroan Terbatas?"],
                        ["Bagaimana prosedur pendaftaran merek dagang di Indonesia?"],
                        ["Apa sanksi pelanggaran UU Perlindungan Data Pribadi?"],
                        ["Ketentuan cuti karyawan menurut UU Ketenagakerjaan"],
                        ["Syarat dan prosedur perceraian menurut hukum Indonesia"],
                        ["Regulasi tentang investasi asing di Indonesia"],
                        ["Ketentuan pajak penghasilan untuk UMKM"],
                        ["Prosedur penyelesaian sengketa konsumen"],
                    ],
                    inputs=query_input,
                    label="Contoh Query"
                )

            # Export Tab
            with gr.TabItem("📥 Export", id="export"):
                gr.Markdown("""
                ## Export Hasil Pencarian

                Download hasil pencarian lengkap dengan semua metadata dan skor.

                **Format yang tersedia:**
                - **Markdown**: Format yang mudah dibaca
                - **JSON**: Data terstruktur lengkap untuk pemrosesan
                - **CSV**: Untuk analisis di spreadsheet
                """)

                with gr.Row():
                    export_format = gr.Radio(
                        choices=["Markdown", "JSON", "CSV"],
                        value="Markdown",
                        label="Format Export"
                    )

                export_btn = gr.Button(
                    "📥 Export Hasil",
                    variant="primary",
                    elem_classes=["export-btn"]
                )

                export_preview = gr.Textbox(
                    label="Preview Export",
                    lines=15,
                    max_lines=20
                    # show_copy_button removed for Gradio 6 compatibility
                )

                export_file = gr.File(
                    label="Download File",
                    visible=True
                )

            # About Tab
            with gr.TabItem("ℹ️ Tentang", id="about"):
                gr.Markdown("""
                ## 🔍 Indonesian Legal Search Engine

                ### Fitur Utama

                **📊 Skor Lengkap**
                - Semantic Score: Kesamaan makna dengan query
                - Keyword Score: Kecocokan kata kunci
                - KG Score: Relevansi dalam Knowledge Graph
                - Authority Score: Tingkat otoritas regulasi
                - Temporal Score: Relevansi temporal
                - Completeness Score: Kelengkapan dokumen

                **📚 Semua Dokumen**
                - Menampilkan SEMUA dokumen yang ditemukan (bukan hanya yang dikutip)
                - Berguna untuk audit dan verifikasi kelengkapan pencarian

                **🔬 Proses Penelitian**
                - Detail fase pencarian
                - Informasi peneliti (persona)
                - Statistik konsensus tim

                **📥 Export**
                - Markdown untuk dokumentasi
                - JSON untuk pemrosesan data
                - CSV untuk analisis spreadsheet

                ### Cara Penggunaan

                1. Masukkan query pencarian dalam Bahasa Indonesia
                2. Klik "Cari" atau tekan Enter
                3. Lihat ringkasan di tab "Ringkasan"
                4. Cek semua dokumen di tab "Semua Dokumen"
                5. Pelajari proses penelitian di tab "Proses Penelitian"
                6. Export hasil jika diperlukan
                """)

        # Event handlers
        search_btn.click(
            fn=search_documents,
            inputs=[query_input, num_results],
            outputs=[summary_output, docs_output, research_output]
        )

        query_input.submit(
            fn=search_documents,
            inputs=[query_input, num_results],
            outputs=[summary_output, docs_output, research_output]
        )

        export_btn.click(
            fn=export_results,
            inputs=[export_format],
            outputs=[export_preview, export_file]
        )

        # Initialize on load
        demo.load(
            fn=lambda: initialize_search_system(),
            outputs=status
        )

    return demo


def launch_search_app(share: bool = False, server_port: int = 7861):
    """Launch search engine app with pre-initialization (NO LLM - retrieval only)"""
    global pipeline

    # Pre-initialize system WITHOUT LLM (retrieval only)
    logger.info("Pre-initializing search system (retrieval only, no LLM)...")

    if pipeline is None:
        logger.info("Initializing RAG pipeline for retrieval only (no LLM loaded)")
        # Initialize with minimal config - retrieval components only
        pipeline = RAGPipeline({
            'llm_provider': 'none',  # Don't load LLM
            'skip_llm': True  # Skip LLM initialization if supported
        })
        
        # Initialize only retrieval components
        if not pipeline.initialize_retrieval_only() if hasattr(pipeline, 'initialize_retrieval_only') else pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            raise RuntimeError("Pipeline initialization failed")
        logger.info("Pipeline initialized successfully (retrieval components only)")

    logger.info("Search system ready, launching UI...")

    # Create and launch demo
    demo = create_search_demo()
    # Gradio 6: css parameter moved to launch()
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share,
        css=SEARCH_CSS  # Moved from Blocks() constructor
    )


if __name__ == "__main__":
    launch_search_app(share=False)
