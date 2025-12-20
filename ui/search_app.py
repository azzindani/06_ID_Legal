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
import pandas as pd
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
from utils.research_transparency import format_detailed_research_process

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

    # Location & Date
    chapter = record.get('chapter', record.get('bab', ''))
    article = record.get('article', record.get('pasal', '')) or record.get('article_number', '')
    location = " | ".join(filter(None, [chapter, article])) or "Dokumen Lengkap"
    eff_date = record.get('effective_date', record.get('tanggal_penetapan', 'N/A'))

    # Build card
    card = f"""
### 📄 {index}. {reg_type} No. {reg_num}/{year}

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
    """Format ALL retrieved documents with complete metadata (Markdown Cards)"""
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


def get_docs_dataframe_data(result: Dict) -> pd.DataFrame:
    """Get document data as a pandas DataFrame for tabular display"""
    all_docs = _extract_all_documents_from_metadata(result)
    
    if not all_docs:
        return pd.DataFrame()
        
    # Sort by final score
    all_docs.sort(
        key=lambda x: x.get('scores', {}).get('final', 
            x.get('final_score', x.get('composite_score', 0))),
        reverse=True
    )
    
    rows = []
    for i, doc in enumerate(all_docs, 1):
        record = doc.get('record', doc)
        scores = doc.get('scores', {})
        
        chapter = record.get('chapter', record.get('bab', ''))
        article = record.get('article', record.get('pasal', '')) or record.get('article_number', '')
        location = " | ".join(filter(None, [chapter, article])) or "Lengkap"
        
        rows.append({
            "No": i,
            "Jenis": record.get('regulation_type', ''),
            "Nomor": record.get('regulation_number', ''),
            "Tahun": record.get('year', ''),
            "Lokasi": location,
            "Tgl Penetapan": record.get('effective_date', record.get('tanggal_penetapan', 'N/A')),
            "Tentang": record.get('about', ''),
            "Skor Final": f"{scores.get('final', doc.get('final_score', 0)):.4f}",
            "Bidang": record.get('kg_primary_domain', ''),
            "Konten": record.get('content', '')[:150] + "..." if record.get('content') else ""
        })
        
    return pd.DataFrame(rows)


def format_research_process_summary(result: Dict) -> str:
    """Format simplified research process details for summary"""
    output = ["## 🔬 Proses Penelitian\n\n"]

    # Phase metadata
    phase_metadata = result.get('phase_metadata', {})
    if phase_metadata:
        # Sort keys to ensure consistent order
        phase_order = ['initial_scan', 'focused_review', 'deep_analysis', 'verification', 'expert_review']
        
        output.append("### 📋 Fase Penelitian\n\n")
        
        total_docs = 0
        researchers = set()
        
        # Collect unique researchers across all phases
        unique_researchers = {}
        
        for phase_key, phase_data in phase_metadata.items():
            if not isinstance(phase_data, dict): continue
            
            researcher = phase_data.get('researcher', 'unknown')
            researchers.add(researcher)
            candidates = phase_data.get('candidates', phase_data.get('results', []))
            total_docs += len(candidates)
            
            if researcher not in unique_researchers:
                unique_researchers[researcher] = 0
            unique_researchers[researcher] += len(candidates)

        # Show summary of phases
        lines = []
        for phase_name in phase_order:
            # Find any entries for this phase
            phase_entries = [v for k, v in phase_metadata.items() if v.get('phase') == phase_name]
            if not phase_entries: continue
            
            doc_count = sum(len(e.get('candidates', e.get('results', []))) for e in phase_entries)
            output.append(f"- **{phase_name.replace('_', ' ').title()}:** {doc_count} dokumen ditemukan\n")

        output.append(f"\n- **Total Peneliti:** {len(researchers)}\n")
        output.append(f"- **Total Dokumen Unik:** {total_docs}\n\n")

    return "".join(output)


def format_summary(result: Dict) -> str:
    """Format search summary with improved overview for search-only mode"""
    output = []

    # Query info
    metadata = result.get('metadata', {})
    query_type = metadata.get('query_type', 'general')
    query_text = metadata.get('query', 'N/A')
    
    output.append(f"## 📋 Ringkasan Hasil Pencarian\n\n")
    output.append(f"**Query:** `{query_text}`\n")
    output.append(f"**Tipe Analisis:** {query_type.title()}\n\n")

    # Result Overview - Use final_results as primary source for 'Summary'
    # This ensures it respects the top_k limit correctly
    final_results = result.get('final_results', [])
    
    if final_results:
        show_count = len(final_results)
        output.append(f"### ⭐ Hasil Relevan ({show_count} dokumen)\n\n")
        
        for i, doc in enumerate(final_results, 1):
            record = doc.get('record', doc)
            
            # Extract metadata
            reg_type = record.get('regulation_type', 'N/A')
            reg_num = record.get('regulation_number', 'N/A')
            year = record.get('year', 'N/A')
            about = record.get('about', 'N/A')
            effective_date = record.get('effective_date', record.get('tanggal_penetapan', ''))
            
            # Location info
            chapter = record.get('chapter', record.get('bab', ''))
            article = record.get('article', record.get('pasal', ''))
            article_num = record.get('article_number', '')
            location_parts = []
            if chapter: location_parts.append(chapter)
            if article: location_parts.append(article)
            elif article_num: location_parts.append(article_num)
            location = " | ".join(location_parts) if location_parts else "Dokumen Lengkap"

            output.append(f"{i}. **{reg_type} No. {reg_num}/{year}**\n")
            output.append(f"   - **Lokasi:** {location}\n")
            if effective_date and effective_date != 'N/A':
                output.append(f"   - **Tgl Penetapan:** {effective_date}\n")
            output.append(f"   - **Tentang:** _{about}_\n")
            
            # Content preview (safe truncation)
            content = record.get('content', '')
            if content:
                preview = content[:300].replace('\n', ' ') + "..." if len(content) > 300 else content
                output.append(f"   - **Konten:** {preview}\n")
            output.append("\n")
        
        # Count phases
        phase_count = len(result.get('phase_metadata', {}))
        if phase_count == 0 and 'research_data' in result:
            phase_count = len(result['research_data'].get('phase_results', {}))
            
        if phase_count > 0:
            output.append(f"Analisis dilakukan melalui **{phase_count}** fase penelitian agen.\n\n")

    # Statistics
    output.append("---\n\n")
    output.append("### 📊 Statistik Sistem\n\n")
    output.append(f"- **Dokumen Dievaluasi:** {len(_extract_all_documents_from_metadata(result))}\n")
    output.append(f"- **Tingkat Konsensus:** {result.get('consensus_data', {}).get('agreement_level', 0):.0%}\n")

    total_time = metadata.get('total_time', metadata.get('processing_time', 0))
    if total_time:
        output.append(f"- **Waktu Proses Total:** {total_time:.2f}detik\n")

    return "".join(output)


def search_documents(query: str, num_results: int = 10, progress=gr.Progress()) -> Tuple[str, str, pd.DataFrame, str]:
    """
    Search for legal documents based on query with progress tracking.
    Returns: (summary, all_documents, df_data, research_process)
    """
    global pipeline, last_search_result

    if not query.strip():
        return "⚠️ Masukkan query pencarian.", "", pd.DataFrame(), ""

    # Yield initial loading state
    loading_md = "### 🔄 Sedang mencari... \nMohon tunggu sejenak, tim peneliti sedang menganalisis query Anda."
    loading_df = pd.DataFrame([{"Status": "Mencari dokumen..."}])
    yield loading_md, loading_md, loading_df, loading_md

    try:
        if pipeline is None:
            progress(0.1, desc="Initializing system...")
            initialize_search_system()

        logger.info(f"Searching for: {query}")
        start_time = time.time()
        
        progress(0.2, desc="🔍 Analyzing query...")
        
        # Use streaming to track progress if possible
        if hasattr(pipeline, 'orchestrator') and hasattr(pipeline.orchestrator, 'stream_run'):
            # Manual streaming through orchestrator nodes
            orchestrator_state = {}
            for state_update in pipeline.orchestrator.stream_run(query, top_k=num_results):
                # Accumulate state updates
                orchestrator_state.update(state_update)
                
                # Check for node completions
                if "query_detection" in state_update:
                    progress(0.4, desc="🔬 Researching documents...")
                elif "research" in state_update:
                    progress(0.6, desc="🤝 Building consensus...")
                elif "consensus" in state_update:
                    progress(0.8, desc="🎯 Reranking results...")
            
            # Extract final data from accumulated state
            # LangGraph outputs state as node_name -> state_update
            # We want to consolidate all updates into a single flat dictionary
            result = {}
            for node_name, node_update in orchestrator_state.items():
                if isinstance(node_update, dict):
                    result.update(node_update)
            
            # Ensure critical keys are at top level
            if not result.get("final_results") and "reranking" in orchestrator_state:
                result.update(orchestrator_state["reranking"])
            
            # Ensure metadata consistency
            if "metadata" not in result:
                result["metadata"] = {}
            
            if "query" not in result["metadata"]:
                result["metadata"]["query"] = query
                
            # If we didn't get a result from streaming, fall back to direct call
            if not result or "final_results" not in result:
                result = pipeline.retrieve_documents(query, top_k=num_results)
        else:
            # Fallback for non-streaming orchestrator
            progress(0.5, desc="🔎 Searching regulations...")
            result = pipeline.retrieve_documents(query, top_k=num_results)
        
        last_search_result = result
        progress(0.9, desc="✨ Formatting results...")

        # Format outputs
        summary = format_summary(result)
        all_docs = format_all_documents(result)
        df_docs = get_docs_dataframe_data(result)
        research = format_detailed_research_process(result, top_n_per_researcher=20, show_content=False)

        yield summary, all_docs, df_docs, research

    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        yield f"❌ Error: {str(e)}", "", pd.DataFrame(), ""


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
                
                # Location
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', ''))
                article_num = record.get('article_number', '')
                location = " | ".join(filter(None, [chapter, article or article_num])) or "N/A"

                export_data['documents'].append({
                    'regulation_type': record.get('regulation_type', ''),
                    'regulation_number': record.get('regulation_number', ''),
                    'year': record.get('year', ''),
                    'about': record.get('about', ''),
                    'location': location,
                    'effective_date': record.get('effective_date', record.get('tanggal_penetapan', '')),
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
                'No', 'Regulation_Type', 'Regulation_Number', 'Year', 'Location', 'Effective_Date', 'About',
                'Final_Score', 'Semantic', 'Keyword', 'KG', 'Authority', 'Temporal',
                'Domain', 'Hierarchy', 'Phase', 'Researcher', 'Consensus', 'Content'
            ])

            # Data
            for i, doc in enumerate(all_docs, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})
                
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', ''))
                article_num = record.get('article_number', '')
                location = " | ".join(filter(None, [chapter, article or article_num])) or "N/A"
                effective_date = record.get('effective_date', record.get('tanggal_penetapan', ''))

                writer.writerow([
                    i,
                    record.get('regulation_type', ''),
                    record.get('regulation_number', ''),
                    record.get('year', ''),
                    location,
                    effective_date,
                    record.get('about', '')[:200],
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
                    'Yes' if doc.get('team_consensus') else 'No',
                    record.get('content', '')[:1000]
                ])

            content = output.getvalue()
            filename = f"search_results_{timestamp}.csv"

        elif export_format == "HTML":
            # HTML export with simple styling
            html_title = f"Laporan Hasil Pencarian Hukum: {last_search_result.get('metadata', {}).get('query', 'Pencarian Legal')}"
            
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
        .doc {{ background: #fff; border: 1px solid #e0e0e0; border-left: 5px solid #1e3a5f; padding: 20px; border-radius: 5px; margin-bottom: 20px; transition: transform 0.2s; }}
        .doc:hover {{ transform: translateX(5px); border-color: #2c5282; }}
        .doc-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }}
        .doc-title {{ font-weight: bold; font-size: 1.2em; color: #1e3a5f; flex-grow: 1; }}
        .doc-score {{ background: #e6fffa; color: #234e52; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; white-space: nowrap; }}
        .doc-meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9em; color: #666; margin-bottom: 15px; background: #fafafa; padding: 10px; border-radius: 4px; }}
        .doc-content {{ background: #fdfdfd; padding: 12px; border: 1px dashed #ddd; font-style: italic; font-size: 0.95em; color: #444; }}
        .footer {{ text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #eee; font-size: 0.8em; color: #999; }}
        .top-tag {{ background: #1e3a5f; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; margin-right: 8px; vertical-align: middle; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{html_title}</h1>
    </div>

    <div class="summary-section">
        <h2>📋 Ringkasan Hasil</h2>
        <div class="meta-grid">
            <div class="meta-item"><span class="meta-label">Total Dokumen:</span> {len(all_docs)}</div>
            <div class="meta-item"><span class="meta-label">Tingkat Konsensus:</span> {last_search_result.get('consensus_data', {}).get('agreement_level', 0):.0%}</div>
            <div class="meta-item"><span class="meta-label">Waktu Proses:</span> {last_search_result.get('metadata', {}).get('total_time', 0):.2f}s</div>
            <div class="meta-item"><span class="meta-label">Metode:</span> RAG Research Agency</div>
        </div>
        
        <h3>⭐ Hasil Utama (Top {min(10, len(all_docs))})</h3>
        <ul style="padding-left: 20px;">
"""]
            
            # Show top 10 in HTML summary list
            for i, doc in enumerate(all_docs[:10], 1):
                record = doc.get('record', doc)
                
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', ''))
                article_num = record.get('article_number', '')
                location = " | ".join(filter(None, [chapter, article or article_num])) or "Dokumen Lengkap"
                
                lines.append(f"            <li><strong>{record.get('regulation_type', 'N/A')} No. {record.get('regulation_number', 'N/A')}/{record.get('year', 'N/A')}</strong> ({location}): {record.get('about', 'N/A')}</li>\n")
            
            lines.append(f"""        </ul>
    </div>

    <div class="doc-list">
        <h2>📚 Daftar Dokumen Lengkap</h2>
""")
            
            for i, doc in enumerate(all_docs, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})
                score = scores.get('final', doc.get('final_score', 0))
                
                lines.append(f"""
    <div class="doc">
        <div class="doc-header">
            <div class="doc-title">{"<span class='top-tag'>TOP</span>" if i <= 3 else ""}{i}. {record.get('regulation_type', 'N/A')} No. {record.get('regulation_number', 'N/A')}/{record.get('year', 'N/A')}</div>
            <div class="doc-score">Skor: {score:.4f}</div>
        </div>
        <div class="doc-meta">
            <div><span class="meta-label">Tentang:</span> {record.get('about', 'N/A')}</div>
            <div><span class="meta-label">Lokasi:</span> {" | ".join(filter(None, [record.get('chapter', record.get('bab', '')), record.get('article', record.get('pasal', '')) or record.get('article_number', '')])) or "Dokumen Lengkap"}</div>
            <div><span class="meta-label">Tgl Penetapan:</span> {record.get('effective_date', record.get('tanggal_penetapan', 'N/A'))}</div>
            <div><span class="meta-label">Lembaga:</span> {record.get('enacting_body', 'N/A')}</div>
            <div><span class="meta-label">Domain:</span> {record.get('kg_primary_domain', 'N/A')}</div>
            <div><span class="meta-label">Fase:</span> {doc.get('_phase', 'N/A')}</div>
        </div>
        <div class="doc-content">
            {record.get('content', '')[:1200]}...
        </div>
    </div>
""")
            
            lines.append(f"""
    <div class="footer">
        &copy; {datetime.now().year} Indonesian Legal RAG System - AI Research Powered
    </div>
</body>
</html>
""")
            content = "".join(lines)
            filename = f"search_results_{timestamp}.html"

        else:  # Markdown
            lines = [f"# Hasil Pencarian Legal\n\n"]
            lines.append(f"**Query:** `{last_search_result.get('metadata', {}).get('query', 'N/A')}`\n")
            lines.append(f"**Timestamp:** {timestamp}\n")
            lines.append(f"**Total Dokumen:** {len(all_docs)}\n\n")
            lines.append("---\n\n")

            # Documents
            lines.append("## Dokumen Ditemukan\n\n")
            for i, doc in enumerate(all_docs, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})
                
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', ''))
                article_num = record.get('article_number', '')
                location = " | ".join(filter(None, [chapter, article or article_num])) or "Dokumen Lengkap"
                effective_date = record.get('effective_date', record.get('tanggal_penetapan', 'N/A'))

                lines.append(f"### {i}. {record.get('regulation_type', 'N/A')} No. {record.get('regulation_number', 'N/A')}/{record.get('year', 'N/A')}\n\n")
                lines.append(f"- **Lokasi:** {location}\n")
                lines.append(f"- **Tgl Penetapan:** {effective_date}\n")
                lines.append(f"- **Tentang:** {record.get('about', 'N/A')}\n\n")
                lines.append(f"**Skor:** Final={scores.get('final', doc.get('final_score', 0)):.4f}, ")
                lines.append(f"Semantic={scores.get('semantic', doc.get('semantic_score', 0)):.4f}, ")
                lines.append(f"KG={scores.get('kg', doc.get('kg_score', 0)):.4f}\n\n")

                content_preview = record.get('content', '')[:1000]
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

        # Status and Search Info
        with gr.Row():
            status = gr.Textbox(
                label="System Status",
                value="Ready to initialize...",
                interactive=False,
                scale=2
            )
            search_info = gr.Markdown("### ⚖️ Indonesian Law Intelligence")

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
                        with gr.Tabs():
                            with gr.TabItem("📊 Tabel Ikhtisar"):
                                docs_df_output = gr.Dataframe(
                                    headers=["No", "Jenis", "Nomor", "Tahun", "Lokasi", "Tgl Penetapan", "Tentang", "Skor Final", "Bidang", "Konten"],
                                    datatype=["number", "str", "str", "str", "str", "str", "str", "str", "str", "str"],
                                    interactive=False,
                                    wrap=True
                                )
                            with gr.TabItem("📄 Kartu Detail"):
                                docs_markdown_output = gr.Markdown(
                                    label="Detail Dokumen",
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
                - **Markdown**: Format yang mudah dibaca (.md)
                - **HTML**: Format profesional dengan styling (.html)
                - **JSON**: Data terstruktur lengkap (.json)
                - **CSV**: Untuk analisis spreadsheet (.csv)
                """)

                with gr.Row():
                    export_format = gr.Radio(
                        choices=["Markdown", "HTML", "JSON", "CSV"],
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
            outputs=[summary_output, docs_markdown_output, docs_df_output, research_output],
            show_progress="full"
        )

        query_input.submit(
            fn=search_documents,
            inputs=[query_input, num_results],
            outputs=[summary_output, docs_markdown_output, docs_df_output, research_output],
            show_progress="full"
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
        pipeline = RAGPipeline()
        
        # Use retrieval-only initialization (skips LLM loading)
        if not pipeline.initialize_retrieval_only():
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
