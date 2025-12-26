"""
Search Formatting Utilities - Shared formatting functions for search results

This module contains formatting functions used by both search_app.py and
unified_app_api.py for consistent output formatting.

File: utils/search_formatting.py
"""

import pandas as pd
from typing import Dict, List, Any

from utils.formatting import _extract_all_documents_from_metadata


def format_score_bar(score: float, label: str, color: str = "#1e3a5f") -> str:
    """Create a visual score bar"""
    percentage = score * 100
    return f"""
**{label}:** {score:.4f}
<div style="background:#e0e0e0;height:8px;border-radius:4px;margin:2px 0;">
<div style="width:{percentage}%;height:100%;background:{color};border-radius:4px;"></div>
</div>
"""


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
        phase_map = {
            'initial_scan': 'Pemindaian Awal',
            'focused_review': 'Tinjauan Terfokus',
            'deep_analysis': 'Analisis Mendalam',
            'verification': 'Verifikasi',
            'expert_review': 'Tinjauan Pakar'
        }
        
        for phase_name in phase_order:
            # Find any entries for this phase
            phase_entries = [v for k, v in phase_metadata.items() if v.get('phase') == phase_name]
            if not phase_entries: continue
            
            doc_count = sum(len(e.get('candidates', e.get('results', []))) for e in phase_entries)
            display_name = phase_map.get(phase_name, phase_name.replace('_', ' ').title())
            output.append(f"- **{display_name}:** {doc_count} dokumen ditemukan\n")

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
