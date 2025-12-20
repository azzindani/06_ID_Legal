"""
Research Transparency - Detailed Step-by-Step Research Process Tracking

Provides comprehensive visibility into:
- What each researcher found (their unique perspective)
- Top documents from each researcher
- Consensus filtering results
- Reranking selection
- Step-by-step document flow

This module creates detailed, auditable research reports showing exactly how
documents were discovered, evaluated, and selected through the RAG pipeline.
"""

from typing import Dict, List, Any, Optional
from config import RESEARCH_TEAM_PERSONAS


def format_detailed_research_process(
    result: Dict[str, Any],
    top_n_per_researcher: int = 20,
    show_content: bool = True
) -> str:
    """
    Format comprehensive step-by-step research process with per-researcher details

    Shows:
    1. Research Team composition
    2. Per-Researcher findings (top N documents each)
    3. Consensus filtering (what passed team validation)
    4. Reranking selection (final top K)
    5. Summary statistics

    Args:
        result: RAG pipeline result with research_data and metadata
        top_n_per_researcher: Show top N documents per researcher (default: 20)
        show_content: Include content preview (default: False for brevity)

    Returns:
        Formatted markdown string with complete research transparency
    """
    lines = []

    #lines.append("-" * 100)
    #lines.append("Step-by-step tracking of document discovery, evaluation, and selection")
    #lines.append("")

    # Extract data
    research_data = result.get('research_data', {})
    consensus_data = result.get('consensus_data', {})
    phase_metadata = result.get('phase_metadata', result.get('all_retrieved_metadata', {}))
    final_results = result.get('final_results', result.get('sources', result.get('citations', [])))

    # RECONSTRUCTION: If phase_metadata is missing but research_data is present, reconstruct it
    if not phase_metadata and research_data:
        # Reconstruct phase_metadata from research_data['phase_results']
        phase_results = research_data.get('phase_results', {})
        if phase_results:
            entry_idx = 0
            for phase_name, results in phase_results.items():
                persona_groups = {}
                for r in results:
                    persona = r.get('metadata', {}).get('persona', 'unknown')
                    if persona not in persona_groups:
                        persona_groups[persona] = []
                    persona_groups[persona].append(r)

                for persona_name, persona_results_list in persona_groups.items():
                    key = f"{entry_idx}_{phase_name}_{persona_name}"
                    
                    # Transform results to candidates format
                    candidates = []
                    for r in persona_results_list:
                        candidates.append({
                            'record': r.get('record', {}),
                            'scores': r.get('scores', {}),
                            'composite_score': r.get('scores', {}).get('final', 0),
                            'team_consensus': r.get('team_consensus', False),
                            'researcher_agreement': r.get('researcher_agreement', 0)
                        })

                    phase_metadata[key] = {
                        'phase': phase_name,
                        'researcher': persona_name,
                        'candidates': candidates,
                        'results': candidates
                    }
                    entry_idx += 1

    if not research_data and not phase_metadata:
        lines.append("⚠️  No research data available")
        return '\n'.join(lines)

    # ============================================================================
    # STEP 1: TIM PENELITI
    # ============================================================================
    lines.append("")
    lines.append("### 👥 LANGKAH 1: Pembentukan Tim Peneliti")
    #lines.append("-" * 80)

    # Extract unique researchers
    researchers = set()
    researcher_personas = {}

    for phase_key, phase_data in phase_metadata.items():
        if isinstance(phase_data, dict):
            researcher_name = phase_data.get('researcher', '')
            if researcher_name:
                researchers.add(researcher_name)
                persona_info = RESEARCH_TEAM_PERSONAS.get(researcher_name, {})
                if persona_info:
                    researcher_personas[researcher_name] = persona_info

    lines.append(f"**Ukuran Tim:** {len(researchers)} anggota")
    lines.append("")

    for researcher in sorted(researchers):
        persona_info = researcher_personas.get(researcher, {})
        name = persona_info.get('name', researcher)
        expertise = persona_info.get('expertise', 'Penelitian Umum')
        lines.append(f"   - **{name}**")
        lines.append(f"     Keahlian: {expertise}")

    #lines.append("")
    lines.append("---")

    # ============================================================================
    # STEP 2: TEMUAN PENELITI INDIVIDUAL
    # ============================================================================
    lines.append("")
    lines.append("### 🔍 LANGKAH 2: Temuan Peneliti Individual")
    #lines.append("-" * 80)
    #lines.append(f"Each researcher independently searches from their unique perspective.")
    #lines.append(f"Showing top {top_n_per_researcher} documents per researcher.")
    #lines.append("")

    # Group by researcher
    researcher_findings = {}

    for phase_key, phase_data in phase_metadata.items():
        if not isinstance(phase_data, dict):
            continue

        researcher = phase_data.get('researcher', 'Unknown')
        phase_name = phase_data.get('phase', 'unknown')
        candidates = phase_data.get('candidates', phase_data.get('results', []))

        if researcher not in researcher_findings:
            researcher_findings[researcher] = {
                'phases': {},
                'total_documents': 0,
                'all_candidates': []
            }

        researcher_findings[researcher]['phases'][phase_name] = candidates
        researcher_findings[researcher]['total_documents'] += len(candidates)
        researcher_findings[researcher]['all_candidates'].extend(candidates)

    # Display per researcher
    for researcher in sorted(researcher_findings.keys()):
        findings = researcher_findings[researcher]
        persona_info = researcher_personas.get(researcher, {})
        name = persona_info.get('name', researcher)

        lines.append(f"#### 👤 {name}")
        lines.append(f"**Total Dokumen Ditemukan:** {findings['total_documents']}")
        lines.append("")

        # Show by phase
        phase_map = {
            'initial_scan': 'Pemindaian Awal',
            'focused_review': 'Tinjauan Terfokus',
            'deep_analysis': 'Analisis Mendalam',
            'verification': 'Verifikasi',
            'expert_review': 'Tinjauan Pakar'
        }

        for phase_name, candidates in findings['phases'].items():
            display_phase = phase_map.get(phase_name, phase_name.replace('_', ' ').title())
            lines.append(f"**Fase: {display_phase}**")
            lines.append(f"   Dokumen: {len(candidates)}")

            # Show top N documents
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get('composite_score', x.get('scores', {}).get('final', 0)),
                reverse=True
            )[:top_n_per_researcher]

            for i, doc in enumerate(sorted_candidates, 1):
                record = doc.get('record', doc)
                scores = doc.get('scores', {})

                reg_type = record.get('regulation_type', 'N/A')
                reg_num = record.get('regulation_number', 'N/A')
                year = record.get('year', 'N/A')
                about = record.get('about', 'N/A')
                enacting_body = record.get('enacting_body', '')
                effective_date = record.get('effective_date', record.get('tanggal_penetapan', ''))

                # Get article/chapter info
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', ''))
                article_number = record.get('article_number', '')

                final_score = scores.get('final', doc.get('composite_score', 0))
                semantic = scores.get('semantic', 0)
                keyword = scores.get('keyword', 0)
                kg = scores.get('kg', 0)

                # Build full regulation name - skip N/A enacting_body
                if enacting_body and enacting_body != 'N/A':
                    reg_name = f"{reg_type} {enacting_body} No. {reg_num} Tahun {year}"
                else:
                    reg_name = f"{reg_type} No. {reg_num} Tahun {year}"

                lines.append(f"   {i}. {reg_name}")
                lines.append(f"      Tentang: {about[:100]}...")

                # Add article/chapter if available
                location_parts = []
                if chapter:
                    location_parts.append(f"{chapter}")
                if article or article_number:
                    location_parts.append(f"{article or article_number}")
                if location_parts:
                    lines.append(f"      Lokasi: {' | '.join(location_parts)}")

                # Only show effective date if it exists and is not N/A
                if effective_date and effective_date != 'N/A':
                    lines.append(f"      Tgl Penetapan: {effective_date}")

                lines.append(f"      Skor: Final: {final_score:.3f} | Semantik: {semantic:.3f} | Keyword: {keyword:.3f} | KG: {kg:.3f}")

                # Team consensus marker
                if doc.get('team_consensus'):
                    lines.append(f"      ⭐ **Konsensus Tim** (Kesepakatan: {doc.get('researcher_agreement', 0):.2f})")

                if show_content:
                    content = record.get('content', '')[:200]
                    lines.append(f"      Konten: {content}...")

                # Don't add separator between list items - they should be adjacent

            lines.append("")

        lines.append("---")

    # ============================================================================
    # STEP 3: PEMBENTUKAN KONSENSUS
    # ============================================================================
    lines.append("")
    lines.append("### 🤝 LANGKAH 3: Konsensus Tim & Validasi Silang")
    #lines.append("-" * 80)
    #lines.append("Documents validated through team agreement and cross-validation.")
    #lines.append("")

    if consensus_data:
        # Handle both key names (consensus_results vs validated_results)
        consensus_results = consensus_data.get('consensus_results', consensus_data.get('validated_results', []))
        
        # Extract statistics safely
        consensus_stats = consensus_data.get('statistics', {})
        threshold = consensus_stats.get('consensus_threshold', consensus_data.get('threshold', 0.6))
        agreement_rate = consensus_stats.get('team_agreement_rate', consensus_data.get('agreement_level', 0))

        lines.append(f"**Dokumen Setelah Konsensus:** {len(consensus_results)}")
        lines.append(f"**Ambang Batas Konsensus:** {threshold:.2f}")
        lines.append(f"**Tingkat Kesepakatan Tim:** {agreement_rate:.2%}")
        lines.append("")

        # Show consensus documents
        lines.append("**Dokumen yang Lolos Konsensus:**")
        for i, doc in enumerate(consensus_results[:20], 1):
            record = doc.get('record', doc)

            reg_type = record.get('regulation_type', 'N/A')
            reg_num = record.get('regulation_number', 'N/A')
            year = record.get('year', 'N/A')
            about = record.get('about', 'N/A')
            enacting_body = record.get('enacting_body', '')
            effective_date = record.get('effective_date', record.get('tanggal_penetapan', ''))

            # Get article/chapter info
            chapter = record.get('chapter', record.get('bab', ''))
            article = record.get('article', record.get('pasal', ''))
            article_number = record.get('article_number', '')

            consensus_score = doc.get('consensus_score', 0)
            agreement = doc.get('researcher_agreement', 0)
            found_by = doc.get('found_by_researchers', [])

            # Build full regulation name - skip N/A enacting_body
            if enacting_body and enacting_body != 'N/A':
                reg_name = f"{reg_type} {enacting_body} No. {reg_num} Tahun {year}"
            else:
                reg_name = f"{reg_type} No. {reg_num} Tahun {year}"

            lines.append(f"{i}. {reg_name}")
            lines.append(f"   Tentang: {about[:100]}...")

            # Add article/chapter if available
            location_parts = []
            if chapter:
                location_parts.append(f"{chapter}")
            if article or article_number:
                location_parts.append(f"{article or article_number}")
            if location_parts:
                lines.append(f"   Lokasi: {' > '.join(location_parts)}")

            # Only show effective date if it exists and is not N/A
            if effective_date and effective_date != 'N/A':
                lines.append(f"   Tgl Penetapan: {effective_date}")

            lines.append(f"   Skor Konsensus: {consensus_score:.3f} | Kesepakatan: {agreement:.2f}")
            lines.append(f"   Ditemukan oleh {len(found_by)} peneliti: {', '.join(found_by[:3])}{'...' if len(found_by) > 3 else ''}")
            # Don't add separator between list items - they should be adjacent
    else:
        lines.append("⚠️  No consensus data available")

    #lines.append("")
    lines.append("---")

    # ============================================================================
    # STEP 4: RERANKING & SELEKSI FINAL
    # ============================================================================
    lines.append("")
    lines.append("### 🎯 LANGKAH 4: Reranking & Seleksi Final")
    #lines.append("-" * 80)
    #lines.append("Final reranking using cross-encoder to select most relevant documents.")
    #lines.append("")

    if final_results:
        lines.append(f"**Dokumen Final yang Terpilih:** {len(final_results)}")
        lines.append("**Dokumen-dokumen ini dikirim ke LLM untuk pembuatan jawaban**")
        lines.append("")

        for i, doc in enumerate(final_results, 1):
            record = doc.get('record', doc)
            
            reg_type = record.get('regulation_type', 'N/A')
            reg_num = record.get('regulation_number', 'N/A')
            year = record.get('year', 'N/A')
            about = record.get('about', 'N/A')
            enacting_body = record.get('enacting_body', '')
            effective_date = record.get('effective_date', record.get('tanggal_penetapan', ''))
            score = doc.get('score', doc.get('final_score', 0))

            # Get article/chapter info
            chapter = record.get('chapter', record.get('bab', ''))
            article = record.get('article', record.get('pasal', ''))
            article_number = record.get('article_number', '')

            # Build full regulation name - skip N/A enacting_body
            if enacting_body and enacting_body != 'N/A':
                reg_name = f"{reg_type} {enacting_body} No. {reg_num} Tahun {year}"
            else:
                reg_name = f"{reg_type} No. {reg_num} Tahun {year}"

            lines.append(f"{i}. {reg_name}")
            lines.append(f"   Tentang: {about}")

            # Add article/chapter if available
            location_parts = []
            if chapter:
                location_parts.append(f"{chapter}")
            if article or article_number:
                location_parts.append(f"{article or article_number}")
            if location_parts:
                lines.append(f"   Lokasi: {' | '.join(location_parts)}")

            # Only show effective date if it exists and is not N/A
            if effective_date and effective_date != 'N/A':
                lines.append(f"   Tgl Penetapan: {effective_date}")

            lines.append(f"   Skor Final: {score:.4f}")

            if show_content:
                content = record.get('content', '')[:300]
                lines.append(f"   Konten: {content}...")

            # Don't add separator between list items - they should be adjacent
    else:
        lines.append("⚠️  No final results available")

    lines.append("---")
    #lines.append("")

    # ============================================================================
    # STEP 5: RINGKASAN STATISTIK
    # ============================================================================
    lines.append("")
    lines.append("### 📈 LANGKAH 5: Ringkasan Statistik")
    #lines.append("-" * 80)

    # Calculate stats
    # Safely get researcher findings counts
    total_retrieved = 0
    if researcher_findings:
        total_retrieved = sum(f['total_documents'] for f in researcher_findings.values())
    elif research_data:
        total_retrieved = research_data.get('total_candidates_evaluated', 0)
        
    total_consensus = len(consensus_data.get('consensus_results', consensus_data.get('validated_results', []))) if consensus_data else 0
    total_final = len(final_results) if final_results else 0

    lines.append(f"**Alur Pipeline Penelitian:**")
    lines.append(f"   1. Pengambilan Awal: {total_retrieved} dokumen ({len(researcher_findings)} peneliti)")
    lines.append(f"   2. Setelah Konsensus: {total_consensus} dokumen")
    lines.append(f"   3. Setelah Reranking: {total_final} dokumen")
    lines.append(f"   4. Tingkat Reduksi: {((total_retrieved - total_final) / total_retrieved * 100) if total_retrieved > 0 else 0:.1f}%")
    lines.append("")

    # Per-researcher stats
    lines.append(f"**Kontribusi Per Peneliti:**")
    for researcher, findings in sorted(researcher_findings.items()):
        name = researcher_personas.get(researcher, {}).get('name', researcher)
        docs = findings['total_documents']
        percentage = (docs / total_retrieved * 100) if total_retrieved > 0 else 0
        lines.append(f"   - {name}: {docs} dokumen ({percentage:.1f}%)")

    lines.append("")

    return '\n'.join(lines)


def format_researcher_summary(phase_metadata: Dict[str, Any]) -> str:
    """
    Quick summary of researchers and their contributions

    Args:
        phase_metadata: Phase metadata dictionary

    Returns:
        Brief formatted summary
    """
    lines = []

    # Extract unique researchers
    researchers = {}

    for phase_key, phase_data in phase_metadata.items():
        if isinstance(phase_data, dict):
            researcher = phase_data.get('researcher', 'Unknown')
            candidates = phase_data.get('candidates', phase_data.get('results', []))

            if researcher not in researchers:
                researchers[researcher] = 0
            researchers[researcher] += len(candidates)

    total_docs = sum(researchers.values())

    lines.append("**Ringkasan Tim Peneliti:**")
    lines.append(f"Total: {len(researchers)} peneliti | {total_docs} dokumen")
    lines.append("")

    for researcher, count in sorted(researchers.items(), key=lambda x: x[1], reverse=True):
        persona_info = RESEARCH_TEAM_PERSONAS.get(researcher, {})
        name = persona_info.get('name', researcher)
        percentage = (count / total_docs * 100) if total_docs > 0 else 0
        lines.append(f"   - {name}: {count} dok ({percentage:.1f}%)")

    return '\n'.join(lines)






