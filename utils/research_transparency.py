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

    lines.append("-" * 100)
    lines.append("Step-by-step tracking of document discovery, evaluation, and selection")
    lines.append("")

    # Extract data
    research_data = result.get('research_data', {})
    consensus_data = result.get('consensus_data', {})
    phase_metadata = result.get('phase_metadata', result.get('all_retrieved_metadata', {}))
    final_results = result.get('sources', result.get('citations', []))

    if not research_data and not phase_metadata:
        lines.append("⚠️  No research data available")
        return '\n'.join(lines)

    # ============================================================================
    # STEP 1: RESEARCH TEAM
    # ============================================================================
    lines.append("### 👥 STEP 1: Research Team Assembly")
    lines.append("-" * 80)

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

    lines.append(f"**Team Size:** {len(researchers)} members")
    lines.append("")

    for researcher in sorted(researchers):
        persona_info = researcher_personas.get(researcher, {})
        name = persona_info.get('name', researcher)
        expertise = persona_info.get('expertise', 'General Research')
        lines.append(f"   - **{name}**")
        lines.append(f"     Expertise: {expertise}")

    lines.append("")

    # ============================================================================
    # STEP 2: INDIVIDUAL RESEARCHER FINDINGS
    # ============================================================================
    lines.append("### 🔍 STEP 2: Individual Researcher Findings")
    lines.append("-" * 80)
    lines.append(f"Each researcher independently searches from their unique perspective.")
    lines.append(f"Showing top {top_n_per_researcher} documents per researcher.")
    lines.append("")

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
        lines.append(f"**Total Documents Found:** {findings['total_documents']}")
        lines.append("")

        # Show by phase
        for phase_name, candidates in findings['phases'].items():
            lines.append(f"**Phase: {phase_name.replace('_', ' ').title()}**")
            lines.append(f"   Documents: {len(candidates)}")

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

                # Build full regulation name
                if enacting_body:
                    reg_name = f"{reg_type} {enacting_body} No. {reg_num} Tahun {year}"
                else:
                    reg_name = f"{reg_type} No. {reg_num} Tahun {year}"

                lines.append(f"   {i}. {reg_name}")
                lines.append(f"      About: {about[:100]}...")

                # Add article/chapter if available
                location_parts = []
                if chapter:
                    location_parts.append(f"Bab {chapter}")
                if article or article_number:
                    location_parts.append(f"Pasal {article or article_number}")
                if location_parts:
                    lines.append(f"      Location: {' > '.join(location_parts)}")
                if effective_date:
                    lines.append(f"      Effective Date: {effective_date}")

                lines.append(f"      Scores: Final={final_score:.3f} | Semantic={semantic:.3f} | Keyword={keyword:.3f} | KG={kg:.3f}")

                # Team consensus marker
                if doc.get('team_consensus'):
                    lines.append(f"      ⭐ **Team Consensus** (Agreement: {doc.get('researcher_agreement', 0):.2f})")

                if show_content:
                    content = record.get('content', '')[:200]
                    lines.append(f"      Content: {content}...")

                lines.append("")

            lines.append("")

        lines.append("-" * 80)
        lines.append("")

    # ============================================================================
    # STEP 3: CONSENSUS BUILDING
    # ============================================================================
    lines.append("### 🤝 STEP 3: Team Consensus & Cross-Validation")
    lines.append("-" * 80)
    lines.append("Documents validated through team agreement and cross-validation.")
    lines.append("")

    if consensus_data:
        consensus_results = consensus_data.get('consensus_results', [])
        consensus_stats = consensus_data.get('statistics', {})

        lines.append(f"**Documents After Consensus:** {len(consensus_results)}")
        lines.append(f"**Consensus Threshold:** {consensus_stats.get('consensus_threshold', 0.6):.2f}")
        lines.append(f"**Team Agreement Rate:** {consensus_stats.get('team_agreement_rate', 0):.2%}")
        lines.append("")

        # Show consensus documents
        lines.append("**Documents That Passed Consensus:**")
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

            # Build full regulation name
            if enacting_body:
                reg_name = f"{reg_type} {enacting_body} No. {reg_num} Tahun {year}"
            else:
                reg_name = f"{reg_type} No. {reg_num} Tahun {year}"

            lines.append(f"{i}. {reg_name}")
            lines.append(f"   About: {about[:100]}...")

            # Add article/chapter if available
            location_parts = []
            if chapter:
                location_parts.append(f"Bab {chapter}")
            if article or article_number:
                location_parts.append(f"Pasal {article or article_number}")
            if location_parts:
                lines.append(f"   Location: {' > '.join(location_parts)}")
            if effective_date:
                lines.append(f"   Effective Date: {effective_date}")

            lines.append(f"   Consensus Score: {consensus_score:.3f} | Agreement: {agreement:.2f}")
            lines.append(f"   Found by {len(found_by)} researchers: {', '.join(found_by[:3])}{'...' if len(found_by) > 3 else ''}")
            lines.append("")
    else:
        lines.append("⚠️  No consensus data available")

    lines.append("")

    # ============================================================================
    # STEP 4: RERANKING & FINAL SELECTION
    # ============================================================================
    lines.append("### 🎯 STEP 4: Reranking & Final Selection")
    lines.append("-" * 80)
    lines.append("Final reranking using cross-encoder to select most relevant documents.")
    lines.append("")

    if final_results:
        lines.append(f"**Final Documents Selected:** {len(final_results)}")
        lines.append("**These documents were sent to the LLM for answer generation**")
        lines.append("")

        for i, doc in enumerate(final_results, 1):
            reg_type = doc.get('regulation_type', 'N/A')
            reg_num = doc.get('regulation_number', 'N/A')
            year = doc.get('year', 'N/A')
            about = doc.get('about', 'N/A')
            enacting_body = doc.get('enacting_body', '')
            effective_date = doc.get('effective_date', doc.get('tanggal_penetapan', ''))
            score = doc.get('score', doc.get('final_score', 0))

            # Get article/chapter info
            chapter = doc.get('chapter', doc.get('bab', ''))
            article = doc.get('article', doc.get('pasal', ''))
            article_number = doc.get('article_number', '')

            # Build full regulation name
            if enacting_body:
                reg_name = f"{reg_type} {enacting_body} No. {reg_num} Tahun {year}"
            else:
                reg_name = f"{reg_type} No. {reg_num} Tahun {year}"

            lines.append(f"{i}. {reg_name}")
            lines.append(f"   About: {about}")

            # Add article/chapter if available
            location_parts = []
            if chapter:
                location_parts.append(f"Bab {chapter}")
            if article or article_number:
                location_parts.append(f"Pasal {article or article_number}")
            if location_parts:
                lines.append(f"   Location: {' > '.join(location_parts)}")
            if effective_date:
                lines.append(f"   Effective Date: {effective_date}")

            lines.append(f"   Final Score: {score:.4f}")

            if show_content:
                content = doc.get('content', '')[:300]
                lines.append(f"   Content: {content}...")

            lines.append("")
    else:
        lines.append("⚠️  No final results available")

    lines.append("")

    # ============================================================================
    # STEP 5: SUMMARY STATISTICS
    # ============================================================================
    lines.append("### 📈 STEP 5: Summary Statistics")
    lines.append("-" * 80)

    # Calculate stats
    total_retrieved = sum(f['total_documents'] for f in researcher_findings.values())
    total_consensus = len(consensus_data.get('consensus_results', [])) if consensus_data else 0
    total_final = len(final_results)

    lines.append(f"**Research Pipeline Flow:**")
    lines.append(f"   1. Initial Retrieval: {total_retrieved} documents ({len(researcher_findings)} researchers)")
    lines.append(f"   2. After Consensus: {total_consensus} documents")
    lines.append(f"   3. After Reranking: {total_final} documents")
    lines.append(f"   4. Reduction Rate: {((total_retrieved - total_final) / total_retrieved * 100) if total_retrieved > 0 else 0:.1f}%")
    lines.append("")

    # Per-researcher stats
    lines.append(f"**Per-Researcher Contribution:**")
    for researcher, findings in sorted(researcher_findings.items()):
        name = researcher_personas.get(researcher, {}).get('name', researcher)
        docs = findings['total_documents']
        percentage = (docs / total_retrieved * 100) if total_retrieved > 0 else 0
        lines.append(f"   - {name}: {docs} documents ({percentage:.1f}%)")

    lines.append("")
    lines.append("=" * 100)

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

    lines.append("**Research Team Summary:**")
    lines.append(f"Total: {len(researchers)} researchers | {total_docs} documents")
    lines.append("")

    for researcher, count in sorted(researchers.items(), key=lambda x: x[1], reverse=True):
        persona_info = RESEARCH_TEAM_PERSONAS.get(researcher, {})
        name = persona_info.get('name', researcher)
        percentage = (count / total_docs * 100) if total_docs > 0 else 0
        lines.append(f"   - {name}: {count} docs ({percentage:.1f}%)")

    return '\n'.join(lines)


