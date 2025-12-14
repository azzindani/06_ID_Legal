"""
Shared Formatting Utilities for Indonesian Legal RAG System

This module contains reusable formatting functions for:
- Document sources and legal references
- Retrieved documents with full metadata
- Research process metadata
- Final candidate selection

These functions are used across:
- Gradio UI (search engine + conversational interface)
- Tests
- Pipelines (search engine, question, conversational)
- API inference
- Other services requiring accurate references

File: utils/formatting.py
"""

from typing import List, Dict, Any, Optional
from config import RESEARCH_TEAM_PERSONAS


def format_sources_info(results: List[Dict], config_dict: Dict) -> str:
    """
    Format LEGAL REFERENCES (Top K Documents Used in LLM Prompt) with FULL details

    Args:
        results: List of document results with scores
        config_dict: Configuration dictionary

    Returns:
        Formatted markdown string with document details
    """
    if not results:
        return "Tidak ada sumber yang ditemukan."

    try:
        # No duplicate header - header is in collapsible summary
        output = [f"**Documents Used in Prompt: {len(results)}**"]
        output.append("")
        output.append("These are the final selected documents sent to the LLM for answer generation.")
        output.append("")

        for i, result in enumerate(results, 1):
            try:
                record = result.get('record', result)

                reg_type = record.get('regulation_type', 'N/A')
                reg_num = record.get('regulation_number', 'N/A')
                year = record.get('year', 'N/A')
                about = record.get('about', 'N/A')
                enacting_body = record.get('enacting_body', 'N/A')
                global_id = record.get('global_id', 'N/A')
                effective_date = record.get('effective_date', record.get('tanggal_penetapan', 'N/A'))

                # Combine regulation type + enacting body + number + year
                regulation_full_name = f"{reg_type} {enacting_body} No. {reg_num} Tahun {year}"

                output.append(f"### {i}. {regulation_full_name}")
                output.append(f"- **Global ID:** {global_id}")
                output.append(f"- **Tentang:** {about}")
                if effective_date != 'N/A':
                    output.append(f"- **Tanggal Penetapan:** {effective_date}")

                # Article/Chapter location - MORE PROMINENT
                chapter = record.get('chapter', record.get('bab', ''))
                article = record.get('article', record.get('pasal', ''))
                article_number = record.get('article_number', '')  # More specific article number
                section = record.get('section', record.get('bagian', ''))
                paragraph = record.get('paragraph', record.get('ayat', ''))

                location_parts = []
                if chapter:
                    location_parts.append(f"Bab {chapter}")
                if section:
                    location_parts.append(f"Bagian {section}")
                if article:
                    location_parts.append(f"Pasal {article}")
                elif article_number:
                    location_parts.append(f"Pasal {article_number}")
                if paragraph:
                    location_parts.append(f"Ayat {paragraph}")

                if location_parts:
                    output.append(f"- **Lokasi:** {' > '.join(location_parts)}")
                else:
                    output.append(f"- **Lokasi:** (Dokumen Lengkap)")

                # All scores
                scores_parts = []
                if 'final_score' in result:
                    scores_parts.append(f"Final: {result['final_score']:.4f}")
                if result.get('semantic_score', 0) > 0:
                    scores_parts.append(f"Semantic: {result.get('semantic_score', 0):.4f}")
                if result.get('keyword_score', 0) > 0:
                    scores_parts.append(f"Keyword: {result.get('keyword_score', 0):.4f}")
                if result.get('rerank_score', 0) > 0:
                    scores_parts.append(f"Rerank: {result['rerank_score']:.4f}")
                if result.get('composite_score', 0) > 0:
                    scores_parts.append(f"Search: {result['composite_score']:.4f}")
                if result.get('kg_score', 0) > 0:
                    scores_parts.append(f"KG: {result['kg_score']:.4f}")
                if result.get('authority_score', 0) > 0:
                    scores_parts.append(f"Authority: {result.get('authority_score', 0):.4f}")
                if result.get('temporal_score', 0) > 0:
                    scores_parts.append(f"Temporal: {result.get('temporal_score', 0):.4f}")
                if result.get('completeness_score', 0) > 0:
                    scores_parts.append(f"Completeness: {result.get('completeness_score', 0):.4f}")

                if scores_parts:
                    output.append(f"- **Skor:** {' | '.join(scores_parts)}")

                # KG metadata
                kg_parts = []
                if record.get('kg_primary_domain'):
                    kg_parts.append(f"Domain: {record['kg_primary_domain']}")
                if record.get('kg_hierarchy_level', 0) > 0:
                    kg_parts.append(f"Hierarchy: Level {record['kg_hierarchy_level']}")
                if record.get('kg_cross_ref_count', 0) > 0:
                    kg_parts.append(f"Cross-refs: {record['kg_cross_ref_count']}")
                if record.get('kg_pagerank', 0) > 0:
                    kg_parts.append(f"PageRank: {record['kg_pagerank']:.4f}")

                if kg_parts:
                    output.append(f"- **Knowledge Graph:** {' | '.join(kg_parts)}")

                # Team consensus info
                if result.get('team_consensus', False):
                    consensus_info = f"Team Consensus: Yes"
                    if 'researcher_agreement' in result:
                        consensus_info += f" (Agreement: {result['researcher_agreement']})"
                    if 'supporting_researchers' in result:
                        researchers = result['supporting_researchers']
                        researcher_names = [RESEARCH_TEAM_PERSONAS.get(r, {}).get('name', r) for r in researchers[:3]]
                        consensus_info += f" | Researchers: {', '.join(researcher_names)}"
                    output.append(f"- **{consensus_info}**")

                # Content preview (800 chars)
                content = record.get('content', '')
                if content:
                    content_preview = content[:800].replace('\n', ' ')
                    output.append(f"- **Isi (preview):** {content_preview}...")

                output.append("")

            except Exception as e:
                output.append(f"Error formatting source {i}: {e}")
                continue

        return "\n".join(output)
    except Exception as e:
        return f"Error formatting sources: {e}"


def _extract_all_documents_from_metadata(metadata: Dict) -> List[Dict]:
    """
    Extract all retrieved documents from various metadata locations - matching test pattern

    This function consolidates documents from different stages of the RAG pipeline:
    - Phase metadata (most complete)
    - Research data
    - Research log
    - Consensus data
    - Top-level sources/citations

    Args:
        metadata: Metadata dictionary from RAG pipeline

    Returns:
        List of document dictionaries with deduplication
    """
    all_docs = []
    seen_ids = set()

    # Try phase_metadata first (most complete)
    phase_metadata = metadata.get('phase_metadata', {})
    for phase_name, phase_data in phase_metadata.items():
        if isinstance(phase_data, dict):
            candidates = phase_data.get('candidates', phase_data.get('results', []))
            for doc in candidates:
                doc_copy = dict(doc) if isinstance(doc, dict) else {'record': doc}
                record = doc_copy.get('record', doc_copy)
                doc_id = record.get('global_id', str(hash(str(doc))))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    doc_copy['_phase'] = phase_data.get('phase', phase_name)
                    doc_copy['_researcher'] = phase_data.get('researcher_name', phase_data.get('researcher', ''))
                    all_docs.append(doc_copy)

    # Try research_data
    if not all_docs:
        research_data = metadata.get('research_data', {})
        all_results = research_data.get('all_results', [])
        for doc in all_results:
            doc_copy = dict(doc) if isinstance(doc, dict) else {'record': doc}
            record = doc_copy.get('record', doc_copy)
            doc_id = record.get('global_id', str(hash(str(doc))))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc_copy)

    # Try research_log
    if not all_docs:
        research_log = metadata.get('research_log', {})
        phase_results = research_log.get('phase_results', {})
        for phase_name, phase_data in phase_results.items():
            if isinstance(phase_data, dict):
                candidates = phase_data.get('candidates', phase_data.get('results', []))
                for doc in candidates:
                    doc_copy = dict(doc) if isinstance(doc, dict) else {'record': doc}
                    record = doc_copy.get('record', doc_copy)
                    doc_id = record.get('global_id', str(hash(str(doc))))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        doc_copy['_phase'] = phase_name
                        all_docs.append(doc_copy)

    # Try consensus_data
    if not all_docs:
        consensus_data = metadata.get('consensus_data', {})
        final_results = consensus_data.get('final_results', [])
        for doc in final_results:
            doc_copy = dict(doc) if isinstance(doc, dict) else {'record': doc}
            record = doc_copy.get('record', doc_copy)
            doc_id = record.get('global_id', str(hash(str(doc))))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc_copy)

    # Try sources/citations at top level (fallback)
    if not all_docs:
        sources = metadata.get('sources', metadata.get('citations', []))
        for doc in sources:
            doc_id = doc.get('global_id', doc.get('regulation_number', str(hash(str(doc)))))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # Convert source format to document format
                all_docs.append({
                    'record': doc,
                    'scores': {
                        'final': doc.get('score', 0),
                        'semantic': doc.get('semantic_score', 0),
                        'keyword': doc.get('keyword_score', 0),
                        'kg': doc.get('kg_score', 0),
                        'authority': doc.get('authority_score', 0),
                        'temporal': doc.get('temporal_score', 0)
                    }
                })

    return all_docs


def format_all_documents(metadata: Dict, max_docs: int = 50) -> str:
    """
    Format ALL Retrieved Documents (Article-Level Details) - Top 50 using existing extraction pattern

    Args:
        metadata: Metadata dictionary from RAG pipeline
        max_docs: Maximum number of documents to format (default: 50)

    Returns:
        Formatted markdown string with all document details
    """
    if not metadata:
        return ""

    try:
        # Use existing extraction function matching test_complete_output.py pattern
        all_docs = _extract_all_documents_from_metadata(metadata)

        if not all_docs:
            return "No documents retrieved during research process."

        # No duplicate header - header is in collapsible summary
        output = [f"**Showing {min(len(all_docs), max_docs)} documents with detailed metadata**"]
        output.append("")

        for i, doc in enumerate(all_docs[:max_docs], 1):
            record = doc.get('record', doc)
            scores = doc.get('scores', {})

            # Basic info
            reg_type = record.get('regulation_type', 'N/A')
            reg_num = record.get('regulation_number', 'N/A')
            year = record.get('year', 'N/A')
            about = record.get('about', 'N/A')
            enacting_body = record.get('enacting_body', 'N/A')
            global_id = record.get('global_id', 'N/A')
            effective_date = record.get('effective_date', record.get('tanggal_penetapan', 'N/A'))

            # Scores
            final_score = scores.get('final', doc.get('final_score', doc.get('composite_score', record.get('score', 0))))
            semantic = scores.get('semantic', doc.get('semantic_score', 0))
            keyword = scores.get('keyword', doc.get('keyword_score', 0))
            kg = scores.get('kg', doc.get('kg_score', 0))
            authority = scores.get('authority', doc.get('authority_score', 0))
            temporal = scores.get('temporal', doc.get('temporal_score', 0))
            completeness = scores.get('completeness', doc.get('completeness_score', 0))

            # Article-level location
            chapter = record.get('chapter', record.get('bab', ''))
            article = record.get('article', record.get('pasal', ''))
            article_number = record.get('article_number', '')
            section = record.get('section', record.get('bagian', ''))
            paragraph = record.get('paragraph', record.get('ayat', ''))

            # KG metadata
            kg_domain = record.get('kg_primary_domain', record.get('primary_domain', ''))
            kg_hierarchy = record.get('kg_hierarchy_level', record.get('hierarchy_level', 0))
            kg_cross_refs = record.get('kg_cross_ref_count', record.get('cross_ref_count', 0))

            # Phase info
            phase = doc.get('_phase', '')
            researcher = doc.get('_researcher', '')

            # Combined regulation full name
            regulation_full_name = f"{reg_type} {enacting_body} No. {reg_num} Tahun {year}"

            output.append(f"**[{i}] {regulation_full_name}**")
            output.append(f"- Global ID: {global_id}")
            output.append(f"- About: {about}")
            if effective_date != 'N/A':
                output.append(f"- Effective Date: {effective_date}")

            # Article-level location
            location_parts = []
            if chapter:
                location_parts.append(f"Bab {chapter}")
            if section:
                location_parts.append(f"Bagian {section}")
            if article:
                location_parts.append(f"Pasal {article}")
            elif article_number:
                location_parts.append(f"Pasal {article_number}")
            if paragraph:
                location_parts.append(f"Ayat {paragraph}")

            if location_parts:
                output.append(f"- Location: {' > '.join(location_parts)}")
            else:
                output.append(f"- Location: (Full Document)")

            # All scores
            output.append(f"- Scores: Final={final_score:.4f} | Semantic={semantic:.4f} | Keyword={keyword:.4f}")
            output.append(f"- Scores: KG={kg:.4f} | Authority={authority:.4f} | Temporal={temporal:.4f} | Completeness={completeness:.4f}")

            # KG metadata
            if kg_domain or kg_hierarchy:
                output.append(f"- Knowledge Graph: Domain={kg_domain or 'N/A'} | Hierarchy={kg_hierarchy} | CrossRefs={kg_cross_refs}")

            # Research info
            if phase or researcher:
                output.append(f"- Discovery: Phase={phase} | Researcher={researcher}")

            # Content (truncated - 300 chars)
            content = record.get('content', '')
            if content:
                content_truncated = content[:300].replace('\n', ' ').strip()
                output.append(f"- Content (truncated): {content_truncated}...")

            output.append("")
            output.append("---")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"Error formatting all documents: {e}"


def format_retrieved_metadata(phase_metadata: Dict, config_dict: Dict) -> str:
    """
    Format all retrieved documents metadata with detailed research process info

    Args:
        phase_metadata: Phase metadata dictionary from research process
        config_dict: Configuration dictionary

    Returns:
        Formatted markdown string with research process details
    """
    if not phase_metadata:
        return ""

    try:
        phase_order = ['initial_scan', 'focused_review', 'deep_analysis', 'verification', 'expert_review']

        phase_groups = {}
        total_kg_enhanced = 0
        unique_researchers = set()

        for phase_key, phase_data in phase_metadata.items():
            if not isinstance(phase_data, dict):
                continue

            phase_name = phase_data.get('phase', phase_key)
            researcher = phase_data.get('researcher', 'unknown')
            unique_researchers.add(researcher)

            if phase_name not in phase_groups:
                phase_groups[phase_name] = {}
            if researcher not in phase_groups[phase_name]:
                phase_groups[phase_name][researcher] = {'candidates': [], 'confidence': 100.0}

            candidates = phase_data.get('candidates', phase_data.get('results', []))
            kg_candidates = [c for c in candidates if c.get('kg_score', 0) > 0.3]
            total_kg_enhanced += len(kg_candidates)

            confidence = phase_data.get('confidence', 100.0)
            phase_groups[phase_name][researcher]['candidates'].extend(candidates)
            phase_groups[phase_name][researcher]['confidence'] = confidence

        # Calculate totals
        total_retrieved = 0
        for phase_name in phase_order:
            if phase_name not in phase_groups:
                continue
            for researcher_data in phase_groups[phase_name].values():
                total_retrieved += len(researcher_data['candidates'])

        # Build output - no duplicate header (header is in collapsible summary)
        output = [f"**Team Members:** {len(unique_researchers)} | **Total Documents:** {total_retrieved:,} | **Phases:** {len(phase_groups)}"]
        output.append("")

        # Detailed phase breakdown
        for phase_name in phase_order:
            if phase_name not in phase_groups:
                continue

            researchers = phase_groups[phase_name]
            output.append(f"#### 📂 PHASE: {phase_name.upper()}")
            output.append("")

            for researcher, researcher_data in researchers.items():
                candidates = researcher_data['candidates']
                confidence = researcher_data['confidence']

                # Get researcher display name with emoji
                if researcher in RESEARCH_TEAM_PERSONAS:
                    persona = RESEARCH_TEAM_PERSONAS[researcher]
                    researcher_name = persona.get('name', researcher)
                    emoji = persona.get('emoji', '👤')
                    full_name = f"**{emoji} {researcher_name}:**"
                else:
                    full_name = f"**👤 {researcher}:**"

                output.append(f"{full_name} {len(candidates)} documents (Confidence: {confidence:.2f}%)")

                # List top documents with scores
                for i, candidate in enumerate(candidates[:5], 1):
                    try:
                        record = candidate.get('record', candidate)
                        score = candidate.get('composite_score', candidate.get('score', 0))
                        kg_score = candidate.get('kg_score', 0)

                        reg_type = record.get('regulation_type', 'N/A')
                        reg_num = record.get('regulation_number', 'N/A')
                        year = record.get('year', 'N/A')
                        enacting_body = record.get('enacting_body', '')

                        # Get article/chapter info
                        chapter = record.get('chapter', record.get('bab', ''))
                        article = record.get('article', record.get('pasal', ''))
                        article_number = record.get('article_number', '')

                        # Build regulation name
                        if enacting_body:
                            reg_name = f"{reg_type} {enacting_body} No. {reg_num} Tahun {year}"
                        else:
                            reg_name = f"{reg_type} No. {reg_num} Tahun {year}"

                        # Add article location if available
                        location = ""
                        if article or article_number:
                            location = f" | Pasal {article or article_number}"
                        elif chapter:
                            location = f" | Bab {chapter}"

                        output.append(f"   {i}. {reg_name}{location} (Score: {score:.3f}, KG: {kg_score:.3f})")
                    except Exception:
                        continue

                if len(candidates) > 5:
                    output.append(f"   ... and {len(candidates) - 5} more documents")
                output.append("")

        # Summary section - removed KG Enhancement Stats as per user request
        output.append(f"**Summary:** {total_retrieved:,} total documents retrieved across {len(phase_groups)} phases")

        return "\n".join(output)
    except Exception as e:
        return f"Error formatting metadata: {e}"


def final_selection_with_kg(candidates: List[Dict], query_type: str, config_dict: Dict) -> List[Dict]:
    """
    Final selection of results with KG enhancement

    Sorts candidates by final score and returns top K results.

    Args:
        candidates: List of candidate documents with scores
        query_type: Type of query (used for future enhancements)
        config_dict: Configuration dictionary with 'final_top_k' parameter

    Returns:
        Top K selected documents sorted by final score
    """
    if not candidates:
        return []

    top_k = config_dict.get('final_top_k', 3)

    # Sort by final score
    sorted_candidates = sorted(
        candidates,
        key=lambda x: x.get('final_score', x.get('rerank_score', x.get('consensus_score', 0))),
        reverse=True
    )

    return sorted_candidates[:top_k]
