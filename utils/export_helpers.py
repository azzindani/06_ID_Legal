"""
Export Helpers - Data Processing and Export Functions

Provides helper functions for formatting search metadata and
exporting conversations to various formats (Markdown, JSON, HTML).
"""

import json
import traceback
from datetime import datetime
from typing import Dict, List, Any

# Optional markdown import for HTML export
try:
    import markdown
except ImportError:
    markdown = None


def format_complete_search_metadata(rag_result: Dict, include_scores: bool = True) -> str:
    """
    Format complete search metadata including ALL documents retrieved

    Args:
        rag_result: Complete RAG result with all_retrieved_metadata
        include_scores: Include detailed scoring information

    Returns:
        Formatted string with all search results
    """
    try:
        if not rag_result or not rag_result.get('all_retrieved_metadata'):
            return "No search metadata available."

        output = []
        output.append("# 📊 COMPLETE SEARCH RESULTS METADATA")
        output.append("=" * 80)
        output.append("")

        all_metadata = rag_result['all_retrieved_metadata']

        # Group by phase
        phase_order = ['initial_scan', 'focused_review', 'deep_analysis', 'verification', 'expert_review']
        phase_groups = {}

        for phase_key, phase_data in all_metadata.items():
            phase_name = phase_data.get('phase', 'unknown')
            if phase_name not in phase_groups:
                phase_groups[phase_name] = []
            phase_groups[phase_name].append((phase_key, phase_data))

        total_docs = 0

        # Process each phase
        for phase_name in phase_order:
            if phase_name not in phase_groups:
                continue

            output.append(f"\n## 🔍 PHASE: {phase_name.upper()}")
            output.append("-" * 80)

            phase_entries = phase_groups[phase_name]

            for phase_key, phase_data in phase_entries:
                researcher_name = phase_data.get('researcher_name', 'Unknown Researcher')
                researcher_id = phase_data.get('researcher', 'unknown')
                candidates = phase_data.get('candidates', [])
                confidence = phase_data.get('confidence', 0)

                output.append(f"\n### Researcher: {researcher_name}")
                output.append(f"- **ID:** `{researcher_id}`")
                output.append(f"- **Documents Found:** {len(candidates)}")
                output.append(f"- **Confidence:** {confidence:.2%}")
                output.append("")

                if candidates:
                    output.append(f"#### Retrieved Documents ({len(candidates)} total):")
                    output.append("")

                    # Show all documents
                    for idx, candidate in enumerate(candidates, 1):
                        try:
                            record = candidate.get('record', {})

                            # Basic info
                            reg_type = record.get('regulation_type', 'N/A')
                            reg_num = record.get('regulation_number', 'N/A')
                            year = record.get('year', 'N/A')
                            about = record.get('about', 'N/A')

                            output.append(f"**{idx}. {reg_type} No. {reg_num}/{year}**")
                            output.append(f"   - About: {about[:100]}{'...' if len(about) > 100 else ''}")

                            if include_scores:
                                # Scores
                                composite_score = candidate.get('composite_score', 0)
                                semantic_score = candidate.get('semantic_score', 0)
                                keyword_score = candidate.get('keyword_score', 0)
                                kg_score = candidate.get('kg_score', 0)

                                output.append(f"   - **Scores:** Composite: {composite_score:.4f} | Semantic: {semantic_score:.4f} | Keyword: {keyword_score:.4f} | KG: {kg_score:.4f}")

                                # KG metadata
                                if kg_score > 0:
                                    kg_domain = record.get('kg_primary_domain', '')
                                    kg_hierarchy = record.get('kg_hierarchy_level', 0)
                                    kg_authority = record.get('kg_authority_score', 0)

                                    output.append(f"   - **KG Metadata:** Domain: {kg_domain} | Hierarchy: {kg_hierarchy} | Authority: {kg_authority:.3f}")

                                # Team consensus
                                if candidate.get('team_consensus'):
                                    agreement = candidate.get('researcher_agreement', 0)
                                    output.append(f"   - **Team Consensus:** ✅ Yes ({agreement} researchers)")

                                # Researcher bias
                                if candidate.get('researcher_bias_applied'):
                                    output.append(f"   - **Researcher:** {candidate['researcher_bias_applied']}")

                            # Content snippet
                            content = record.get('content', '')
                            if content:
                                snippet = content[:200] + "..." if len(content) > 200 else content
                                output.append(f"   - **Content:** {snippet}")

                            output.append("")

                        except Exception as e:
                            output.append(f"   Error formatting document {idx}: {e}")
                            output.append("")

                    total_docs += len(candidates)

                output.append("")

        # Summary
        output.append("\n## 📈 SEARCH SUMMARY")
        output.append("=" * 80)
        output.append(f"- **Total Documents Retrieved:** {total_docs}")
        output.append(f"- **Phases Executed:** {len(phase_groups)}")
        output.append(f"- **Unique Researchers:** {len(set(pd.get('researcher', 'unknown') for phases in phase_groups.values() for _, pd in phases))}")

        return "\n".join(output)

    except Exception as e:
        error_details = traceback.format_exc()
        return f"Error formatting search metadata: {str(e)}\n\n{error_details}"


def export_conversation_to_markdown(conversation_history: List[Dict], include_metadata: bool = True, include_research_process: bool = True) -> str:
    """FIXED: Export with thinking process properly included"""
    try:
        md_parts = []

        md_parts.append("# Legal Consultation Export")
        md_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_parts.append("=" * 80)
        md_parts.append("")

        for idx, entry in enumerate(conversation_history, 1):
            md_parts.append(f"\n## Exchange {idx}")
            md_parts.append("-" * 80)

            # Query
            query = entry.get('query', '')
            md_parts.append(f"\n**Question:** {query}\n")

            # Query metadata
            query_type = entry.get('query_type', 'general')
            query_entities = entry.get('query_entities', [])
            if query_type:
                md_parts.append(f"**Query Type:** {query_type}")
                if query_entities:
                    md_parts.append(f"**Key Entities:** {', '.join(query_entities[:5])}")
                md_parts.append("")

            # *** FIXED: Thinking process properly extracted and displayed ***
            thinking = entry.get('thinking', '')
            if thinking and include_research_process:
                md_parts.append("### 🧠 Thinking Process")
                md_parts.append("-" * 40)
                # Clean up thinking content
                thinking_clean = thinking.strip()
                if thinking_clean:
                    md_parts.append(thinking_clean)
                md_parts.append("")

            # Response
            response = entry.get('response', '')
            if response:
                md_parts.append("### ✅ Answer")
                md_parts.append("-" * 40)
                md_parts.append(response)
                md_parts.append("")

            # Legal References
            sources_used = entry.get('sources_used', [])
            if sources_used:
                md_parts.append(f"### 📚 Legal References ({len(sources_used)} documents)")
                md_parts.append("-" * 80)

                for i, source in enumerate(sources_used, 1):
                    try:
                        regulation = f"{source.get('regulation_type', 'N/A')} No. {source.get('regulation_number', 'N/A')}/{source.get('year', 'N/A')}"
                        md_parts.append(f"\n**{i}. {regulation}**")
                        md_parts.append(f"   - About: {source.get('about', 'N/A')}")
                        md_parts.append(f"   - Enacting Body: {source.get('enacting_body', 'N/A')}")

                        if source.get('final_score'):
                            md_parts.append(f"   - Score: {source.get('final_score', 0):.4f}")
                        if source.get('kg_score'):
                            md_parts.append(f"   - KG Score: {source.get('kg_score', 0):.4f}")
                        if source.get('kg_primary_domain'):
                            md_parts.append(f"   - Domain: {source.get('kg_primary_domain')}")
                        if source.get('team_consensus'):
                            md_parts.append(f"   - Team Consensus: ✓ Yes")

                        content = source.get('content', '')
                        if content:
                            snippet = content[:200] + "..." if len(content) > 200 else content
                            md_parts.append(f"   - Content: {snippet}")
                    except Exception as e:
                        print(f"Error processing source {i}: {e}")
                        continue

                md_parts.append("")

            # Research process details with complete metadata
            if include_research_process and include_metadata and entry.get('research_log'):
                research_log = entry['research_log']
                md_parts.append("### 🔍 Research Process Details")
                md_parts.append("-" * 80)
                md_parts.append(f"- **Team Members:** {len(research_log.get('team_members', []))}")
                md_parts.append(f"- **Total Documents Retrieved:** {research_log.get('total_documents_retrieved', 0)}")

                phase_results = research_log.get('phase_results', {})
                if phase_results:
                    md_parts.append(f"- **Phases Executed:** {len(phase_results)}")

                    # Detailed phase breakdown
                    phase_order = ['initial_scan', 'focused_review', 'deep_analysis', 'verification', 'expert_review']
                    phase_groups = {}

                    for phase_key, phase_data in phase_results.items():
                        phase_name = phase_key.split('_', 1)[-1] if '_' in phase_key else phase_key
                        for base_phase in phase_order:
                            if base_phase in phase_key:
                                phase_name = base_phase
                                break

                        if phase_name not in phase_groups:
                            phase_groups[phase_name] = []
                        phase_groups[phase_name].append((phase_key, phase_data))

                    total_retrieved = 0

                    for phase_name in phase_order:
                        if phase_name not in phase_groups:
                            continue

                        md_parts.append(f"\n#### 📂 PHASE: {phase_name.upper()}")

                        phase_entries = phase_groups[phase_name]
                        phase_total = sum(len(pd.get('candidates', [])) for _, pd in phase_entries)
                        total_retrieved += phase_total

                        for phase_key, phase_data in phase_entries:
                            researcher_name = phase_data.get('researcher_name', 'Unknown')
                            candidates = phase_data.get('candidates', [])
                            confidence = phase_data.get('confidence', 0)

                            md_parts.append(f"\n**{researcher_name}:** {len(candidates)} documents (Confidence: {confidence:.2%})")

                            # Show top 5 documents per researcher
                            for idx_doc, candidate in enumerate(candidates[:5], 1):
                                try:
                                    record = candidate.get('record', {})
                                    reg_info = f"{record.get('regulation_type', 'N/A')} No. {record.get('regulation_number', 'N/A')}/{record.get('year', 'N/A')}"
                                    composite = candidate.get('composite_score', 0)
                                    kg_score = candidate.get('kg_score', 0)

                                    md_parts.append(f"   {idx_doc}. {reg_info} (Score: {composite:.3f}, KG: {kg_score:.3f})")
                                except Exception:
                                    continue

                            if len(candidates) > 5:
                                md_parts.append(f"   ... and {len(candidates) - 5} more documents")

                    md_parts.append(f"\n**Total Documents Retrieved:** {total_retrieved}")
                md_parts.append("")

        md_parts.append("\n" + "=" * 80)
        md_parts.append("\n*Export completed successfully*")

        return "\n".join(md_parts)

    except Exception as e:
        error_details = traceback.format_exc()
        return f"# Error Generating Markdown Export\n\n```\n{error_details}\n```"


def export_conversation_to_json(conversation_history: List[Dict], include_full_content: bool = True) -> str:
    """FIXED: JSON export with proper error handling and serialization"""
    try:
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_exchanges": len(conversation_history),
                "version": "2.1-fixed"
            },
            "conversation": []
        }

        for idx, entry in enumerate(conversation_history, 1):
            try:
                exchange = {
                    "exchange_number": idx,
                    "query": str(entry.get('query', '')),
                    "query_type": str(entry.get('query_type', 'general')),
                    "query_entities": [str(e) for e in entry.get('query_entities', [])],
                    "response": {
                        "thinking": str(entry.get('thinking', '')),
                        "answer": str(entry.get('response', ''))
                    },
                    "legal_references": {
                        "count": len(entry.get('sources_used', [])),
                        "sources": []
                    },
                    "research_metadata": {
                        "available": bool(entry.get('research_log')),
                        "total_documents_retrieved": 0,
                        "phases": {}
                    }
                }

                # Add legal references
                if entry.get('sources_used'):
                    for source in entry['sources_used']:
                        try:
                            source_info = {
                                "regulation": {
                                    "type": str(source.get('regulation_type', '')),
                                    "number": str(source.get('regulation_number', '')),
                                    "year": str(source.get('year', '')),
                                    "citation": f"{source.get('regulation_type', '')} No. {source.get('regulation_number', '')}/{source.get('year', '')}"
                                },
                                "about": str(source.get('about', ''))[:500],  # Limit length
                                "enacting_body": str(source.get('enacting_body', '')),
                                "scores": {
                                    "final": float(source.get('final_score', 0)),
                                    "kg": float(source.get('kg_score', 0))
                                },
                                "kg_metadata": {
                                    "primary_domain": str(source.get('kg_primary_domain', '')),
                                    "hierarchy_level": int(source.get('kg_hierarchy_level', 0)),
                                    "team_consensus": bool(source.get('team_consensus', False))
                                }
                            }

                            if include_full_content:
                                content = str(source.get('content', ''))
                                source_info["content"] = content[:1000] if len(content) > 1000 else content

                            exchange["legal_references"]["sources"].append(source_info)
                        except Exception as e:
                            print(f"Error processing source: {e}")
                            continue

                # Add research metadata
                if entry.get('research_log') and entry['research_log'].get('phase_results'):
                    try:
                        phase_results = entry['research_log']['phase_results']

                        for phase_key, phase_data in phase_results.items():
                            phase_name = str(phase_key.split('_', 1)[-1] if '_' in phase_key else phase_key)

                            if phase_name not in exchange["research_metadata"]["phases"]:
                                exchange["research_metadata"]["phases"][phase_name] = {
                                    "researchers": [],
                                    "total_documents": 0
                                }

                            researcher_data = {
                                "researcher_id": str(phase_data.get('researcher', 'unknown')),
                                "researcher_name": str(phase_data.get('researcher_name', 'Unknown')),
                                "documents_found": int(len(phase_data.get('candidates', []))),
                                "confidence": float(phase_data.get('confidence', 0)),
                                "documents": []
                            }

                            # Add top documents
                            for candidate in phase_data.get('candidates', [])[:10]:
                                try:
                                    record = candidate.get('record', {})

                                    doc_data = {
                                        "regulation": {
                                            "type": str(record.get('regulation_type', '')),
                                            "number": str(record.get('regulation_number', '')),
                                            "year": str(record.get('year', '')),
                                        },
                                        "scores": {
                                            "composite": float(candidate.get('composite_score', 0)),
                                            "semantic": float(candidate.get('semantic_score', 0)),
                                            "keyword": float(candidate.get('keyword_score', 0)),
                                            "kg": float(candidate.get('kg_score', 0))
                                        },
                                        "team_consensus": bool(candidate.get('team_consensus', False))
                                    }

                                    if include_full_content:
                                        content = str(record.get('content', ''))
                                        doc_data["content"] = content[:500] if len(content) > 500 else content

                                    researcher_data["documents"].append(doc_data)

                                except Exception as e:
                                    print(f"Error processing document: {e}")
                                    continue

                            exchange["research_metadata"]["phases"][phase_name]["researchers"].append(researcher_data)
                            exchange["research_metadata"]["phases"][phase_name]["total_documents"] += len(phase_data.get('candidates', []))

                        exchange["research_metadata"]["total_documents_retrieved"] = sum(
                            phase["total_documents"]
                            for phase in exchange["research_metadata"]["phases"].values()
                        )
                    except Exception as e:
                        print(f"Error processing research metadata: {e}")

                export_data["conversation"].append(exchange)

            except Exception as e:
                print(f"Error processing exchange {idx}: {e}")
                # Add placeholder for failed exchange
                export_data["conversation"].append({
                    "exchange_number": idx,
                    "error": str(e),
                    "query": str(entry.get('query', 'Error processing query'))
                })
                continue

        # *** FIXED: Ensure proper JSON serialization ***
        return json.dumps(export_data, indent=2, ensure_ascii=False, default=str)

    except Exception as e:
        # *** FIXED: Return valid JSON even on error ***
        error_details = traceback.format_exc()
        return json.dumps({
            "error": "Export failed",
            "message": str(e),
            "traceback": error_details
        }, indent=2, ensure_ascii=False)


def export_conversation_to_html(conversation_history: List[Dict], include_metadata: bool = True) -> str:
    """FIXED: HTML export with thinking process and table rendering support"""
    try:
        html_content = []

        # Enhanced CSS with table support
        html_content.append("""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legal Consultation Export</title>
    <style>
        * { box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            line-height: 1.8;
            color: #2c3e50;
        }

        .container {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        h1 { color: #1a237e; border-bottom: 4px solid #3f51b5; padding-bottom: 15px; margin-bottom: 30px; }
        h2 { color: #283593; margin-top: 40px; padding: 15px; background: linear-gradient(to right, #e8eaf6, transparent); border-left: 5px solid #3f51b5; }
        h3 { color: #3949ab; margin-top: 25px; padding-left: 10px; border-left: 3px solid #5c6bc0; }

        .exchange {
            background: #f8f9fa;
            padding: 25px;
            margin: 30px 0;
            border-radius: 12px;
            border-left: 4px solid #3f51b5;
        }

        .question {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #ff9800;
        }

        /* *** FIXED: Thinking process styling *** */
        .thinking {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }

        .thinking h4 {
            margin-top: 0;
            color: #1976d2;
        }

        .answer {
            background: #f0f7ff;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #4caf50;
        }

        /* *** FIXED: Table styling *** */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        table thead {
            background: #3f51b5;
            color: white;
        }

        table th, table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }

        table tbody tr:hover {
            background: #f5f5f5;
        }

        table th {
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }

        /* Collapsible details */
        details {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #e0e0e0;
        }

        details summary {
            font-weight: 600;
            color: #1565c0;
            cursor: pointer;
            user-select: none;
        }

        details[open] summary {
            margin-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }

        .doc-item {
            background: #fafafa;
            padding: 12px;
            margin: 10px 0;
            border-radius: 6px;
            border-left: 3px solid #1976d2;
        }

        .score-badge {
            display: inline-block;
            background: #4caf50;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            margin-right: 5px;
            font-weight: 600;
        }

        code {
            background-color: #f5f5f5;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            color: #d32f2f;
        }

        @media print {
            body { background: white; }
            .exchange { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Legal Consultation Export</h1>
        <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        <hr>
""")

        # Process each exchange
        for idx, entry in enumerate(conversation_history, 1):
            html_content.append(f'<div class="exchange">')
            html_content.append(f'<h2>Exchange {idx}</h2>')

            # Query
            query = entry.get('query', '')
            html_content.append(f'<div class="question"><strong>Question:</strong> {query}</div>')

            # *** FIXED: Thinking process display ***
            thinking = entry.get('thinking', '')
            if thinking:
                thinking_html = thinking.replace('<', '&lt;').replace('>', '&gt;')
                html_content.append(f'<details open>')
                html_content.append(f'<summary>Thinking Process</summary>')
                html_content.append(f'<div class="thinking">{thinking_html}</div>')
                html_content.append(f'</details>')

            # Response - convert markdown to HTML including tables
            response = entry.get('response', '')
            if response:
                # *** FIXED: Markdown conversion with table extension ***
                if markdown:
                    response_html = markdown.markdown(
                        response,
                        extensions=['tables', 'fenced_code', 'nl2br']
                    )
                else:
                    # Fallback if markdown not available
                    response_html = response.replace('\n', '<br>')
                html_content.append(f'<div class="answer"><strong>Answer:</strong><br>{response_html}</div>')

            # Legal References
            sources_used = entry.get('sources_used', [])
            if sources_used:
                html_content.append(f'<details>')
                html_content.append(f'<summary>Legal References ({len(sources_used)} documents)</summary>')

                for i, source in enumerate(sources_used, 1):
                    try:
                        regulation = f"{source.get('regulation_type', 'N/A')} No. {source.get('regulation_number', 'N/A')}/{source.get('year', 'N/A')}"
                        html_content.append(f'<div class="doc-item">')
                        html_content.append(f'<strong>{i}. {regulation}</strong>')

                        if source.get('about'):
                            html_content.append(f'<br><em>{source.get("about")}</em>')

                        scores_html = ''
                        if source.get('final_score'):
                            scores_html += f'<span class="score-badge">Score: {source.get("final_score", 0):.4f}</span>'
                        if source.get('kg_score'):
                            scores_html += f'<span class="score-badge">KG: {source.get("kg_score", 0):.4f}</span>'
                        if scores_html:
                            html_content.append(f'<br>{scores_html}')

                        html_content.append(f'</div>')
                    except Exception:
                        continue

                html_content.append(f'</details>')

            # All search results metadata - DETAILED with COLLAPSIBLE PHASES
            if include_metadata and entry.get('research_log') and entry['research_log'].get('phase_results'):
                html_content.append('<details>')
                html_content.append('<summary>Complete Search Results (All Retrieved Documents)</summary>')

                phase_results = entry['research_log']['phase_results']
                phase_order = ['initial_scan', 'focused_review', 'deep_analysis', 'verification', 'expert_review']
                phase_groups = {}

                for phase_key, phase_data in phase_results.items():
                    phase_name = phase_key.split('_', 1)[-1] if '_' in phase_key else phase_key
                    for base_phase in phase_order:
                        if base_phase in phase_key:
                            phase_name = base_phase
                            break

                    if phase_name not in phase_groups:
                        phase_groups[phase_name] = []
                    phase_groups[phase_name].append((phase_key, phase_data))

                total_retrieved = 0

                for phase_name in phase_order:
                    if phase_name not in phase_groups:
                        continue

                    html_content.append(f'<div class="phase-section">')
                    html_content.append(f'<div class="phase-title">PHASE: {phase_name.upper()}</div>')

                    phase_entries = phase_groups[phase_name]
                    phase_total = sum(len(pd.get('candidates', [])) for _, pd in phase_entries)
                    total_retrieved += phase_total

                    for phase_key, phase_data in phase_entries:
                        researcher_name = phase_data.get('researcher_name', 'Unknown')
                        candidates = phase_data.get('candidates', [])
                        confidence = phase_data.get('confidence', 0)

                        html_content.append(f'<details>')
                        html_content.append(f'<summary>{researcher_name}: {len(candidates)} documents (Confidence: {confidence:.2%})</summary>')

                        html_content.append(f'<div class="researcher-section">')

                        # Display ALL documents with complete details
                        for idx_doc, candidate in enumerate(candidates, 1):
                            try:
                                record = candidate.get('record', {})
                                reg_info = f"{record.get('regulation_type', 'N/A')} No. {record.get('regulation_number', 'N/A')}/{record.get('year', 'N/A')}"
                                about = record.get('about', 'N/A')
                                composite = candidate.get('composite_score', 0)
                                kg_score = candidate.get('kg_score', 0)

                                html_content.append(f'<div class="document-item">')

                                # Main document line with scores
                                score_badges = f'<span class="score-badge">{composite:.3f}</span>'
                                score_badges += f'<span class="score-badge">KG: {kg_score:.3f}</span>'

                                html_content.append(f'<div class="doc-title"><span class="doc-number">{idx_doc}.</span> {reg_info}</div>')
                                html_content.append(f'<div class="doc-detail">{score_badges}</div>')

                                # Additional details
                                if about:
                                    snippet = about[:100] + "..." if len(about) > 100 else about
                                    html_content.append(f'<div class="doc-detail"><strong>About:</strong> {snippet}</div>')

                                if record.get('enacting_body'):
                                    html_content.append(f'<div class="doc-detail"><strong>Body:</strong> {record.get("enacting_body")}</div>')

                                # KG metadata
                                kg_details = []
                                if record.get('kg_primary_domain'):
                                    kg_details.append(f"Domain: {record.get('kg_primary_domain')}")
                                if record.get('kg_hierarchy_level'):
                                    kg_details.append(f"Hierarchy: Level {record.get('kg_hierarchy_level')}")
                                if kg_details:
                                    html_content.append(f'<div class="doc-detail"><strong>KG:</strong> {", ".join(kg_details)}</div>')

                                # Team consensus
                                if candidate.get('team_consensus'):
                                    html_content.append(f'<div class="doc-detail"><span class="consensus-badge">Team Consensus ({candidate.get("researcher_agreement", 0)} researchers)</span></div>')

                                html_content.append(f'</div>')
                            except Exception as e:
                                print(f"Error processing document {idx_doc}: {e}")
                                continue

                        html_content.append(f'</div>')
                        html_content.append(f'</details>')

                    html_content.append(f'</div>')

                html_content.append(f'<p><strong>Total Documents Retrieved:</strong> {total_retrieved}</p>')
                html_content.append(f'</details>')

            html_content.append('</div>')

        html_content.append("""
        <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 2px solid #e0e0e0; color: #757575;">
            <p><strong>Generated by Enhanced KG Indonesian Legal RAG System</strong></p>
        </div>
    </div>
</body>
</html>
""")

        return "\n".join(html_content)

    except Exception as e:
        error_details = traceback.format_exc()
        return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Export Error</title></head>
<body>
    <h1>Error Generating HTML Export</h1>
    <pre>{error_details}</pre>
</body>
</html>"""
