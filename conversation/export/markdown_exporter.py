"""
Markdown Exporter - Export Conversations to Markdown Format

File: conversation/export/markdown_exporter.py
"""

from typing import Dict, Any, List, Optional
from .base_exporter import BaseExporter


class MarkdownExporter(BaseExporter):
    """
    Export conversations to Markdown format

    Features:
    - Collapsible sections for metadata
    - Formatted tables for sources
    - Code blocks for technical content
    """

    def get_file_extension(self) -> str:
        return '.md'

    def export(self, session_data: Dict[str, Any]) -> str:
        """
        Export session to Markdown

        Args:
            session_data: Session data from ConversationManager

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append(f"# Konsultasi Hukum Indonesia")
        lines.append("")
        lines.append(f"**Session ID:** `{session_data.get('id', 'N/A')}`")
        lines.append(f"**Tanggal:** {self._format_timestamp(session_data.get('created_at', ''))}")
        lines.append("")

        # Session summary
        if self.include_metadata:
            metadata = session_data.get('metadata', {})
            lines.append("## Ringkasan Sesi")
            lines.append("")
            lines.append(f"- **Total Pertanyaan:** {metadata.get('total_queries', 0)}")
            lines.append(f"- **Total Token:** {metadata.get('total_tokens', 0)}")
            lines.append(f"- **Total Waktu:** {self._format_duration(metadata.get('total_time', 0))}")

            regulations = metadata.get('regulations_cited', [])
            if regulations:
                lines.append(f"- **Peraturan Dikutip:** {len(regulations)}")

            lines.append("")

        # Conversation turns
        lines.append("## Percakapan")
        lines.append("")

        for turn in session_data.get('turns', []):
            lines.extend(self._format_turn(turn))
            lines.append("")

        # Citations summary
        if self.include_sources:
            all_citations = self._collect_citations(session_data)
            if all_citations:
                lines.append("## Daftar Referensi")
                lines.append("")
                lines.extend(self._format_citations_table(all_citations))
                lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Diekspor pada {self._format_timestamp(session_data.get('updated_at', ''))}*")

        return '\n'.join(lines)

    def _format_turn(self, turn: Dict[str, Any]) -> List[str]:
        """Format a single conversation turn"""
        lines = []

        turn_num = turn.get('turn_number', '?')
        timestamp = self._format_timestamp(turn.get('timestamp', ''))

        # Question
        lines.append(f"### Pertanyaan {turn_num}")
        lines.append("")
        lines.append(f"> {turn.get('query', '')}")
        lines.append("")

        # Answer
        lines.append(f"### Jawaban {turn_num}")
        lines.append("")
        lines.append(turn.get('answer', ''))
        lines.append("")

        # Metadata (collapsible)
        if self.include_metadata and turn.get('metadata'):
            meta = turn['metadata']
            lines.append("<details>")
            lines.append(f"<summary>Detail ({timestamp})</summary>")
            lines.append("")

            if self.include_timing:
                if 'total_time' in meta:
                    lines.append(f"- **Waktu Total:** {self._format_duration(meta['total_time'])}")
                if 'retrieval_time' in meta:
                    lines.append(f"- **Waktu Retrieval:** {self._format_duration(meta['retrieval_time'])}")
                if 'generation_time' in meta:
                    lines.append(f"- **Waktu Generasi:** {self._format_duration(meta['generation_time'])}")

            if 'tokens_generated' in meta:
                lines.append(f"- **Token:** {meta['tokens_generated']}")

            if 'query_type' in meta:
                lines.append(f"- **Tipe Query:** {meta['query_type']}")

            if 'results_count' in meta:
                lines.append(f"- **Hasil Ditemukan:** {meta['results_count']}")

            # Sources
            if self.include_sources and 'citations' in meta:
                lines.append("")
                lines.append("**Sumber:**")
                for i, citation in enumerate(meta['citations'], 1):
                    cite_text = citation.get('citation_text', '')
                    if not cite_text:
                        cite_text = f"{citation.get('regulation_type', '')} No. {citation.get('regulation_number', '')}/{citation.get('year', '')}"
                    lines.append(f"{i}. {cite_text}")

            lines.append("")
            lines.append("</details>")

        return lines

    def _collect_citations(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect all unique citations from session"""
        citations = {}

        for turn in session_data.get('turns', []):
            meta = turn.get('metadata', {})
            for citation in meta.get('citations', []):
                key = f"{citation.get('regulation_type', '')}-{citation.get('regulation_number', '')}-{citation.get('year', '')}"
                if key not in citations:
                    citations[key] = citation

        return list(citations.values())

    def _format_citations_table(self, citations: List[Dict[str, Any]]) -> List[str]:
        """Format citations as markdown table"""
        lines = []

        lines.append("| No | Jenis | Nomor | Tahun | Tentang |")
        lines.append("|----|----|----|----|-----|")

        for i, citation in enumerate(citations, 1):
            reg_type = citation.get('regulation_type', 'N/A')
            reg_num = citation.get('regulation_number', 'N/A')
            year = citation.get('year', 'N/A')
            about = citation.get('about', 'N/A')

            # Truncate about if too long
            if len(about) > 50:
                about = about[:47] + "..."

            lines.append(f"| {i} | {reg_type} | {reg_num} | {year} | {about} |")

        return lines

    def format_complete_search_metadata(
        self,
        rag_result: Dict[str, Any],
        include_scores: bool = True
    ) -> str:
        """
        Format complete search metadata including ALL documents retrieved
        Ported from original Kaggle_Demo.ipynb

        Args:
            rag_result: Complete RAG result with research_log or all_retrieved_metadata
            include_scores: Include detailed scoring information

        Returns:
            Formatted string with all search results
        """
        lines = []

        lines.append("# COMPLETE SEARCH RESULTS METADATA")
        lines.append("=" * 80)
        lines.append("")

        # Get research log or all_retrieved_metadata
        research_log = rag_result.get('research_log', {})
        phase_results = research_log.get('phase_results', rag_result.get('all_retrieved_metadata', {}))

        if not phase_results:
            return "No search metadata available."

        # Group by phase
        phase_order = ['initial_scan', 'focused_review', 'deep_analysis', 'verification', 'expert_review']
        phase_groups = {}

        for phase_key, phase_data in phase_results.items():
            # Determine phase name
            phase_name = 'unknown'
            for base_phase in phase_order:
                if base_phase in phase_key:
                    phase_name = base_phase
                    break

            if phase_name not in phase_groups:
                phase_groups[phase_name] = []
            phase_groups[phase_name].append((phase_key, phase_data))

        total_docs = 0

        # Process each phase
        for phase_name in phase_order:
            if phase_name not in phase_groups:
                continue

            lines.append(f"\n## PHASE: {phase_name.upper()}")
            lines.append("-" * 80)

            phase_entries = phase_groups[phase_name]

            for phase_key, phase_data in phase_entries:
                researcher_name = phase_data.get('researcher_name', 'Unknown Researcher')
                researcher_id = phase_data.get('researcher', 'unknown')
                candidates = phase_data.get('candidates', phase_data.get('results', []))
                confidence = phase_data.get('confidence', 0)

                lines.append(f"\n### Researcher: {researcher_name}")
                lines.append(f"- **ID:** `{researcher_id}`")
                lines.append(f"- **Documents Found:** {len(candidates)}")
                lines.append(f"- **Confidence:** {confidence:.2%}")
                lines.append("")

                if candidates:
                    lines.append(f"#### Retrieved Documents ({len(candidates)} total):")
                    lines.append("")

                    # Show all documents
                    for idx, candidate in enumerate(candidates, 1):
                        try:
                            record = candidate.get('record', candidate)

                            # Basic info
                            reg_type = record.get('regulation_type', 'N/A')
                            reg_num = record.get('regulation_number', 'N/A')
                            year = record.get('year', 'N/A')
                            about = record.get('about', 'N/A')

                            lines.append(f"**{idx}. {reg_type} No. {reg_num}/{year}**")

                            about_preview = about[:100] + '...' if len(about) > 100 else about
                            lines.append(f"   - About: {about_preview}")

                            if include_scores:
                                # Scores
                                scores = candidate.get('scores', {})
                                composite_score = scores.get('final', candidate.get('composite_score', candidate.get('final_score', 0)))
                                semantic_score = scores.get('semantic', candidate.get('semantic_score', 0))
                                keyword_score = scores.get('keyword', candidate.get('keyword_score', 0))
                                kg_score = scores.get('kg', candidate.get('kg_score', 0))

                                lines.append(f"   - **Scores:** Composite: {composite_score:.4f} | Semantic: {semantic_score:.4f} | Keyword: {keyword_score:.4f} | KG: {kg_score:.4f}")

                                # KG metadata
                                if kg_score > 0:
                                    kg_domain = record.get('kg_primary_domain', '')
                                    kg_hierarchy = record.get('kg_hierarchy_level', 0)
                                    kg_authority = record.get('kg_authority_score', 0)

                                    if kg_domain or kg_hierarchy or kg_authority:
                                        lines.append(f"   - **KG Metadata:** Domain: {kg_domain} | Hierarchy: {kg_hierarchy} | Authority: {kg_authority:.3f}")

                                # Team consensus
                                if candidate.get('team_consensus'):
                                    agreement = candidate.get('researcher_agreement', len(candidate.get('personas_agreed', [])))
                                    lines.append(f"   - **Team Consensus:** Yes ({agreement} researchers)")

                            # Content snippet
                            content = record.get('content', '')
                            if content:
                                snippet = content[:200] + "..." if len(content) > 200 else content
                                lines.append(f"   - **Content:** {snippet}")

                            lines.append("")

                        except Exception as e:
                            lines.append(f"   Error formatting document {idx}: {e}")
                            lines.append("")

                    total_docs += len(candidates)

                lines.append("")

        # Summary
        lines.append("\n## SEARCH SUMMARY")
        lines.append("=" * 80)
        lines.append(f"- **Total Documents Retrieved:** {total_docs}")
        lines.append(f"- **Phases Executed:** {len(phase_groups)}")

        return "\n".join(lines)

    def export_with_research_details(
        self,
        session_data: Dict[str, Any],
        include_research_process: bool = True
    ) -> str:
        """
        Export session with full research process details
        Similar to original notebook's export_conversation_to_markdown

        Args:
            session_data: Session data from ConversationManager
            include_research_process: Include detailed research process

        Returns:
            Markdown string with research details
        """
        lines = []

        # Header
        lines.append("# Legal Consultation Export")
        lines.append(f"Generated: {self._format_timestamp(session_data.get('updated_at', ''))}")
        lines.append("=" * 80)
        lines.append("")

        # Process each turn
        for idx, turn in enumerate(session_data.get('turns', []), 1):
            lines.append(f"\n## Exchange {idx}")
            lines.append("-" * 80)

            # Query
            query = turn.get('query', '')
            lines.append(f"\n**Question:** {query}\n")

            # Query metadata
            meta = turn.get('metadata', {})
            query_type = meta.get('query_type', 'general')
            if query_type:
                lines.append(f"**Query Type:** {query_type}")
                lines.append("")

            # Thinking process
            thinking = turn.get('thinking', meta.get('thinking', ''))
            if thinking and include_research_process:
                lines.append("### Thinking Process")
                lines.append("-" * 40)
                lines.append(thinking)
                lines.append("")

            # Response
            answer = turn.get('answer', '')
            if answer:
                lines.append("### Answer")
                lines.append("-" * 40)
                lines.append(answer)
                lines.append("")

            # Legal References
            citations = meta.get('citations', [])
            sources = meta.get('sources', [])
            all_sources = citations or sources

            if all_sources:
                lines.append(f"### Legal References ({len(all_sources)} documents)")
                lines.append("-" * 80)

                for i, source in enumerate(all_sources, 1):
                    try:
                        regulation = f"{source.get('regulation_type', 'N/A')} No. {source.get('regulation_number', 'N/A')}/{source.get('year', 'N/A')}"
                        lines.append(f"\n**{i}. {regulation}**")
                        lines.append(f"   - About: {source.get('about', 'N/A')}")

                        if source.get('relevance_score'):
                            lines.append(f"   - Score: {source.get('relevance_score', 0):.4f}")
                        if source.get('kg_score'):
                            lines.append(f"   - KG Score: {source.get('kg_score', 0):.4f}")

                    except Exception as e:
                        lines.append(f"   Error processing source {i}: {e}")

                lines.append("")

            # Research process details
            if include_research_process and meta.get('research_log'):
                research_log = meta['research_log']
                lines.append("### Research Process Details")
                lines.append("-" * 80)
                lines.append(f"- **Team Members:** {len(research_log.get('team_members', []))}")
                lines.append(f"- **Total Documents Retrieved:** {research_log.get('total_documents_retrieved', 0)}")

                # Add complete search metadata
                search_metadata = self.format_complete_search_metadata(
                    {'research_log': research_log},
                    include_scores=True
                )
                lines.append("")
                lines.append(search_metadata)

        lines.append("\n" + "=" * 80)
        lines.append("\n*Export completed successfully*")

        return "\n".join(lines)
