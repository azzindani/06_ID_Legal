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
