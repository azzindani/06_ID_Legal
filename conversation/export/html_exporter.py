"""
HTML Exporter - Export Conversations to HTML Format

File: conversation/export/html_exporter.py
"""

from typing import Dict, Any, List, Optional
from .base_exporter import BaseExporter

# Optional markdown import for proper markdown-to-HTML conversion
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


class HTMLExporter(BaseExporter):
    """
    Export conversations to styled HTML format

    Features:
    - Responsive design
    - Print-friendly styles
    - Collapsible sections
    - Syntax highlighting ready
    """

    def get_file_extension(self) -> str:
        return '.html'

    def export(self, session_data: Dict[str, Any]) -> str:
        """
        Export session to HTML

        Args:
            session_data: Session data from ConversationManager

        Returns:
            HTML string
        """
        html_parts = []

        # HTML header with styles
        html_parts.append(self._get_html_header(session_data))

        # Body content
        html_parts.append('<body>')
        html_parts.append('<div class="container">')

        # Title and summary
        html_parts.append(self._get_title_section(session_data))

        # Session summary
        if self.include_metadata:
            html_parts.append(self._get_summary_section(session_data))

        # Conversation turns
        html_parts.append('<section class="conversation">')
        html_parts.append('<h2>Percakapan</h2>')

        for turn in session_data.get('turns', []):
            html_parts.append(self._format_turn(turn))

        html_parts.append('</section>')

        # Citations
        if self.include_sources:
            citations = self._collect_citations(session_data)
            if citations:
                html_parts.append(self._get_citations_section(citations))

        # Footer
        html_parts.append(self._get_footer(session_data))

        html_parts.append('</div>')
        html_parts.append('</body>')
        html_parts.append('</html>')

        return '\n'.join(html_parts)

    def _get_html_header(self, session_data: Dict[str, Any]) -> str:
        """Get HTML header with CSS styles"""
        return f'''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Konsultasi Hukum - {session_data.get('id', 'N/A')}</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --background-color: #f5f6fa;
            --card-background: #ffffff;
            --text-color: #2c3e50;
            --border-color: #dcdde1;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid var(--secondary-color);
            margin-bottom: 30px;
        }}

        h1 {{
            color: var(--primary-color);
            margin-bottom: 10px;
        }}

        h2 {{
            color: var(--primary-color);
            margin: 20px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }}

        h3 {{
            color: var(--secondary-color);
            margin: 15px 0 10px 0;
            font-size: 1.2em;
        }}

        h4 {{
            color: var(--text-color);
            margin: 10px 0 8px 0;
            font-size: 1.1em;
        }}

        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}

        li {{
            margin: 5px 0;
        }}

        p {{
            margin: 10px 0;
        }}

        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}

        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
        }}

        pre code {{
            background-color: transparent;
            padding: 0;
        }}

        blockquote {{
            border-left: 4px solid var(--secondary-color);
            margin: 15px 0;
            padding: 10px 20px;
            background-color: var(--background-color);
            font-style: italic;
        }}

        hr {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 20px 0;
        }}

        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}

        .summary {{
            background: var(--card-background);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .summary ul {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}

        .summary li {{
            padding: 8px;
            background: var(--background-color);
            border-radius: 4px;
        }}

        .turn {{
            background: var(--card-background);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .question {{
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid var(--secondary-color);
            margin-bottom: 15px;
        }}

        .question-label {{
            font-weight: bold;
            color: var(--secondary-color);
            margin-bottom: 8px;
        }}

        .answer {{
            padding: 15px 0;
        }}

        .answer-label {{
            font-weight: bold;
            color: var(--primary-color);
            margin-bottom: 8px;
        }}

        .details {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid var(--border-color);
        }}

        details {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}

        summary {{
            cursor: pointer;
            font-weight: bold;
            padding: 5px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: var(--background-color);
            font-weight: bold;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
            border-top: 1px solid var(--border-color);
            margin-top: 30px;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            .container {{
                max-width: 100%;
            }}

            .turn, .summary {{
                box-shadow: none;
                border: 1px solid var(--border-color);
            }}

            details {{
                display: block;
            }}
        }}
    </style>
</head>'''

    def _get_title_section(self, session_data: Dict[str, Any]) -> str:
        """Get title section HTML"""
        return f'''<header>
    <h1>Konsultasi Hukum Indonesia</h1>
    <p class="meta">
        Session ID: <code>{session_data.get('id', 'N/A')}</code><br>
        Tanggal: {self._format_timestamp(session_data.get('created_at', ''))}
    </p>
</header>'''

    def _get_summary_section(self, session_data: Dict[str, Any]) -> str:
        """Get summary section HTML"""
        metadata = session_data.get('metadata', {})

        return f'''<section class="summary">
    <h2>Ringkasan Sesi</h2>
    <ul>
        <li><strong>Total Pertanyaan:</strong> {metadata.get('total_queries', 0)}</li>
        <li><strong>Total Token:</strong> {metadata.get('total_tokens', 0)}</li>
        <li><strong>Total Waktu:</strong> {self._format_duration(metadata.get('total_time', 0))}</li>
        <li><strong>Peraturan Dikutip:</strong> {len(metadata.get('regulations_cited', []))}</li>
    </ul>
</section>'''

    def _format_turn(self, turn: Dict[str, Any]) -> str:
        """Format a single turn as HTML"""
        turn_num = turn.get('turn_number', '?')
        timestamp = self._format_timestamp(turn.get('timestamp', ''))

        html = f'''<div class="turn">
    <div class="question">
        <div class="question-label">Pertanyaan {turn_num}</div>
        <div>{turn.get('query', '')}</div>
    </div>
    <div class="answer">
        <div class="answer-label">Jawaban {turn_num}</div>
        <div>{self._format_answer_text(turn.get('answer', ''))}</div>
    </div>'''

        # Add metadata details
        if self.include_metadata and turn.get('metadata'):
            html += self._format_turn_details(turn['metadata'], timestamp)

        html += '</div>'
        return html

    def _format_answer_text(self, text: str) -> str:
        """
        Format answer text with proper markdown-to-HTML conversion

        Converts markdown syntax to HTML including:
        - Headers (###, ##, etc.)
        - Bold (**text**)
        - Lists
        - Code blocks
        - Tables
        - Details/summary tags
        """
        if not text:
            return ''

        # Convert markdown to HTML if library is available
        if MARKDOWN_AVAILABLE:
            # Use markdown with extensions for tables, code, and line breaks
            html_content = markdown.markdown(
                text,
                extensions=['tables', 'fenced_code', 'nl2br', 'extra']
            )
            return html_content
        else:
            # Fallback: basic formatting with paragraph breaks and line breaks
            # Replace common markdown patterns manually
            html_parts = []
            for paragraph in text.split('\n\n'):
                if paragraph.strip():
                    # Convert line breaks within paragraphs
                    formatted = paragraph.replace('\n', '<br>')
                    # Basic bold conversion
                    formatted = formatted.replace('**', '<strong>').replace('**', '</strong>')
                    html_parts.append(f'<p>{formatted}</p>')
            return ''.join(html_parts)

    def _format_turn_details(self, metadata: Dict[str, Any], timestamp: str) -> str:
        """Format turn metadata as collapsible details"""
        details = f'''<div class="details">
    <details>
        <summary>Detail ({timestamp})</summary>
        <ul>'''

        if self.include_timing:
            if 'total_time' in metadata:
                details += f'<li>Waktu Total: {self._format_duration(metadata["total_time"])}</li>'
            if 'retrieval_time' in metadata:
                details += f'<li>Waktu Retrieval: {self._format_duration(metadata["retrieval_time"])}</li>'

        if 'tokens_generated' in metadata:
            details += f'<li>Token: {metadata["tokens_generated"]}</li>'

        if 'query_type' in metadata:
            details += f'<li>Tipe Query: {metadata["query_type"]}</li>'

        details += '</ul></details></div>'
        return details

    def _collect_citations(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect all unique citations"""
        citations = {}
        for turn in session_data.get('turns', []):
            for citation in turn.get('metadata', {}).get('citations', []):
                key = f"{citation.get('regulation_type')}-{citation.get('regulation_number')}-{citation.get('year')}"
                if key not in citations:
                    citations[key] = citation
        return list(citations.values())

    def _get_citations_section(self, citations: List[Dict[str, Any]]) -> str:
        """Get citations section HTML"""
        html = '''<section class="citations">
    <h2>Daftar Referensi</h2>
    <table>
        <thead>
            <tr>
                <th>No</th>
                <th>Jenis</th>
                <th>Nomor</th>
                <th>Tahun</th>
                <th>Tentang</th>
            </tr>
        </thead>
        <tbody>'''

        for i, citation in enumerate(citations, 1):
            about = citation.get('about', 'N/A')
            if len(about) > 50:
                about = about[:47] + '...'

            html += f'''<tr>
                <td>{i}</td>
                <td>{citation.get('regulation_type', 'N/A')}</td>
                <td>{citation.get('regulation_number', 'N/A')}</td>
                <td>{citation.get('year', 'N/A')}</td>
                <td>{about}</td>
            </tr>'''

        html += '</tbody></table></section>'
        return html

    def _get_footer(self, session_data: Dict[str, Any]) -> str:
        """Get footer HTML"""
        return f'''<footer>
    Diekspor pada {self._format_timestamp(session_data.get('updated_at', ''))}
</footer>'''
