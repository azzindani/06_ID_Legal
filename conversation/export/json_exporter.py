"""
JSON Exporter - Export Conversations to JSON Format

File: conversation/export/json_exporter.py
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
from .base_exporter import BaseExporter


class JSONExporter(BaseExporter):
    """
    Export conversations to JSON format

    Features:
    - Complete metadata preservation
    - Structured for easy parsing
    - Optional pretty printing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pretty_print = self.config.get('pretty_print', True)
        self.indent = self.config.get('indent', 2)

    def get_file_extension(self) -> str:
        return '.json'

    def export(self, session_data: Dict[str, Any]) -> str:
        """
        Export session to JSON

        Args:
            session_data: Session data from ConversationManager

        Returns:
            JSON string
        """
        export_data = {
            'export_info': {
                'format': 'json',
                'version': '1.0',
                'exported_at': datetime.now().isoformat(),
                'exporter': 'JSONExporter'
            },
            'session': self._prepare_session_data(session_data)
        }

        if self.pretty_print:
            return json.dumps(export_data, ensure_ascii=False, indent=self.indent)
        else:
            return json.dumps(export_data, ensure_ascii=False)

    def _prepare_session_data(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare session data for JSON export"""
        prepared = {
            'id': session_data.get('id'),
            'created_at': session_data.get('created_at'),
            'updated_at': session_data.get('updated_at'),
            'summary': self._create_summary(session_data),
            'turns': []
        }

        # Process turns
        for turn in session_data.get('turns', []):
            turn_data = {
                'turn_number': turn.get('turn_number'),
                'timestamp': turn.get('timestamp'),
                'query': turn.get('query'),
                'thinking': turn.get('thinking', ''),
                'answer': turn.get('answer'),
                'sources': turn.get('sources_text', ''),  # Full formatted sources text
                'research_process': turn.get('research_text', ''),  # Full formatted research text
            }

            # Include additional metadata if enabled
            if self.include_metadata:
                meta = turn.get('metadata', {})
                turn_data['metadata'] = {
                    'timing': {},
                    'stats': {}
                }
                # Timing info
                if self.include_timing:
                    for key in ['total_time', 'retrieval_time', 'generation_time']:
                        if key in meta:
                            turn_data['metadata']['timing'][key] = meta[key]
                # Stats
                for key in ['tokens_generated', 'query_type', 'results_count']:
                    if key in meta:
                        turn_data['metadata']['stats'][key] = meta[key]

            prepared['turns'].append(turn_data)

        return prepared

    def _create_summary(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create session summary"""
        metadata = session_data.get('metadata', {})

        summary = {
            'total_turns': len(session_data.get('turns', [])),
            'total_queries': metadata.get('total_queries', 0),
            'total_tokens': metadata.get('total_tokens', 0),
            'total_time_seconds': round(metadata.get('total_time', 0), 3)
        }

        if metadata.get('regulations_cited'):
            summary['regulations_cited'] = list(metadata['regulations_cited'])
            summary['unique_regulations_count'] = len(metadata['regulations_cited'])

        return summary

    def _prepare_turn_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare turn metadata for export"""
        prepared = {}

        # Timing
        if self.include_timing:
            if 'total_time' in metadata:
                prepared['total_time'] = round(metadata['total_time'], 3)
            if 'retrieval_time' in metadata:
                prepared['retrieval_time'] = round(metadata['retrieval_time'], 3)
            if 'generation_time' in metadata:
                prepared['generation_time'] = round(metadata['generation_time'], 3)

        # Other metadata
        for key in ['tokens_generated', 'query_type', 'results_count']:
            if key in metadata:
                prepared[key] = metadata[key]

        # Sources
        if self.include_sources and 'citations' in metadata:
            prepared['citations'] = metadata['citations']

        if self.include_sources and 'sources' in metadata:
            prepared['sources'] = metadata['sources']

        return prepared
