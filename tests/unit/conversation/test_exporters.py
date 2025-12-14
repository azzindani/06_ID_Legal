"""
Exporter Tests

Unit tests for conversation export modules.

Run with: pytest conversation/tests/test_exporters.py -v
"""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conversation.export import MarkdownExporter, JSONExporter, HTMLExporter


@pytest.fixture
def sample_session_data():
    """Sample session data for testing"""
    return {
        'id': 'test-session-123',
        'created_at': '2024-01-15T10:30:00',
        'updated_at': '2024-01-15T10:35:00',
        'turns': [
            {
                'turn_number': 1,
                'timestamp': '2024-01-15T10:30:00',
                'query': 'Apa sanksi pelanggaran UU Ketenagakerjaan?',
                'answer': 'Sanksi pelanggaran UU Ketenagakerjaan meliputi sanksi administratif dan pidana.',
                'metadata': {
                    'total_time': 5.2,
                    'retrieval_time': 2.1,
                    'generation_time': 3.1,
                    'tokens_generated': 150,
                    'query_type': 'sanctions',
                    'results_count': 3,
                    'citations': [
                        {
                            'regulation_type': 'Undang-Undang',
                            'regulation_number': '13',
                            'year': '2003',
                            'about': 'Ketenagakerjaan',
                            'citation_text': 'UU No. 13 Tahun 2003 tentang Ketenagakerjaan'
                        }
                    ]
                }
            },
            {
                'turn_number': 2,
                'timestamp': '2024-01-15T10:32:00',
                'query': 'Berapa denda maksimalnya?',
                'answer': 'Denda maksimal adalah Rp 400.000.000.',
                'metadata': {
                    'total_time': 3.5,
                    'tokens_generated': 80,
                    'query_type': 'sanctions'
                }
            }
        ],
        'metadata': {
            'total_queries': 2,
            'total_tokens': 230,
            'total_time': 8.7,
            'regulations_cited': ['UU 13/2003']
        }
    }


class TestMarkdownExporter:
    """Test Markdown export"""

    def test_export_basic(self, sample_session_data):
        """Test basic markdown export"""
        exporter = MarkdownExporter()
        content = exporter.export(sample_session_data)

        assert '# Konsultasi Hukum Indonesia' in content
        assert 'test-session-123' in content
        assert 'Pertanyaan 1' in content
        assert 'Jawaban 1' in content

    def test_export_includes_metadata(self, sample_session_data):
        """Test that export includes metadata"""
        exporter = MarkdownExporter({'include_metadata': True})
        content = exporter.export(sample_session_data)

        assert 'Ringkasan Sesi' in content
        assert 'Total Pertanyaan' in content

    def test_export_excludes_metadata(self, sample_session_data):
        """Test that export can exclude metadata"""
        exporter = MarkdownExporter({'include_metadata': False})
        content = exporter.export(sample_session_data)

        assert 'Ringkasan Sesi' not in content

    def test_export_includes_citations(self, sample_session_data):
        """Test that export includes citations"""
        exporter = MarkdownExporter({'include_sources': True})
        content = exporter.export(sample_session_data)

        assert 'Daftar Referensi' in content
        assert 'Ketenagakerjaan' in content

    def test_export_table_format(self, sample_session_data):
        """Test citations table format"""
        exporter = MarkdownExporter()
        content = exporter.export(sample_session_data)

        assert '| No | Jenis | Nomor | Tahun | Tentang |' in content

    def test_file_extension(self):
        """Test file extension"""
        exporter = MarkdownExporter()
        assert exporter.get_file_extension() == '.md'

    def test_save_to_file(self, sample_session_data):
        """Test saving to file"""
        exporter = MarkdownExporter()
        content = exporter.export(sample_session_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.save_to_file(content, 'test.md', tmpdir)

            assert path.exists()
            assert path.suffix == '.md'

            with open(path, 'r', encoding='utf-8') as f:
                saved_content = f.read()

            assert saved_content == content


class TestJSONExporter:
    """Test JSON export"""

    def test_export_basic(self, sample_session_data):
        """Test basic JSON export"""
        exporter = JSONExporter()
        content = exporter.export(sample_session_data)

        # Should be valid JSON
        data = json.loads(content)

        assert 'export_info' in data
        assert 'session' in data
        assert data['session']['id'] == 'test-session-123'

    def test_export_pretty_print(self, sample_session_data):
        """Test pretty printed JSON"""
        exporter = JSONExporter({'pretty_print': True, 'indent': 4})
        content = exporter.export(sample_session_data)

        # Should have indentation
        assert '\n    ' in content

    def test_export_compact(self, sample_session_data):
        """Test compact JSON"""
        exporter = JSONExporter({'pretty_print': False})
        content = exporter.export(sample_session_data)

        # Should be single line (no unnecessary newlines)
        lines = content.strip().split('\n')
        assert len(lines) == 1

    def test_export_includes_summary(self, sample_session_data):
        """Test that export includes summary"""
        exporter = JSONExporter()
        content = exporter.export(sample_session_data)
        data = json.loads(content)

        assert 'summary' in data['session']
        assert data['session']['summary']['total_turns'] == 2

    def test_export_includes_turns(self, sample_session_data):
        """Test that export includes turns"""
        exporter = JSONExporter()
        content = exporter.export(sample_session_data)
        data = json.loads(content)

        assert len(data['session']['turns']) == 2
        assert data['session']['turns'][0]['query'] == 'Apa sanksi pelanggaran UU Ketenagakerjaan?'

    def test_file_extension(self):
        """Test file extension"""
        exporter = JSONExporter()
        assert exporter.get_file_extension() == '.json'

    def test_export_and_save(self, sample_session_data):
        """Test export and save convenience method"""
        exporter = JSONExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_and_save(sample_session_data, 'test', tmpdir)

            assert path.exists()
            assert path.suffix == '.json'

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert data['session']['id'] == 'test-session-123'


class TestHTMLExporter:
    """Test HTML export"""

    def test_export_basic(self, sample_session_data):
        """Test basic HTML export"""
        exporter = HTMLExporter()
        content = exporter.export(sample_session_data)

        assert '<!DOCTYPE html>' in content
        assert '<html' in content
        assert '</html>' in content
        assert 'Konsultasi Hukum Indonesia' in content

    def test_export_includes_css(self, sample_session_data):
        """Test that export includes CSS styles"""
        exporter = HTMLExporter()
        content = exporter.export(sample_session_data)

        assert '<style>' in content
        assert 'font-family' in content

    def test_export_includes_turns(self, sample_session_data):
        """Test that export includes turns"""
        exporter = HTMLExporter()
        content = exporter.export(sample_session_data)

        assert 'Pertanyaan 1' in content
        assert 'Jawaban 1' in content
        assert 'class="turn"' in content

    def test_export_includes_summary(self, sample_session_data):
        """Test that export includes summary"""
        exporter = HTMLExporter({'include_metadata': True})
        content = exporter.export(sample_session_data)

        assert 'Ringkasan Sesi' in content
        assert 'class="summary"' in content

    def test_export_includes_citations_table(self, sample_session_data):
        """Test that export includes citations table"""
        exporter = HTMLExporter({'include_sources': True})
        content = exporter.export(sample_session_data)

        assert 'Daftar Referensi' in content
        assert '<table>' in content
        assert 'Ketenagakerjaan' in content

    def test_export_responsive(self, sample_session_data):
        """Test that export includes responsive meta tag"""
        exporter = HTMLExporter()
        content = exporter.export(sample_session_data)

        assert 'viewport' in content
        assert 'width=device-width' in content

    def test_file_extension(self):
        """Test file extension"""
        exporter = HTMLExporter()
        assert exporter.get_file_extension() == '.html'

    def test_save_to_file(self, sample_session_data):
        """Test saving HTML to file"""
        exporter = HTMLExporter()
        content = exporter.export(sample_session_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.save_to_file(content, 'test.html', tmpdir)

            assert path.exists()
            assert path.suffix == '.html'


class TestExporterConfiguration:
    """Test exporter configuration options"""

    def test_exclude_timing(self, sample_session_data):
        """Test excluding timing information"""
        config = {'include_timing': False}

        # Test with Markdown
        md_exporter = MarkdownExporter(config)
        md_content = md_exporter.export(sample_session_data)

        # Should not have timing details
        assert 'Waktu Total' not in md_content or 'include_timing' not in md_content

    def test_exclude_sources(self, sample_session_data):
        """Test excluding sources"""
        config = {'include_sources': False}

        md_exporter = MarkdownExporter(config)
        md_content = md_exporter.export(sample_session_data)

        assert 'Daftar Referensi' not in md_content


class TestEmptySession:
    """Test export of empty sessions"""

    @pytest.fixture
    def empty_session_data(self):
        return {
            'id': 'empty-session',
            'created_at': '2024-01-15T10:00:00',
            'updated_at': '2024-01-15T10:00:00',
            'turns': [],
            'metadata': {
                'total_queries': 0,
                'total_tokens': 0,
                'total_time': 0,
                'regulations_cited': []
            }
        }

    def test_markdown_empty(self, empty_session_data):
        """Test markdown export of empty session"""
        exporter = MarkdownExporter()
        content = exporter.export(empty_session_data)

        assert '# Konsultasi Hukum Indonesia' in content
        assert 'empty-session' in content

    def test_json_empty(self, empty_session_data):
        """Test JSON export of empty session"""
        exporter = JSONExporter()
        content = exporter.export(empty_session_data)

        data = json.loads(content)
        assert len(data['session']['turns']) == 0

    def test_html_empty(self, empty_session_data):
        """Test HTML export of empty session"""
        exporter = HTMLExporter()
        content = exporter.export(empty_session_data)

        assert '<!DOCTYPE html>' in content


class TestAutoFilename:
    """Test automatic filename generation"""

    def test_auto_filename_md(self, sample_session_data):
        """Test auto filename for Markdown"""
        exporter = MarkdownExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_and_save(sample_session_data, directory=tmpdir)

            assert path.name.startswith('conversation_')
            assert path.suffix == '.md'

    def test_auto_filename_json(self, sample_session_data):
        """Test auto filename for JSON"""
        exporter = JSONExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_and_save(sample_session_data, directory=tmpdir)

            assert path.name.startswith('conversation_')
            assert path.suffix == '.json'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
