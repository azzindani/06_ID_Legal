"""
Unit tests for exporters

Run with: pytest tests/unit/test_exporters.py -v
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conversation import MarkdownExporter, JSONExporter, HTMLExporter


@pytest.mark.unit
class TestMarkdownExporter:
    """Test Markdown exporter"""

    @pytest.fixture
    def exporter(self):
        return MarkdownExporter()

    @pytest.fixture
    def session_data(self):
        return {
            'id': 'test-123',
            'created_at': '2024-01-15T10:00:00',
            'updated_at': '2024-01-15T10:30:00',
            'turns': [{
                'turn_number': 1,
                'timestamp': '2024-01-15T10:00:00',
                'query': 'Test question?',
                'answer': 'Test answer.',
                'metadata': {}
            }],
            'metadata': {
                'total_queries': 1,
                'total_tokens': 100,
                'total_time': 5.0,
                'regulations_cited': []
            }
        }

    def test_export_returns_string(self, exporter, session_data):
        """Test that export returns string"""
        result = exporter.export(session_data)
        assert isinstance(result, str)

    def test_export_contains_header(self, exporter, session_data):
        """Test that export contains header"""
        result = exporter.export(session_data)
        assert '# Konsultasi Hukum Indonesia' in result

    def test_file_extension(self, exporter):
        """Test file extension"""
        assert exporter.get_file_extension() == '.md'


@pytest.mark.unit
class TestJSONExporter:
    """Test JSON exporter"""

    @pytest.fixture
    def exporter(self):
        return JSONExporter()

    @pytest.fixture
    def session_data(self):
        return {
            'id': 'test-123',
            'created_at': '2024-01-15T10:00:00',
            'updated_at': '2024-01-15T10:30:00',
            'turns': [],
            'metadata': {}
        }

    def test_export_valid_json(self, exporter, session_data):
        """Test that export produces valid JSON"""
        result = exporter.export(session_data)
        data = json.loads(result)
        assert 'session' in data

    def test_file_extension(self, exporter):
        """Test file extension"""
        assert exporter.get_file_extension() == '.json'


@pytest.mark.unit
class TestHTMLExporter:
    """Test HTML exporter"""

    @pytest.fixture
    def exporter(self):
        return HTMLExporter()

    @pytest.fixture
    def session_data(self):
        return {
            'id': 'test-123',
            'created_at': '2024-01-15T10:00:00',
            'updated_at': '2024-01-15T10:30:00',
            'turns': [],
            'metadata': {}
        }

    def test_export_valid_html(self, exporter, session_data):
        """Test that export produces valid HTML"""
        result = exporter.export(session_data)
        assert '<!DOCTYPE html>' in result
        assert '</html>' in result

    def test_file_extension(self, exporter):
        """Test file extension"""
        assert exporter.get_file_extension() == '.html'
