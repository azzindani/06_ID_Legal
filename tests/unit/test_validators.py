"""
Unit Tests for API Validators

Tests the shared validation logic in api/validators.py
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.validators import validate_session_id, validate_query


class TestSessionIDValidator:
    """Test validate_session_id() function"""

    def test_valid_session_id_alphanumeric(self):
        """Valid alphanumeric session ID"""
        result = validate_session_id("session123")
        assert result == "session123"

    def test_valid_session_id_with_hyphens(self):
        """Valid session ID with hyphens"""
        result = validate_session_id("session-123-456")
        assert result == "session-123-456"

    def test_valid_session_id_with_underscores(self):
        """Valid session ID with underscores"""
        result = validate_session_id("session_123_456")
        assert result == "session_123_456"

    def test_valid_session_id_mixed(self):
        """Valid session ID with alphanumeric, hyphens, and underscores"""
        result = validate_session_id("user_session-2024-01")
        assert result == "user_session-2024-01"

    def test_none_session_id(self):
        """None session ID should return None"""
        result = validate_session_id(None)
        assert result is None

    def test_invalid_session_id_with_spaces(self):
        """Session ID with spaces should raise ValueError"""
        with pytest.raises(ValueError, match="must contain only alphanumeric"):
            validate_session_id("session 123")

    def test_invalid_session_id_with_special_chars(self):
        """Session ID with special characters should raise ValueError"""
        with pytest.raises(ValueError, match="must contain only alphanumeric"):
            validate_session_id("session@123")

    def test_invalid_session_id_with_dots(self):
        """Session ID with dots should raise ValueError"""
        with pytest.raises(ValueError, match="must contain only alphanumeric"):
            validate_session_id("session.123")

    def test_invalid_session_id_with_slash(self):
        """Session ID with slashes should raise ValueError"""
        with pytest.raises(ValueError, match="must contain only alphanumeric"):
            validate_session_id("session/123")


class TestQueryValidator:
    """Test validate_query() function"""

    def test_valid_query_simple(self):
        """Valid simple query"""
        result = validate_query("Apa itu UU Ketenagakerjaan?")
        assert result == "Apa itu UU Ketenagakerjaan?"

    def test_valid_query_complex(self):
        """Valid complex legal query"""
        query = "Bagaimana ketentuan dalam UU No. 13 Tahun 2003 Pasal 1 ayat (1)?"
        result = validate_query(query)
        assert result == query

    def test_query_strips_whitespace(self):
        """Query should be stripped of leading/trailing whitespace"""
        result = validate_query("  test query  ")
        assert result == "test query"

    def test_query_strips_newlines(self):
        """Query should be stripped of newlines"""
        result = validate_query("\ntest query\n")
        assert result == "test query"

    def test_empty_query_raises_error(self):
        """Empty query should raise ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_query("")

    def test_whitespace_only_query_raises_error(self):
        """Whitespace-only query should raise ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_query("   ")

    def test_newline_only_query_raises_error(self):
        """Newline-only query should raise ValueError"""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_query("\n\n")

    def test_query_with_script_tag_raises_error(self):
        """Query with <script> tag should raise ValueError (XSS prevention)"""
        with pytest.raises(ValueError, match="dangerous content"):
            validate_query("test <script>alert('xss')</script>")

    def test_query_with_script_uppercase_raises_error(self):
        """Query with <SCRIPT> tag (uppercase) should raise ValueError"""
        with pytest.raises(ValueError, match="dangerous content"):
            validate_query("test <SCRIPT>alert('xss')</SCRIPT>")

    def test_query_with_javascript_protocol_raises_error(self):
        """Query with javascript: protocol should raise ValueError"""
        with pytest.raises(ValueError, match="dangerous content"):
            validate_query("test javascript:alert('xss')")

    def test_query_with_onerror_raises_error(self):
        """Query with onerror= attribute should raise ValueError"""
        with pytest.raises(ValueError, match="dangerous content"):
            validate_query("test onerror=alert('xss')")

    def test_query_with_onclick_raises_error(self):
        """Query with onclick= attribute should raise ValueError"""
        with pytest.raises(ValueError, match="dangerous content"):
            validate_query("test onclick=alert('xss')")

    def test_valid_query_with_safe_html_like_text(self):
        """Query with safe HTML-like text (not actual tags) should pass"""
        # This tests that we're case-insensitive but only blocking actual dangerous patterns
        result = validate_query("Apa itu scriptum dalam hukum romawi?")
        assert result == "Apa itu scriptum dalam hukum romawi?"

    def test_valid_query_with_indonesian_chars(self):
        """Query with Indonesian characters should pass"""
        query = "Apa sanksi pelanggaran peraturan pemerintah?"
        result = validate_query(query)
        assert result == query


class TestValidatorIntegration:
    """Integration tests for validators"""

    def test_session_id_validator_preserves_valid_input(self):
        """Session ID validator should not modify valid input"""
        valid_ids = [
            "abc123",
            "user-session-2024",
            "test_session_123",
            "a1b2c3",
            "SESSION_2024-01-15"
        ]
        for session_id in valid_ids:
            assert validate_session_id(session_id) == session_id

    def test_query_validator_preserves_legal_queries(self):
        """Query validator should not modify valid legal queries"""
        legal_queries = [
            "UU No. 13 Tahun 2003",
            "Pasal 1 ayat (1)",
            "Peraturan Pemerintah tentang Pengupahan",
            "Apa hak dan kewajiban pekerja?",
            "Bagaimana prosedur PHK?"
        ]
        for query in legal_queries:
            assert validate_query(query) == query

    def test_validators_reject_malicious_input(self):
        """Validators should reject common attack patterns"""
        malicious_queries = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "test onerror=alert(1)",
            "test onclick=malicious()"
        ]
        for query in malicious_queries:
            with pytest.raises(ValueError):
                validate_query(query)

        malicious_session_ids = [
            "session@123",
            "session 123",
            "session/123",
            "session.123",
            "../../../etc/passwd"
        ]
        for session_id in malicious_session_ids:
            with pytest.raises(ValueError):
                validate_session_id(session_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
