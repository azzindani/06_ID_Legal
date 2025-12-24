"""
Unit Tests for Path Setup Utility

Tests the centralized path setup module.

File: tests/unit/test_path_setup.py
"""

import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestPathSetup:
    """Tests for utils/path_setup.py"""
    
    def test_project_root_exists(self):
        """Test that PROJECT_ROOT points to an existing directory"""
        from utils.path_setup import PROJECT_ROOT
        
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()
    
    def test_project_root_is_correct(self):
        """Test that PROJECT_ROOT points to the actual project root"""
        from utils.path_setup import PROJECT_ROOT
        
        # The project root should contain these key files/directories
        expected_markers = ['config.py', 'main.py', 'pipeline', 'core', 'api']
        
        for marker in expected_markers:
            assert (PROJECT_ROOT / marker).exists(), f"Expected {marker} in project root"
    
    def test_ensure_project_path(self):
        """Test ensure_project_path function"""
        from utils.path_setup import ensure_project_path, PROJECT_ROOT
        
        result = ensure_project_path()
        
        assert result == PROJECT_ROOT
        assert str(PROJECT_ROOT) in sys.path
    
    def test_get_project_root(self):
        """Test get_project_root function"""
        from utils.path_setup import get_project_root, PROJECT_ROOT
        
        result = get_project_root()
        
        assert result == PROJECT_ROOT
        assert isinstance(result, Path)
    
    def test_path_is_absolute(self):
        """Test that PROJECT_ROOT is an absolute path"""
        from utils.path_setup import PROJECT_ROOT
        
        assert PROJECT_ROOT.is_absolute()
    
    def test_can_import_project_modules(self):
        """Test that path setup enables project imports"""
        from utils.path_setup import ensure_project_path
        ensure_project_path()
        
        # These imports should work after path setup
        try:
            from config import DEFAULT_CONFIG
            from utils.logger_utils import get_logger
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed after path setup: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
