"""
Unit tests for Memory File Operations

Tests all functionality of the MemoryFileOperations class including
YAML frontmatter parsing, file I/O, validation, and error handling.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, mock_open
import yaml

from .file_operations import (
    MemoryFileOperations,
    read_memory_file,
    write_memory_file,
    update_memory_importance
)


class TestMemoryFileOperations:
    """Test suite for MemoryFileOperations class."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test_memory.md"
        
        # Sample frontmatter and content
        self.sample_frontmatter = {
            'created': '2024-01-01T12:00:00',
            'last_updated': '2024-01-01T12:00:00',
            'file_type': 'core_memory',
            'importance_score': 8,
            'category': 'personal'
        }
        
        self.sample_content = "# Test Memory\n\nThis is test content for memory operations."
        
        # Create sample file
        self._create_sample_file()
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_sample_file(self):
        """Create a sample memory file for testing."""
        yaml_content = yaml.dump(self.sample_frontmatter, default_flow_style=False, sort_keys=False)
        full_content = f"---\n{yaml_content}---\n\n{self.sample_content}"
        
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
    
    def test_parse_memory_file_success(self):
        """Test successful parsing of a memory file."""
        frontmatter, content = MemoryFileOperations.parse_memory_file(self.test_file)
        
        assert frontmatter['created'] == '2024-01-01T12:00:00'
        assert frontmatter['importance_score'] == 8
        assert frontmatter['category'] == 'personal'
        assert content == self.sample_content
    
    def test_parse_memory_file_no_frontmatter(self):
        """Test parsing a file without frontmatter."""
        # Create file without frontmatter
        no_frontmatter_file = self.test_dir / "no_frontmatter.md"
        with open(no_frontmatter_file, 'w', encoding='utf-8') as f:
            f.write("Just content without frontmatter")
        
        frontmatter, content = MemoryFileOperations.parse_memory_file(no_frontmatter_file)
        
        assert 'created' in frontmatter
        assert 'last_updated' in frontmatter
        assert frontmatter['importance_score'] == 5
        assert frontmatter['category'] == 'general'
        assert content == "Just content without frontmatter"
    
    def test_parse_memory_file_not_found(self):
        """Test parsing a non-existent file."""
        non_existent = self.test_dir / "does_not_exist.md"
        
        with pytest.raises(FileNotFoundError):
            MemoryFileOperations.parse_memory_file(non_existent)
    
    def test_parse_memory_file_invalid_yaml(self):
        """Test parsing a file with invalid YAML frontmatter."""
        invalid_yaml_file = self.test_dir / "invalid_yaml.md"
        content = "---\n[invalid: yaml: content\n---\n\nContent"
        
        with open(invalid_yaml_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        with pytest.raises(ValueError, match="Invalid YAML frontmatter"):
            MemoryFileOperations.parse_memory_file(invalid_yaml_file)
    
    def test_write_memory_file_success(self):
        """Test successful writing of a memory file."""
        new_file = self.test_dir / "new_memory.md"
        test_frontmatter = {
            'created': '2024-01-02T10:00:00',
            'importance_score': 6,
            'category': 'test'
        }
        test_content = "New test content"
        
        result = MemoryFileOperations.write_memory_file(new_file, test_frontmatter, test_content)
        
        assert result is True
        assert new_file.exists()
        
        # Verify content
        frontmatter, content = MemoryFileOperations.parse_memory_file(new_file)
        assert frontmatter['created'] == '2024-01-02T10:00:00'
        assert frontmatter['importance_score'] == 6
        assert 'last_updated' in frontmatter  # Should be auto-added
        assert content == test_content
    
    def test_write_memory_file_with_backup(self):
        """Test writing a file with backup creation."""
        original_content = "Original content"
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        new_frontmatter = {'importance_score': 9}
        new_content = "Updated content"
        
        result = MemoryFileOperations.write_memory_file(
            self.test_file, new_frontmatter, new_content, backup=True
        )
        
        assert result is True
        
        # Check backup was created
        backup_files = list(self.test_dir.glob("*.bak"))
        assert len(backup_files) > 0
    
    def test_update_frontmatter_success(self):
        """Test successful frontmatter update."""
        updates = {
            'importance_score': 9,
            'category': 'updated',
            'new_field': 'new_value'
        }
        
        result = MemoryFileOperations.update_frontmatter(self.test_file, updates)
        
        assert result is True
        
        # Verify updates
        frontmatter, content = MemoryFileOperations.parse_memory_file(self.test_file)
        assert frontmatter['importance_score'] == 9
        assert frontmatter['category'] == 'updated'
        assert frontmatter['new_field'] == 'new_value'
        assert content == self.sample_content  # Content unchanged
    
    def test_append_to_memory_file_success(self):
        """Test successful content appending."""
        new_content = "This is appended content"
        section_header = "New Section"
        
        result = MemoryFileOperations.append_to_memory_file(
            self.test_file, new_content, section_header, importance_score=9
        )
        
        assert result is True
        
        # Verify content
        frontmatter, content = MemoryFileOperations.parse_memory_file(self.test_file)
        assert frontmatter['importance_score'] == 9
        assert "## New Section" in content
        assert "This is appended content" in content
        assert self.sample_content in content  # Original content preserved
    
    def test_get_file_metadata_success(self):
        """Test successful metadata retrieval."""
        metadata = MemoryFileOperations.get_file_metadata(self.test_file)
        
        assert metadata['exists'] is True
        assert metadata['path'] == str(self.test_file)
        assert metadata['size_bytes'] > 0
        assert metadata['importance_score'] == 8
        assert metadata['category'] == 'personal'
        assert metadata['word_count'] > 0
        assert metadata['line_count'] > 0
        assert metadata['char_count'] > 0
        assert 'frontmatter' in metadata
    
    def test_get_file_metadata_not_found(self):
        """Test metadata for non-existent file."""
        non_existent = self.test_dir / "does_not_exist.md"
        metadata = MemoryFileOperations.get_file_metadata(non_existent)
        
        assert metadata['exists'] is False
    
    def test_search_file_content_success(self):
        """Test successful content search."""
        matches = MemoryFileOperations.search_file_content(self.test_file, "Test Memory")
        
        assert len(matches) == 1
        assert matches[0]['line_number'] == 1
        assert "Test Memory" in matches[0]['line_content']
        assert matches[0]['importance_score'] == 8
    
    def test_search_file_content_case_insensitive(self):
        """Test case-insensitive content search."""
        matches = MemoryFileOperations.search_file_content(self.test_file, "test memory", case_sensitive=False)
        
        assert len(matches) == 1
        assert "Test Memory" in matches[0]['line_content']
    
    def test_search_file_content_no_matches(self):
        """Test content search with no matches."""
        matches = MemoryFileOperations.search_file_content(self.test_file, "nonexistent text")
        
        assert len(matches) == 0
    
    def test_validate_memory_file_valid(self):
        """Test validation of a valid memory file."""
        validation = MemoryFileOperations.validate_memory_file(self.test_file)
        
        assert validation['valid'] is True
        assert len(validation['errors']) == 0
        assert validation['file_path'] == str(self.test_file)
    
    def test_validate_memory_file_missing_fields(self):
        """Test validation with missing required fields."""
        # Create file with incomplete frontmatter
        incomplete_frontmatter = {'created': '2024-01-01T12:00:00'}
        yaml_content = yaml.dump(incomplete_frontmatter, default_flow_style=False)
        full_content = f"---\n{yaml_content}---\n\nContent"
        
        incomplete_file = self.test_dir / "incomplete.md"
        with open(incomplete_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        validation = MemoryFileOperations.validate_memory_file(incomplete_file)
        
        assert validation['valid'] is False
        assert any("Missing required frontmatter field" in error for error in validation['errors'])
    
    def test_validate_memory_file_invalid_importance_score(self):
        """Test validation with invalid importance score."""
        invalid_frontmatter = {
            'created': '2024-01-01T12:00:00',
            'last_updated': '2024-01-01T12:00:00',
            'importance_score': 15  # Invalid - should be 1-10
        }
        yaml_content = yaml.dump(invalid_frontmatter, default_flow_style=False)
        full_content = f"---\n{yaml_content}---\n\nContent"
        
        invalid_file = self.test_dir / "invalid_score.md"
        with open(invalid_file, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        validation = MemoryFileOperations.validate_memory_file(invalid_file)
        
        assert validation['valid'] is False
        assert any("Importance score must be a number between 1 and 10" in error for error in validation['errors'])
    
    def test_create_memory_entry(self):
        """Test creating a new memory entry."""
        content = "Test memory entry content"
        importance_score = 7
        category = "test_category"
        additional_metadata = {'custom_field': 'custom_value'}
        
        frontmatter, parsed_content = MemoryFileOperations.create_memory_entry(
            content, importance_score, category, additional_metadata=additional_metadata
        )
        
        assert frontmatter['importance_score'] == 7
        assert frontmatter['category'] == 'test_category'
        assert frontmatter['custom_field'] == 'custom_value'
        assert 'created' in frontmatter
        assert 'last_updated' in frontmatter
        assert parsed_content == content
    
    def test_list_memory_files(self):
        """Test listing memory files in a directory."""
        # Create additional test files
        file1 = self.test_dir / "file1.md"
        file2 = self.test_dir / "file2.md"
        file3 = self.test_dir / "file3.txt"  # Different extension
        
        for file_path in [file1, file2, file3]:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Test content")
        
        # List markdown files only
        files = MemoryFileOperations.list_memory_files(self.test_dir, "*.md")
        
        assert len(files) == 3  # Original test file + 2 new files
        file_names = [f['name'] for f in files]
        assert 'test_memory.md' in file_names
        assert 'file1.md' in file_names
        assert 'file2.md' in file_names
        assert 'file3.txt' not in file_names
    
    def test_list_memory_files_with_metadata(self):
        """Test listing files with metadata included."""
        files = MemoryFileOperations.list_memory_files(self.test_dir, include_metadata=True)
        
        assert len(files) == 1
        file_info = files[0]
        assert 'importance_score' in file_info
        assert 'category' in file_info
        assert file_info['exists'] is True
    
    def test_clean_old_backups(self):
        """Test cleaning up old backup files."""
        # Create some backup files with different timestamps
        base_time = datetime.now()
        
        backup_files = []
        for i in range(15):  # Create more than max_backups (10)
            timestamp = (base_time - timedelta(days=i)).strftime('%Y%m%d_%H%M%S')
            backup_file = self.test_dir / f"test_memory.{timestamp}.bak"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(f"Backup content {i}")
            backup_files.append(backup_file)
        
        # Clean up with max_backups=5
        removed_count = MemoryFileOperations.clean_old_backups(
            self.test_dir, max_backups=5, max_age_days=7
        )
        
        assert removed_count > 0
        
        # Check remaining backups
        remaining_backups = list(self.test_dir.glob("*.bak"))
        assert len(remaining_backups) <= 5


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "convenience_test.md"
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_read_memory_file_convenience(self):
        """Test read_memory_file convenience function."""
        # Create test file
        frontmatter = {'importance_score': 5, 'created': datetime.now().isoformat()}
        content = "Test content"
        MemoryFileOperations.write_memory_file(self.test_file, frontmatter, content)
        
        # Test convenience function
        parsed_frontmatter, parsed_content = read_memory_file(self.test_file)
        
        assert parsed_frontmatter['importance_score'] == 5
        assert parsed_content == content
    
    def test_write_memory_file_convenience(self):
        """Test write_memory_file convenience function."""
        frontmatter = {'importance_score': 7, 'category': 'test'}
        content = "Convenience test content"
        
        result = write_memory_file(self.test_file, frontmatter, content)
        
        assert result is True
        assert self.test_file.exists()
    
    def test_update_memory_importance_convenience(self):
        """Test update_memory_importance convenience function."""
        # Create initial file
        frontmatter = {'importance_score': 5, 'created': datetime.now().isoformat()}
        content = "Test content"
        MemoryFileOperations.write_memory_file(self.test_file, frontmatter, content)
        
        # Update importance
        result = update_memory_importance(self.test_file, 8)
        
        assert result is True
        
        # Verify update
        updated_frontmatter, _ = read_memory_file(self.test_file)
        assert updated_frontmatter['importance_score'] == 8


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_write_memory_file_permission_error(self):
        """Test handling of permission errors during write."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(IOError, match="Error writing memory file"):
                MemoryFileOperations.write_memory_file(
                    self.test_dir / "test.md", 
                    {'importance_score': 5}, 
                    "content"
                )
    
    def test_parse_memory_file_io_error(self):
        """Test handling of I/O errors during parsing."""
        test_file = self.test_dir / "test.md"
        test_file.touch()  # Create empty file
        
        with patch('builtins.open', side_effect=IOError("I/O error")):
            with pytest.raises(ValueError, match="Error parsing memory file"):
                MemoryFileOperations.parse_memory_file(test_file)
    
    def test_search_file_content_with_error(self):
        """Test search handling when file has errors."""
        test_file = self.test_dir / "test.md"
        
        # File doesn't exist
        matches = MemoryFileOperations.search_file_content(test_file, "search term")
        
        assert len(matches) == 1
        assert 'error' in matches[0]
        assert matches[0]['file_path'] == str(test_file)


if __name__ == "__main__":
    pytest.main([__file__]) 