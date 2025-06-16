"""
Unit tests for Memory Validation and Error Handling

Tests all functionality of the validation and error handling system including
validation rules, error recovery mechanisms, backup operations, and data integrity checks.
"""

import pytest
import tempfile
import shutil
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Use absolute imports to avoid import issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

try:
    from src.memory.validation import (
        ValidationSeverity,
        ErrorType,
        ValidationIssue,
        ValidationResult,
        MemoryValidator,
        MemoryErrorHandler,
        validate_memory_system
    )
    from src.memory.memory_initializer import MemoryInitializer
    from src.memory.file_operations import write_memory_file
    from src.memory.importance_scoring import TimestampManager
except ImportError:
    # Fallback for direct execution
    from validation import (
        ValidationSeverity,
        ErrorType,
        ValidationIssue,
        ValidationResult,
        MemoryValidator,
        MemoryErrorHandler,
        validate_memory_system
    )
    import sys
    sys.path.append(os.path.dirname(__file__))
    from memory_initializer import MemoryInitializer
    from file_operations import write_memory_file
    from importance_scoring import TimestampManager


class TestValidationIssue:
    """Test ValidationIssue dataclass."""
    
    def test_validation_issue_creation(self):
        """Test creating a ValidationIssue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            error_type=ErrorType.MISSING_DATA,
            message="Test error message",
            file_path=Path("test.md"),
            field="test_field",
            value="test_value",
            suggestion="Test suggestion"
        )
        
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.error_type == ErrorType.MISSING_DATA
        assert issue.message == "Test error message"
        assert issue.file_path == Path("test.md")
        assert issue.field == "test_field"
        assert issue.value == "test_value"
        assert issue.suggestion == "Test suggestion"
    
    def test_validation_issue_str(self):
        """Test string representation of ValidationIssue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            error_type=ErrorType.VALIDATION_ERROR,
            message="Test warning",
            file_path=Path("memory.md"),
            field="importance_score"
        )
        
        str_repr = str(issue)
        assert "[WARNING]" in str_repr
        assert "Test warning" in str_repr
        assert "memory.md" in str_repr
        assert "importance_score" in str_repr


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        issues = [
            ValidationIssue(ValidationSeverity.WARNING, ErrorType.VALIDATION_ERROR, "Warning"),
            ValidationIssue(ValidationSeverity.ERROR, ErrorType.MISSING_DATA, "Error"),
        ]
        
        result = ValidationResult(is_valid=True, issues=issues)
        
        assert result.warnings_count == 1
        assert result.errors_count == 1
        assert result.is_valid is False  # Should be False due to error
    
    def test_validation_result_add_issue(self):
        """Test adding issues to ValidationResult."""
        result = ValidationResult(is_valid=True, issues=[])
        
        # Add warning
        warning = ValidationIssue(ValidationSeverity.WARNING, ErrorType.VALIDATION_ERROR, "Warning")
        result.add_issue(warning)
        assert result.warnings_count == 1
        assert result.errors_count == 0
        assert result.is_valid is True
        
        # Add error
        error = ValidationIssue(ValidationSeverity.ERROR, ErrorType.MISSING_DATA, "Error")
        result.add_issue(error)
        assert result.warnings_count == 1
        assert result.errors_count == 1
        assert result.is_valid is False
    
    def test_validation_result_summary(self):
        """Test validation result summary."""
        # All passed
        result = ValidationResult(is_valid=True, issues=[])
        assert "✅ All validations passed" in result.get_summary()
        
        # Valid with warnings
        result = ValidationResult(is_valid=True, issues=[
            ValidationIssue(ValidationSeverity.WARNING, ErrorType.VALIDATION_ERROR, "Warning")
        ])
        summary = result.get_summary()
        assert "⚠️" in summary and "1 warning(s)" in summary
        
        # Invalid with errors
        result = ValidationResult(is_valid=False, issues=[
            ValidationIssue(ValidationSeverity.ERROR, ErrorType.MISSING_DATA, "Error")
        ])
        summary = result.get_summary()
        assert "❌" in summary and "1 error(s)" in summary


class TestMemoryValidator:
    """Test suite for MemoryValidator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
        
        self.validator = MemoryValidator(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert self.validator.memory_base_path == self.test_dir
        assert hasattr(self.validator, 'logger')
    
    def test_validate_valid_memory_file(self):
        """Test validating a valid memory file."""
        # Create valid memory file
        file_path = self.test_dir / 'interactions' / 'test.md'
        frontmatter = {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'interaction',
            'importance_score': 5
        }
        content = "This is a valid memory file."
        
        write_memory_file(file_path, frontmatter, content)
        
        result = self.validator.validate_memory_file(file_path)
        assert result.is_valid is True
        assert result.errors_count == 0
    
    def test_validate_missing_file(self):
        """Test validating a non-existent file."""
        file_path = self.test_dir / 'nonexistent.md'
        
        result = self.validator.validate_memory_file(file_path)
        assert result.is_valid is False
        assert result.errors_count == 1
        assert any(issue.error_type == ErrorType.MISSING_DATA for issue in result.issues)
    
    def test_validate_file_missing_required_fields(self):
        """Test validating file with missing required fields."""
        file_path = self.test_dir / 'interactions' / 'incomplete.md'
        frontmatter = {
            'created': TimestampManager.create_timestamp()
            # Missing memory_type and importance_score
        }
        content = "Incomplete memory file."
        
        write_memory_file(file_path, frontmatter, content)
        
        result = self.validator.validate_memory_file(file_path)
        assert result.is_valid is False
        assert result.errors_count >= 2  # Missing memory_type and importance_score
        
        missing_fields = [issue.field for issue in result.issues 
                         if issue.error_type == ErrorType.MISSING_DATA]
        assert 'memory_type' in missing_fields
        assert 'importance_score' in missing_fields
    
    def test_validate_invalid_field_types(self):
        """Test validating file with invalid field types."""
        file_path = self.test_dir / 'interactions' / 'invalid_types.md'
        frontmatter = {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 123,  # Should be string
            'importance_score': "high",  # Should be int
            'access_count': -5  # Should be >= 0
        }
        content = "Memory with invalid field types."
        
        write_memory_file(file_path, frontmatter, content)
        
        result = self.validator.validate_memory_file(file_path)
        assert result.is_valid is False
        assert result.errors_count >= 3
        
        # Check specific validation errors
        type_errors = [issue for issue in result.issues 
                      if issue.error_type == ErrorType.VALIDATION_ERROR and "wrong type" in issue.message]
        assert len(type_errors) >= 2
    
    def test_validate_invalid_field_values(self):
        """Test validating file with invalid field values."""
        file_path = self.test_dir / 'interactions' / 'invalid_values.md'
        frontmatter = {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'invalid_type',  # Not in allowed values
            'importance_score': 15  # Exceeds maximum
        }
        content = "Memory with invalid field values."
        
        write_memory_file(file_path, frontmatter, content)
        
        result = self.validator.validate_memory_file(file_path)
        assert result.is_valid is False
        assert result.errors_count >= 2
        
        # Check for invalid value errors
        value_errors = [issue for issue in result.issues 
                       if "invalid value" in issue.message or "exceeds maximum" in issue.message]
        assert len(value_errors) >= 2
    
    def test_validate_invalid_timestamps(self):
        """Test validating file with invalid timestamps."""
        file_path = self.test_dir / 'interactions' / 'invalid_time.md'
        frontmatter = {
            'created': "not-a-timestamp",
            'memory_type': 'interaction',
            'importance_score': 5,
            'last_updated': "2024-13-45T25:70:99"  # Invalid datetime
        }
        content = "Memory with invalid timestamps."
        
        write_memory_file(file_path, frontmatter, content)
        
        result = self.validator.validate_memory_file(file_path)
        assert result.is_valid is False
        
        # Check for timestamp format errors
        format_errors = [issue for issue in result.issues 
                        if issue.error_type == ErrorType.FORMAT_ERROR]
        assert len(format_errors) >= 2
    
    def test_validate_content_issues(self):
        """Test validating file with content issues."""
        file_path = self.test_dir / 'interactions' / 'content_issues.md'
        frontmatter = {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'interaction',
            'importance_score': 5
        }
        
        # Very long content
        long_content = "x" * 200000  # Exceeds max length
        
        write_memory_file(file_path, frontmatter, long_content)
        
        result = self.validator.validate_memory_file(file_path)
        
        # Should have warning about content length
        length_warnings = [issue for issue in result.issues 
                          if "exceeds maximum length" in issue.message]
        assert len(length_warnings) >= 1
    
    def test_validate_empty_content(self):
        """Test validating file with empty content."""
        file_path = self.test_dir / 'interactions' / 'empty.md'
        frontmatter = {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'interaction',
            'importance_score': 5
        }
        content = ""  # Empty content
        
        write_memory_file(file_path, frontmatter, content)
        
        result = self.validator.validate_memory_file(file_path)
        
        # Should have warning about empty content
        empty_warnings = [issue for issue in result.issues 
                         if "empty content" in issue.message]
        assert len(empty_warnings) >= 1
    
    def test_validate_memory_type_specific_rules(self):
        """Test memory type-specific validation rules."""
        # Test condensed memory with low source_count
        condensed_path = self.test_dir / 'condensed' / 'test_condensed.md'
        condensed_frontmatter = {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'condensed',
            'importance_score': 6,
            'source_count': 1  # Should be >= 2 for condensed
        }
        
        write_memory_file(condensed_path, condensed_frontmatter, "Condensed memory")
        
        result = self.validator.validate_memory_file(condensed_path)
        
        # Should have warning about low source_count
        source_warnings = [issue for issue in result.issues 
                          if "low source_count" in issue.message]
        assert len(source_warnings) >= 1
        
        # Test core memory with low importance
        core_path = self.test_dir / 'core' / 'user_profile.md'
        core_frontmatter = {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'user_profile',
            'importance_score': 3  # Should be >= 5 for core memories
        }
        
        write_memory_file(core_path, core_frontmatter, "User profile")
        
        result = self.validator.validate_memory_file(core_path)
        
        # Should have warning about low importance
        importance_warnings = [issue for issue in result.issues 
                              if "low importance score" in issue.message]
        assert len(importance_warnings) >= 1
    
    def test_validate_directory(self):
        """Test validating a memory directory."""
        # Create multiple files with various issues
        interactions_dir = self.test_dir / 'interactions'
        
        # Valid file
        valid_file = interactions_dir / 'valid.md'
        write_memory_file(valid_file, {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'interaction',
            'importance_score': 5
        }, "Valid content")
        
        # Invalid file
        invalid_file = interactions_dir / 'invalid.md'
        write_memory_file(invalid_file, {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'invalid_type',  # Invalid
            'importance_score': 15  # Invalid
        }, "Invalid content")
        
        result = self.validator.validate_memory_directory(interactions_dir)
        
        # Should find issues from invalid file but not valid file
        assert result.errors_count >= 2
        assert len(result.issues) >= 2
    
    def test_validate_nonexistent_directory(self):
        """Test validating a non-existent directory."""
        nonexistent_dir = self.test_dir / 'nonexistent'
        
        result = self.validator.validate_memory_directory(nonexistent_dir)
        assert result.is_valid is False
        assert any(issue.error_type == ErrorType.MISSING_DATA for issue in result.issues)
    
    def test_validate_empty_directory(self):
        """Test validating an empty directory."""
        empty_dir = self.test_dir / 'empty_test'
        empty_dir.mkdir()
        
        result = self.validator.validate_memory_directory(empty_dir)
        
        # Should have warning about no files
        no_files_warnings = [issue for issue in result.issues 
                            if "No memory files found" in issue.message]
        assert len(no_files_warnings) >= 1
    
    def test_validate_memory_system(self):
        """Test validating the entire memory system."""
        # Create some test files
        interactions_dir = self.test_dir / 'interactions'
        core_dir = self.test_dir / 'core'
        
        # Valid interaction
        write_memory_file(interactions_dir / 'interaction1.md', {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'interaction',
            'importance_score': 5
        }, "Interaction content")
        
        # Valid core memory
        write_memory_file(core_dir / 'user_profile.md', {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'user_profile',
            'importance_score': 8
        }, "User profile content")
        
        result = self.validator.validate_memory_system()
        
        # Should validate directory structure and all files
        assert isinstance(result, ValidationResult)
        # Might have some warnings but should be generally valid
    
    def test_validate_cross_file_consistency(self):
        """Test cross-file consistency validation."""
        interactions_dir = self.test_dir / 'interactions'
        
        # Create files with duplicate memory IDs
        write_memory_file(interactions_dir / 'dup1.md', {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'interaction',
            'importance_score': 5,
            'memory_id': 'duplicate_id'
        }, "First file")
        
        write_memory_file(interactions_dir / 'dup2.md', {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'interaction',
            'importance_score': 5,
            'memory_id': 'duplicate_id'  # Duplicate!
        }, "Second file")
        
        result = self.validator.validate_memory_directory(interactions_dir)
        
        # Should detect duplicate ID
        duplicate_errors = [issue for issue in result.issues 
                           if "Duplicate memory ID" in issue.message]
        assert len(duplicate_errors) >= 1


class TestMemoryErrorHandler:
    """Test suite for MemoryErrorHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
        
        self.error_handler = MemoryErrorHandler(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        assert self.error_handler.memory_base_path == self.test_dir
        assert self.error_handler.backup_dir.exists()
        assert self.error_handler.backup_dir == self.test_dir / '.backups'
    
    def test_create_backup(self):
        """Test creating a backup file."""
        # Create test file
        test_file = self.test_dir / 'test.md'
        test_content = "Test file content"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Create backup
        backup_path = self.error_handler.create_backup(test_file)
        
        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.parent == self.error_handler.backup_dir
        
        # Verify backup content
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_content = f.read()
        
        assert backup_content == test_content
    
    def test_create_backup_nonexistent_file(self):
        """Test creating backup of non-existent file."""
        nonexistent_file = self.test_dir / 'nonexistent.md'
        
        backup_path = self.error_handler.create_backup(nonexistent_file)
        assert backup_path is None
    
    def test_restore_from_backup(self):
        """Test restoring a file from backup."""
        # Create original file
        test_file = self.test_dir / 'test.md'
        original_content = "Original content"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Create backup
        backup_path = self.error_handler.create_backup(test_file)
        assert backup_path is not None
        
        # Modify original file
        modified_content = "Modified content"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        # Restore from backup
        success = self.error_handler.restore_from_backup(test_file, backup_path)
        assert success is True
        
        # Verify restoration
        with open(test_file, 'r', encoding='utf-8') as f:
            restored_content = f.read()
        
        assert restored_content == original_content
    
    def test_restore_from_latest_backup(self):
        """Test restoring from latest backup when no specific backup provided."""
        test_file = self.test_dir / 'test.md'
        
        # Create original and backup
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Original")
        
        backup1 = self.error_handler.create_backup(test_file)
        
        # Wait a moment and create another backup
        import time
        time.sleep(0.1)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Updated")
        
        backup2 = self.error_handler.create_backup(test_file)
        
        # Modify file again
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Final")
        
        # Restore from latest backup (should be backup2)
        success = self.error_handler.restore_from_backup(test_file)
        assert success is True
        
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert content == "Updated"
    
    def test_repair_corrupted_file(self):
        """Test repairing a corrupted memory file."""
        corrupted_file = self.test_dir / 'corrupted.md'
        
        # Create corrupted YAML content
        corrupted_content = """---
created: 2024-01-01T10:00:00
memory_type: interaction
importance_score: 5
broken_yaml: [unclosed bracket
---
This is the content part.
"""
        
        with open(corrupted_file, 'w', encoding='utf-8') as f:
            f.write(corrupted_content)
        
        # Attempt repair
        success = self.error_handler.repair_corrupted_file(corrupted_file)
        assert success is True
        
        # Verify file is now readable
        try:
            from src.memory.file_operations import read_memory_file
            frontmatter, content = read_memory_file(corrupted_file)
            assert isinstance(frontmatter, dict)
            assert 'created' in frontmatter
            assert 'memory_type' in frontmatter
            assert 'importance_score' in frontmatter
        except Exception:
            # If we can't import, just check file exists and has content
            assert corrupted_file.exists()
            with open(corrupted_file, 'r', encoding='utf-8') as f:
                repaired_content = f.read()
            assert '---' in repaired_content
    
    def test_handle_validation_result_auto_fix(self):
        """Test handling validation results with auto-fix."""
        # Create file with missing required fields
        test_file = self.test_dir / 'interactions' / 'to_fix.md'
        frontmatter = {
            'created': TimestampManager.create_timestamp()
            # Missing memory_type and importance_score
        }
        content = "Content to fix"
        
        write_memory_file(test_file, frontmatter, content)
        
        # Create validation result with fixable issues
        result = ValidationResult(is_valid=False, issues=[
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                error_type=ErrorType.MISSING_DATA,
                message="Missing required field: memory_type",
                file_path=test_file,
                field='memory_type'
            ),
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                error_type=ErrorType.MISSING_DATA,
                message="Missing required field: importance_score",
                file_path=test_file,
                field='importance_score'
            )
        ])
        
        # Handle with auto-fix
        fixed_result = self.error_handler.handle_validation_result(result, auto_fix=True)
        
        # Should have fewer issues after fixing
        assert len(fixed_result.issues) < len(result.issues)
    
    def test_cleanup_old_backups(self):
        """Test cleaning up old backup files."""
        # Create some test files and backups
        test_file = self.test_dir / 'test.md'
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Test content")
        
        # Create recent backup
        recent_backup = self.error_handler.create_backup(test_file)
        
        # Create old backup by manipulating timestamp
        old_backup = self.error_handler.backup_dir / 'old_backup.md'
        with open(old_backup, 'w', encoding='utf-8') as f:
            f.write("Old backup content")
        
        # Set old timestamp
        old_time = datetime.now() - timedelta(days=40)
        old_timestamp = old_time.timestamp()
        os.utime(old_backup, (old_timestamp, old_timestamp))
        
        # Cleanup backups older than 30 days
        cleaned_count = self.error_handler.cleanup_old_backups(days_to_keep=30)
        
        assert cleaned_count >= 1
        assert not old_backup.exists()
        assert recent_backup.exists()
    
    def test_can_auto_fix(self):
        """Test checking if issues can be auto-fixed."""
        # Fixable issue
        fixable_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            error_type=ErrorType.MISSING_DATA,
            message="Missing field",
            field='memory_type'
        )
        
        assert self.error_handler._can_auto_fix(fixable_issue) is True
        
        # Non-fixable issue
        non_fixable_issue = ValidationIssue(
            severity=ValidationSeverity.CRITICAL,
            error_type=ErrorType.SYSTEM_ERROR,
            message="System error"
        )
        
        assert self.error_handler._can_auto_fix(non_fixable_issue) is False


class TestValidationConvenienceFunction:
    """Test the convenience validation function."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_validate_memory_system_function(self):
        """Test the convenience validate_memory_system function."""
        # Create some test files
        interactions_dir = self.test_dir / 'interactions'
        
        write_memory_file(interactions_dir / 'test.md', {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'interaction',
            'importance_score': 5
        }, "Test content")
        
        # Test without auto-fix
        result = validate_memory_system(self.test_dir, auto_fix=False)
        assert isinstance(result, ValidationResult)
        
        # Test with auto-fix
        result_with_fix = validate_memory_system(self.test_dir, auto_fix=True)
        assert isinstance(result_with_fix, ValidationResult)
    
    def test_validate_with_auto_fix(self):
        """Test validation with auto-fix enabled."""
        # Create file with fixable issues
        interactions_dir = self.test_dir / 'interactions'
        test_file = interactions_dir / 'fixable.md'
        
        # Write file with missing fields
        frontmatter = {
            'created': TimestampManager.create_timestamp()
            # Missing memory_type and importance_score
        }
        write_memory_file(test_file, frontmatter, "Content")
        
        # Validate with auto-fix
        result = validate_memory_system(self.test_dir, auto_fix=True)
        
        # Should attempt to fix issues
        assert isinstance(result, ValidationResult)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.validator = MemoryValidator(self.test_dir)
        self.error_handler = MemoryErrorHandler(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_validate_binary_file(self):
        """Test validating a binary file (should handle gracefully)."""
        # Create a binary file
        binary_file = self.test_dir / 'binary.md'
        binary_content = b'\x00\x01\x02\x03\x04\x05'
        
        with open(binary_file, 'wb') as f:
            f.write(binary_content)
        
        result = self.validator.validate_memory_file(binary_file)
        
        # Should handle gracefully and report error
        assert result.is_valid is False
        assert result.errors_count >= 1
    
    def test_validate_permission_denied(self):
        """Test handling permission errors."""
        # This test is platform-dependent and might not work on all systems
        test_file = self.test_dir / 'protected.md'
        
        # Create file first
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("---\ncreated: test\n---\nContent")
        
        # Try to make file unreadable (might not work on all systems)
        try:
            test_file.chmod(0o000)
            
            result = self.validator.validate_memory_file(test_file)
            
            # Should handle permission error gracefully
            assert isinstance(result, ValidationResult)
            
        except (OSError, PermissionError):
            # Permission manipulation might not work, skip test
            pytest.skip("Cannot manipulate file permissions on this system")
        finally:
            # Restore permissions for cleanup
            try:
                test_file.chmod(0o644)
            except:
                pass
    
    def test_validate_very_large_file(self):
        """Test validating a very large file."""
        large_file = self.test_dir / 'large.md'
        
        # Create file with large content
        frontmatter = {
            'created': TimestampManager.create_timestamp(),
            'memory_type': 'interaction',
            'importance_score': 5
        }
        large_content = "x" * 500000  # 500KB content
        
        write_memory_file(large_file, frontmatter, large_content)
        
        result = self.validator.validate_memory_file(large_file)
        
        # Should warn about large content but still validate
        length_warnings = [issue for issue in result.issues 
                          if "exceeds maximum length" in issue.message]
        assert len(length_warnings) >= 1
    
    def test_malformed_yaml_handling(self):
        """Test handling completely malformed YAML."""
        malformed_file = self.test_dir / 'malformed.md'
        
        # Create file with completely broken YAML
        malformed_content = """---
{invalid yaml content: [broken
---
Content here
"""
        
        with open(malformed_file, 'w', encoding='utf-8') as f:
            f.write(malformed_content)
        
        result = self.validator.validate_memory_file(malformed_file)
        
        # Should detect file corruption
        assert result.is_valid is False
        corruption_errors = [issue for issue in result.issues 
                            if issue.error_type == ErrorType.FILE_CORRUPTION]
        assert len(corruption_errors) >= 1


if __name__ == "__main__":
    pytest.main([__file__]) 