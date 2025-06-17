"""
Memory Validation and Error Handling

This module provides comprehensive validation and error handling for the memory system,
ensuring data integrity, graceful error recovery, and system reliability.
"""

import re
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

# Handle imports gracefully for both package and standalone execution
try:
    from .file_operations import read_memory_file, write_memory_file
    from .importance_scoring import TimestampManager
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from file_operations import read_memory_file, write_memory_file
    from importance_scoring import TimestampManager


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Types of errors that can occur in the memory system."""
    VALIDATION_ERROR = "validation_error"
    FILE_CORRUPTION = "file_corruption"
    PERMISSION_ERROR = "permission_error"
    FORMAT_ERROR = "format_error"
    CONSISTENCY_ERROR = "consistency_error"
    MISSING_DATA = "missing_data"
    INVALID_REFERENCE = "invalid_reference"
    SYSTEM_ERROR = "system_error"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during checks."""
    severity: ValidationSeverity
    error_type: ErrorType
    message: str
    file_path: Optional[Path] = None
    field: Optional[str] = None
    value: Any = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of the validation issue."""
        location = f" in {self.file_path}" if self.file_path else ""
        field_info = f" (field: {self.field})" if self.field else ""
        return f"[{self.severity.value.upper()}] {self.message}{location}{field_info}"


@dataclass
class ValidationResult:
    """Results of a validation operation."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int = 0
    errors_count: int = 0
    
    def __post_init__(self):
        """Calculate counts after initialization."""
        self.warnings_count = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)
        self.errors_count = sum(1 for issue in self.issues 
                               if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        self.is_valid = self.errors_count == 0
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue to the result."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.WARNING:
            self.warnings_count += 1
        elif issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.errors_count += 1
            self.is_valid = False
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        if self.is_valid and self.warnings_count == 0:
            return "✅ All validations passed"
        elif self.is_valid:
            return f"⚠️  Valid with {self.warnings_count} warning(s)"
        else:
            return f"❌ Invalid with {self.errors_count} error(s) and {self.warnings_count} warning(s)"


class MemoryValidator:
    """Comprehensive memory validation system."""
    
    # Required frontmatter fields by memory type
    REQUIRED_FIELDS = {
        'core': ['memory_type', 'created', 'importance_score'],
        'interaction': ['created', 'importance_score', 'memory_type'],
        'condensed': ['created', 'importance_score', 'memory_type', 'source_count'],
        'system': ['created', 'memory_type']
    }
    
    # Field validation rules
    FIELD_RULES = {
        'memory_type': {
            'type': str,
            'allowed_values': [
                'user_profile', 'active_context', 'relationship_evolution',
                'preferences_patterns', 'life_context', 'interaction', 
                'condensed', 'system'
            ]
        },
        'importance_score': {
            'type': int,
            'min_value': 1,
            'max_value': 10
        },
        'created': {
            'type': str,
            'format': 'iso_datetime'
        },
        'last_updated': {
            'type': str,
            'format': 'iso_datetime'
        },
        'last_accessed': {
            'type': str,
            'format': 'iso_datetime'
        },
        'access_count': {
            'type': int,
            'min_value': 0
        },
        'categories': {
            'type': list,
            'item_type': str
        },
        'tags': {
            'type': list,
            'item_type': str
        },
        'source_count': {
            'type': int,
            'min_value': 1
        }
    }
    
    # Content validation limits
    CONTENT_LIMITS = {
        'max_content_length': 100000,  # 100KB
        'max_title_length': 200,
        'max_category_length': 50,
        'max_tag_length': 30,
        'max_categories': 10,
        'max_tags': 20
    }
    
    def __init__(self, memory_base_path: Path):
        """
        Initialize the memory validator.
        
        Args:
            memory_base_path: Base path to the memory directory
        """
        self.memory_base_path = Path(memory_base_path)
        self.logger = logging.getLogger(__name__)
    
    def validate_memory_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a single memory file.
        
        Args:
            file_path: Path to the memory file to validate
            
        Returns:
            ValidationResult with detailed findings
        """
        result = ValidationResult(is_valid=True, issues=[])
        
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    error_type=ErrorType.MISSING_DATA,
                    message="Memory file does not exist",
                    file_path=file_path
                ))
                return result
            
            if not file_path.is_file():
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    error_type=ErrorType.VALIDATION_ERROR,
                    message="Path is not a file",
                    file_path=file_path
                ))
                return result
            
            # Try to read the file
            try:
                frontmatter, content = read_memory_file(file_path)
            except Exception as e:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    error_type=ErrorType.FILE_CORRUPTION,
                    message=f"Failed to read memory file: {str(e)}",
                    file_path=file_path,
                    suggestion="File may be corrupted. Consider restoring from backup."
                ))
                return result
            
            # Validate frontmatter
            self._validate_frontmatter(frontmatter, file_path, result)
            
            # Validate content
            self._validate_content(content, file_path, result)
            
            # Validate file-specific rules
            memory_type = frontmatter.get('memory_type', 'unknown')
            self._validate_memory_type_specific(frontmatter, content, memory_type, file_path, result)
            
        except Exception as e:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                error_type=ErrorType.SYSTEM_ERROR,
                message=f"Unexpected error during validation: {str(e)}",
                file_path=file_path
            ))
        
        return result
    
    def validate_memory_directory(self, directory_path: Path) -> ValidationResult:
        """
        Validate all memory files in a directory.
        
        Args:
            directory_path: Path to the directory to validate
            
        Returns:
            ValidationResult with findings from all files
        """
        result = ValidationResult(is_valid=True, issues=[])
        
        if not directory_path.exists():
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                error_type=ErrorType.MISSING_DATA,
                message="Memory directory does not exist",
                file_path=directory_path
            ))
            return result
        
        # Find all markdown files, excluding documentation files
        all_md_files = list(directory_path.rglob("*.md"))
        md_files = [f for f in all_md_files if not self._is_documentation_file(f)]
        
        if not md_files:
            # Check if we excluded some files
            excluded_count = len(all_md_files) - len(md_files)
            if excluded_count > 0:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    error_type=ErrorType.VALIDATION_ERROR,
                    message=f"Found {excluded_count} documentation file(s), no memory files in directory",
                    file_path=directory_path
                ))
            else:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    error_type=ErrorType.MISSING_DATA,
                    message="No memory files found in directory",
                    file_path=directory_path
                ))
            return result
        
        # Validate each memory file
        for file_path in md_files:
            file_result = self.validate_memory_file(file_path)
            result.issues.extend(file_result.issues)
        
        # Perform cross-file validation
        self._validate_cross_file_consistency(md_files, result)
        
        # Recalculate counts
        result.__post_init__()
        
        return result
    
    def validate_memory_system(self) -> ValidationResult:
        """
        Validate the entire memory system.
        
        Returns:
            ValidationResult with comprehensive findings
        """
        result = ValidationResult(is_valid=True, issues=[])
        
        # Validate directory structure
        self._validate_directory_structure(result)
        
        # Validate all memory directories
        memory_dirs = ['core', 'interactions', 'condensed', 'system']
        
        for dir_name in memory_dirs:
            dir_path = self.memory_base_path / dir_name
            if dir_path.exists():
                dir_result = self.validate_memory_directory(dir_path)
                result.issues.extend(dir_result.issues)
        
        # Validate system integrity
        self._validate_system_integrity(result)
        
        # Recalculate counts
        result.__post_init__()
        
        return result
    
    def _validate_frontmatter(self, frontmatter: Dict[str, Any], file_path: Path, result: ValidationResult) -> None:
        """Validate frontmatter structure and fields."""
        if not isinstance(frontmatter, dict):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                error_type=ErrorType.FORMAT_ERROR,
                message="Frontmatter is not a valid dictionary",
                file_path=file_path
            ))
            return
        
        # Determine memory category
        memory_category = self._get_memory_category(file_path)
        
        # Check required fields
        required_fields = self.REQUIRED_FIELDS.get(memory_category, [])
        for field in required_fields:
            if field not in frontmatter:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    error_type=ErrorType.MISSING_DATA,
                    message=f"Missing required field: {field}",
                    file_path=file_path,
                    field=field,
                    suggestion=f"Add the required {field} field to frontmatter"
                ))
        
        # Validate individual fields
        for field, value in frontmatter.items():
            self._validate_field(field, value, file_path, result)
    
    def _validate_content(self, content: str, file_path: Path, result: ValidationResult) -> None:
        """Validate memory content."""
        if not isinstance(content, str):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                error_type=ErrorType.FORMAT_ERROR,
                message="Content is not a string",
                file_path=file_path
            ))
            return
        
        # Check content length
        if len(content) > self.CONTENT_LIMITS['max_content_length']:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Content exceeds maximum length ({len(content)} > {self.CONTENT_LIMITS['max_content_length']})",
                file_path=file_path,
                suggestion="Consider condensing or splitting the content"
            ))
        
        # Check for common encoding issues
        try:
            content.encode('utf-8')
        except UnicodeEncodeError as e:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                error_type=ErrorType.FORMAT_ERROR,
                message=f"Content contains invalid UTF-8 characters: {str(e)}",
                file_path=file_path
            ))
        
        # Check for empty content in non-system files
        if not content.strip() and not self._is_system_file(file_path):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                error_type=ErrorType.VALIDATION_ERROR,
                message="Memory file has empty content",
                file_path=file_path
            ))
    
    def _validate_field(self, field: str, value: Any, file_path: Path, result: ValidationResult) -> None:
        """Validate a specific frontmatter field."""
        if field not in self.FIELD_RULES:
            return  # Skip unknown fields (they might be custom)
        
        rules = self.FIELD_RULES[field]
        
        # Check type
        expected_type = rules.get('type')
        if expected_type and not isinstance(value, expected_type):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Field {field} has wrong type: expected {expected_type.__name__}, got {type(value).__name__}",
                file_path=file_path,
                field=field,
                value=value
            ))
            return
        
        # Check allowed values
        allowed_values = rules.get('allowed_values')
        if allowed_values and value not in allowed_values:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Field {field} has invalid value: {value}. Allowed: {allowed_values}",
                file_path=file_path,
                field=field,
                value=value
            ))
        
        # Check numeric ranges
        min_value = rules.get('min_value')
        max_value = rules.get('max_value')
        if isinstance(value, (int, float)):
            if min_value is not None and value < min_value:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    error_type=ErrorType.VALIDATION_ERROR,
                    message=f"Field {field} value {value} is below minimum {min_value}",
                    file_path=file_path,
                    field=field,
                    value=value
                ))
            if max_value is not None and value > max_value:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    error_type=ErrorType.VALIDATION_ERROR,
                    message=f"Field {field} value {value} exceeds maximum {max_value}",
                    file_path=file_path,
                    field=field,
                    value=value
                ))
        
        # Check format-specific rules
        format_type = rules.get('format')
        if format_type == 'iso_datetime':
            if not self._validate_iso_datetime(value):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    error_type=ErrorType.FORMAT_ERROR,
                    message=f"Field {field} is not a valid ISO datetime: {value}",
                    file_path=file_path,
                    field=field,
                    value=value
                ))
        
        # Check list items
        if isinstance(value, list):
            item_type = rules.get('item_type')
            if item_type:
                for i, item in enumerate(value):
                    if not isinstance(item, item_type):
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            error_type=ErrorType.VALIDATION_ERROR,
                            message=f"Field {field}[{i}] has wrong type: expected {item_type.__name__}, got {type(item).__name__}",
                            file_path=file_path,
                            field=f"{field}[{i}]",
                            value=item
                        ))
    
    def _validate_memory_type_specific(self, frontmatter: Dict[str, Any], content: str, 
                                     memory_type: str, file_path: Path, result: ValidationResult) -> None:
        """Validate memory type-specific rules."""
        if memory_type == 'condensed':
            # Condensed memories should have source_count
            source_count = frontmatter.get('source_count', 0)
            if source_count < 2:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    error_type=ErrorType.VALIDATION_ERROR,
                    message=f"Condensed memory has low source_count: {source_count}",
                    file_path=file_path,
                    field='source_count',
                    suggestion="Condensed memories should typically combine 2+ sources"
                ))
        
        elif memory_type in ['user_profile', 'relationship_evolution', 'preferences_patterns', 'life_context']:
            # Core memories should have higher importance scores
            importance = frontmatter.get('importance_score', 1)
            if importance < 5:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    error_type=ErrorType.VALIDATION_ERROR,
                    message=f"Core memory has low importance score: {importance}",
                    file_path=file_path,
                    field='importance_score',
                    suggestion="Core memories typically have importance scores >= 5"
                ))
        
        elif memory_type == 'interaction':
            # Interaction memories should have timestamps
            if 'created' not in frontmatter:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    error_type=ErrorType.MISSING_DATA,
                    message="Interaction memory missing creation timestamp",
                    file_path=file_path,
                    field='created'
                ))
    
    def _validate_cross_file_consistency(self, file_paths: List[Path], result: ValidationResult) -> None:
        """Validate consistency across multiple memory files."""
        seen_ids = set()
        memory_types = defaultdict(int)
        
        for file_path in file_paths:
            try:
                frontmatter, _ = read_memory_file(file_path)
                
                # Check for duplicate IDs if present
                memory_id = frontmatter.get('memory_id')
                if memory_id:
                    if memory_id in seen_ids:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            error_type=ErrorType.CONSISTENCY_ERROR,
                            message=f"Duplicate memory ID: {memory_id}",
                            file_path=file_path,
                            field='memory_id',
                            value=memory_id
                        ))
                    else:
                        seen_ids.add(memory_id)
                
                # Track memory type distribution
                memory_type = frontmatter.get('memory_type', 'unknown')
                memory_types[memory_type] += 1
                
            except Exception as e:
                # File reading errors are handled elsewhere
                continue
        
        # Check for unusual memory type distributions
        total_files = len(file_paths)
        if total_files > 10:  # Only check if we have enough files
            interaction_ratio = memory_types.get('interaction', 0) / total_files
            if interaction_ratio > 0.9:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    error_type=ErrorType.VALIDATION_ERROR,
                    message=f"High ratio of interaction memories ({interaction_ratio:.1%})",
                    suggestion="Consider condensing old interactions"
                ))
    
    def _validate_directory_structure(self, result: ValidationResult) -> None:
        """Validate the memory directory structure."""
        required_dirs = ['core', 'interactions']
        optional_dirs = ['condensed', 'system']
        
        for dir_name in required_dirs:
            dir_path = self.memory_base_path / dir_name
            if not dir_path.exists():
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    error_type=ErrorType.MISSING_DATA,
                    message=f"Required memory directory missing: {dir_name}",
                    file_path=dir_path,
                    suggestion=f"Create the {dir_name} directory"
                ))
            elif not dir_path.is_dir():
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    error_type=ErrorType.VALIDATION_ERROR,
                    message=f"Path exists but is not a directory: {dir_name}",
                    file_path=dir_path
                ))
    
    def _validate_system_integrity(self, result: ValidationResult) -> None:
        """Validate overall system integrity."""
        # Check for orphaned files
        all_files = list(self.memory_base_path.rglob("*"))
        
        # Count different file types
        md_files = [f for f in all_files if f.suffix == '.md']
        other_files = [f for f in all_files if f.suffix not in ['.md', ''] and f.is_file()]
        
        if other_files:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Found {len(other_files)} non-markdown files in memory directory",
                suggestion="Consider cleaning up non-memory files"
            ))
        
        # Check file age distribution
        now = datetime.now()
        very_old_files = []
        
        for file_path in md_files:
            try:
                frontmatter, _ = read_memory_file(file_path)
                created_str = frontmatter.get('created', '')
                if created_str:
                    created_dt = TimestampManager.parse_timestamp(created_str)
                    if created_dt and (now - created_dt).days > 365:
                        very_old_files.append(file_path)
            except Exception:
                continue
        
        if len(very_old_files) > 50:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                error_type=ErrorType.VALIDATION_ERROR,
                message=f"Found {len(very_old_files)} files older than 1 year",
                suggestion="Consider archiving or condensing very old memories"
            ))
    
    def _get_memory_category(self, file_path: Path) -> str:
        """Get the memory category based on file path."""
        path_parts = file_path.parts
        if 'core' in path_parts:
            return 'core'
        elif 'interactions' in path_parts:
            return 'interaction'
        elif 'condensed' in path_parts:
            return 'condensed'
        elif 'system' in path_parts:
            return 'system'
        else:
            return 'unknown'
    
    def _is_system_file(self, file_path: Path) -> bool:
        """Check if a file is a system file."""
        return 'system' in file_path.parts
    
    def _is_documentation_file(self, file_path: Path) -> bool:
        """Check if a file is a documentation file that should be excluded from validation."""
        # Exclude README files and other documentation
        filename = file_path.name.lower()
        return (
            filename == 'readme.md' or
            filename.startswith('readme') or
            filename.startswith('doc') or
            filename.startswith('help') or
            filename.endswith('-docs.md') or
            filename.endswith('-documentation.md')
        )
    
    def _validate_iso_datetime(self, value: str) -> bool:
        """Validate ISO datetime format."""
        try:
            TimestampManager.parse_timestamp(value)
            return True
        except (ValueError, TypeError):
            return False


class MemoryErrorHandler:
    """Handles memory system errors with recovery strategies."""
    
    def __init__(self, memory_base_path: Path):
        """
        Initialize the error handler.
        
        Args:
            memory_base_path: Base path to the memory directory
        """
        self.memory_base_path = Path(memory_base_path)
        self.logger = logging.getLogger(__name__)
        self.backup_dir = self.memory_base_path / '.backups'
        self.backup_dir.mkdir(exist_ok=True)
    
    def handle_validation_result(self, result: ValidationResult, auto_fix: bool = False) -> ValidationResult:
        """
        Handle validation results with appropriate recovery actions.
        
        Args:
            result: ValidationResult to handle
            auto_fix: Whether to attempt automatic fixes
            
        Returns:
            Updated ValidationResult after handling
        """
        if result.is_valid:
            return result
        
        fixed_issues = []
        
        for issue in result.issues:
            if auto_fix and self._can_auto_fix(issue):
                try:
                    if self._attempt_auto_fix(issue):
                        fixed_issues.append(issue)
                        self.logger.info(f"Auto-fixed issue: {issue}")
                except Exception as e:
                    self.logger.error(f"Failed to auto-fix issue {issue}: {e}")
        
        # Remove fixed issues from result
        result.issues = [issue for issue in result.issues if issue not in fixed_issues]
        result.__post_init__()
        
        return result
    
    def create_backup(self, file_path: Path) -> Optional[Path]:
        """
        Create a backup of a memory file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to the backup file or None if failed
        """
        try:
            if not file_path.exists():
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            # Copy file content
            with open(file_path, 'r', encoding='utf-8') as src:
                content = src.read()
            
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(content)
            
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def restore_from_backup(self, file_path: Path, backup_path: Optional[Path] = None) -> bool:
        """
        Restore a file from backup.
        
        Args:
            file_path: Path to the file to restore
            backup_path: Specific backup to restore from, or None for latest
            
        Returns:
            True if restoration was successful
        """
        try:
            if backup_path is None:
                # Find the latest backup
                backup_pattern = f"{file_path.stem}_*{file_path.suffix}"
                backups = list(self.backup_dir.glob(backup_pattern))
                if not backups:
                    self.logger.error(f"No backups found for {file_path}")
                    return False
                backup_path = max(backups, key=lambda p: p.stat().st_mtime)
            
            if not backup_path.exists():
                self.logger.error(f"Backup file does not exist: {backup_path}")
                return False
            
            # Copy backup content to original file
            with open(backup_path, 'r', encoding='utf-8') as src:
                content = src.read()
            
            with open(file_path, 'w', encoding='utf-8') as dst:
                dst.write(content)
            
            self.logger.info(f"Restored {file_path} from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore {file_path} from backup: {e}")
            return False
    
    def repair_corrupted_file(self, file_path: Path) -> bool:
        """
        Attempt to repair a corrupted memory file.
        
        Args:
            file_path: Path to the corrupted file
            
        Returns:
            True if repair was successful
        """
        try:
            # Create backup first
            backup_path = self.create_backup(file_path)
            if not backup_path:
                self.logger.warning(f"Could not create backup before repair: {file_path}")
            
            # Try to read raw content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_content = f.read()
            
            # Attempt to extract YAML frontmatter and content
            if raw_content.startswith('---\n'):
                parts = raw_content.split('---\n', 2)
                if len(parts) >= 3:
                    yaml_content = parts[1]
                    md_content = parts[2]
                    
                    # Try to repair YAML
                    try:
                        frontmatter = yaml.safe_load(yaml_content)
                        if not isinstance(frontmatter, dict):
                            frontmatter = {}
                    except yaml.YAMLError:
                        # Create minimal frontmatter
                        frontmatter = {
                            'created': TimestampManager.create_timestamp(),
                            'memory_type': 'interaction',
                            'importance_score': 1,
                            'repaired': True,
                            'repair_timestamp': TimestampManager.create_timestamp()
                        }
                    
                    # Ensure required fields
                    if 'created' not in frontmatter:
                        frontmatter['created'] = TimestampManager.create_timestamp()
                    if 'memory_type' not in frontmatter:
                        frontmatter['memory_type'] = 'interaction'
                    if 'importance_score' not in frontmatter:
                        frontmatter['importance_score'] = 1
                    
                    # Write repaired file
                    repaired_yaml = yaml.dump(frontmatter, default_flow_style=False)
                    repaired_content = f"---\n{repaired_yaml}---\n{md_content}"
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(repaired_content)
                    
                    self.logger.info(f"Repaired corrupted file: {file_path}")
                    return True
            
            # If repair failed, try to create a minimal valid file
            minimal_frontmatter = {
                'created': TimestampManager.create_timestamp(),
                'memory_type': 'interaction',
                'importance_score': 1,
                'repaired': True,
                'repair_timestamp': TimestampManager.create_timestamp(),
                'original_content_corrupted': True
            }
            
            minimal_yaml = yaml.dump(minimal_frontmatter, default_flow_style=False)
            minimal_content = f"---\n{minimal_yaml}---\n# Repaired File\n\nOriginal content was corrupted and could not be recovered.\n"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(minimal_content)
            
            self.logger.warning(f"Created minimal replacement for corrupted file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to repair corrupted file {file_path}: {e}")
            return False
    
    def _can_auto_fix(self, issue: ValidationIssue) -> bool:
        """Check if an issue can be automatically fixed."""
        # Auto-fixable issues
        fixable_combinations = [
            (ValidationSeverity.WARNING, ErrorType.MISSING_DATA),
            (ValidationSeverity.ERROR, ErrorType.MISSING_DATA),
            (ValidationSeverity.ERROR, ErrorType.FORMAT_ERROR)
        ]
        
        return (issue.severity, issue.error_type) in fixable_combinations
    
    def _attempt_auto_fix(self, issue: ValidationIssue) -> bool:
        """Attempt to automatically fix a validation issue."""
        if not issue.file_path or not issue.file_path.exists():
            return False
        
        try:
            # Create backup first
            self.create_backup(issue.file_path)
            
            # Read current file
            frontmatter, content = read_memory_file(issue.file_path)
            modified = False
            
            # Fix missing required fields
            if issue.error_type == ErrorType.MISSING_DATA and issue.field:
                if issue.field == 'created':
                    frontmatter['created'] = TimestampManager.create_timestamp()
                    modified = True
                elif issue.field == 'memory_type':
                    # Infer memory type from file path
                    if 'core' in issue.file_path.parts:
                        frontmatter['memory_type'] = 'user_profile'
                    elif 'interactions' in issue.file_path.parts:
                        frontmatter['memory_type'] = 'interaction'
                    else:
                        frontmatter['memory_type'] = 'interaction'
                    modified = True
                elif issue.field == 'importance_score':
                    frontmatter['importance_score'] = 5  # Default medium importance
                    modified = True
            
            # Fix format errors
            elif issue.error_type == ErrorType.FORMAT_ERROR and issue.field:
                if issue.field in ['created', 'last_updated', 'last_accessed']:
                    # Try to fix invalid timestamps
                    frontmatter[issue.field] = TimestampManager.create_timestamp()
                    modified = True
            
            # Write back if modified
            if modified:
                write_memory_file(issue.file_path, frontmatter, content)
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to auto-fix issue {issue}: {e}")
        
        return False
    
    def cleanup_old_backups(self, days_to_keep: int = 30) -> int:
        """
        Clean up old backup files.
        
        Args:
            days_to_keep: Number of days to keep backups
            
        Returns:
            Number of backups cleaned up
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            cleaned_count = 0
            
            for backup_file in self.backup_dir.glob("*"):
                if backup_file.is_file():
                    file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        backup_file.unlink()
                        cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old backup files")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
            return 0


def validate_memory_system(memory_base_path: Path, auto_fix: bool = False) -> ValidationResult:
    """
    Convenience function to validate the entire memory system.
    
    Args:
        memory_base_path: Base path to the memory directory
        auto_fix: Whether to attempt automatic fixes
        
    Returns:
        ValidationResult with comprehensive findings
    """
    validator = MemoryValidator(memory_base_path)
    error_handler = MemoryErrorHandler(memory_base_path)
    
    result = validator.validate_memory_system()
    
    if auto_fix and not result.is_valid:
        result = error_handler.handle_validation_result(result, auto_fix=True)
    
    return result 