"""
Memory File Operations

This module provides core functionality for reading, writing, and manipulating
memory files with YAML frontmatter. It handles the low-level file operations
that the memory management system depends on.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import yaml


class MemoryFileOperations:
    """Handles low-level memory file operations with YAML frontmatter."""
    
    @staticmethod
    def parse_memory_file(file_path: Path) -> Tuple[Dict[str, Any], str]:
        """
        Parse a memory file with YAML frontmatter.
        
        Args:
            file_path: Path to the memory file
            
        Returns:
            Tuple of (frontmatter_dict, content_string)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Memory file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has frontmatter
            if not content.startswith('---\n'):
                # File without frontmatter - create default metadata
                frontmatter = {
                    'created': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'file_type': 'memory',
                    'importance_score': 5,
                    'category': 'general'
                }
                return frontmatter, content.strip()
            
            # Parse frontmatter
            parts = content.split('---\n', 2)
            if len(parts) < 3:
                raise ValueError(f"Invalid frontmatter format in {file_path}")
            
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML frontmatter in {file_path}: {e}")
            
            content_text = parts[2].strip()
            
            return frontmatter, content_text
            
        except Exception as e:
            raise ValueError(f"Error parsing memory file {file_path}: {e}")
    
    @staticmethod
    def write_memory_file(
        file_path: Path, 
        frontmatter: Dict[str, Any], 
        content: str,
        backup: bool = True
    ) -> bool:
        """
        Write a memory file with YAML frontmatter.
        
        Args:
            file_path: Path to write the file
            frontmatter: YAML frontmatter dictionary
            content: File content (markdown text)
            backup: Whether to create backup of existing file
            
        Returns:
            bool: True if successful
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            # Create backup if requested and file exists
            if backup and file_path.exists():
                MemoryFileOperations._create_backup(file_path)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update last_updated timestamp
            frontmatter['last_updated'] = datetime.now().isoformat()
            
            # Generate YAML frontmatter
            yaml_content = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
            
            # Combine frontmatter and content
            full_content = f"---\n{yaml_content}---\n\n{content}"
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            return True
            
        except Exception as e:
            raise IOError(f"Error writing memory file {file_path}: {e}")
    
    @staticmethod
    def update_frontmatter(
        file_path: Path, 
        updates: Dict[str, Any],
        backup: bool = True
    ) -> bool:
        """
        Update only the frontmatter of an existing memory file.
        
        Args:
            file_path: Path to the memory file
            updates: Dictionary of frontmatter fields to update
            backup: Whether to create backup of existing file
            
        Returns:
            bool: True if successful
        """
        try:
            # Read existing file
            frontmatter, content = MemoryFileOperations.parse_memory_file(file_path)
            
            # Update frontmatter
            frontmatter.update(updates)
            
            # Write back to file
            return MemoryFileOperations.write_memory_file(
                file_path, frontmatter, content, backup
            )
            
        except Exception as e:
            raise IOError(f"Error updating frontmatter in {file_path}: {e}")
    
    @staticmethod
    def append_to_memory_file(
        file_path: Path,
        new_content: str,
        section_header: Optional[str] = None,
        importance_score: Optional[int] = None,
        backup: bool = True
    ) -> bool:
        """
        Append content to an existing memory file.
        
        Args:
            file_path: Path to the memory file
            new_content: Content to append
            section_header: Optional section header for the new content
            importance_score: Update importance score if provided
            backup: Whether to create backup
            
        Returns:
            bool: True if successful
        """
        try:
            # Read existing file
            frontmatter, existing_content = MemoryFileOperations.parse_memory_file(file_path)
            
            # Prepare new content
            content_to_add = new_content.strip()
            if section_header:
                content_to_add = f"\n\n## {section_header}\n\n{content_to_add}"
            else:
                content_to_add = f"\n\n{content_to_add}"
            
            # Combine content
            updated_content = existing_content + content_to_add
            
            # Update frontmatter
            if importance_score is not None:
                frontmatter['importance_score'] = importance_score
            
            # Write updated file
            return MemoryFileOperations.write_memory_file(
                file_path, frontmatter, updated_content, backup
            )
            
        except Exception as e:
            raise IOError(f"Error appending to memory file {file_path}: {e}")
    
    @staticmethod
    def get_file_metadata(file_path: Path) -> Dict[str, Any]:
        """
        Get metadata about a memory file.
        
        Args:
            file_path: Path to the memory file
            
        Returns:
            Dict containing file metadata
        """
        try:
            if not file_path.exists():
                return {'exists': False}
            
            # Get file stats
            stat = file_path.stat()
            
            # Parse frontmatter
            frontmatter, content = MemoryFileOperations.parse_memory_file(file_path)
            
            # Calculate content metrics
            word_count = len(content.split())
            line_count = len(content.split('\n'))
            char_count = len(content)
            
            return {
                'exists': True,
                'path': str(file_path),
                'size_bytes': stat.st_size,
                'created_timestamp': frontmatter.get('created'),
                'last_updated': frontmatter.get('last_updated'),
                'importance_score': frontmatter.get('importance_score'),
                'category': frontmatter.get('category'),
                'file_type': frontmatter.get('file_type'),
                'word_count': word_count,
                'line_count': line_count,
                'char_count': char_count,
                'frontmatter': frontmatter
            }
            
        except Exception as e:
            return {
                'exists': True,
                'error': str(e),
                'path': str(file_path)
            }
    
    @staticmethod
    def search_file_content(
        file_path: Path, 
        search_term: str, 
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for text within a memory file.
        
        Args:
            file_path: Path to the memory file
            search_term: Text to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of dictionaries containing match information
        """
        try:
            frontmatter, content = MemoryFileOperations.parse_memory_file(file_path)
            
            # Prepare search
            search_text = search_term if case_sensitive else search_term.lower()
            content_to_search = content if case_sensitive else content.lower()
            
            matches = []
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line_to_search = line if case_sensitive else line.lower()
                if search_text in line_to_search:
                    matches.append({
                        'line_number': line_num,
                        'line_content': line.strip(),
                        'file_path': str(file_path),
                        'importance_score': frontmatter.get('importance_score', 0)
                    })
            
            return matches
            
        except Exception as e:
            return [{'error': str(e), 'file_path': str(file_path)}]
    
    @staticmethod
    def validate_memory_file(file_path: Path) -> Dict[str, Any]:
        """
        Validate the structure and content of a memory file.
        
        Args:
            file_path: Path to the memory file
            
        Returns:
            Dict containing validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_path': str(file_path)
        }
        
        try:
            if not file_path.exists():
                validation['valid'] = False
                validation['errors'].append("File does not exist")
                return validation
            
            # Check file size
            file_size = file_path.stat().st_size
            max_size = 5 * 1024 * 1024  # 5MB default
            
            if file_size > max_size:
                validation['warnings'].append(f"File size ({file_size} bytes) exceeds recommended maximum")
            
            # Parse file
            frontmatter, content = MemoryFileOperations.parse_memory_file(file_path)
            
            # Validate required frontmatter fields
            required_fields = ['created', 'last_updated', 'importance_score']
            for field in required_fields:
                if field not in frontmatter:
                    validation['errors'].append(f"Missing required frontmatter field: {field}")
                    validation['valid'] = False
            
            # Validate importance score
            importance = frontmatter.get('importance_score')
            if importance is not None:
                if not isinstance(importance, (int, float)) or not (1 <= importance <= 10):
                    validation['errors'].append("Importance score must be a number between 1 and 10")
                    validation['valid'] = False
            
            # Validate timestamps
            for timestamp_field in ['created', 'last_updated']:
                if timestamp_field in frontmatter:
                    try:
                        datetime.fromisoformat(frontmatter[timestamp_field].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        validation['warnings'].append(f"Invalid timestamp format in {timestamp_field}")
            
            # Check content
            if not content.strip():
                validation['warnings'].append("File has no content")
            
            # Check for markdown syntax issues
            if '---' in content:
                validation['warnings'].append("Content contains '---' which might interfere with frontmatter")
            
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"Error validating file: {e}")
        
        return validation
    
    @staticmethod
    def create_memory_entry(
        content: str,
        importance_score: int,
        category: str = 'general',
        file_type: str = 'interaction',
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], str]:
        """
        Create a new memory entry with proper frontmatter.
        
        Args:
            content: The memory content
            importance_score: Importance score (1-10)
            category: Category of the memory
            file_type: Type of memory file
            additional_metadata: Additional frontmatter fields
            
        Returns:
            Tuple of (frontmatter, content) ready for writing
        """
        timestamp = datetime.now().isoformat()
        
        frontmatter = {
            'created': timestamp,
            'last_updated': timestamp,
            'file_type': file_type,
            'importance_score': importance_score,
            'category': category
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            frontmatter.update(additional_metadata)
        
        return frontmatter, content.strip()
    
    @staticmethod
    def list_memory_files(
        directory: Path, 
        pattern: str = "*.md",
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all memory files in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            include_metadata: Whether to include file metadata
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        try:
            if not directory.exists():
                return files
            
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_info = {
                        'path': str(file_path),
                        'name': file_path.name,
                        'size': file_path.stat().st_size
                    }
                    
                    if include_metadata:
                        file_info.update(MemoryFileOperations.get_file_metadata(file_path))
                    
                    files.append(file_info)
            
        except Exception as e:
            # Return empty list on error, but could log the error
            pass
        
        return files
    
    @staticmethod
    def _create_backup(file_path: Path) -> None:
        """
        Create a backup of an existing file.
        
        Args:
            file_path: Path to the file to backup
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = file_path.with_suffix(f'.{timestamp}.bak')
            
            # Read and write to create backup
            with open(file_path, 'r', encoding='utf-8') as src:
                content = src.read()
            
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(content)
                
        except Exception:
            # Backup failure shouldn't stop the main operation
            pass
    
    @staticmethod
    def clean_old_backups(
        directory: Path, 
        max_backups: int = 10,
        max_age_days: int = 30
    ) -> int:
        """
        Clean up old backup files.
        
        Args:
            directory: Directory to clean
            max_backups: Maximum number of backups to keep per file
            max_age_days: Maximum age of backups in days
            
        Returns:
            Number of backups removed
        """
        removed_count = 0
        
        try:
            backup_files = list(directory.glob('*.bak'))
            current_time = datetime.now()
            
            # Group backups by original filename
            backup_groups = {}
            for backup_file in backup_files:
                # Extract original filename (remove timestamp and .bak)
                name_parts = backup_file.name.split('.')
                if len(name_parts) >= 3:
                    original_name = '.'.join(name_parts[:-2])  # Remove timestamp.bak
                    if original_name not in backup_groups:
                        backup_groups[original_name] = []
                    backup_groups[original_name].append(backup_file)
            
            # Process each group
            for original_name, backups in backup_groups.items():
                # Sort by modification time (newest first)
                backups.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                
                # Remove excess backups
                if len(backups) > max_backups:
                    for backup in backups[max_backups:]:
                        backup.unlink()
                        removed_count += 1
                
                # Remove old backups
                for backup in backups[:max_backups]:
                    file_age = current_time - datetime.fromtimestamp(backup.stat().st_mtime)
                    if file_age.days > max_age_days:
                        backup.unlink()
                        removed_count += 1
            
        except Exception:
            # Cleanup errors shouldn't affect main functionality
            pass
        
        return removed_count


# Convenience functions for common operations
def read_memory_file(file_path: Path) -> Tuple[Dict[str, Any], str]:
    """Convenience function to read a memory file."""
    return MemoryFileOperations.parse_memory_file(file_path)


def write_memory_file(file_path: Path, frontmatter: Dict[str, Any], content: str) -> bool:
    """Convenience function to write a memory file."""
    return MemoryFileOperations.write_memory_file(file_path, frontmatter, content)


def update_memory_importance(file_path: Path, importance_score: int) -> bool:
    """Convenience function to update a file's importance score."""
    return MemoryFileOperations.update_frontmatter(
        file_path, {'importance_score': importance_score}
    ) 