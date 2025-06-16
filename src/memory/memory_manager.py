"""
Memory Manager

This module provides the high-level memory management interface for the Personal AI Assistant.
It orchestrates all memory operations including CRUD operations, search, organization, and
maintenance of the persistent memory system.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .file_operations import MemoryFileOperations, read_memory_file, write_memory_file
from .memory_initializer import MemoryInitializer

# Handle imports gracefully for both package and standalone execution
try:
    from ..utils.config import (
        get_memory_base_path,
        get_memory_recent_days,
        get_memory_medium_days,
        get_memory_archive_days,
        get_memory_high_importance_threshold,
        get_memory_medium_importance_threshold
    )
except ImportError:
    # Fallback for standalone execution
    import os
    def get_memory_base_path() -> str:
        return os.getenv('MEMORY_BASE_PATH', '~/assistant_memory')
    def get_memory_recent_days() -> int:
        return int(os.getenv('MEMORY_RECENT_DAYS', '30'))
    def get_memory_medium_days() -> int:
        return int(os.getenv('MEMORY_MEDIUM_DAYS', '180'))
    def get_memory_archive_days() -> int:
        return int(os.getenv('MEMORY_ARCHIVE_DAYS', '180'))
    def get_memory_high_importance_threshold() -> int:
        return int(os.getenv('MEMORY_HIGH_IMPORTANCE_THRESHOLD', '7'))
    def get_memory_medium_importance_threshold() -> int:
        return int(os.getenv('MEMORY_MEDIUM_IMPORTANCE_THRESHOLD', '4'))


class MemoryType(Enum):
    """Types of memory files."""
    CORE = "core"
    INTERACTION = "interaction"
    CONDENSED = "condensed"
    SYSTEM = "system"


class ImportanceLevel(Enum):
    """Importance levels for memories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryEntry:
    """Represents a memory entry with metadata."""
    content: str
    importance_score: int
    category: str
    file_type: str = "interaction"
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """Represents a search result."""
    file_path: str
    content_snippet: str
    importance_score: int
    category: str
    created: str
    relevance_score: float = 0.0
    line_number: Optional[int] = None


class MemoryManager:
    """High-level memory management interface."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the memory manager.
        
        Args:
            base_path: Optional override for memory base path
        """
        self.base_path = Path(base_path or get_memory_base_path()).expanduser()
        self.initializer = MemoryInitializer(str(self.base_path))
        
        # Memory configuration
        self.recent_days = get_memory_recent_days()
        self.medium_days = get_memory_medium_days()
        self.archive_days = get_memory_archive_days()
        self.high_importance_threshold = get_memory_high_importance_threshold()
        self.medium_importance_threshold = get_memory_medium_importance_threshold()
        
        # Core memory file paths
        self.core_files = {
            'user_profile': self.base_path / 'core' / 'user_profile.md',
            'active_context': self.base_path / 'core' / 'active_context.md',
            'relationship_evolution': self.base_path / 'core' / 'relationship_evolution.md',
            'preferences_patterns': self.base_path / 'core' / 'preferences_patterns.md',
            'life_context': self.base_path / 'core' / 'life_context.md'
        }
        
        # Ensure memory system is initialized
        self._ensure_initialized()
    
    def _ensure_initialized(self) -> None:
        """Ensure the memory system is properly initialized."""
        if not self.initializer.is_initialized():
            success = self.initializer.initialize_memory_structure()
            if not success:
                raise RuntimeError("Failed to initialize memory system")
    
    # Core Memory Operations
    
    def get_core_memory(self, memory_type: str) -> Tuple[Dict[str, Any], str]:
        """
        Get a core memory file.
        
        Args:
            memory_type: Type of core memory (user_profile, active_context, etc.)
            
        Returns:
            Tuple of (frontmatter, content)
            
        Raises:
            ValueError: If memory type is invalid
            FileNotFoundError: If memory file doesn't exist
        """
        if memory_type not in self.core_files:
            raise ValueError(f"Invalid core memory type: {memory_type}")
        
        file_path = self.core_files[memory_type]
        return read_memory_file(file_path)
    
    def update_core_memory(
        self, 
        memory_type: str, 
        content: str, 
        section: Optional[str] = None,
        importance_score: Optional[int] = None
    ) -> bool:
        """
        Update a core memory file.
        
        Args:
            memory_type: Type of core memory
            content: New content to add or replace
            section: Optional section to update (appends if provided)
            importance_score: Optional importance score update
            
        Returns:
            bool: True if successful
        """
        if memory_type not in self.core_files:
            raise ValueError(f"Invalid core memory type: {memory_type}")
        
        file_path = self.core_files[memory_type]
        
        if section:
            # Append to existing content
            return MemoryFileOperations.append_to_memory_file(
                file_path, content, section, importance_score
            )
        else:
            # Replace content
            frontmatter, _ = read_memory_file(file_path)
            if importance_score is not None:
                frontmatter['importance_score'] = importance_score
            return write_memory_file(file_path, frontmatter, content)
    
    def get_all_core_memories(self) -> Dict[str, Tuple[Dict[str, Any], str]]:
        """
        Get all core memory files.
        
        Returns:
            Dict mapping memory type to (frontmatter, content) tuples
        """
        memories = {}
        for memory_type in self.core_files:
            try:
                memories[memory_type] = self.get_core_memory(memory_type)
            except Exception as e:
                # Log error but continue with other memories
                memories[memory_type] = ({}, f"Error loading memory: {e}")
        
        return memories
    
    # Interaction Memory Operations
    
    def create_interaction_memory(
        self, 
        entry: MemoryEntry,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Create a new interaction memory.
        
        Args:
            entry: Memory entry to create
            conversation_id: Optional conversation identifier
            
        Returns:
            str: Path to created memory file
        """
        # Generate filename
        timestamp = datetime.now()
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_interaction.md"
        
        if conversation_id:
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{conversation_id}.md"
        
        # Determine directory based on age
        month_dir = self.base_path / 'interactions' / timestamp.strftime('%Y-%m')
        month_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = month_dir / filename
        
        # Create frontmatter
        additional_metadata = {
            'conversation_id': conversation_id,
            'tags': entry.tags,
            **entry.metadata
        }
        
        frontmatter, content = MemoryFileOperations.create_memory_entry(
            entry.content,
            entry.importance_score,
            entry.category,
            entry.file_type,
            additional_metadata
        )
        
        # Write file
        success = write_memory_file(file_path, frontmatter, content)
        if not success:
            raise RuntimeError(f"Failed to create interaction memory: {file_path}")
        
        return str(file_path)
    
    def get_interaction_memory(self, file_path: str) -> Tuple[Dict[str, Any], str]:
        """
        Get an interaction memory by file path.
        
        Args:
            file_path: Path to the memory file
            
        Returns:
            Tuple of (frontmatter, content)
        """
        return read_memory_file(Path(file_path))
    
    def update_interaction_memory(
        self, 
        file_path: str, 
        content: str,
        importance_score: Optional[int] = None,
        append: bool = False
    ) -> bool:
        """
        Update an interaction memory.
        
        Args:
            file_path: Path to the memory file
            content: New content
            importance_score: Optional importance score update
            append: Whether to append or replace content
            
        Returns:
            bool: True if successful
        """
        path = Path(file_path)
        
        if append:
            return MemoryFileOperations.append_to_memory_file(
                path, content, importance_score=importance_score
            )
        else:
            frontmatter, _ = read_memory_file(path)
            if importance_score is not None:
                frontmatter['importance_score'] = importance_score
            return write_memory_file(path, frontmatter, content)
    
    def delete_interaction_memory(self, file_path: str) -> bool:
        """
        Delete an interaction memory.
        
        Args:
            file_path: Path to the memory file
            
        Returns:
            bool: True if successful
        """
        try:
            Path(file_path).unlink()
            return True
        except Exception:
            return False
    
    # Search Operations
    
    def search_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        importance_threshold: Optional[int] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        categories: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[SearchResult]:
        """
        Search across all memory files.
        
        Args:
            query: Search query
            memory_types: Types of memories to search
            importance_threshold: Minimum importance score
            date_range: Date range to search within
            categories: Categories to filter by
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        results = []
        
        # Default to searching all memory types
        if memory_types is None:
            memory_types = [MemoryType.CORE, MemoryType.INTERACTION, MemoryType.CONDENSED]
        
        # Search core memories
        if MemoryType.CORE in memory_types:
            results.extend(self._search_core_memories(query, importance_threshold, categories))
        
        # Search interaction memories
        if MemoryType.INTERACTION in memory_types:
            results.extend(self._search_interaction_memories(
                query, importance_threshold, date_range, categories
            ))
        
        # Search condensed memories
        if MemoryType.CONDENSED in memory_types:
            results.extend(self._search_condensed_memories(
                query, importance_threshold, date_range, categories
            ))
        
        # Sort by relevance and importance
        results.sort(key=lambda r: (r.relevance_score, r.importance_score), reverse=True)
        
        return results[:limit]
    
    def _search_core_memories(
        self, 
        query: str, 
        importance_threshold: Optional[int],
        categories: Optional[List[str]]
    ) -> List[SearchResult]:
        """Search core memory files."""
        results = []
        core_path = self.base_path / 'core'
        
        for file_path in core_path.glob('*.md'):
            try:
                matches = MemoryFileOperations.search_file_content(file_path, query, case_sensitive=False)
                
                if matches:
                    frontmatter, _ = read_memory_file(file_path)
                    
                    # Apply filters
                    if importance_threshold and frontmatter.get('importance_score', 0) < importance_threshold:
                        continue
                    
                    if categories and frontmatter.get('category') not in categories:
                        continue
                    
                    for match in matches:
                        if 'error' not in match:
                            results.append(SearchResult(
                                file_path=str(file_path),
                                content_snippet=match['line_content'],
                                importance_score=frontmatter.get('importance_score', 0),
                                category=frontmatter.get('category', 'unknown'),
                                created=frontmatter.get('created', ''),
                                relevance_score=self._calculate_relevance(query, match['line_content']),
                                line_number=match['line_number']
                            ))
            
            except Exception:
                continue
        
        return results
    
    def _search_interaction_memories(
        self,
        query: str,
        importance_threshold: Optional[int],
        date_range: Optional[Tuple[datetime, datetime]],
        categories: Optional[List[str]]
    ) -> List[SearchResult]:
        """Search interaction memory files."""
        results = []
        interactions_path = self.base_path / 'interactions'
        
        for month_dir in interactions_path.iterdir():
            if not month_dir.is_dir():
                continue
            
            for file_path in month_dir.glob('*.md'):
                try:
                    frontmatter, _ = read_memory_file(file_path)
                    
                    # Apply date filter
                    if date_range:
                        created_str = frontmatter.get('created', '')
                        if created_str:
                            try:
                                created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                                if not (date_range[0] <= created_date <= date_range[1]):
                                    continue
                            except ValueError:
                                continue
                    
                    # Apply other filters
                    if importance_threshold and frontmatter.get('importance_score', 0) < importance_threshold:
                        continue
                    
                    if categories and frontmatter.get('category') not in categories:
                        continue
                    
                    matches = MemoryFileOperations.search_file_content(file_path, query, case_sensitive=False)
                    
                    for match in matches:
                        if 'error' not in match:
                            results.append(SearchResult(
                                file_path=str(file_path),
                                content_snippet=match['line_content'],
                                importance_score=frontmatter.get('importance_score', 0),
                                category=frontmatter.get('category', 'unknown'),
                                created=frontmatter.get('created', ''),
                                relevance_score=self._calculate_relevance(query, match['line_content']),
                                line_number=match['line_number']
                            ))
                
                except Exception:
                    continue
        
        return results
    
    def _search_condensed_memories(
        self,
        query: str,
        importance_threshold: Optional[int],
        date_range: Optional[Tuple[datetime, datetime]],
        categories: Optional[List[str]]
    ) -> List[SearchResult]:
        """Search condensed memory files."""
        results = []
        condensed_path = self.base_path / 'condensed'
        
        for subdir in ['recent', 'medium', 'archive']:
            subdir_path = condensed_path / subdir
            if not subdir_path.exists():
                continue
            
            for file_path in subdir_path.glob('*.md'):
                try:
                    frontmatter, _ = read_memory_file(file_path)
                    
                    # Apply filters (similar to interaction memories)
                    if date_range:
                        created_str = frontmatter.get('created', '')
                        if created_str:
                            try:
                                created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                                if not (date_range[0] <= created_date <= date_range[1]):
                                    continue
                            except ValueError:
                                continue
                    
                    if importance_threshold and frontmatter.get('importance_score', 0) < importance_threshold:
                        continue
                    
                    if categories and frontmatter.get('category') not in categories:
                        continue
                    
                    matches = MemoryFileOperations.search_file_content(file_path, query, case_sensitive=False)
                    
                    for match in matches:
                        if 'error' not in match:
                            results.append(SearchResult(
                                file_path=str(file_path),
                                content_snippet=match['line_content'],
                                importance_score=frontmatter.get('importance_score', 0),
                                category=frontmatter.get('category', 'unknown'),
                                created=frontmatter.get('created', ''),
                                relevance_score=self._calculate_relevance(query, match['line_content']),
                                line_number=match['line_number']
                            ))
                
                except Exception:
                    continue
        
        return results
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """
        Calculate relevance score for search results.
        
        Args:
            query: Search query
            content: Content to score
            
        Returns:
            float: Relevance score (0.0 to 1.0)
        """
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Simple relevance scoring
        if query_lower == content_lower:
            return 1.0
        elif query_lower in content_lower:
            # Score based on query length vs content length
            return len(query) / len(content) if len(content) > 0 else 0.0
        else:
            # Check for word matches
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            
            if query_words and content_words:
                overlap = len(query_words.intersection(content_words))
                return overlap / len(query_words)
        
        return 0.0
    
    # Memory Organization and Maintenance
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory system statistics.
        
        Returns:
            Dict containing memory statistics
        """
        stats = {
            'core_memories': {},
            'interaction_memories': {
                'total_files': 0,
                'total_size_mb': 0,
                'by_month': {},
                'by_importance': {'high': 0, 'medium': 0, 'low': 0}
            },
            'condensed_memories': {
                'recent': 0,
                'medium': 0,
                'archive': 0
            },
            'system_health': {
                'initialized': self.initializer.is_initialized(),
                'last_maintenance': None,
                'backup_count': 0
            }
        }
        
        # Core memory stats
        for memory_type, file_path in self.core_files.items():
            if file_path.exists():
                metadata = MemoryFileOperations.get_file_metadata(file_path)
                stats['core_memories'][memory_type] = {
                    'size_bytes': metadata.get('size_bytes', 0),
                    'word_count': metadata.get('word_count', 0),
                    'importance_score': metadata.get('importance_score', 0),
                    'last_updated': metadata.get('last_updated', '')
                }
        
        # Interaction memory stats
        interactions_path = self.base_path / 'interactions'
        if interactions_path.exists():
            for month_dir in interactions_path.iterdir():
                if month_dir.is_dir():
                    month_files = list(month_dir.glob('*.md'))
                    stats['interaction_memories']['by_month'][month_dir.name] = len(month_files)
                    stats['interaction_memories']['total_files'] += len(month_files)
                    
                    # Calculate size and importance distribution
                    for file_path in month_files:
                        try:
                            metadata = MemoryFileOperations.get_file_metadata(file_path)
                            stats['interaction_memories']['total_size_mb'] += metadata.get('size_bytes', 0)
                            
                            importance = metadata.get('importance_score', 0)
                            if importance >= self.high_importance_threshold:
                                stats['interaction_memories']['by_importance']['high'] += 1
                            elif importance >= self.medium_importance_threshold:
                                stats['interaction_memories']['by_importance']['medium'] += 1
                            else:
                                stats['interaction_memories']['by_importance']['low'] += 1
                        except Exception:
                            continue
            
            # Convert bytes to MB
            stats['interaction_memories']['total_size_mb'] = round(
                stats['interaction_memories']['total_size_mb'] / (1024 * 1024), 2
            )
        
        # Condensed memory stats
        condensed_path = self.base_path / 'condensed'
        if condensed_path.exists():
            for subdir in ['recent', 'medium', 'archive']:
                subdir_path = condensed_path / subdir
                if subdir_path.exists():
                    stats['condensed_memories'][subdir] = len(list(subdir_path.glob('*.md')))
        
        # System health
        backup_count = 0
        for backup_file in self.base_path.rglob('*.bak'):
            backup_count += 1
        stats['system_health']['backup_count'] = backup_count
        
        return stats
    
    def cleanup_old_memories(
        self, 
        dry_run: bool = True,
        max_backup_age_days: int = 30
    ) -> Dict[str, Any]:
        """
        Clean up old memories and backups.
        
        Args:
            dry_run: If True, only report what would be cleaned
            max_backup_age_days: Maximum age for backup files
            
        Returns:
            Dict containing cleanup results
        """
        results = {
            'backups_removed': 0,
            'old_interactions_found': 0,
            'files_processed': 0,
            'errors': []
        }
        
        try:
            # Clean up old backups
            if not dry_run:
                results['backups_removed'] = MemoryFileOperations.clean_old_backups(
                    self.base_path, max_age_days=max_backup_age_days
                )
            else:
                # Count backups that would be removed
                cutoff_date = datetime.now() - timedelta(days=max_backup_age_days)
                for backup_file in self.base_path.rglob('*.bak'):
                    try:
                        if datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff_date:
                            results['backups_removed'] += 1
                    except Exception:
                        continue
            
            # Identify old interaction files that could be condensed
            interactions_path = self.base_path / 'interactions'
            if interactions_path.exists():
                cutoff_date = datetime.now() - timedelta(days=self.archive_days)
                
                for month_dir in interactions_path.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    for file_path in month_dir.glob('*.md'):
                        try:
                            frontmatter, _ = read_memory_file(file_path)
                            created_str = frontmatter.get('created', '')
                            
                            if created_str:
                                created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                                if created_date < cutoff_date:
                                    results['old_interactions_found'] += 1
                            
                            results['files_processed'] += 1
                        
                        except Exception as e:
                            results['errors'].append(f"Error processing {file_path}: {e}")
        
        except Exception as e:
            results['errors'].append(f"Cleanup error: {e}")
        
        return results
    
    def validate_memory_system(self) -> Dict[str, Any]:
        """
        Validate the entire memory system.
        
        Returns:
            Dict containing validation results
        """
        validation = {
            'valid': True,
            'core_memories': {},
            'interaction_memories': {'valid': 0, 'invalid': 0, 'errors': []},
            'system_files': {'valid': 0, 'invalid': 0, 'errors': []},
            'overall_errors': []
        }
        
        try:
            # Validate core memories
            for memory_type, file_path in self.core_files.items():
                if file_path.exists():
                    file_validation = MemoryFileOperations.validate_memory_file(file_path)
                    validation['core_memories'][memory_type] = file_validation
                    
                    if not file_validation['valid']:
                        validation['valid'] = False
                else:
                    validation['core_memories'][memory_type] = {
                        'valid': False,
                        'errors': ['File does not exist'],
                        'file_path': str(file_path)
                    }
                    validation['valid'] = False
            
            # Validate interaction memories (sample)
            interactions_path = self.base_path / 'interactions'
            if interactions_path.exists():
                file_count = 0
                for month_dir in interactions_path.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    for file_path in month_dir.glob('*.md'):
                        if file_count >= 100:  # Limit validation to avoid performance issues
                            break
                        
                        try:
                            file_validation = MemoryFileOperations.validate_memory_file(file_path)
                            if file_validation['valid']:
                                validation['interaction_memories']['valid'] += 1
                            else:
                                validation['interaction_memories']['invalid'] += 1
                                validation['interaction_memories']['errors'].append({
                                    'file': str(file_path),
                                    'errors': file_validation['errors']
                                })
                            
                            file_count += 1
                        
                        except Exception as e:
                            validation['interaction_memories']['invalid'] += 1
                            validation['interaction_memories']['errors'].append({
                                'file': str(file_path),
                                'errors': [str(e)]
                            })
            
            # Check system configuration
            system_config_path = self.base_path / 'system' / 'config.json'
            if system_config_path.exists():
                try:
                    with open(system_config_path, 'r', encoding='utf-8') as f:
                        json.load(f)  # Validate JSON format
                    validation['system_files']['valid'] += 1
                except Exception as e:
                    validation['system_files']['invalid'] += 1
                    validation['system_files']['errors'].append(f"Invalid system config: {e}")
                    validation['valid'] = False
        
        except Exception as e:
            validation['overall_errors'].append(str(e))
            validation['valid'] = False
        
        return validation
    
    # Utility Methods
    
    def get_recent_interactions(self, days: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent interaction memories.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction metadata
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        interactions = []
        
        interactions_path = self.base_path / 'interactions'
        if not interactions_path.exists():
            return interactions
        
        for month_dir in interactions_path.iterdir():
            if not month_dir.is_dir():
                continue
            
            for file_path in month_dir.glob('*.md'):
                try:
                    metadata = MemoryFileOperations.get_file_metadata(file_path)
                    created_str = metadata.get('created_timestamp', '')
                    
                    if created_str:
                        created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                        if created_date >= cutoff_date:
                            interactions.append({
                                'file_path': str(file_path),
                                'created': created_str,
                                'importance_score': metadata.get('importance_score', 0),
                                'category': metadata.get('category', 'unknown'),
                                'word_count': metadata.get('word_count', 0)
                            })
                
                except Exception:
                    continue
        
        # Sort by creation date (newest first)
        interactions.sort(key=lambda x: x['created'], reverse=True)
        
        return interactions[:limit]
    
    def get_importance_distribution(self) -> Dict[str, int]:
        """
        Get distribution of memories by importance level.
        
        Returns:
            Dict mapping importance levels to counts
        """
        distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        # Check all memory files
        for memory_path in [self.base_path / 'core', self.base_path / 'interactions']:
            if not memory_path.exists():
                continue
            
            for file_path in memory_path.rglob('*.md'):
                try:
                    metadata = MemoryFileOperations.get_file_metadata(file_path)
                    importance = metadata.get('importance_score', 0)
                    
                    if importance >= 9:
                        distribution['critical'] += 1
                    elif importance >= self.high_importance_threshold:
                        distribution['high'] += 1
                    elif importance >= self.medium_importance_threshold:
                        distribution['medium'] += 1
                    else:
                        distribution['low'] += 1
                
                except Exception:
                    continue
        
        return distribution 