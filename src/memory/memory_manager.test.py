"""
Unit tests for Memory Manager

Tests all functionality of the MemoryManager class including CRUD operations,
search capabilities, memory organization, and system maintenance.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import json

from .memory_manager import (
    MemoryManager,
    MemoryEntry,
    SearchResult,
    MemoryType,
    ImportanceLevel
)


class TestMemoryManager:
    """Test suite for MemoryManager class."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.memory_manager = MemoryManager(str(self.test_dir))
        
        # Verify initialization worked
        assert self.memory_manager.initializer.is_initialized()
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    # Core Memory Tests
    
    def test_get_core_memory_success(self):
        """Test successful retrieval of core memory."""
        frontmatter, content = self.memory_manager.get_core_memory('user_profile')
        
        assert isinstance(frontmatter, dict)
        assert isinstance(content, str)
        assert 'created' in frontmatter
        assert 'importance_score' in frontmatter
    
    def test_get_core_memory_invalid_type(self):
        """Test error handling for invalid core memory type."""
        with pytest.raises(ValueError, match="Invalid core memory type"):
            self.memory_manager.get_core_memory('invalid_type')
    
    def test_update_core_memory_replace(self):
        """Test updating core memory by replacing content."""
        new_content = "Updated user profile content"
        result = self.memory_manager.update_core_memory('user_profile', new_content, importance_score=9)
        
        assert result is True
        
        # Verify update
        frontmatter, content = self.memory_manager.get_core_memory('user_profile')
        assert content == new_content
        assert frontmatter['importance_score'] == 9
    
    def test_update_core_memory_append(self):
        """Test updating core memory by appending content."""
        original_frontmatter, original_content = self.memory_manager.get_core_memory('user_profile')
        
        new_content = "Additional information"
        section = "New Section"
        result = self.memory_manager.update_core_memory(
            'user_profile', new_content, section=section, importance_score=8
        )
        
        assert result is True
        
        # Verify append
        frontmatter, content = self.memory_manager.get_core_memory('user_profile')
        assert original_content in content
        assert f"## {section}" in content
        assert new_content in content
        assert frontmatter['importance_score'] == 8
    
    def test_get_all_core_memories(self):
        """Test retrieving all core memories."""
        all_memories = self.memory_manager.get_all_core_memories()
        
        expected_types = ['user_profile', 'active_context', 'relationship_evolution', 
                         'preferences_patterns', 'life_context']
        
        assert len(all_memories) == len(expected_types)
        for memory_type in expected_types:
            assert memory_type in all_memories
            frontmatter, content = all_memories[memory_type]
            assert isinstance(frontmatter, dict)
            assert isinstance(content, str)
    
    # Interaction Memory Tests
    
    def test_create_interaction_memory(self):
        """Test creating a new interaction memory."""
        entry = MemoryEntry(
            content="Test interaction content",
            importance_score=7,
            category="conversation",
            tags=["test", "interaction"],
            metadata={"source": "test"}
        )
        
        file_path = self.memory_manager.create_interaction_memory(entry, "test_conv_123")
        
        assert file_path is not None
        assert Path(file_path).exists()
        assert "test_conv_123" in file_path
        
        # Verify content
        frontmatter, content = self.memory_manager.get_interaction_memory(file_path)
        assert content == "Test interaction content"
        assert frontmatter['importance_score'] == 7
        assert frontmatter['category'] == "conversation"
        assert frontmatter['conversation_id'] == "test_conv_123"
        assert frontmatter['tags'] == ["test", "interaction"]
        assert frontmatter['source'] == "test"
    
    def test_get_interaction_memory(self):
        """Test retrieving an interaction memory."""
        entry = MemoryEntry(
            content="Test content for retrieval",
            importance_score=6,
            category="test"
        )
        
        file_path = self.memory_manager.create_interaction_memory(entry)
        frontmatter, content = self.memory_manager.get_interaction_memory(file_path)
        
        assert content == "Test content for retrieval"
        assert frontmatter['importance_score'] == 6
        assert frontmatter['category'] == "test"
    
    def test_update_interaction_memory_replace(self):
        """Test updating interaction memory by replacing content."""
        entry = MemoryEntry(
            content="Original content",
            importance_score=5,
            category="test"
        )
        
        file_path = self.memory_manager.create_interaction_memory(entry)
        
        # Update with replacement
        new_content = "Updated content"
        result = self.memory_manager.update_interaction_memory(
            file_path, new_content, importance_score=8, append=False
        )
        
        assert result is True
        
        # Verify update
        frontmatter, content = self.memory_manager.get_interaction_memory(file_path)
        assert content == new_content
        assert frontmatter['importance_score'] == 8
    
    def test_update_interaction_memory_append(self):
        """Test updating interaction memory by appending content."""
        entry = MemoryEntry(
            content="Original content",
            importance_score=5,
            category="test"
        )
        
        file_path = self.memory_manager.create_interaction_memory(entry)
        
        # Update with append
        additional_content = "Additional content"
        result = self.memory_manager.update_interaction_memory(
            file_path, additional_content, importance_score=7, append=True
        )
        
        assert result is True
        
        # Verify append
        frontmatter, content = self.memory_manager.get_interaction_memory(file_path)
        assert "Original content" in content
        assert "Additional content" in content
        assert frontmatter['importance_score'] == 7
    
    def test_delete_interaction_memory(self):
        """Test deleting an interaction memory."""
        entry = MemoryEntry(
            content="Content to be deleted",
            importance_score=5,
            category="test"
        )
        
        file_path = self.memory_manager.create_interaction_memory(entry)
        assert Path(file_path).exists()
        
        # Delete the memory
        result = self.memory_manager.delete_interaction_memory(file_path)
        
        assert result is True
        assert not Path(file_path).exists()
    
    # Search Tests
    
    def test_search_memories_basic(self):
        """Test basic memory search functionality."""
        # Create some test memories
        entry1 = MemoryEntry(
            content="Python programming discussion",
            importance_score=8,
            category="technical"
        )
        entry2 = MemoryEntry(
            content="Machine learning concepts",
            importance_score=7,
            category="technical"
        )
        entry3 = MemoryEntry(
            content="Personal conversation about hobbies",
            importance_score=5,
            category="personal"
        )
        
        self.memory_manager.create_interaction_memory(entry1)
        self.memory_manager.create_interaction_memory(entry2)
        self.memory_manager.create_interaction_memory(entry3)
        
        # Search for "programming"
        results = self.memory_manager.search_memories("programming")
        
        assert len(results) >= 1
        assert any("programming" in result.content_snippet.lower() for result in results)
        
        # Verify result structure
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.file_path
        assert result.content_snippet
        assert isinstance(result.importance_score, int)
        assert result.category
        assert result.created
    
    def test_search_memories_with_filters(self):
        """Test memory search with various filters."""
        # Create test memories with different characteristics
        entry1 = MemoryEntry(
            content="High importance technical discussion",
            importance_score=9,
            category="technical"
        )
        entry2 = MemoryEntry(
            content="Low importance technical note",
            importance_score=3,
            category="technical"
        )
        entry3 = MemoryEntry(
            content="Personal technical thoughts",
            importance_score=6,
            category="personal"
        )
        
        self.memory_manager.create_interaction_memory(entry1)
        self.memory_manager.create_interaction_memory(entry2)
        self.memory_manager.create_interaction_memory(entry3)
        
        # Search with importance threshold
        results = self.memory_manager.search_memories(
            "technical", 
            importance_threshold=7
        )
        
        # Should only return high importance results
        assert len(results) >= 1
        for result in results:
            assert result.importance_score >= 7
        
        # Search with category filter
        results = self.memory_manager.search_memories(
            "technical",
            categories=["personal"]
        )
        
        # Should only return personal category results
        assert len(results) >= 1
        for result in results:
            assert result.category == "personal"
    
    def test_search_memories_core_only(self):
        """Test searching only core memories."""
        # Update a core memory with searchable content
        self.memory_manager.update_core_memory(
            'user_profile', 
            "User enjoys Python programming and machine learning"
        )
        
        # Search only core memories
        results = self.memory_manager.search_memories(
            "Python programming",
            memory_types=[MemoryType.CORE]
        )
        
        assert len(results) >= 1
        # Verify result is from core memory
        core_path = str(self.test_dir / 'core')
        assert any(core_path in result.file_path for result in results)
    
    def test_search_memories_date_range(self):
        """Test memory search with date range filter."""
        # Create memories at different times
        entry = MemoryEntry(
            content="Recent memory content",
            importance_score=6,
            category="test"
        )
        
        file_path = self.memory_manager.create_interaction_memory(entry)
        
        # Search with date range (last 24 hours)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        results = self.memory_manager.search_memories(
            "Recent memory",
            date_range=(start_date, end_date)
        )
        
        assert len(results) >= 1
        assert any("Recent memory" in result.content_snippet for result in results)
    
    # Statistics and Maintenance Tests
    
    def test_get_memory_statistics(self):
        """Test memory statistics generation."""
        # Create some test memories
        entry1 = MemoryEntry(content="Test memory 1", importance_score=8, category="test")
        entry2 = MemoryEntry(content="Test memory 2", importance_score=5, category="test")
        
        self.memory_manager.create_interaction_memory(entry1)
        self.memory_manager.create_interaction_memory(entry2)
        
        stats = self.memory_manager.get_memory_statistics()
        
        # Verify structure
        assert 'core_memories' in stats
        assert 'interaction_memories' in stats
        assert 'condensed_memories' in stats
        assert 'system_health' in stats
        
        # Verify core memories stats
        assert len(stats['core_memories']) == 5  # 5 core memory types
        
        # Verify interaction memories stats
        assert stats['interaction_memories']['total_files'] >= 2
        assert 'by_month' in stats['interaction_memories']
        assert 'by_importance' in stats['interaction_memories']
        
        # Verify system health
        assert stats['system_health']['initialized'] is True
    
    def test_cleanup_old_memories_dry_run(self):
        """Test cleanup operation in dry run mode."""
        # Create some test memories
        entry = MemoryEntry(content="Test memory", importance_score=5, category="test")
        self.memory_manager.create_interaction_memory(entry)
        
        # Run cleanup in dry run mode
        results = self.memory_manager.cleanup_old_memories(dry_run=True)
        
        assert 'backups_removed' in results
        assert 'old_interactions_found' in results
        assert 'files_processed' in results
        assert 'errors' in results
        assert isinstance(results['errors'], list)
    
    def test_validate_memory_system(self):
        """Test memory system validation."""
        # Create some test memories
        entry = MemoryEntry(content="Valid test memory", importance_score=7, category="test")
        self.memory_manager.create_interaction_memory(entry)
        
        validation_result = self.memory_manager.validate_memory_system()
        
        # Verify it returns a ValidationResult object
        assert hasattr(validation_result, 'is_valid')
        assert hasattr(validation_result, 'issues')
        assert hasattr(validation_result, 'warnings_count')
        assert hasattr(validation_result, 'errors_count')
        assert hasattr(validation_result, 'get_summary')
        
        # Should be valid for a fresh system
        assert validation_result.is_valid is True
        
        # Should have no errors
        assert validation_result.errors_count == 0
    
    def test_get_recent_interactions(self):
        """Test retrieving recent interactions."""
        # Create some test interactions
        entry1 = MemoryEntry(content="Recent interaction 1", importance_score=6, category="test")
        entry2 = MemoryEntry(content="Recent interaction 2", importance_score=7, category="test")
        
        self.memory_manager.create_interaction_memory(entry1)
        self.memory_manager.create_interaction_memory(entry2)
        
        recent = self.memory_manager.get_recent_interactions(days=1, limit=10)
        
        assert len(recent) >= 2
        
        # Verify structure
        for interaction in recent:
            assert 'file_path' in interaction
            assert 'created' in interaction
            assert 'importance_score' in interaction
            assert 'category' in interaction
            assert 'word_count' in interaction
        
        # Should be sorted by creation date (newest first)
        if len(recent) > 1:
            created_dates = [interaction['created'] for interaction in recent]
            assert created_dates == sorted(created_dates, reverse=True)
    
    def test_get_importance_distribution(self):
        """Test importance distribution calculation."""
        # Create memories with different importance scores
        entries = [
            MemoryEntry(content="Critical memory", importance_score=10, category="test"),
            MemoryEntry(content="High importance memory", importance_score=8, category="test"),
            MemoryEntry(content="Medium importance memory", importance_score=5, category="test"),
            MemoryEntry(content="Low importance memory", importance_score=2, category="test")
        ]
        
        for entry in entries:
            self.memory_manager.create_interaction_memory(entry)
        
        distribution = self.memory_manager.get_importance_distribution()
        
        # Verify structure
        assert 'critical' in distribution
        assert 'high' in distribution
        assert 'medium' in distribution
        assert 'low' in distribution
        
        # Verify counts (at least the ones we created)
        assert distribution['critical'] >= 1  # score 10
        assert distribution['high'] >= 1      # score 8
        assert distribution['medium'] >= 1    # score 5
        assert distribution['low'] >= 1       # score 2


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""
    
    def test_memory_entry_creation(self):
        """Test creating a MemoryEntry."""
        entry = MemoryEntry(
            content="Test content",
            importance_score=7,
            category="test",
            file_type="interaction",
            tags=["tag1", "tag2"],
            metadata={"key": "value"}
        )
        
        assert entry.content == "Test content"
        assert entry.importance_score == 7
        assert entry.category == "test"
        assert entry.file_type == "interaction"
        assert entry.tags == ["tag1", "tag2"]
        assert entry.metadata == {"key": "value"}
    
    def test_memory_entry_defaults(self):
        """Test MemoryEntry with default values."""
        entry = MemoryEntry(
            content="Test content",
            importance_score=5,
            category="test"
        )
        
        assert entry.file_type == "interaction"
        assert entry.tags == []
        assert entry.metadata == {}


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            file_path="/path/to/file.md",
            content_snippet="Test content snippet",
            importance_score=8,
            category="test",
            created="2024-01-01T12:00:00",
            relevance_score=0.85,
            line_number=5
        )
        
        assert result.file_path == "/path/to/file.md"
        assert result.content_snippet == "Test content snippet"
        assert result.importance_score == 8
        assert result.category == "test"
        assert result.created == "2024-01-01T12:00:00"
        assert result.relevance_score == 0.85
        assert result.line_number == 5


class TestMemoryManagerErrorHandling:
    """Test error handling in MemoryManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization_failure(self):
        """Test handling of initialization failure."""
        # Try to initialize in a non-existent directory with no permissions
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(RuntimeError, match="Failed to initialize memory system"):
                MemoryManager(str(self.test_dir / "nonexistent"))
    
    def test_search_with_corrupted_files(self):
        """Test search handling when files are corrupted."""
        memory_manager = MemoryManager(str(self.test_dir))
        
        # Create a corrupted file
        corrupted_file = self.test_dir / 'interactions' / '2024-01' / 'corrupted.md'
        corrupted_file.parent.mkdir(parents=True, exist_ok=True)
        with open(corrupted_file, 'w', encoding='utf-8') as f:
            f.write("---\ninvalid: yaml: [content\n---\nContent")
        
        # Search should handle the corrupted file gracefully
        results = memory_manager.search_memories("test query")
        
        # Should not crash and return empty results
        assert isinstance(results, list)
    
    def test_get_interaction_memory_not_found(self):
        """Test error handling when interaction memory file doesn't exist."""
        memory_manager = MemoryManager(str(self.test_dir))
        
        with pytest.raises(FileNotFoundError):
            memory_manager.get_interaction_memory("/nonexistent/path.md")


if __name__ == "__main__":
    pytest.main([__file__]) 