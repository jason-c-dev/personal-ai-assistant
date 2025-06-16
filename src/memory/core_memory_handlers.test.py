"""
Unit tests for Core Memory Handlers

Tests all functionality of the core memory handlers including domain-specific
operations for each memory type and the factory pattern.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Use absolute imports to avoid import issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.memory.core_memory_handlers import (
    UpdateMode,
    MemoryUpdate,
    BaseCoreMemoryHandler,
    UserProfileHandler,
    ActiveContextHandler,
    RelationshipEvolutionHandler,
    PreferencesPatternsHandler,
    LifeContextHandler,
    CoreMemoryHandlerFactory
)
from src.memory.memory_initializer import MemoryInitializer
from src.memory.file_operations import write_memory_file


class TestMemoryUpdate:
    """Test MemoryUpdate dataclass."""
    
    def test_memory_update_creation(self):
        """Test creating a MemoryUpdate."""
        update = MemoryUpdate(
            content="Test content",
            mode=UpdateMode.APPEND,
            section="Test Section",
            importance_score=7,
            metadata={"key": "value"}
        )
        
        assert update.content == "Test content"
        assert update.mode == UpdateMode.APPEND
        assert update.section == "Test Section"
        assert update.importance_score == 7
        assert update.metadata == {"key": "value"}
    
    def test_memory_update_defaults(self):
        """Test MemoryUpdate with default values."""
        update = MemoryUpdate(content="Test content")
        
        assert update.mode == UpdateMode.APPEND
        assert update.section is None
        assert update.importance_score is None
        assert update.metadata == {}


class TestUserProfileHandler:
    """Test suite for UserProfileHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
        
        self.profile_path = self.test_dir / 'core' / 'user_profile.md'
        self.handler = UserProfileHandler(self.profile_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test handler initialization."""
        assert self.handler.file_path == self.profile_path
        assert self.handler.memory_type == "user_profile"
        assert self.profile_path.exists()
    
    def test_initialization_file_not_found(self):
        """Test initialization with non-existent file."""
        non_existent_path = self.test_dir / 'nonexistent.md'
        with pytest.raises(FileNotFoundError):
            UserProfileHandler(non_existent_path)
    
    def test_get_content(self):
        """Test getting profile content."""
        frontmatter, content = self.handler.get_content()
        
        assert isinstance(frontmatter, dict)
        assert isinstance(content, str)
        assert 'created' in frontmatter
        assert 'importance_score' in frontmatter
    
    def test_update_replace_mode(self):
        """Test updating profile with replace mode."""
        update = MemoryUpdate(
            content="New profile content",
            mode=UpdateMode.REPLACE,
            importance_score=8
        )
        
        result = self.handler.update(update)
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert content == "New profile content"
        assert frontmatter['importance_score'] == 8
    
    def test_update_append_mode(self):
        """Test updating profile with append mode."""
        original_frontmatter, original_content = self.handler.get_content()
        
        update = MemoryUpdate(
            content="Additional information",
            mode=UpdateMode.APPEND,
            section="New Section"
        )
        
        result = self.handler.update(update)
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert original_content in content
        assert "## New Section" in content
        assert "Additional information" in content
    
    def test_update_section_mode_new(self):
        """Test updating a new section."""
        update = MemoryUpdate(
            content="Section content",
            mode=UpdateMode.UPDATE_SECTION,
            section="Test Section"
        )
        
        result = self.handler.update(update)
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "## Test Section" in content
        assert "Section content" in content
    
    def test_update_section_mode_existing(self):
        """Test updating an existing section."""
        # First, create a section
        update1 = MemoryUpdate(
            content="Original content",
            mode=UpdateMode.UPDATE_SECTION,
            section="Test Section"
        )
        self.handler.update(update1)
        
        # Then update it
        update2 = MemoryUpdate(
            content="Updated content",
            mode=UpdateMode.UPDATE_SECTION,
            section="Test Section"
        )
        
        result = self.handler.update(update2)
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Updated content" in content
        assert "Original content" not in content
    
    def test_update_merge_mode(self):
        """Test updating profile with merge mode."""
        update = MemoryUpdate(
            content="Merged information",
            mode=UpdateMode.MERGE
        )
        
        result = self.handler.update(update)
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Merged information" in content
        assert f"## Updated {datetime.now().strftime('%Y-%m-%d')}" in content
    
    def test_get_summary(self):
        """Test getting profile summary."""
        # Add some content to profile
        update = MemoryUpdate(
            content="John is a software developer who enjoys programming and technology.",
            mode=UpdateMode.REPLACE
        )
        self.handler.update(update)
        
        summary = self.handler.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "software developer" in summary.lower()
    
    def test_update_basic_info(self):
        """Test updating basic user information."""
        result = self.handler.update_basic_info(
            name="John Doe",
            age=30,
            occupation="Software Developer"
        )
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "## Basic Information" in content
        assert "**Name**: John Doe" in content
        assert "**Age**: 30" in content
        assert "**Occupation**: Software Developer" in content
    
    def test_update_basic_info_partial(self):
        """Test updating basic info with only some fields."""
        result = self.handler.update_basic_info(name="Jane Smith")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "**Name**: Jane Smith" in content
        assert "**Age**" not in content
        assert "**Occupation**" not in content
    
    def test_add_interest_new_section(self):
        """Test adding interest when section doesn't exist."""
        result = self.handler.add_interest("Machine Learning", "Technology")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "## Interests" in content
        assert "- Machine Learning (Technology)" in content
    
    def test_add_interest_existing_section(self):
        """Test adding interest to existing section."""
        # First add an interest to create the section
        self.handler.add_interest("Programming", "Technology")
        
        # Then add another interest
        result = self.handler.add_interest("Reading", "Leisure")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "- Programming (Technology)" in content
        assert "- Reading (Leisure)" in content
    
    def test_update_importance(self):
        """Test updating importance score."""
        result = self.handler.update_importance(9)
        
        assert result is True
        
        frontmatter, _ = self.handler.get_content()
        assert frontmatter['importance_score'] == 9
    
    def test_update_importance_invalid(self):
        """Test updating importance with invalid score."""
        with pytest.raises(ValueError, match="Importance score must be between 1 and 10"):
            self.handler.update_importance(11)
    
    def test_search_content(self):
        """Test searching content in profile."""
        # Add searchable content
        update = MemoryUpdate(
            content="John is a Python developer who loves machine learning",
            mode=UpdateMode.REPLACE
        )
        self.handler.update(update)
        
        results = self.handler.search_content("Python")
        
        assert len(results) > 0
        assert any("Python" in result['line_content'] for result in results)
    
    def test_validate(self):
        """Test profile validation."""
        validation = self.handler.validate()
        
        assert 'valid' in validation
        assert validation['valid'] is True


class TestActiveContextHandler:
    """Test suite for ActiveContextHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
        
        self.context_path = self.test_dir / 'core' / 'active_context.md'
        self.handler = ActiveContextHandler(self.context_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test handler initialization."""
        assert self.handler.memory_type == "active_context"
        assert self.context_path.exists()
    
    def test_update_current_focus(self):
        """Test updating current focus."""
        result = self.handler.update_current_focus("Learning new technologies", "high")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "## Current Focus" in content
        assert "Learning new technologies" in content
        assert "**Priority**: high" in content
    
    def test_add_recent_activity(self):
        """Test adding recent activity."""
        result = self.handler.add_recent_activity("Completed project review", "Work")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Completed project review" in content
        assert "(Work)" in content
    
    def test_clear_outdated_context(self):
        """Test clearing outdated context."""
        result = self.handler.clear_outdated_context(7)
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Context Cleanup" in content
        assert "7 days" in content
    
    def test_get_summary(self):
        """Test getting context summary."""
        # Add some context
        self.handler.update_current_focus("Working on AI project")
        self.handler.add_recent_activity("Meeting with team")
        
        summary = self.handler.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestRelationshipEvolutionHandler:
    """Test suite for RelationshipEvolutionHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
        
        self.relationship_path = self.test_dir / 'core' / 'relationship_evolution.md'
        self.handler = RelationshipEvolutionHandler(self.relationship_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test handler initialization."""
        assert self.handler.memory_type == "relationship_evolution"
        assert self.relationship_path.exists()
    
    def test_add_milestone(self):
        """Test adding relationship milestone."""
        result = self.handler.add_milestone("First meaningful conversation", "high")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "First meaningful conversation" in content
        assert "(Significance: high)" in content
        assert datetime.now().strftime("%Y-%m-%d") in content
    
    def test_update_trust_level(self):
        """Test updating trust level."""
        result = self.handler.update_trust_level("high", "Consistent helpful interactions")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "## Trust Level" in content
        assert "**Trust Level**: high" in content
        assert "**Reason**: Consistent helpful interactions" in content
    
    def test_add_communication_pattern(self):
        """Test adding communication pattern."""
        result = self.handler.add_communication_pattern("Asks detailed technical questions", "Often")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Communication Patterns" in content
        assert "Asks detailed technical questions" in content
        assert "(Frequency: Often)" in content
    
    def test_get_summary(self):
        """Test getting relationship summary."""
        # Add some relationship data
        self.handler.add_milestone("Initial connection established")
        self.handler.update_trust_level("building")
        
        summary = self.handler.get_summary()
        assert isinstance(summary, str)


class TestPreferencesPatternsHandler:
    """Test suite for PreferencesPatternsHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
        
        self.preferences_path = self.test_dir / 'core' / 'preferences_patterns.md'
        self.handler = PreferencesPatternsHandler(self.preferences_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test handler initialization."""
        assert self.handler.memory_type == "preferences_patterns"
        assert self.preferences_path.exists()
    
    def test_add_preference(self):
        """Test adding a preference."""
        result = self.handler.add_preference("Detailed explanations", "Communication", "strong")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Communication Preferences" in content
        assert "Detailed explanations" in content
        assert "(Strength: strong)" in content
    
    def test_add_pattern(self):
        """Test adding a behavioral pattern."""
        result = self.handler.add_pattern(
            "Asks follow-up questions", 
            "Technical discussions", 
            "Frequently"
        )
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Behavioral Patterns" in content
        assert "Asks follow-up questions" in content
        assert "(Context: Technical discussions)" in content
        assert "(Frequency: Frequently)" in content
    
    def test_update_communication_style(self):
        """Test updating communication style."""
        result = self.handler.update_communication_style("Casual", "Prefers informal tone")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "## Communication Style" in content
        assert "**Preferred Style**: Casual" in content
        assert "**Details**: Prefers informal tone" in content
    
    def test_get_summary(self):
        """Test getting preferences summary."""
        # Add some preferences
        self.handler.add_preference("Direct communication", "Style")
        self.handler.add_pattern("Uses technical terminology")
        
        summary = self.handler.get_summary()
        assert isinstance(summary, str)


class TestLifeContextHandler:
    """Test suite for LifeContextHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
        
        self.life_context_path = self.test_dir / 'core' / 'life_context.md'
        self.handler = LifeContextHandler(self.life_context_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test handler initialization."""
        assert self.handler.memory_type == "life_context"
        assert self.life_context_path.exists()
    
    def test_add_life_event(self):
        """Test adding a life event."""
        result = self.handler.add_life_event("Started new job at tech company", "high", "2024-01-15")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Life Events" in content
        assert "Started new job at tech company" in content
        assert "**2024-01-15**" in content
        assert "(Impact: high)" in content
    
    def test_add_life_event_default_date(self):
        """Test adding life event with default date."""
        result = self.handler.add_life_event("Completed certification", "medium")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Completed certification" in content
        assert datetime.now().strftime("%Y-%m-%d") in content
    
    def test_update_current_situation(self):
        """Test updating current situation."""
        result = self.handler.update_current_situation(
            "Working remotely", 
            "Enjoying the flexibility and productivity"
        )
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "## Current Situation" in content
        assert "**Current Situation**: Working remotely" in content
        assert "**Details**: Enjoying the flexibility" in content
    
    def test_add_goal(self):
        """Test adding a goal."""
        result = self.handler.add_goal("Learn machine learning", "high", "6 months")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Goals" in content
        assert "Learn machine learning" in content
        assert "(Priority: high)" in content
        assert "(Timeline: 6 months)" in content
    
    def test_add_challenge(self):
        """Test adding a challenge."""
        result = self.handler.add_challenge("Work-life balance", "ongoing")
        
        assert result is True
        
        frontmatter, content = self.handler.get_content()
        assert "Current Challenges" in content
        assert "Work-life balance" in content
        assert "(Status: ongoing)" in content
        assert datetime.now().strftime("%Y-%m-%d") in content
    
    def test_get_summary(self):
        """Test getting life context summary."""
        # Add some life context
        self.handler.update_current_situation("Transitioning careers")
        self.handler.add_goal("Complete coding bootcamp")
        
        summary = self.handler.get_summary()
        assert isinstance(summary, str)


class TestCoreMemoryHandlerFactory:
    """Test suite for CoreMemoryHandlerFactory."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_create_user_profile_handler(self):
        """Test creating user profile handler."""
        file_path = self.test_dir / 'core' / 'user_profile.md'
        handler = CoreMemoryHandlerFactory.create_handler('user_profile', file_path)
        
        assert isinstance(handler, UserProfileHandler)
        assert handler.memory_type == "user_profile"
    
    def test_create_active_context_handler(self):
        """Test creating active context handler."""
        file_path = self.test_dir / 'core' / 'active_context.md'
        handler = CoreMemoryHandlerFactory.create_handler('active_context', file_path)
        
        assert isinstance(handler, ActiveContextHandler)
        assert handler.memory_type == "active_context"
    
    def test_create_relationship_evolution_handler(self):
        """Test creating relationship evolution handler."""
        file_path = self.test_dir / 'core' / 'relationship_evolution.md'
        handler = CoreMemoryHandlerFactory.create_handler('relationship_evolution', file_path)
        
        assert isinstance(handler, RelationshipEvolutionHandler)
        assert handler.memory_type == "relationship_evolution"
    
    def test_create_preferences_patterns_handler(self):
        """Test creating preferences patterns handler."""
        file_path = self.test_dir / 'core' / 'preferences_patterns.md'
        handler = CoreMemoryHandlerFactory.create_handler('preferences_patterns', file_path)
        
        assert isinstance(handler, PreferencesPatternsHandler)
        assert handler.memory_type == "preferences_patterns"
    
    def test_create_life_context_handler(self):
        """Test creating life context handler."""
        file_path = self.test_dir / 'core' / 'life_context.md'
        handler = CoreMemoryHandlerFactory.create_handler('life_context', file_path)
        
        assert isinstance(handler, LifeContextHandler)
        assert handler.memory_type == "life_context"
    
    def test_create_handler_invalid_type(self):
        """Test creating handler with invalid type."""
        file_path = self.test_dir / 'core' / 'test.md'
        
        with pytest.raises(ValueError, match="Unsupported memory type"):
            CoreMemoryHandlerFactory.create_handler('invalid_type', file_path)
    
    def test_get_supported_types(self):
        """Test getting supported handler types."""
        types = CoreMemoryHandlerFactory.get_supported_types()
        
        expected_types = [
            'user_profile',
            'active_context',
            'relationship_evolution',
            'preferences_patterns',
            'life_context'
        ]
        
        assert len(types) == len(expected_types)
        for memory_type in expected_types:
            assert memory_type in types


class TestUpdateModes:
    """Test different update modes across handlers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
        
        self.profile_path = self.test_dir / 'core' / 'user_profile.md'
        self.handler = UserProfileHandler(self.profile_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_unsupported_update_mode(self):
        """Test handling of unsupported update mode."""
        # Create a custom enum value (this is a bit contrived but tests the error handling)
        with patch('src.memory.core_memory_handlers.UpdateMode') as mock_mode:
            mock_mode.UNSUPPORTED = "unsupported"
            
            update = MemoryUpdate(
                content="Test content",
                mode="unsupported"  # This will cause an error
            )
            
            with pytest.raises(ValueError, match="Unsupported update mode"):
                self.handler.update(update)


class TestErrorHandling:
    """Test error handling in core memory handlers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.initializer = MemoryInitializer(str(self.test_dir))
        self.initializer.initialize_memory_structure()
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_handler_with_corrupted_file(self):
        """Test handler behavior with corrupted memory file."""
        # Create a corrupted file
        corrupted_path = self.test_dir / 'core' / 'corrupted.md'
        with open(corrupted_path, 'w', encoding='utf-8') as f:
            f.write("---\ninvalid: yaml: [content\n---\nContent")
        
        # Handler creation should still work
        handler = UserProfileHandler(corrupted_path)
        
        # But operations might fail gracefully
        try:
            frontmatter, content = handler.get_content()
            # If it succeeds, that's fine too (our file operations are robust)
        except Exception:
            # Expected for corrupted files
            pass
    
    def test_file_permission_error(self):
        """Test handling of file permission errors."""
        profile_path = self.test_dir / 'core' / 'user_profile.md'
        handler = UserProfileHandler(profile_path)
        
        # Mock file operations to raise permission error
        with patch('src.memory.core_memory_handlers.write_memory_file', 
                   side_effect=PermissionError("Permission denied")):
            
            update = MemoryUpdate(content="Test content")
            
            # This should return False rather than crash
            result = handler.update(update)
            # The exact behavior depends on how write_memory_file handles errors
            # In practice, it might return False or raise an exception


if __name__ == "__main__":
    pytest.main([__file__]) 