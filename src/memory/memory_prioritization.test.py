"""
Unit tests for memory_prioritization module

Tests for intelligent memory prioritization including adaptive learning,
priority scoring algorithms, and integration with condensation systems.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Handle imports gracefully for both package and standalone execution
try:
    from .memory_prioritization import (
        IntelligentMemoryPrioritizer, PriorityLevel, PriorityFactor,
        MemoryPriority, UserPattern, PrioritizationConfig
    )
    from .importance_scoring import ImportanceScore
    from .file_operations import write_memory_file
except ImportError:
    from memory_prioritization import (
        IntelligentMemoryPrioritizer, PriorityLevel, PriorityFactor,
        MemoryPriority, UserPattern, PrioritizationConfig
    )
    from importance_scoring import ImportanceScore
    from file_operations import write_memory_file


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create a temporary memory directory for testing"""
    memory_dir = tmp_path / "test_memory"
    memory_dir.mkdir()
    
    # Create required subdirectories
    (memory_dir / "interactions").mkdir()
    (memory_dir / "condensed").mkdir()
    (memory_dir / "core").mkdir()
    (memory_dir / "system").mkdir()
    
    return str(memory_dir)


@pytest.fixture
def prioritizer(temp_memory_dir):
    """Create an IntelligentMemoryPrioritizer instance for testing"""
    config = PrioritizationConfig(
        critical_threshold=90.0,
        high_threshold=75.0,
        medium_threshold=50.0,
        low_threshold=25.0,
        learning_enabled=True
    )
    return IntelligentMemoryPrioritizer(temp_memory_dir, config)


@pytest.fixture
def sample_memory_files(temp_memory_dir):
    """Create sample memory files for testing"""
    base_path = Path(temp_memory_dir)
    
    # Critical importance memory
    critical_path = base_path / "core" / "user_profile.md"
    critical_frontmatter = {
        'created': datetime.now().isoformat(),
        'importance_score': 9,
        'memory_type': 'user_profile',
        'category': 'personal',
        'user_marked_important': True
    }
    critical_content = "My name is John Smith and I'm a software engineer. I have two children and live in San Francisco."
    write_memory_file(critical_path, critical_frontmatter, critical_content)
    
    # High importance memory
    high_path = base_path / "interactions" / "2024-01" / "important_decision.md"
    high_path.parent.mkdir(exist_ok=True)
    high_frontmatter = {
        'created': (datetime.now() - timedelta(days=2)).isoformat(),
        'importance_score': 8,
        'memory_type': 'interaction',
        'category': 'decision'
    }
    high_content = "I need to decide whether to accept the job offer from TechCorp. This is a crucial career decision."
    write_memory_file(high_path, high_frontmatter, high_content)
    
    # Medium importance memory
    medium_path = base_path / "interactions" / "2024-01" / "conversation.md"
    medium_frontmatter = {
        'created': (datetime.now() - timedelta(days=7)).isoformat(),
        'importance_score': 6,
        'memory_type': 'interaction',
        'category': 'conversation'
    }
    medium_content = "We discussed machine learning algorithms and their applications in data science."
    write_memory_file(medium_path, medium_frontmatter, medium_content)
    
    # Low importance memory
    low_path = base_path / "interactions" / "2023-12" / "casual_chat.md"
    low_path.parent.mkdir(exist_ok=True)
    low_frontmatter = {
        'created': (datetime.now() - timedelta(days=30)).isoformat(),
        'importance_score': 3,
        'memory_type': 'interaction',
        'category': 'casual'
    }
    low_content = "Weather was nice today. Had a good walk in the park."
    write_memory_file(low_path, low_frontmatter, low_content)
    
    return {
        'critical': str(critical_path),
        'high': str(high_path),
        'medium': str(medium_path),
        'low': str(low_path)
    }


class TestPrioritizationConfig:
    """Test cases for PrioritizationConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = PrioritizationConfig()
        
        assert config.critical_threshold == 95.0
        assert config.high_threshold == 80.0
        assert config.medium_threshold == 60.0
        assert config.low_threshold == 40.0
        assert config.learning_enabled is True
        assert config.pattern_confidence_threshold == 0.7
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = PrioritizationConfig(
            critical_threshold=90.0,
            high_threshold=70.0,
            learning_enabled=False
        )
        
        assert config.critical_threshold == 90.0
        assert config.high_threshold == 70.0
        assert config.learning_enabled is False


class TestIntelligentMemoryPrioritizerInitialization:
    """Test cases for IntelligentMemoryPrioritizer initialization"""
    
    def test_initialization_default_config(self, temp_memory_dir):
        """Test initialization with default configuration"""
        prioritizer = IntelligentMemoryPrioritizer(temp_memory_dir)
        
        assert prioritizer.base_path == Path(temp_memory_dir)
        assert isinstance(prioritizer.config, PrioritizationConfig)
        assert prioritizer.importance_scorer is not None
        assert prioritizer.time_organizer is not None
        assert isinstance(prioritizer.user_patterns, dict)
        assert isinstance(prioritizer.access_log, list)
    
    def test_initialization_custom_config(self, temp_memory_dir):
        """Test initialization with custom configuration"""
        config = PrioritizationConfig(learning_enabled=False)
        prioritizer = IntelligentMemoryPrioritizer(temp_memory_dir, config)
        
        assert prioritizer.config.learning_enabled is False
    
    def test_directory_creation(self, temp_memory_dir):
        """Test that required directories are created"""
        prioritizer = IntelligentMemoryPrioritizer(temp_memory_dir)
        
        expected_dirs = [
            prioritizer.patterns_dir,
            prioritizer.access_log_dir,
            prioritizer.priorities_dir
        ]
        
        for dir_path in expected_dirs:
            assert dir_path.exists()
            assert dir_path.is_dir()


class TestPriorityFactorCalculation:
    """Test cases for individual priority factor calculations"""
    
    def test_calculate_recency_score(self, prioritizer):
        """Test recency score calculation"""
        # Recent memory (1 day old)
        recent_frontmatter = {
            'created': (datetime.now() - timedelta(days=1)).isoformat()
        }
        recent_score = prioritizer._calculate_recency_score(recent_frontmatter)
        
        # Old memory (100 days old)
        old_frontmatter = {
            'created': (datetime.now() - timedelta(days=100)).isoformat()
        }
        old_score = prioritizer._calculate_recency_score(old_frontmatter)
        
        # Recent should have higher score
        assert recent_score > old_score
        assert 0.0 <= recent_score <= 1.0
        assert 0.0 <= old_score <= 1.0
    
    def test_calculate_recency_score_no_date(self, prioritizer):
        """Test recency score with missing date"""
        frontmatter = {}
        score = prioritizer._calculate_recency_score(frontmatter)
        
        assert score == 0.5  # Default middle score
    
    def test_calculate_frequency_score_empty_log(self, prioritizer):
        """Test frequency score with empty access log"""
        score = prioritizer._calculate_frequency_score("/test/path.md")
        
        assert score == 0.0
    
    def test_calculate_frequency_score_with_accesses(self, prioritizer):
        """Test frequency score with access history"""
        # Add some access entries
        test_path = "/test/path.md"
        for i in range(5):
            prioritizer.access_log.append({
                'file_path': test_path,
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat()
            })
        
        score = prioritizer._calculate_frequency_score(test_path)
        
        assert score > 0.0
        assert score <= 1.0
    
    def test_calculate_uniqueness_score(self, prioritizer):
        """Test uniqueness score calculation"""
        # Unique content
        unique_content = "This is a very specific and unique personal story that never happened before in my life."
        unique_frontmatter = {'category': 'personal'}
        unique_score = prioritizer._calculate_uniqueness_score(unique_content, unique_frontmatter)
        
        # Common content
        common_content = "Hello, how are you today?"
        common_frontmatter = {'category': 'casual'}
        common_score = prioritizer._calculate_uniqueness_score(common_content, common_frontmatter)
        
        assert unique_score > common_score
        assert 0.0 <= unique_score <= 1.0
        assert 0.0 <= common_score <= 1.0
    
    def test_calculate_user_preference_score_no_preferences(self, prioritizer):
        """Test user preference score without explicit preferences"""
        content = "Test content"
        frontmatter = {'category': 'test'}
        
        score = prioritizer._calculate_user_preference_score(content, frontmatter, None)
        
        assert 0.0 <= score <= 1.0
    
    def test_calculate_user_preference_score_with_preferences(self, prioritizer):
        """Test user preference score with preferences"""
        content = "Discussion about machine learning and AI algorithms"
        frontmatter = {'category': 'technology'}
        preferences = {
            'preferred_topics': ['machine learning', 'AI'],
            'preferred_categories': ['technology']
        }
        
        score = prioritizer._calculate_user_preference_score(content, frontmatter, preferences)
        
        assert score > 0.5  # Should have higher score due to matches
        assert score <= 1.0
    
    def test_calculate_context_relevance_score_no_context(self, prioritizer):
        """Test context relevance score without context"""
        content = "Test content"
        frontmatter = {}
        
        score = prioritizer._calculate_context_relevance_score(content, frontmatter, None)
        
        assert score == 0.5  # Default middle score
    
    def test_calculate_context_relevance_score_with_context(self, prioritizer):
        """Test context relevance score with relevant context"""
        content = "Discussion about machine learning algorithms and neural networks"
        frontmatter = {'category': 'technology'}
        context = {
            'current_topic': 'machine learning',
            'user_activity': 'learning'
        }
        
        score = prioritizer._calculate_context_relevance_score(content, frontmatter, context)
        
        assert score > 0.5  # Should have higher score due to relevance
        assert score <= 1.0


class TestMemoryPrioritization:
    """Test cases for complete memory prioritization"""
    
    def test_prioritize_memory_basic(self, prioritizer, sample_memory_files):
        """Test basic memory prioritization"""
        # Test critical memory
        critical_priority = prioritizer.prioritize_memory(sample_memory_files['critical'])
        
        assert isinstance(critical_priority, MemoryPriority)
        assert critical_priority.overall_priority == PriorityLevel.CRITICAL
        assert critical_priority.priority_score > 90.0
        assert not critical_priority.condensation_eligibility
        assert critical_priority.confidence > 0.0
    
    def test_prioritize_memory_with_context(self, prioritizer, sample_memory_files):
        """Test memory prioritization with context"""
        context = {
            'current_topic': 'career decisions',
            'user_activity': 'planning'
        }
        
        priority = prioritizer.prioritize_memory(sample_memory_files['high'], context)
        
        assert isinstance(priority, MemoryPriority)
        # Should have high context relevance due to decision content
        assert priority.factor_scores[PriorityFactor.CONTEXT_RELEVANCE.value] > 0.5
    
    def test_prioritize_memory_with_user_preferences(self, prioritizer, sample_memory_files):
        """Test memory prioritization with user preferences"""
        user_preferences = {
            'preferred_categories': ['conversation'],
            'preferred_topics': ['machine learning']
        }
        
        priority = prioritizer.prioritize_memory(
            sample_memory_files['medium'],
            user_preferences=user_preferences
        )
        
        assert isinstance(priority, MemoryPriority)
        # Should have higher user preference score
        assert priority.factor_scores[PriorityFactor.USER_PREFERENCE.value] > 0.3
    
    def test_prioritize_memory_error_handling(self, prioritizer):
        """Test prioritization error handling for non-existent files"""
        priority = prioritizer.prioritize_memory("/non/existent/file.md")
        
        assert isinstance(priority, MemoryPriority)
        assert priority.overall_priority == PriorityLevel.LOW
        assert priority.confidence < 0.5
        assert "Error in analysis" in priority.preservation_reason
    
    def test_determine_priority_levels(self, prioritizer):
        """Test priority level determination from scores"""
        test_scores = [
            (95.0, PriorityLevel.CRITICAL),
            (85.0, PriorityLevel.HIGH),
            (65.0, PriorityLevel.MEDIUM),
            (45.0, PriorityLevel.LOW),
            (20.0, PriorityLevel.ARCHIVAL)
        ]
        
        for score, expected_level in test_scores:
            level = prioritizer._determine_priority_level(score)
            assert level == expected_level
    
    def test_condensation_eligibility_rules(self, prioritizer):
        """Test condensation eligibility determination"""
        # Mock importance score
        high_importance = ImportanceScore(0.9, 9, {}, 0.8, [])
        low_importance = ImportanceScore(0.3, 3, {}, 0.7, [])
        
        # Critical priority - not eligible
        critical_eligible = prioritizer._determine_condensation_eligibility(
            PriorityLevel.CRITICAL, low_importance, {}
        )
        assert not critical_eligible
        
        # High importance score - not eligible
        high_score_eligible = prioritizer._determine_condensation_eligibility(
            PriorityLevel.MEDIUM, high_importance, {}
        )
        assert not high_score_eligible
        
        # User marked important - not eligible
        user_marked_eligible = prioritizer._determine_condensation_eligibility(
            PriorityLevel.MEDIUM, low_importance, {'user_marked_important': True}
        )
        assert not user_marked_eligible
        
        # Protected memory type - not eligible
        protected_type_eligible = prioritizer._determine_condensation_eligibility(
            PriorityLevel.MEDIUM, low_importance, {'memory_type': 'user_profile'}
        )
        assert not protected_type_eligible
        
        # Normal case - eligible
        normal_eligible = prioritizer._determine_condensation_eligibility(
            PriorityLevel.MEDIUM, low_importance, {'memory_type': 'interaction'}
        )
        assert normal_eligible


class TestAdaptiveLearning:
    """Test cases for adaptive learning functionality"""
    
    def test_record_access_basic(self, prioritizer, sample_memory_files):
        """Test basic access recording"""
        file_path = sample_memory_files['medium']
        initial_log_size = len(prioritizer.access_log)
        
        prioritizer.record_access(file_path, "read")
        
        assert len(prioritizer.access_log) == initial_log_size + 1
        
        latest_entry = prioritizer.access_log[-1]
        assert latest_entry['file_path'] == file_path
        assert latest_entry['access_type'] == "read"
        assert 'timestamp' in latest_entry
    
    def test_record_access_with_context(self, prioritizer, sample_memory_files):
        """Test access recording with context"""
        file_path = sample_memory_files['high']
        context = {'current_topic': 'career planning'}
        
        prioritizer.record_access(file_path, "read", context)
        
        latest_entry = prioritizer.access_log[-1]
        assert latest_entry['context'] == context
    
    def test_pattern_learning_disabled(self, temp_memory_dir):
        """Test behavior when pattern learning is disabled"""
        config = PrioritizationConfig(learning_enabled=False)
        prioritizer = IntelligentMemoryPrioritizer(temp_memory_dir, config)
        
        # Apply learned patterns should return default
        pattern_score = prioritizer._apply_learned_patterns("test content", {})
        assert pattern_score == 0.5
    
    def test_update_category_preference_pattern(self, prioritizer, sample_memory_files):
        """Test category preference pattern learning"""
        # Record multiple accesses to same category
        for _ in range(10):
            prioritizer.record_access(sample_memory_files['medium'], "read")
        
        # Check if pattern was learned
        if 'category_preference' in prioritizer.user_patterns:
            pattern = prioritizer.user_patterns['category_preference']
            assert pattern.observation_count >= 10
            assert 'conversation' in pattern.pattern_data.get('categories', [])
    
    def test_get_access_patterns(self, prioritizer):
        """Test access pattern analysis"""
        test_path = "/test/path.md"
        
        # Add some access entries
        for i in range(5):
            prioritizer.access_log.append({
                'file_path': test_path,
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat()
            })
        
        patterns = prioritizer._get_access_patterns(test_path)
        
        assert patterns['total_accesses'] == 5
        assert patterns['recent_accesses'] >= 0
        assert 'last_access' in patterns
        assert 'access_frequency' in patterns
        assert patterns['access_trend'] in ['stable', 'increasing', 'decreasing']
    
    def test_pattern_confidence_calculation(self, prioritizer):
        """Test pattern confidence calculation"""
        # Create a pattern with consistent data
        pattern = UserPattern(
            pattern_type='category_preference',
            pattern_data={'categories': ['technology', 'personal']},
            confidence=0.8,
            observation_count=20,
            last_updated=datetime.now().isoformat(),
            examples=[]
        )
        
        prioritizer.user_patterns['test_pattern'] = pattern
        
        # Test pattern application
        content = "Discussion about technology and programming"
        frontmatter = {'category': 'technology'}
        
        score = prioritizer._apply_learned_patterns(content, frontmatter)
        assert 0.0 <= score <= 1.0


class TestMemoryListAndRecommendations:
    """Test cases for memory listing and condensation recommendations"""
    
    def test_get_prioritized_memories(self, prioritizer, sample_memory_files):
        """Test getting prioritized memory list"""
        memories = prioritizer.get_prioritized_memories(limit=10)
        
        assert isinstance(memories, list)
        assert len(memories) <= 10
        
        # Should be sorted by priority score (highest first)
        if len(memories) > 1:
            for i in range(len(memories) - 1):
                assert memories[i].priority_score >= memories[i + 1].priority_score
    
    def test_get_prioritized_memories_with_filter(self, prioritizer, sample_memory_files):
        """Test getting prioritized memories with priority filter"""
        memories = prioritizer.get_prioritized_memories(
            limit=10,
            priority_filter=[PriorityLevel.CRITICAL, PriorityLevel.HIGH]
        )
        
        for memory in memories:
            assert memory.overall_priority in [PriorityLevel.CRITICAL, PriorityLevel.HIGH]
    
    def test_get_prioritized_memories_with_context(self, prioritizer, sample_memory_files):
        """Test getting prioritized memories with context"""
        context = {'current_topic': 'career decisions'}
        
        memories = prioritizer.get_prioritized_memories(limit=10, context=context)
        
        assert isinstance(memories, list)
        # Context should influence prioritization
        for memory in memories:
            assert PriorityFactor.CONTEXT_RELEVANCE.value in memory.factor_scores
    
    def test_get_condensation_recommendations(self, prioritizer, sample_memory_files):
        """Test getting condensation recommendations"""
        recommendations = prioritizer.get_condensation_recommendations()
        
        assert isinstance(recommendations, list)
        
        for file_path, recommendation in recommendations:
            assert isinstance(file_path, str)
            assert isinstance(recommendation, str)
            assert any(keyword in recommendation for keyword in [
                'PRESERVE', 'LIGHT_CONDENSATION', 'STANDARD_CONDENSATION',
                'AGGRESSIVE_CONDENSATION', 'ARCHIVE'
            ])
    
    def test_get_prioritization_stats(self, prioritizer, sample_memory_files):
        """Test getting prioritization statistics"""
        stats = prioritizer.get_prioritization_stats()
        
        assert isinstance(stats, dict)
        
        required_keys = [
            'total_memories', 'priority_distribution', 'average_priority_score',
            'condensation_eligible_count', 'condensation_eligible_percentage',
            'learned_patterns_count', 'access_log_size', 'learning_enabled'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert stats['total_memories'] >= 0
        assert isinstance(stats['priority_distribution'], dict)
        assert stats['average_priority_score'] >= 0
        assert stats['condensation_eligible_percentage'] >= 0
        assert stats['learning_enabled'] == prioritizer.config.learning_enabled


class TestPatternPersistence:
    """Test cases for pattern and access log persistence"""
    
    def test_save_and_load_user_patterns(self, prioritizer):
        """Test saving and loading user patterns"""
        # Create a test pattern
        test_pattern = UserPattern(
            pattern_type='test_pattern',
            pattern_data={'test_data': ['value1', 'value2']},
            confidence=0.8,
            observation_count=10,
            last_updated=datetime.now().isoformat(),
            examples=['example1', 'example2']
        )
        
        prioritizer.user_patterns['test'] = test_pattern
        
        # Save patterns
        prioritizer._save_user_patterns()
        
        # Create new prioritizer instance to test loading
        new_prioritizer = IntelligentMemoryPrioritizer(str(prioritizer.base_path))
        
        # Check if pattern was loaded
        assert 'test' in new_prioritizer.user_patterns
        loaded_pattern = new_prioritizer.user_patterns['test']
        assert loaded_pattern.pattern_type == test_pattern.pattern_type
        assert loaded_pattern.confidence == test_pattern.confidence
    
    def test_save_and_load_access_log(self, prioritizer):
        """Test saving and loading access log"""
        # Add test entries to access log
        test_entries = [
            {
                'file_path': '/test/path1.md',
                'access_type': 'read',
                'timestamp': datetime.now().isoformat(),
                'context': {}
            },
            {
                'file_path': '/test/path2.md',
                'access_type': 'write',
                'timestamp': datetime.now().isoformat(),
                'context': {'test': 'data'}
            }
        ]
        
        prioritizer.access_log.extend(test_entries)
        
        # Save access log
        prioritizer._save_access_log()
        
        # Create new prioritizer instance to test loading
        new_prioritizer = IntelligentMemoryPrioritizer(str(prioritizer.base_path))
        
        # Check if access log was loaded
        assert len(new_prioritizer.access_log) >= len(test_entries)
        
        # Check if test entries are present
        for test_entry in test_entries:
            found = False
            for loaded_entry in new_prioritizer.access_log:
                if (loaded_entry['file_path'] == test_entry['file_path'] and
                    loaded_entry['access_type'] == test_entry['access_type']):
                    found = True
                    break
            assert found


class TestErrorHandling:
    """Test cases for error handling in prioritization"""
    
    def test_corrupted_pattern_file_handling(self, temp_memory_dir):
        """Test handling of corrupted pattern files"""
        # Create corrupted pattern file
        patterns_dir = Path(temp_memory_dir) / 'system' / 'user_patterns'
        patterns_dir.mkdir(parents=True, exist_ok=True)
        
        corrupted_file = patterns_dir / 'learned_patterns.json'
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content {")
        
        # Should handle gracefully without crashing
        prioritizer = IntelligentMemoryPrioritizer(temp_memory_dir)
        assert isinstance(prioritizer.user_patterns, dict)
    
    def test_corrupted_access_log_handling(self, temp_memory_dir):
        """Test handling of corrupted access log files"""
        # Create corrupted access log file
        access_log_dir = Path(temp_memory_dir) / 'system' / 'access_logs'
        access_log_dir.mkdir(parents=True, exist_ok=True)
        
        corrupted_file = access_log_dir / 'access_log.json'
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content [")
        
        # Should handle gracefully without crashing
        prioritizer = IntelligentMemoryPrioritizer(temp_memory_dir)
        assert isinstance(prioritizer.access_log, list)
    
    def test_missing_memory_file_handling(self, prioritizer):
        """Test handling of missing memory files during prioritization"""
        # This should not crash and should return a default low priority
        priority = prioritizer.prioritize_memory("/nonexistent/file.md")
        
        assert isinstance(priority, MemoryPriority)
        assert priority.overall_priority == PriorityLevel.LOW
        assert priority.confidence < 0.5


class TestIntegrationScenarios:
    """Test cases for integration with other systems"""
    
    def test_integration_with_importance_scorer(self, prioritizer, sample_memory_files):
        """Test integration with importance scoring system"""
        priority = prioritizer.prioritize_memory(sample_memory_files['critical'])
        
        # Should have valid importance score
        assert isinstance(priority.importance_score, ImportanceScore)
        assert priority.importance_score.normalized_score >= 1
        assert priority.importance_score.normalized_score <= 10
        assert len(priority.importance_score.reasoning) > 0
    
    def test_realistic_workflow(self, prioritizer, sample_memory_files):
        """Test realistic workflow with multiple operations"""
        # Record some accesses
        for file_path in sample_memory_files.values():
            prioritizer.record_access(file_path, "read")
        
        # Get prioritized memories
        memories = prioritizer.get_prioritized_memories(limit=5)
        assert len(memories) > 0
        
        # Get condensation recommendations
        recommendations = prioritizer.get_condensation_recommendations()
        assert len(recommendations) > 0
        
        # Get statistics
        stats = prioritizer.get_prioritization_stats()
        assert stats['total_memories'] > 0
        assert stats['access_log_size'] > 0
    
    def test_pattern_learning_over_time(self, prioritizer, sample_memory_files):
        """Test pattern learning over multiple interactions"""
        # Simulate user preferring certain categories
        preferred_file = sample_memory_files['medium']  # 'conversation' category
        
        # Record multiple accesses to preferred category
        for _ in range(8):  # Above min_observations_for_pattern
            prioritizer.record_access(preferred_file, "read")
        
        # Check if pattern confidence increased
        if 'category_preference' in prioritizer.user_patterns:
            pattern = prioritizer.user_patterns['category_preference']
            assert pattern.observation_count >= 8
            
            # Test if the pattern influences prioritization
            priority = prioritizer.prioritize_memory(preferred_file)
            user_preference_score = priority.factor_scores.get(
                PriorityFactor.USER_PREFERENCE.value, 0
            )
            assert user_preference_score > 0.3  # Should have some preference score


if __name__ == "__main__":
    pytest.main([__file__]) 