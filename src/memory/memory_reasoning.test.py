"""
Unit tests for Memory Decision Reasoning Engine

Tests for comprehensive chain-of-thought reasoning for all memory operations
including storage, retrieval, modification, and context integration decisions.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Handle imports gracefully for both package and standalone execution
try:
    from .memory_reasoning import (
        MemoryReasoningEngine, ReasoningType, ConfidenceLevel,
        DecisionReasoning, ReasoningStep, ReasoningConfig
    )
    from .importance_scoring import ImportanceScore
    from .file_operations import write_memory_file
except ImportError:
    from memory_reasoning import (
        MemoryReasoningEngine, ReasoningType, ConfidenceLevel,
        DecisionReasoning, ReasoningStep, ReasoningConfig
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
    (memory_dir / "core").mkdir()
    (memory_dir / "system").mkdir()
    
    return str(memory_dir)


@pytest.fixture
def reasoning_engine(temp_memory_dir):
    """Create a MemoryReasoningEngine instance for testing"""
    config = ReasoningConfig(
        enable_detailed_reasoning=True,
        enable_alternative_analysis=True,
        reasoning_timeout_seconds=10
    )
    return MemoryReasoningEngine(temp_memory_dir, config)


@pytest.fixture
def sample_memory_files(temp_memory_dir):
    """Create sample memory files for testing"""
    base_path = Path(temp_memory_dir)
    
    # User profile memory
    profile_path = base_path / "core" / "user_profile.md"
    profile_frontmatter = {
        'created': datetime.now().isoformat(),
        'importance_score': 9,
        'memory_type': 'user_profile',
        'category': 'personal'
    }
    profile_content = "User is a software engineer named John who loves machine learning and has two cats."
    write_memory_file(profile_path, profile_frontmatter, profile_content)
    
    # Interaction memory
    interaction_path = base_path / "interactions" / "conversation.md"
    interaction_frontmatter = {
        'created': (datetime.now() - timedelta(days=5)).isoformat(),
        'importance_score': 6,
        'memory_type': 'interaction',
        'category': 'technical'
    }
    interaction_content = "Discussed neural networks and deep learning applications in computer vision."
    write_memory_file(interaction_path, interaction_frontmatter, interaction_content)
    
    return {
        'profile': str(profile_path),
        'interaction': str(interaction_path)
    }


class TestReasoningConfig:
    """Test cases for ReasoningConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ReasoningConfig()
        
        assert config.enable_detailed_reasoning is True
        assert config.enable_alternative_analysis is True
        assert config.enable_risk_assessment is True
        assert config.min_confidence_threshold == 0.3
        assert config.max_reasoning_steps == 10
        assert config.reasoning_timeout_seconds == 5
        assert config.log_all_decisions is True
        assert config.save_reasoning_history is True
        assert config.reasoning_history_days == 30
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = ReasoningConfig(
            enable_detailed_reasoning=False,
            max_reasoning_steps=5,
            reasoning_timeout_seconds=10
        )
        
        assert config.enable_detailed_reasoning is False
        assert config.max_reasoning_steps == 5
        assert config.reasoning_timeout_seconds == 10


class TestMemoryReasoningEngineInitialization:
    """Test cases for MemoryReasoningEngine initialization"""
    
    def test_initialization_default_config(self, temp_memory_dir):
        """Test initialization with default configuration"""
        engine = MemoryReasoningEngine(temp_memory_dir)
        
        assert engine.base_path == Path(temp_memory_dir)
        assert isinstance(engine.config, ReasoningConfig)
        assert engine.importance_scorer is not None
        assert engine.reasoning_dir.exists()
        assert isinstance(engine.reasoning_cache, dict)
    
    def test_initialization_custom_config(self, temp_memory_dir):
        """Test initialization with custom configuration"""
        config = ReasoningConfig(enable_detailed_reasoning=False)
        engine = MemoryReasoningEngine(temp_memory_dir, config)
        
        assert engine.config.enable_detailed_reasoning is False
    
    def test_directory_creation(self, temp_memory_dir):
        """Test that reasoning directory is created"""
        engine = MemoryReasoningEngine(temp_memory_dir)
        
        expected_dir = Path(temp_memory_dir) / 'system' / 'reasoning'
        assert expected_dir.exists()
        assert expected_dir.is_dir()


class TestStorageDecisionReasoning:
    """Test cases for storage decision reasoning"""
    
    @patch('memory_reasoning.ImportanceScorer')
    def test_reason_storage_decision_basic(self, mock_scorer, reasoning_engine):
        """Test basic storage decision reasoning"""
        # Mock importance scorer
        mock_importance = ImportanceScore(0.7, 7, {}, 0.8, ["High importance content"])
        mock_scorer.return_value.calculate_importance.return_value = mock_importance
        reasoning_engine.importance_scorer = mock_scorer.return_value
        
        content = "This is important information about the user's preferences."
        metadata = {
            'memory_type': 'interaction',
            'category': 'personal'
        }
        context = {'current_topic': 'preferences'}
        
        reasoning = reasoning_engine.reason_storage_decision(content, metadata, context)
        
        assert isinstance(reasoning, DecisionReasoning)
        assert reasoning.reasoning_type == ReasoningType.STORAGE
        assert len(reasoning.reasoning_steps) >= 3
        assert reasoning.overall_confidence > 0
        assert reasoning.confidence_level in ConfidenceLevel
        assert "STORE" in reasoning.decision_summary or "SKIP" in reasoning.decision_summary
    
    @patch('memory_reasoning.ImportanceScorer')
    def test_reason_storage_decision_high_importance(self, mock_scorer, reasoning_engine):
        """Test storage decision with high importance content"""
        # Mock high importance score
        mock_importance = ImportanceScore(0.9, 9, {}, 0.9, ["Critical information"])
        mock_scorer.return_value.calculate_importance.return_value = mock_importance
        reasoning_engine.importance_scorer = mock_scorer.return_value
        
        content = "User's critical personal information and preferences."
        metadata = {'memory_type': 'user_profile', 'category': 'personal'}
        
        reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        
        assert "STORE" in reasoning.decision_summary
        assert reasoning.overall_confidence > 0.7
        assert len(reasoning.expected_outcomes) > 0
        assert len(reasoning.potential_risks) > 0
    
    @patch('memory_reasoning.ImportanceScorer')
    def test_reason_storage_decision_low_importance(self, mock_scorer, reasoning_engine):
        """Test storage decision with low importance content"""
        # Mock low importance score
        mock_importance = ImportanceScore(0.2, 2, {}, 0.6, ["Low importance content"])
        mock_scorer.return_value.calculate_importance.return_value = mock_importance
        reasoning_engine.importance_scorer = mock_scorer.return_value
        
        content = "Just casual small talk about the weather."
        metadata = {'memory_type': 'interaction', 'category': 'casual'}
        
        reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        
        assert reasoning.reasoning_type == ReasoningType.STORAGE
        assert len(reasoning.reasoning_steps) >= 3
        # Decision could be SKIP or STORE depending on other factors
    
    def test_storage_reasoning_steps_structure(self, reasoning_engine):
        """Test that storage reasoning steps have proper structure"""
        content = "Test content for reasoning structure validation."
        metadata = {'memory_type': 'interaction'}
        
        reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        
        for step in reasoning.reasoning_steps:
            assert isinstance(step, ReasoningStep)
            assert step.step_number > 0
            assert step.description is not None
            assert isinstance(step.evidence, list)
            assert 0 <= step.confidence <= 1
            assert isinstance(step.alternatives_considered, list)
            assert step.timestamp is not None
    
    def test_storage_reasoning_caching(self, reasoning_engine):
        """Test that storage reasoning is cached"""
        content = "Test content for caching validation."
        metadata = {'memory_type': 'interaction'}
        
        reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        
        assert reasoning.reasoning_id in reasoning_engine.reasoning_cache
        cached_reasoning = reasoning_engine.reasoning_cache[reasoning.reasoning_id]
        assert cached_reasoning.reasoning_id == reasoning.reasoning_id


class TestRetrievalDecisionReasoning:
    """Test cases for retrieval decision reasoning"""
    
    def test_reason_retrieval_decision_basic(self, reasoning_engine, sample_memory_files):
        """Test basic retrieval decision reasoning"""
        query_context = {
            'type': 'information_lookup',
            'keywords': ['machine learning', 'neural networks'],
            'temporal': 'recent',
            'urgency': 'normal'
        }
        available_memories = list(sample_memory_files.values())
        
        reasoning = reasoning_engine.reason_retrieval_decision(
            query_context, available_memories, max_memories=5
        )
        
        assert isinstance(reasoning, DecisionReasoning)
        assert reasoning.reasoning_type == ReasoningType.RETRIEVAL
        assert len(reasoning.reasoning_steps) >= 2
        assert "RETRIEVE" in reasoning.decision_summary
        assert reasoning.overall_confidence > 0
    
    def test_reason_retrieval_decision_empty_memories(self, reasoning_engine):
        """Test retrieval decision with no available memories"""
        query_context = {'type': 'general', 'keywords': []}
        available_memories = []
        
        reasoning = reasoning_engine.reason_retrieval_decision(
            query_context, available_memories
        )
        
        assert reasoning.reasoning_type == ReasoningType.RETRIEVAL
        assert len(reasoning.reasoning_steps) >= 2
        assert "0 memories" in reasoning.decision_summary
    
    def test_reason_retrieval_decision_max_limit(self, reasoning_engine, sample_memory_files):
        """Test retrieval decision with memory limit"""
        query_context = {'type': 'comprehensive', 'keywords': ['test']}
        available_memories = list(sample_memory_files.values()) * 10  # Create many memories
        
        reasoning = reasoning_engine.reason_retrieval_decision(
            query_context, available_memories, max_memories=3
        )
        
        # Should respect the max_memories limit
        assert "3 memories" in reasoning.decision_summary or "RETRIEVE 3" in reasoning.decision_summary
    
    def test_retrieval_reasoning_factors(self, reasoning_engine, sample_memory_files):
        """Test that retrieval reasoning considers proper factors"""
        query_context = {
            'type': 'specific_lookup',
            'keywords': ['software', 'engineering'],
            'temporal': 'any',
            'urgency': 'high'
        }
        available_memories = list(sample_memory_files.values())
        
        reasoning = reasoning_engine.reason_retrieval_decision(
            query_context, available_memories
        )
        
        # Check that proper factors are considered
        assert 'context_analysis' in reasoning.factors_considered
        assert 'memory_scores' in reasoning.factors_considered
        assert 'retrieval_strategy' in reasoning.factors_considered


class TestModificationDecisionReasoning:
    """Test cases for modification decision reasoning"""
    
    def test_reason_modification_decision_basic(self, reasoning_engine, sample_memory_files):
        """Test basic modification decision reasoning"""
        existing_path = sample_memory_files['interaction']
        proposed_changes = {
            'content': 'Updated content about neural networks and computer vision applications.',
            'importance_score': 7
        }
        context = {'reason': 'content_update'}
        
        reasoning = reasoning_engine.reason_modification_decision(
            existing_path, proposed_changes, context
        )
        
        assert isinstance(reasoning, DecisionReasoning)
        assert reasoning.reasoning_type == ReasoningType.MODIFICATION
        assert len(reasoning.reasoning_steps) >= 3
        assert reasoning.overall_confidence > 0
        assert any(action in reasoning.decision_summary for action in ['UPDATE', 'CREATE_NEW', 'MERGE'])
    
    def test_reason_modification_decision_nonexistent_file(self, reasoning_engine):
        """Test modification decision with non-existent file"""
        nonexistent_path = "/path/to/nonexistent/file.md"
        proposed_changes = {'content': 'New content'}
        
        reasoning = reasoning_engine.reason_modification_decision(
            nonexistent_path, proposed_changes
        )
        
        assert reasoning.reasoning_type == ReasoningType.MODIFICATION
        # Should handle gracefully even with missing file
        assert len(reasoning.reasoning_steps) >= 3
    
    def test_modification_reasoning_change_impact(self, reasoning_engine, sample_memory_files):
        """Test that modification reasoning assesses change impact"""
        existing_path = sample_memory_files['profile']
        proposed_changes = {
            'content': 'Completely different content about cooking instead of software engineering.',
            'category': 'cooking'
        }
        
        reasoning = reasoning_engine.reason_modification_decision(
            existing_path, proposed_changes
        )
        
        # Should analyze change impact
        assert 'change_impact' in reasoning.factors_considered
        assert 'modification_strategy' in reasoning.factors_considered
        
        # Major changes might suggest creating new memory
        change_impact = reasoning.factors_considered['change_impact']
        assert 'magnitude' in change_impact
        assert 'similarity' in change_impact


class TestContextIntegrationReasoning:
    """Test cases for context integration reasoning"""
    
    def test_reason_context_integration_basic(self, reasoning_engine):
        """Test basic context integration reasoning"""
        retrieved_memories = [
            {
                'path': '/test/memory1.md',
                'content': 'User likes machine learning',
                'relevance_score': 0.8
            },
            {
                'path': '/test/memory2.md',
                'content': 'User works as software engineer',
                'relevance_score': 0.6
            }
        ]
        current_context = {
            'topic': 'career advice',
            'conversation_type': 'consultation'
        }
        response_goal = "Provide personalized career guidance"
        
        reasoning = reasoning_engine.reason_context_integration_decision(
            retrieved_memories, current_context, response_goal
        )
        
        assert isinstance(reasoning, DecisionReasoning)
        assert reasoning.reasoning_type == ReasoningType.CONTEXT_INTEGRATION
        assert len(reasoning.reasoning_steps) >= 2
        assert "INTEGRATE" in reasoning.decision_summary
        assert reasoning.overall_confidence > 0
    
    def test_context_integration_empty_memories(self, reasoning_engine):
        """Test context integration with no memories"""
        retrieved_memories = []
        current_context = {'topic': 'general'}
        response_goal = "Provide helpful response"
        
        reasoning = reasoning_engine.reason_context_integration_decision(
            retrieved_memories, current_context, response_goal
        )
        
        assert reasoning.reasoning_type == ReasoningType.CONTEXT_INTEGRATION
        assert "0 memories" in reasoning.decision_summary
    
    def test_context_integration_strategy_selection(self, reasoning_engine):
        """Test that context integration selects appropriate strategy"""
        retrieved_memories = [
            {'path': '/test/memory.md', 'content': 'Test content', 'relevance_score': 0.9}
        ]
        current_context = {'topic': 'technical_discussion'}
        response_goal = "Provide technical explanation"
        
        reasoning = reasoning_engine.reason_context_integration_decision(
            retrieved_memories, current_context, response_goal
        )
        
        # Should have integration strategy
        assert 'integration_strategy' in reasoning.factors_considered
        strategy = reasoning.factors_considered['integration_strategy']
        assert 'approach' in strategy
        assert 'style' in strategy
        assert 'selected_memories' in strategy


class TestReasoningUtilities:
    """Test cases for reasoning utility methods"""
    
    def test_get_reasoning_summary(self, reasoning_engine):
        """Test getting reasoning summary"""
        content = "Test content for summary"
        metadata = {'memory_type': 'interaction'}
        
        reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        summary = reasoning_engine.get_reasoning_summary(reasoning.reasoning_id)
        
        assert summary is not None
        assert summary['id'] == reasoning.reasoning_id
        assert summary['type'] == ReasoningType.STORAGE.value
        assert 'decision' in summary
        assert 'confidence' in summary
        assert 'confidence_level' in summary
        assert 'steps' in summary
        assert 'timestamp' in summary
    
    def test_get_reasoning_summary_nonexistent(self, reasoning_engine):
        """Test getting summary for non-existent reasoning"""
        summary = reasoning_engine.get_reasoning_summary("nonexistent_id")
        assert summary is None
    
    def test_get_reasoning_history(self, reasoning_engine):
        """Test getting reasoning history"""
        # Create some reasoning entries
        for i in range(3):
            content = f"Test content {i}"
            metadata = {'memory_type': 'interaction'}
            reasoning_engine.reason_storage_decision(content, metadata)
        
        history = reasoning_engine.get_reasoning_history(days=1)
        
        assert isinstance(history, list)
        assert len(history) >= 3
        
        for entry in history:
            assert 'id' in entry
            assert 'type' in entry
            assert 'decision' in entry
            assert 'confidence' in entry
            assert 'timestamp' in entry
    
    def test_get_reasoning_history_filtered(self, reasoning_engine):
        """Test getting filtered reasoning history"""
        # Create storage reasoning
        content = "Test storage content"
        metadata = {'memory_type': 'interaction'}
        reasoning_engine.reason_storage_decision(content, metadata)
        
        # Get only storage reasoning
        history = reasoning_engine.get_reasoning_history(
            reasoning_type=ReasoningType.STORAGE,
            days=1
        )
        
        assert isinstance(history, list)
        for entry in history:
            assert entry['type'] == ReasoningType.STORAGE.value
    
    def test_cleanup_old_reasoning(self, reasoning_engine):
        """Test cleanup of old reasoning files"""
        # Create some reasoning entries
        content = "Test content for cleanup"
        metadata = {'memory_type': 'interaction'}
        reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        
        # Verify file was created
        reasoning_file = reasoning_engine.reasoning_dir / f"{reasoning.reasoning_id}.json"
        assert reasoning_file.exists()
        
        # Test cleanup (should not delete recent files)
        cleaned_count = reasoning_engine.cleanup_old_reasoning(days=30)
        assert reasoning_file.exists()  # Should still exist
        
        # Test cleanup with 0 days (should delete all)
        cleaned_count = reasoning_engine.cleanup_old_reasoning(days=0)
        assert cleaned_count >= 0  # Should return count of cleaned files


class TestConfidenceLevelDetermination:
    """Test cases for confidence level determination"""
    
    def test_determine_confidence_level_very_high(self, reasoning_engine):
        """Test very high confidence level determination"""
        level = reasoning_engine._determine_confidence_level(0.95)
        assert level == ConfidenceLevel.VERY_HIGH
    
    def test_determine_confidence_level_high(self, reasoning_engine):
        """Test high confidence level determination"""
        level = reasoning_engine._determine_confidence_level(0.8)
        assert level == ConfidenceLevel.HIGH
    
    def test_determine_confidence_level_medium(self, reasoning_engine):
        """Test medium confidence level determination"""
        level = reasoning_engine._determine_confidence_level(0.6)
        assert level == ConfidenceLevel.MEDIUM
    
    def test_determine_confidence_level_low(self, reasoning_engine):
        """Test low confidence level determination"""
        level = reasoning_engine._determine_confidence_level(0.4)
        assert level == ConfidenceLevel.LOW
    
    def test_determine_confidence_level_very_low(self, reasoning_engine):
        """Test very low confidence level determination"""
        level = reasoning_engine._determine_confidence_level(0.2)
        assert level == ConfidenceLevel.VERY_LOW


class TestReasoningPersistence:
    """Test cases for reasoning persistence"""
    
    def test_save_reasoning(self, reasoning_engine):
        """Test saving reasoning to file"""
        content = "Test content for persistence"
        metadata = {'memory_type': 'interaction'}
        
        reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        
        # Check if reasoning file was created
        reasoning_file = reasoning_engine.reasoning_dir / f"{reasoning.reasoning_id}.json"
        assert reasoning_file.exists()
        
        # Verify file content
        with open(reasoning_file, 'r') as f:
            saved_data = json.load(f)
            
        assert saved_data['reasoning_id'] == reasoning.reasoning_id
        assert saved_data['reasoning_type'] == reasoning.reasoning_type.value
        assert saved_data['decision_summary'] == reasoning.decision_summary
    
    def test_load_reasoning_history_from_files(self, reasoning_engine):
        """Test loading reasoning history from saved files"""
        # Create and save some reasoning
        content = "Test content for file loading"
        metadata = {'memory_type': 'interaction'}
        reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        
        # Clear cache to force loading from files
        reasoning_engine.reasoning_cache.clear()
        
        # Get history (should load from files)
        history = reasoning_engine.get_reasoning_history(days=1)
        
        assert len(history) > 0
        assert any(entry['id'] == reasoning.reasoning_id for entry in history)


class TestErrorHandling:
    """Test cases for error handling in reasoning"""
    
    def test_reasoning_with_invalid_memory_path(self, reasoning_engine):
        """Test reasoning with invalid memory file path"""
        invalid_path = "/completely/invalid/path/file.md"
        proposed_changes = {'content': 'New content'}
        
        # Should not crash, should handle gracefully
        reasoning = reasoning_engine.reason_modification_decision(
            invalid_path, proposed_changes
        )
        
        assert isinstance(reasoning, DecisionReasoning)
        assert reasoning.reasoning_type == ReasoningType.MODIFICATION
        # Should have lower confidence due to error
        assert reasoning.overall_confidence < 0.8
    
    def test_reasoning_with_corrupted_config(self, temp_memory_dir):
        """Test reasoning engine with minimal/corrupted config"""
        # Create engine with minimal config
        config = ReasoningConfig(
            enable_detailed_reasoning=False,
            max_reasoning_steps=1
        )
        engine = MemoryReasoningEngine(temp_memory_dir, config)
        
        content = "Test content"
        metadata = {'memory_type': 'interaction'}
        
        # Should still work with limited config
        reasoning = engine.reason_storage_decision(content, metadata)
        assert isinstance(reasoning, DecisionReasoning)


class TestIntegrationScenarios:
    """Test cases for integration scenarios"""
    
    def test_full_reasoning_workflow(self, reasoning_engine, sample_memory_files):
        """Test complete reasoning workflow"""
        # 1. Storage decision
        content = "User mentioned they're learning Python programming"
        metadata = {'memory_type': 'interaction', 'category': 'learning'}
        storage_reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        
        # 2. Retrieval decision
        query_context = {'keywords': ['programming', 'learning'], 'type': 'help_request'}
        available_memories = list(sample_memory_files.values())
        retrieval_reasoning = reasoning_engine.reason_retrieval_decision(
            query_context, available_memories
        )
        
        # 3. Context integration
        retrieved_memories = [{'path': sample_memory_files['profile'], 'relevance_score': 0.8}]
        current_context = {'topic': 'programming_help'}
        integration_reasoning = reasoning_engine.reason_context_integration_decision(
            retrieved_memories, current_context, "Provide programming guidance"
        )
        
        # All reasoning should be successful
        assert storage_reasoning.reasoning_type == ReasoningType.STORAGE
        assert retrieval_reasoning.reasoning_type == ReasoningType.RETRIEVAL
        assert integration_reasoning.reasoning_type == ReasoningType.CONTEXT_INTEGRATION
        
        # Should have different reasoning IDs
        reasoning_ids = {
            storage_reasoning.reasoning_id,
            retrieval_reasoning.reasoning_id,
            integration_reasoning.reasoning_id
        }
        assert len(reasoning_ids) == 3
    
    def test_reasoning_performance(self, reasoning_engine):
        """Test reasoning performance and timing"""
        content = "Performance test content"
        metadata = {'memory_type': 'interaction'}
        
        start_time = datetime.now()
        reasoning = reasoning_engine.reason_storage_decision(content, metadata)
        end_time = datetime.now()
        
        execution_time_seconds = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert execution_time_seconds < 1.0  # Less than 1 second
        assert reasoning.execution_time_ms > 0
        assert reasoning.execution_time_ms < 1000  # Less than 1000ms


if __name__ == "__main__":
    pytest.main([__file__]) 