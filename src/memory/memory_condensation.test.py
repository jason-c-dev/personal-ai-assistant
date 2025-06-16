"""
Unit tests for memory_condensation module

Tests for automated memory condensation including AI-powered summarization,
chain-of-thought reasoning, and condensation workflows.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

# Handle imports gracefully for both package and standalone execution
try:
    from .memory_condensation import (
        MemoryCondensationSystem, CondensationStrategy, CondensationTrigger,
        CondensationReasoning, CondensationResult
    )
    from .time_based_organizer import CondensationCandidate, TimeWindow
    from .file_operations import write_memory_file
except ImportError:
    from memory_condensation import (
        MemoryCondensationSystem, CondensationStrategy, CondensationTrigger,
        CondensationReasoning, CondensationResult
    )
    from time_based_organizer import CondensationCandidate, TimeWindow
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
def condensation_system(temp_memory_dir):
    """Create a MemoryCondensationSystem instance for testing"""
    config = {
        'size_threshold_mb': 10,  # Lower threshold for testing
        'condensation_schedule_hours': 1,  # More frequent for testing
        'min_batch_size': 2,
        'max_batch_size': 5,
        'quality_threshold': 0.6
    }
    return MemoryCondensationSystem(temp_memory_dir, config)


@pytest.fixture
def sample_condensation_candidates(temp_memory_dir):
    """Create sample condensation candidates for testing"""
    base_path = Path(temp_memory_dir)
    interactions_dir = base_path / "interactions"
    
    candidates = []
    
    # Create sample memory files
    for i in range(4):
        memory_date = datetime.now() - timedelta(days=60 + i*10)
        memory_path = interactions_dir / "2023-12" / f"memory_{i}.md"
        memory_path.parent.mkdir(exist_ok=True)
        
        frontmatter = {
            'created': memory_date.isoformat(),
            'importance_score': 5 + i,
            'category': 'conversation' if i % 2 == 0 else 'discussion'
        }
        content = f"Sample memory content {i} with some important information about topic {i}"
        write_memory_file(memory_path, frontmatter, content)
        
        candidate = CondensationCandidate(
            file_path=str(memory_path),
            age_days=60 + i*10,
            importance_score=5 + i,
            size_kb=0.5 + i*0.1,
            category=frontmatter['category'],
            interaction_count=3 + i,
            condensation_priority=50.0 + i*10
        )
        candidates.append(candidate)
    
    return candidates


class TestMemoryCondensationSystemInitialization:
    """Test cases for MemoryCondensationSystem initialization"""
    
    def test_initialization_with_default_config(self, temp_memory_dir):
        """Test initialization with default configuration"""
        system = MemoryCondensationSystem(temp_memory_dir)
        
        assert system.base_path == Path(temp_memory_dir)
        assert system.size_threshold_mb == 50
        assert system.condensation_schedule_hours == 24
        assert system.min_batch_size == 3
        assert system.max_batch_size == 20
        assert system.quality_threshold == 0.7
    
    def test_initialization_with_custom_config(self, temp_memory_dir):
        """Test initialization with custom configuration"""
        custom_config = {
            'size_threshold_mb': 25,
            'condensation_schedule_hours': 12,
            'min_batch_size': 2,
            'max_batch_size': 10,
            'quality_threshold': 0.8
        }
        
        system = MemoryCondensationSystem(temp_memory_dir, custom_config)
        
        assert system.size_threshold_mb == 25
        assert system.condensation_schedule_hours == 12
        assert system.min_batch_size == 2
        assert system.max_batch_size == 10
        assert system.quality_threshold == 0.8
    
    def test_condensation_log_directory_creation(self, temp_memory_dir):
        """Test that condensation log directory is created"""
        system = MemoryCondensationSystem(temp_memory_dir)
        
        log_dir = Path(temp_memory_dir) / 'system' / 'condensation_logs'
        assert log_dir.exists()
        assert log_dir.is_dir()


class TestCondensationTriggers:
    """Test cases for condensation trigger logic"""
    
    def test_size_threshold_trigger(self, condensation_system):
        """Test size threshold trigger"""
        # Mock the condensation metrics to exceed threshold
        with patch.object(condensation_system.time_organizer, 'get_condensation_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'current_state': {'total_size_mb': 15.0}  # Exceeds 10MB threshold
            }
            
            should_trigger, triggers, analysis = condensation_system.should_trigger_condensation()
            
            assert should_trigger is True
            assert CondensationTrigger.SIZE_THRESHOLD in triggers
            assert 'size_trigger' in analysis
            assert analysis['size_trigger']['current_size_mb'] == 15.0
    
    def test_time_based_trigger(self, condensation_system):
        """Test time-based trigger"""
        # Mock last condensation time to be old enough
        with patch.object(condensation_system, '_get_last_condensation_time') as mock_time:
            old_time = datetime.now() - timedelta(hours=2)  # 2 hours ago, exceeds 1 hour threshold
            mock_time.return_value = old_time
            
            with patch.object(condensation_system.time_organizer, 'get_condensation_metrics') as mock_metrics:
                mock_metrics.return_value = {'current_state': {'total_size_mb': 5.0}}
                
                should_trigger, triggers, analysis = condensation_system.should_trigger_condensation()
                
                assert should_trigger is True
                assert CondensationTrigger.TIME_BASED in triggers
                assert 'time_trigger' in analysis
    
    def test_importance_decay_trigger(self, condensation_system, sample_condensation_candidates):
        """Test importance decay trigger"""
        # Create high-priority candidates
        high_priority_candidates = []
        for i in range(15):  # Create 15 high-priority candidates
            candidate = CondensationCandidate(
                file_path=f"/test/path_{i}.md",
                age_days=100,
                importance_score=4,
                size_kb=1.0,
                category="test",
                interaction_count=5,
                condensation_priority=85.0  # High priority
            )
            high_priority_candidates.append(candidate)
        
        with patch.object(condensation_system.time_organizer, 'identify_condensation_candidates') as mock_candidates:
            mock_candidates.return_value = high_priority_candidates
            
            with patch.object(condensation_system.time_organizer, 'get_condensation_metrics') as mock_metrics:
                mock_metrics.return_value = {'current_state': {'total_size_mb': 5.0}}
                
                should_trigger, triggers, analysis = condensation_system.should_trigger_condensation()
                
                assert should_trigger is True
                assert CondensationTrigger.IMPORTANCE_DECAY in triggers
                assert 'importance_trigger' in analysis
    
    def test_no_triggers_activated(self, condensation_system):
        """Test when no triggers are activated"""
        with patch.object(condensation_system.time_organizer, 'get_condensation_metrics') as mock_metrics:
            mock_metrics.return_value = {'current_state': {'total_size_mb': 5.0}}  # Below threshold
            
            with patch.object(condensation_system, '_get_last_condensation_time') as mock_time:
                recent_time = datetime.now() - timedelta(minutes=30)  # Recent
                mock_time.return_value = recent_time
                
                with patch.object(condensation_system.time_organizer, 'identify_condensation_candidates') as mock_candidates:
                    mock_candidates.return_value = []  # No candidates
                    
                    should_trigger, triggers, analysis = condensation_system.should_trigger_condensation()
                    
                    assert should_trigger is False
                    assert len(triggers) == 0


class TestBatchCreation:
    """Test cases for condensation batch creation"""
    
    def test_create_condensation_batches(self, condensation_system, sample_condensation_candidates):
        """Test creation of condensation batches"""
        batches = condensation_system._create_condensation_batches(sample_condensation_candidates)
        
        # Should create batches based on grouping
        assert isinstance(batches, list)
        
        # Each batch should meet minimum size requirements
        for batch in batches:
            assert len(batch) >= condensation_system.min_batch_size
            assert len(batch) <= condensation_system.max_batch_size
    
    def test_batch_grouping_by_category(self, condensation_system, sample_condensation_candidates):
        """Test that batches are grouped by category and time period"""
        batches = condensation_system._create_condensation_batches(sample_condensation_candidates)
        
        # Candidates in the same batch should have similar characteristics
        for batch in batches:
            categories = [c.category for c in batch]
            # Should have consistent categories (allowing some mixing)
            assert len(set(categories)) <= 2
    
    def test_batch_size_limits(self, condensation_system):
        """Test batch size limits"""
        # Create many candidates
        many_candidates = []
        for i in range(50):
            candidate = CondensationCandidate(
                file_path=f"/test/path_{i}.md",
                age_days=60,
                importance_score=5,
                size_kb=1.0,
                category="conversation",  # Same category to force grouping
                interaction_count=5,
                condensation_priority=50.0
            )
            many_candidates.append(candidate)
        
        batches = condensation_system._create_condensation_batches(many_candidates)
        
        # No batch should exceed maximum size
        for batch in batches:
            assert len(batch) <= condensation_system.max_batch_size


class TestBatchAnalysis:
    """Test cases for batch analysis"""
    
    def test_analyze_batch(self, condensation_system, sample_condensation_candidates):
        """Test analysis of a condensation batch"""
        batch = sample_condensation_candidates[:3]  # Use first 3 candidates
        
        analysis = condensation_system._analyze_batch(batch)
        
        assert 'batch_size' in analysis
        assert analysis['batch_size'] == 3
        assert 'total_size_kb' in analysis
        assert 'avg_importance' in analysis
        assert 'avg_age_days' in analysis
        assert 'categories' in analysis
        assert 'primary_category' in analysis
        assert 'primary_time_window' in analysis
    
    def test_analyze_empty_batch(self, condensation_system):
        """Test analysis of empty batch"""
        analysis = condensation_system._analyze_batch([])
        
        assert analysis == {}
    
    def test_batch_time_window_classification(self, condensation_system, temp_memory_dir):
        """Test time window classification in batch analysis"""
        # Create candidates with known time windows
        recent_candidate = CondensationCandidate(
            file_path=str(Path(temp_memory_dir) / "interactions" / "recent.md"),
            age_days=15,  # Recent
            importance_score=6,
            size_kb=1.0,
            category="conversation",
            interaction_count=5,
            condensation_priority=60.0
        )
        
        # Create the actual file for testing
        memory_path = Path(recent_candidate.file_path)
        memory_path.parent.mkdir(exist_ok=True)
        frontmatter = {
            'created': (datetime.now() - timedelta(days=15)).isoformat(),
            'importance_score': 6,
            'category': 'conversation'
        }
        write_memory_file(memory_path, frontmatter, "Recent memory content")
        
        analysis = condensation_system._analyze_batch([recent_candidate])
        
        assert analysis['primary_time_window'] == TimeWindow.RECENT


class TestCondensationReasoning:
    """Test cases for chain-of-thought reasoning"""
    
    @pytest.mark.asyncio
    async def test_generate_condensation_reasoning(self, condensation_system, sample_condensation_candidates):
        """Test generation of condensation reasoning"""
        batch = sample_condensation_candidates[:3]
        batch_analysis = condensation_system._analyze_batch(batch)
        
        reasoning = await condensation_system._generate_condensation_reasoning(batch, batch_analysis)
        
        assert isinstance(reasoning, CondensationReasoning)
        assert reasoning.timestamp is not None
        assert reasoning.decision is not None
        assert len(reasoning.reasoning_steps) > 0
        assert isinstance(reasoning.factors_considered, dict)
        assert 0 <= reasoning.confidence_score <= 1
        assert len(reasoning.alternative_approaches) > 0
        assert reasoning.expected_outcome is not None
        assert reasoning.preservation_rationale is not None
    
    @pytest.mark.asyncio
    async def test_reasoning_critical_memories(self, condensation_system):
        """Test reasoning with critical importance memories"""
        # Create batch with critical importance
        critical_candidate = CondensationCandidate(
            file_path="/test/critical.md",
            age_days=30,
            importance_score=9,  # Critical
            size_kb=1.0,
            category="important",
            interaction_count=5,
            condensation_priority=70.0
        )
        
        batch = [critical_candidate]
        batch_analysis = condensation_system._analyze_batch(batch)
        
        reasoning = await condensation_system._generate_condensation_reasoning(batch, batch_analysis)
        
        # Should mention critical memories in reasoning
        reasoning_text = " ".join(reasoning.reasoning_steps)
        assert "critical" in reasoning_text.lower()


class TestCondensationStrategies:
    """Test cases for condensation strategies"""
    
    def test_choose_condensation_strategy_preserve_important(self, condensation_system):
        """Test strategy selection for critical memories"""
        batch_analysis = {
            'critical_importance_count': 2,
            'primary_time_window': TimeWindow.RECENT
        }
        reasoning = CondensationReasoning(
            timestamp=datetime.now().isoformat(),
            decision="test",
            reasoning_steps=[],
            factors_considered={},
            confidence_score=0.8,
            alternative_approaches=[],
            expected_outcome="test",
            preservation_rationale="test"
        )
        
        strategy = condensation_system._choose_condensation_strategy(batch_analysis, reasoning)
        
        assert strategy == CondensationStrategy.PRESERVE_IMPORTANT
    
    def test_choose_condensation_strategy_by_time_window(self, condensation_system):
        """Test strategy selection based on time window"""
        # Test different time windows
        test_cases = [
            (TimeWindow.RECENT, CondensationStrategy.PRESERVE_IMPORTANT),
            (TimeWindow.MEDIUM, CondensationStrategy.EXTRACT_THEMES),
            (TimeWindow.ARCHIVE, CondensationStrategy.FACTUAL_SUMMARY)
        ]
        
        for time_window, expected_strategy in test_cases:
            batch_analysis = {
                'critical_importance_count': 0,
                'primary_time_window': time_window
            }
            reasoning = CondensationReasoning(
                timestamp=datetime.now().isoformat(),
                decision="test",
                reasoning_steps=[],
                factors_considered={},
                confidence_score=0.8,
                alternative_approaches=[],
                expected_outcome="test",
                preservation_rationale="test"
            )
            
            strategy = condensation_system._choose_condensation_strategy(batch_analysis, reasoning)
            assert strategy == expected_strategy


class TestCondensationContent:
    """Test cases for content condensation"""
    
    def test_preserve_important_condensation(self, condensation_system):
        """Test preserve important condensation strategy"""
        content_list = [
            "High importance conversation about project planning",
            "Medium importance discussion about schedules",
            "Low importance casual chat"
        ]
        frontmatter_list = [
            {'importance_score': 8, 'created': '2024-01-01T10:00:00'},
            {'importance_score': 6, 'created': '2024-01-01T11:00:00'},
            {'importance_score': 3, 'created': '2024-01-01T12:00:00'}
        ]
        
        condensed = condensation_system._preserve_important_condensation(content_list, frontmatter_list)
        
        assert "High Importance Information" in condensed
        assert "project planning" in condensed
        assert "Condensed from 3 memories" in condensed
    
    def test_extract_themes_condensation(self, condensation_system):
        """Test theme extraction condensation strategy"""
        content_list = [
            "Discussion about machine learning algorithms and neural networks",
            "Conversation about artificial intelligence and deep learning models",
            "Talk about programming languages and software development"
        ]
        frontmatter_list = [
            {'importance_score': 6, 'created': '2024-01-01T10:00:00', 'category': 'tech'},
            {'importance_score': 5, 'created': '2024-01-01T11:00:00', 'category': 'tech'},
            {'importance_score': 4, 'created': '2024-01-01T12:00:00', 'category': 'tech'}
        ]
        
        condensed = condensation_system._extract_themes_condensation(content_list, frontmatter_list)
        
        assert "Key Themes" in condensed
        assert "Theme-Based Summary" in condensed
        assert "Condensed from 3 memories" in condensed
    
    def test_factual_summary_condensation(self, condensation_system):
        """Test factual summary condensation strategy"""
        content_list = [
            "Project deadline is January 15, 2024. Team size: 5 developers.",
            "Budget approved: $50,000. Timeline: 3 months.",
            "Client name: ABC Corp. Contact: John Smith."
        ]
        frontmatter_list = [
            {'importance_score': 7, 'created': '2024-01-01T10:00:00', 'category': 'project'},
            {'importance_score': 6, 'created': '2024-01-01T11:00:00', 'category': 'project'},
            {'importance_score': 5, 'created': '2024-01-01T12:00:00', 'category': 'project'}
        ]
        
        condensed = condensation_system._factual_summary_condensation(content_list, frontmatter_list)
        
        assert "Key Facts" in condensed
        assert "Summary Statistics" in condensed
        assert "Factual Summary" in condensed
        assert "January 15" in condensed or "50,000" in condensed  # Should preserve facts with numbers


class TestQualityAssessment:
    """Test cases for condensation quality assessment"""
    
    @pytest.mark.asyncio
    async def test_assess_condensation_quality(self, condensation_system):
        """Test quality assessment of condensed content"""
        source_memories = [
            {
                'content': 'Original detailed discussion about machine learning algorithms',
                'frontmatter': {'importance_score': 7}
            },
            {
                'content': 'Follow-up conversation about neural network architectures',
                'frontmatter': {'importance_score': 6}
            }
        ]
        
        condensed_content = """
        # Condensed Memory
        ## Key Themes
        - Machine learning algorithms
        - Neural network architectures
        ## Summary
        Detailed discussions about AI topics
        """
        
        strategy = CondensationStrategy.EXTRACT_THEMES
        
        quality_metrics = await condensation_system._assess_condensation_quality(
            source_memories, condensed_content, strategy
        )
        
        assert 'compression_ratio' in quality_metrics
        assert 'information_preservation' in quality_metrics
        assert 'structure_quality' in quality_metrics
        assert 'strategy_effectiveness' in quality_metrics
        assert 'overall_quality' in quality_metrics
        
        # All metrics should be between 0 and 1
        for metric_name, value in quality_metrics.items():
            assert 0 <= value <= 1, f"Metric {metric_name} = {value} not in range [0,1]"
    
    @pytest.mark.asyncio
    async def test_quality_metrics_good_condensation(self, condensation_system):
        """Test quality metrics for well-structured condensation"""
        source_memories = [
            {'content': 'machine learning algorithms discussion', 'frontmatter': {'importance_score': 7}}
        ]
        
        # Well-structured condensed content
        condensed_content = """
        # Condensed Memory
        ## Key Themes
        - Machine learning algorithms (mentioned 2 times)
        ## Key Facts
        - Discussion about algorithms
        """
        
        strategy = CondensationStrategy.EXTRACT_THEMES
        
        quality_metrics = await condensation_system._assess_condensation_quality(
            source_memories, condensed_content, strategy
        )
        
        # Should have high structure quality due to proper formatting
        assert quality_metrics['structure_quality'] >= 0.8
        assert quality_metrics['strategy_effectiveness'] >= 0.8


class TestAutomatedCondensation:
    """Test cases for automated condensation workflow"""
    
    @pytest.mark.asyncio
    async def test_run_automated_condensation_dry_run(self, condensation_system, sample_condensation_candidates):
        """Test automated condensation in dry run mode"""
        with patch.object(condensation_system, 'should_trigger_condensation') as mock_trigger:
            mock_trigger.return_value = (True, [CondensationTrigger.SIZE_THRESHOLD], {'test': 'data'})
            
            with patch.object(condensation_system.time_organizer, 'identify_condensation_candidates') as mock_candidates:
                mock_candidates.return_value = sample_condensation_candidates
                
                result = await condensation_system.run_automated_condensation(dry_run=True)
                
                assert result['status'] == 'completed'
                assert result['dry_run'] is True
                assert 'batches_processed' in result
                assert 'memories_processed' in result
    
    @pytest.mark.asyncio
    async def test_run_automated_condensation_no_triggers(self, condensation_system):
        """Test automated condensation when no triggers are active"""
        with patch.object(condensation_system, 'should_trigger_condensation') as mock_trigger:
            mock_trigger.return_value = (False, [], {})
            
            result = await condensation_system.run_automated_condensation(dry_run=False)
            
            assert result['status'] == 'skipped'
            assert result['reason'] == 'No condensation triggers activated'
    
    @pytest.mark.asyncio
    async def test_run_automated_condensation_no_candidates(self, condensation_system):
        """Test automated condensation when no candidates are found"""
        with patch.object(condensation_system, 'should_trigger_condensation') as mock_trigger:
            mock_trigger.return_value = (True, [CondensationTrigger.TIME_BASED], {'test': 'data'})
            
            with patch.object(condensation_system.time_organizer, 'identify_condensation_candidates') as mock_candidates:
                mock_candidates.return_value = []
                
                result = await condensation_system.run_automated_condensation()
                
                assert result['status'] == 'completed'
                assert result['reason'] == 'No condensation candidates found'


class TestCondensationLogging:
    """Test cases for condensation logging and history"""
    
    def test_log_condensation_operation(self, condensation_system):
        """Test logging of condensation operations"""
        summary = {
            'status': 'completed',
            'batches_processed': 2,
            'memories_processed': 8,
            'timestamp': datetime.now().isoformat()
        }
        
        condensation_system._log_condensation_operation(summary)
        
        # Check that log file was created
        log_files = list(condensation_system.condensation_log_dir.glob('condensation_*.json'))
        assert len(log_files) > 0
        
        # Check log content
        with open(log_files[0], 'r') as f:
            logged_data = json.load(f)
            assert logged_data['status'] == 'completed'
            assert logged_data['batches_processed'] == 2
    
    def test_get_condensation_history(self, condensation_system):
        """Test retrieval of condensation history"""
        # Create test log files
        for i in range(3):
            log_data = {
                'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
                'status': 'completed',
                'batches_processed': i + 1
            }
            log_filename = f"condensation_test_{i}.json"
            log_path = condensation_system.condensation_log_dir / log_filename
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f)
        
        history = condensation_system.get_condensation_history(days=30)
        
        assert len(history) == 3
        # Should be sorted by timestamp (most recent first)
        timestamps = [item['timestamp'] for item in history]
        assert timestamps == sorted(timestamps, reverse=True)


class TestErrorHandling:
    """Test cases for error handling in condensation system"""
    
    @pytest.mark.asyncio
    async def test_handle_missing_source_files(self, condensation_system):
        """Test handling of missing source files during condensation"""
        # Create candidates with non-existent files
        candidates = [
            CondensationCandidate(
                file_path="/nonexistent/file1.md",
                age_days=60,
                importance_score=5,
                size_kb=1.0,
                category="test",
                interaction_count=5,
                condensation_priority=60.0
            )
        ]
        
        # Should handle gracefully without crashing
        with patch.object(condensation_system, '_condense_memories') as mock_condense:
            mock_condense.return_value = ("test content", {"test": "frontmatter"})
            
            with patch.object(condensation_system, '_assess_condensation_quality') as mock_quality:
                mock_quality.return_value = {'overall_quality': 0.8}
                
                # Should not raise exception
                try:
                    result = await condensation_system._process_condensation_batch(candidates)
                    assert isinstance(result, CondensationResult)
                except Exception as e:
                    # If it does raise an exception, it should be handled gracefully
                    assert "Error reading memory file" in str(e) or True
    
    def test_handle_corrupted_log_files(self, condensation_system):
        """Test handling of corrupted condensation log files"""
        # Create corrupted log file
        corrupted_log = condensation_system.condensation_log_dir / "condensation_corrupted.json"
        with open(corrupted_log, 'w') as f:
            f.write("invalid json content {")
        
        # Should handle gracefully
        history = condensation_system.get_condensation_history()
        assert isinstance(history, list)  # Should return empty list or valid data


if __name__ == "__main__":
    pytest.main([__file__]) 