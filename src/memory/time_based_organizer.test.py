"""
Unit tests for time_based_organizer module

Tests for intelligent time-based memory organization including
memory classification, condensation, and archival strategies.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from time_based_organizer import (
    TimeBasedOrganizer, TimeWindow, MemoryTimeMetrics, 
    CondensationCandidate
)
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
def organizer(temp_memory_dir):
    """Create a TimeBasedOrganizer instance for testing"""
    return TimeBasedOrganizer(temp_memory_dir)


@pytest.fixture
def sample_memories(temp_memory_dir):
    """Create sample memory files for testing"""
    base_path = Path(temp_memory_dir)
    interactions_dir = base_path / "interactions"
    
    # Create memories of different ages
    memories = []
    
    # Recent memory (5 days old)
    recent_date = datetime.now() - timedelta(days=5)
    recent_path = interactions_dir / "2024-01" / "recent_memory.md"
    recent_path.parent.mkdir(exist_ok=True)
    recent_frontmatter = {
        'created': recent_date.isoformat(),
        'importance_score': 6,
        'category': 'conversation'
    }
    write_memory_file(recent_path, recent_frontmatter, "Recent conversation about Python")
    memories.append(('recent', recent_path, recent_frontmatter))
    
    # Medium memory (60 days old)
    medium_date = datetime.now() - timedelta(days=60)
    medium_path = interactions_dir / "2023-12" / "medium_memory.md"
    medium_path.parent.mkdir(exist_ok=True)
    medium_frontmatter = {
        'created': medium_date.isoformat(),
        'importance_score': 5,
        'category': 'discussion'
    }
    write_memory_file(medium_path, medium_frontmatter, "Discussion about AI development")
    memories.append(('medium', medium_path, medium_frontmatter))
    
    # Archive memory (200 days old)
    archive_date = datetime.now() - timedelta(days=200)
    archive_path = interactions_dir / "2023-06" / "archive_memory.md"
    archive_path.parent.mkdir(exist_ok=True)
    archive_frontmatter = {
        'created': archive_date.isoformat(),
        'importance_score': 4,
        'category': 'casual'
    }
    write_memory_file(archive_path, archive_frontmatter, "Old casual conversation")
    memories.append(('archive', archive_path, archive_frontmatter))
    
    return memories


class TestTimeBasedOrganizerInitialization:
    """Test cases for TimeBasedOrganizer initialization"""
    
    def test_initialization_with_default_config(self, temp_memory_dir):
        """Test initialization with default configuration"""
        organizer = TimeBasedOrganizer(temp_memory_dir)
        
        assert organizer.base_path == Path(temp_memory_dir)
        assert organizer.recent_days == 30
        assert organizer.medium_days == 180
        assert organizer.critical_threshold == 9
        assert organizer.high_threshold == 7
        assert organizer.medium_threshold == 4
    
    def test_initialization_with_custom_config(self, temp_memory_dir):
        """Test initialization with custom configuration"""
        custom_config = {
            'recent_days': 14,
            'medium_days': 90,
            'critical_threshold': 8,
            'high_threshold': 6,
            'condensation_batch_size': 20
        }
        
        organizer = TimeBasedOrganizer(temp_memory_dir, custom_config)
        
        assert organizer.recent_days == 14
        assert organizer.medium_days == 90
        assert organizer.critical_threshold == 8
        assert organizer.high_threshold == 6
        assert organizer.condensation_batch_size == 20
    
    def test_condensed_structure_creation(self, temp_memory_dir):
        """Test that condensed directory structure is created"""
        organizer = TimeBasedOrganizer(temp_memory_dir)
        
        condensed_dir = Path(temp_memory_dir) / "condensed"
        assert (condensed_dir / "recent").exists()
        assert (condensed_dir / "medium").exists()
        assert (condensed_dir / "archive").exists()


class TestMemoryClassification:
    """Test cases for memory classification by age"""
    
    def test_classify_recent_memory(self, organizer):
        """Test classification of recent memories"""
        recent_date = datetime.now() - timedelta(days=15)
        
        time_window = organizer.classify_memory_by_age(recent_date)
        
        assert time_window == TimeWindow.RECENT
    
    def test_classify_medium_memory(self, organizer):
        """Test classification of medium-age memories"""
        medium_date = datetime.now() - timedelta(days=90)
        
        time_window = organizer.classify_memory_by_age(medium_date)
        
        assert time_window == TimeWindow.MEDIUM
    
    def test_classify_archive_memory(self, organizer):
        """Test classification of archive memories"""
        archive_date = datetime.now() - timedelta(days=300)
        
        time_window = organizer.classify_memory_by_age(archive_date)
        
        assert time_window == TimeWindow.ARCHIVE
    
    def test_classify_boundary_cases(self, organizer):
        """Test classification at time window boundaries"""
        # Exactly at recent boundary (30 days)
        boundary_recent = datetime.now() - timedelta(days=30)
        assert organizer.classify_memory_by_age(boundary_recent) == TimeWindow.RECENT
        
        # Just past recent boundary (31 days)
        past_recent = datetime.now() - timedelta(days=31)
        assert organizer.classify_memory_by_age(past_recent) == TimeWindow.MEDIUM
        
        # Exactly at medium boundary (180 days)
        boundary_medium = datetime.now() - timedelta(days=180)
        assert organizer.classify_memory_by_age(boundary_medium) == TimeWindow.MEDIUM
        
        # Just past medium boundary (181 days)
        past_medium = datetime.now() - timedelta(days=181)
        assert organizer.classify_memory_by_age(past_medium) == TimeWindow.ARCHIVE


class TestMemoryAgeDistribution:
    """Test cases for memory age distribution analysis"""
    
    def test_get_memory_age_distribution_empty(self, organizer):
        """Test age distribution with no memories"""
        distribution = organizer.get_memory_age_distribution()
        
        assert len(distribution) == 3
        for window in [TimeWindow.RECENT.value, TimeWindow.MEDIUM.value, TimeWindow.ARCHIVE.value]:
            assert window in distribution
            assert distribution[window].total_memories == 0
            assert distribution[window].memory_size_mb == 0.0
    
    def test_get_memory_age_distribution_with_memories(self, organizer, sample_memories):
        """Test age distribution with sample memories"""
        distribution = organizer.get_memory_age_distribution()
        
        # Should have memories in all three time windows
        assert distribution[TimeWindow.RECENT.value].total_memories >= 1
        assert distribution[TimeWindow.MEDIUM.value].total_memories >= 1
        assert distribution[TimeWindow.ARCHIVE.value].total_memories >= 1
        
        # Should have valid metrics
        for window, metrics in distribution.items():
            if metrics.total_memories > 0:
                assert metrics.average_age_days >= 0
                assert metrics.oldest_memory_days >= 0
                assert metrics.memory_size_mb >= 0
    
    def test_memory_size_calculation(self, organizer, sample_memories):
        """Test that memory sizes are calculated correctly"""
        distribution = organizer.get_memory_age_distribution()
        
        total_size = sum(metrics.memory_size_mb for metrics in distribution.values())
        assert total_size > 0  # Should have some size from sample memories


class TestCondensationCandidates:
    """Test cases for condensation candidate identification"""
    
    def test_identify_condensation_candidates_empty(self, organizer):
        """Test candidate identification with no memories"""
        candidates = organizer.identify_condensation_candidates()
        
        assert len(candidates) == 0
    
    def test_identify_condensation_candidates_with_memories(self, organizer, sample_memories):
        """Test candidate identification with sample memories"""
        candidates = organizer.identify_condensation_candidates()
        
        # Should find some candidates (depends on importance scores and ages)
        assert isinstance(candidates, list)
        
        for candidate in candidates:
            assert isinstance(candidate, CondensationCandidate)
            assert candidate.file_path is not None
            assert candidate.age_days >= 0
            assert 1 <= candidate.importance_score <= 10
            assert candidate.size_kb >= 0
            assert candidate.condensation_priority >= 0
    
    def test_filter_candidates_by_time_window(self, organizer, sample_memories):
        """Test filtering candidates by specific time window"""
        medium_candidates = organizer.identify_condensation_candidates(
            time_window=TimeWindow.MEDIUM
        )
        
        for candidate in medium_candidates:
            # Verify the candidate is actually in the medium time window
            file_path = Path(candidate.file_path)
            if file_path.exists():
                from file_operations import read_memory_file
                frontmatter, _ = read_memory_file(file_path)
                created_str = frontmatter.get('created', '')
                if created_str:
                    created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    window = organizer.classify_memory_by_age(created_date)
                    assert window == TimeWindow.MEDIUM
    
    def test_filter_candidates_by_importance(self, organizer, sample_memories):
        """Test filtering candidates by minimum importance"""
        high_importance_candidates = organizer.identify_condensation_candidates(
            min_importance=7
        )
        
        # Should have fewer candidates when filtering for high importance
        all_candidates = organizer.identify_condensation_candidates()
        assert len(high_importance_candidates) <= len(all_candidates)
    
    def test_candidate_priority_calculation(self, organizer):
        """Test condensation priority calculation"""
        # Test various scenarios
        test_cases = [
            # (age_days, importance_score, size_kb, interaction_count, expected_range)
            (365, 3, 1024, 10, (60, 100)),  # Old, low importance, large, many interactions
            (7, 9, 100, 2, (0, 30)),        # Recent, high importance, small, few interactions
            (90, 5, 500, 5, (20, 80)),      # Medium age, medium importance
        ]
        
        for age_days, importance, size_kb, interaction_count, (min_priority, max_priority) in test_cases:
            priority = organizer._calculate_condensation_priority(
                age_days, importance, size_kb, interaction_count
            )
            
            assert 0 <= priority <= 100
            assert min_priority <= priority <= max_priority
    
    def test_candidates_sorted_by_priority(self, organizer, sample_memories):
        """Test that candidates are returned sorted by priority"""
        candidates = organizer.identify_condensation_candidates()
        
        if len(candidates) > 1:
            # Should be sorted by priority (highest first)
            for i in range(len(candidates) - 1):
                assert candidates[i].condensation_priority >= candidates[i + 1].condensation_priority


class TestMemoryOrganization:
    """Test cases for memory organization by time"""
    
    def test_organize_memories_dry_run(self, organizer, sample_memories):
        """Test memory organization in dry run mode"""
        summary = organizer.organize_memories_by_time(dry_run=True)
        
        assert summary['dry_run'] is True
        assert 'analyzed_files' in summary
        assert 'condensation_candidates' in summary
        assert 'errors' in summary
        assert isinstance(summary['errors'], list)
    
    def test_organize_memories_no_interactions(self, organizer):
        """Test organization with no interaction files"""
        summary = organizer.organize_memories_by_time(dry_run=True)
        
        assert summary['analyzed_files'] == 0
        assert summary['condensation_candidates'] == 0
        assert summary['moved_to_recent'] == 0
        assert summary['moved_to_medium'] == 0
        assert summary['moved_to_archive'] == 0
    
    @patch('time_based_organizer.TimeBasedOrganizer._create_condensed_summaries')
    def test_organize_memories_actual_run(self, mock_create_summaries, organizer, sample_memories):
        """Test actual memory organization (not dry run)"""
        summary = organizer.organize_memories_by_time(dry_run=False)
        
        assert summary['dry_run'] is False
        
        # Should have called summary creation if there were candidates
        if summary['condensation_candidates'] > 0:
            mock_create_summaries.assert_called()


class TestCondensedSummaryCreation:
    """Test cases for condensed summary creation"""
    
    def test_group_candidates_for_condensation(self, organizer, temp_memory_dir):
        """Test grouping of candidates for condensation"""
        # Create test candidates
        candidates = [
            CondensationCandidate(
                file_path=str(Path(temp_memory_dir) / "interactions" / "2024-01" / "file1.md"),
                age_days=45,
                importance_score=5,
                size_kb=10,
                category="conversation",
                interaction_count=5
            ),
            CondensationCandidate(
                file_path=str(Path(temp_memory_dir) / "interactions" / "2024-01" / "file2.md"),
                age_days=50,
                importance_score=4,
                size_kb=15,
                category="conversation",
                interaction_count=3
            ),
            CondensationCandidate(
                file_path=str(Path(temp_memory_dir) / "interactions" / "2024-02" / "file3.md"),
                age_days=30,
                importance_score=6,
                size_kb=8,
                category="discussion",
                interaction_count=4
            )
        ]
        
        groups = organizer._group_candidates_for_condensation(candidates)
        
        # Should group by category and date
        assert isinstance(groups, dict)
        
        # Only groups with sufficient candidates should be included
        for group_key, group_candidates in groups.items():
            assert len(group_candidates) >= organizer.min_interactions_for_condensation
    
    def test_condense_memory_group_recent(self, organizer, temp_memory_dir):
        """Test condensation of recent memory group"""
        # Create test content
        content_list = [
            {
                'content': 'Important discussion about Python programming',
                'importance': 8,
                'date': '2024-01-15T10:00:00',
                'category': 'programming'
            },
            {
                'content': 'Follow-up questions about Python best practices',
                'importance': 6,
                'date': '2024-01-15T11:00:00',
                'category': 'programming'
            }
        ]
        
        summary = organizer._create_recent_summary(content_list)
        
        assert "Recent Memory Summary" in summary
        assert "Condensed from 2 interactions" in summary
        assert "Important discussion about Python" in summary
    
    def test_condense_memory_group_medium(self, organizer):
        """Test condensation of medium memory group"""
        content_list = [
            {
                'content': 'Detailed discussion about machine learning algorithms',
                'importance': 7,
                'date': '2023-12-15T10:00:00',
                'category': 'ai'
            },
            {
                'content': 'Questions about neural network architecture',
                'importance': 5,
                'date': '2023-12-15T11:00:00',
                'category': 'ai'
            }
        ]
        
        summary = organizer._create_medium_summary(content_list)
        
        assert "Medium-Term Memory Summary" in summary
        assert "Condensed from 2 interactions" in summary
        assert "Important Facts" in summary or "Key Themes" in summary
    
    def test_condense_memory_group_archive(self, organizer):
        """Test condensation of archive memory group"""
        content_list = [
            {
                'content': 'Critical information about user preferences',
                'importance': 9,
                'date': '2023-06-15T10:00:00',
                'category': 'profile'
            },
            {
                'content': 'General conversation about weather',
                'importance': 3,
                'date': '2023-06-15T11:00:00',
                'category': 'casual'
            }
        ]
        
        summary = organizer._create_archive_summary(content_list)
        
        assert "Archived Memory Summary" in summary
        assert "Condensed from 2 interactions" in summary
        # Archive should focus on critical information only
        if any(item['importance'] >= organizer.critical_threshold for item in content_list):
            assert "Critical Information" in summary


class TestCondensationMetrics:
    """Test cases for condensation metrics and recommendations"""
    
    def test_get_condensation_metrics_empty(self, organizer):
        """Test condensation metrics with no memories"""
        metrics = organizer.get_condensation_metrics()
        
        assert 'current_state' in metrics
        assert 'condensation_opportunities' in metrics
        
        current_state = metrics['current_state']
        assert current_state['total_memories'] == 0
        assert current_state['total_size_mb'] == 0
        assert 'age_distribution' in current_state
        
        opportunities = metrics['condensation_opportunities']
        assert opportunities['total_candidates'] == 0
        assert opportunities['potential_savings_mb'] == 0
        assert 'recommended_action' in opportunities
    
    def test_get_condensation_metrics_with_memories(self, organizer, sample_memories):
        """Test condensation metrics with sample memories"""
        metrics = organizer.get_condensation_metrics()
        
        current_state = metrics['current_state']
        assert current_state['total_memories'] > 0
        assert current_state['total_size_mb'] > 0
        
        opportunities = metrics['condensation_opportunities']
        assert opportunities['total_candidates'] >= 0
        assert opportunities['potential_savings_mb'] >= 0
    
    def test_condensation_recommendations(self, organizer):
        """Test condensation recommendations"""
        # Test various scenarios
        test_cases = [
            (150, 60, "URGENT"),  # Large size, many candidates
            (50, 30, "HIGH"),     # Normal size, many candidates
            (20, 15, "MEDIUM"),   # Small size, few candidates
            (10, 5, "LOW")        # Very small, very few candidates
        ]
        
        for total_size_mb, candidate_count, expected_level in test_cases:
            recommendation = organizer._get_condensation_recommendation(total_size_mb, candidate_count)
            
            assert isinstance(recommendation, str)
            assert expected_level in recommendation


class TestErrorHandling:
    """Test cases for error handling in time-based organization"""
    
    def test_invalid_memory_file_handling(self, organizer, temp_memory_dir):
        """Test handling of invalid memory files"""
        # Create an invalid memory file (not proper markdown with frontmatter)
        interactions_dir = Path(temp_memory_dir) / "interactions"
        invalid_file = interactions_dir / "invalid.md"
        invalid_file.parent.mkdir(exist_ok=True)
        
        with open(invalid_file, 'w') as f:
            f.write("This is not a valid memory file format")
        
        # Should handle gracefully and not crash
        candidates = organizer.identify_condensation_candidates()
        assert isinstance(candidates, list)
        
        distribution = organizer.get_memory_age_distribution()
        assert isinstance(distribution, dict)
    
    def test_missing_created_date_handling(self, organizer, temp_memory_dir):
        """Test handling of memory files without created date"""
        interactions_dir = Path(temp_memory_dir) / "interactions"
        no_date_file = interactions_dir / "no_date.md"
        no_date_file.parent.mkdir(exist_ok=True)
        
        # Create file with frontmatter but no created date
        frontmatter = {
            'importance_score': 5,
            'category': 'test'
        }
        write_memory_file(no_date_file, frontmatter, "Test content")
        
        # Should handle gracefully
        candidates = organizer.identify_condensation_candidates()
        assert isinstance(candidates, list)


if __name__ == "__main__":
    pytest.main([__file__]) 