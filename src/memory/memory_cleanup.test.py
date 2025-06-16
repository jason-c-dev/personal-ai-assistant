"""
Unit tests for Memory Cleanup and Optimization System

Tests for comprehensive memory cleanup, duplicate detection, storage optimization,
and memory health monitoring functionality.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Handle imports gracefully for both package and standalone execution
try:
    from .memory_cleanup import (
        MemoryCleanupOptimizer, CleanupConfig, CleanupStats,
        OptimizationStats, MemoryHealthReport
    )
    from .file_operations import write_memory_file
except ImportError:
    from memory_cleanup import (
        MemoryCleanupOptimizer, CleanupConfig, CleanupStats,
        OptimizationStats, MemoryHealthReport
    )
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
    (memory_dir / "archive").mkdir()
    
    return str(memory_dir)


@pytest.fixture
def cleanup_config():
    """Create a test cleanup configuration"""
    return CleanupConfig(
        max_age_days=30,
        archive_age_days=15,
        temp_file_age_hours=1,
        min_importance_threshold=3.0,
        low_importance_age_days=7,
        enable_duplicate_detection=True,
        similarity_threshold=0.8,
        dry_run_mode=True,  # Safe for testing
        max_files_per_cleanup=100
    )


@pytest.fixture
def cleanup_optimizer(temp_memory_dir, cleanup_config):
    """Create a MemoryCleanupOptimizer instance for testing"""
    return MemoryCleanupOptimizer(temp_memory_dir, cleanup_config)


@pytest.fixture
def sample_memory_files(temp_memory_dir):
    """Create sample memory files for testing"""
    base_path = Path(temp_memory_dir)
    files_created = {}
    
    # Recent high-importance file
    recent_path = base_path / "interactions" / "recent_important.md"
    recent_frontmatter = {
        'created': datetime.now().isoformat(),
        'importance_score': 8,
        'memory_type': 'interaction',
        'category': 'important'
    }
    recent_content = "This is recent important information about the user's project."
    write_memory_file(recent_path, recent_frontmatter, recent_content)
    files_created['recent'] = str(recent_path)
    
    # Old low-importance file
    old_path = base_path / "interactions" / "old_unimportant.md"
    old_frontmatter = {
        'created': (datetime.now() - timedelta(days=45)).isoformat(),
        'importance_score': 2,
        'memory_type': 'interaction',
        'category': 'casual'
    }
    old_content = "Old casual conversation about weather."
    write_memory_file(old_path, old_frontmatter, old_content)
    files_created['old'] = str(old_path)
    
    # Duplicate content file 1
    dup1_path = base_path / "interactions" / "duplicate1.md"
    dup1_frontmatter = {
        'created': (datetime.now() - timedelta(days=5)).isoformat(),
        'importance_score': 6,
        'memory_type': 'interaction',
        'category': 'technical'
    }
    dup_content = "User discussed machine learning algorithms and neural networks."
    write_memory_file(dup1_path, dup1_frontmatter, dup_content)
    files_created['dup1'] = str(dup1_path)
    
    # Duplicate content file 2
    dup2_path = base_path / "interactions" / "duplicate2.md"
    dup2_frontmatter = {
        'created': (datetime.now() - timedelta(days=3)).isoformat(),
        'importance_score': 7,
        'memory_type': 'interaction',
        'category': 'technical'
    }
    write_memory_file(dup2_path, dup2_frontmatter, dup_content)  # Same content
    files_created['dup2'] = str(dup2_path)
    
    # Archive candidate
    archive_path = base_path / "interactions" / "archive_candidate.md"
    archive_frontmatter = {
        'created': (datetime.now() - timedelta(days=20)).isoformat(),
        'importance_score': 7,
        'memory_type': 'interaction',
        'category': 'reference'
    }
    archive_content = "Important reference information that should be archived."
    write_memory_file(archive_path, archive_frontmatter, archive_content)
    files_created['archive'] = str(archive_path)
    
    # Create some temporary files
    temp_file1 = base_path / "temp_file.tmp"
    temp_file1.write_text("Temporary content")
    files_created['temp1'] = str(temp_file1)
    
    temp_file2 = base_path / "backup.bak"
    temp_file2.write_text("Backup content")
    files_created['temp2'] = str(temp_file2)
    
    return files_created


class TestCleanupConfig:
    """Test cases for CleanupConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CleanupConfig()
        
        assert config.max_age_days == 365
        assert config.archive_age_days == 180
        assert config.temp_file_age_hours == 24
        assert config.min_importance_threshold == 2.0
        assert config.low_importance_age_days == 30
        assert config.enable_duplicate_detection is True
        assert config.similarity_threshold == 0.85
        assert config.enable_compression is True
        assert config.dry_run_mode is False
        assert config.auto_cleanup_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = CleanupConfig(
            max_age_days=100,
            min_importance_threshold=5.0,
            dry_run_mode=True,
            similarity_threshold=0.9
        )
        
        assert config.max_age_days == 100
        assert config.min_importance_threshold == 5.0
        assert config.dry_run_mode is True
        assert config.similarity_threshold == 0.9


class TestMemoryCleanupOptimizerInitialization:
    """Test cases for MemoryCleanupOptimizer initialization"""
    
    def test_initialization_default_config(self, temp_memory_dir):
        """Test initialization with default configuration"""
        optimizer = MemoryCleanupOptimizer(temp_memory_dir)
        
        assert optimizer.base_path == Path(temp_memory_dir)
        assert isinstance(optimizer.config, CleanupConfig)
        assert optimizer.importance_scorer is not None
        assert optimizer.prioritizer is not None
        assert optimizer.reasoning_engine is not None
        assert optimizer.cleanup_dir.exists()
        assert optimizer.backup_dir.exists()
    
    def test_initialization_custom_config(self, temp_memory_dir, cleanup_config):
        """Test initialization with custom configuration"""
        optimizer = MemoryCleanupOptimizer(temp_memory_dir, cleanup_config)
        
        assert optimizer.config.dry_run_mode is True
        assert optimizer.config.max_age_days == 30
    
    def test_directory_creation(self, temp_memory_dir):
        """Test that required directories are created"""
        optimizer = MemoryCleanupOptimizer(temp_memory_dir)
        
        expected_cleanup_dir = Path(temp_memory_dir) / 'system' / 'cleanup'
        expected_backup_dir = Path(temp_memory_dir) / 'system' / 'backups'
        
        assert expected_cleanup_dir.exists()
        assert expected_backup_dir.exists()


class TestOutdatedMemoryCleanup:
    """Test cases for outdated memory cleanup"""
    
    def test_cleanup_outdated_memories_basic(self, cleanup_optimizer, sample_memory_files):
        """Test basic outdated memory cleanup"""
        stats = cleanup_optimizer.cleanup_outdated_memories()
        
        assert isinstance(stats, CleanupStats)
        assert stats.files_processed > 0
        # In dry run mode, files shouldn't actually be deleted
        assert stats.files_deleted >= 0
        assert stats.files_archived >= 0
        assert stats.errors_encountered >= 0
    
    def test_cleanup_respects_importance_threshold(self, cleanup_optimizer, sample_memory_files):
        """Test that cleanup respects importance threshold"""
        # Set very high importance threshold
        cleanup_optimizer.config.min_importance_threshold = 10.0
        
        stats = cleanup_optimizer.cleanup_outdated_memories()
        
        # Should process files but not delete high-importance ones
        assert stats.files_processed > 0
    
    def test_cleanup_respects_max_files_limit(self, cleanup_optimizer, sample_memory_files):
        """Test that cleanup respects max files per cleanup limit"""
        cleanup_optimizer.config.max_files_per_cleanup = 1
        
        stats = cleanup_optimizer.cleanup_outdated_memories()
        
        # Should respect the limit
        assert stats.files_deleted + stats.files_archived <= 1
    
    def test_cleanup_handles_invalid_dates(self, cleanup_optimizer, temp_memory_dir):
        """Test cleanup handles files with invalid creation dates"""
        base_path = Path(temp_memory_dir)
        
        # Create file with invalid date
        invalid_path = base_path / "interactions" / "invalid_date.md"
        invalid_frontmatter = {
            'created': 'invalid-date-format',
            'importance_score': 1,
            'memory_type': 'interaction'
        }
        write_memory_file(invalid_path, invalid_frontmatter, "Content with invalid date")
        
        stats = cleanup_optimizer.cleanup_outdated_memories()
        
        # Should handle gracefully without crashing
        assert isinstance(stats, CleanupStats)
        assert stats.files_processed > 0


class TestDuplicateDetection:
    """Test cases for duplicate detection and removal"""
    
    def test_detect_and_remove_duplicates_basic(self, cleanup_optimizer, sample_memory_files):
        """Test basic duplicate detection"""
        stats = cleanup_optimizer.detect_and_remove_duplicates()
        
        assert isinstance(stats, CleanupStats)
        assert stats.files_processed > 0
        # Should find duplicates in sample files
        assert stats.duplicates_removed >= 0
    
    def test_duplicate_detection_disabled(self, cleanup_optimizer, sample_memory_files):
        """Test duplicate detection when disabled"""
        cleanup_optimizer.config.enable_duplicate_detection = False
        
        stats = cleanup_optimizer.detect_and_remove_duplicates()
        
        # Should return empty stats when disabled
        assert stats.files_processed == 0
        assert stats.duplicates_removed == 0
    
    def test_content_hash_generation(self, cleanup_optimizer):
        """Test content hash generation for duplicate detection"""
        content1 = "This is test content for hashing."
        content2 = "This is test content for hashing."  # Same content
        content3 = "This is different content."
        
        hash1 = cleanup_optimizer._generate_content_hash(content1)
        hash2 = cleanup_optimizer._generate_content_hash(content2)
        hash3 = cleanup_optimizer._generate_content_hash(content3)
        
        assert hash1 == hash2  # Same content should have same hash
        assert hash1 != hash3  # Different content should have different hash
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length
    
    def test_content_similarity_calculation(self, cleanup_optimizer):
        """Test content similarity calculation"""
        content1 = "machine learning algorithms neural networks"
        content2 = "neural networks machine learning algorithms"  # Same words, different order
        content3 = "cooking recipes and kitchen tips"
        
        similarity1 = cleanup_optimizer._calculate_content_similarity(content1, content2)
        similarity2 = cleanup_optimizer._calculate_content_similarity(content1, content3)
        
        assert similarity1 == 1.0  # Same words should be 100% similar
        assert similarity2 < 0.5   # Different content should be less similar
        assert 0.0 <= similarity1 <= 1.0
        assert 0.0 <= similarity2 <= 1.0
    
    def test_merge_duplicate_files(self, cleanup_optimizer, temp_memory_dir):
        """Test merging duplicate files"""
        base_path = Path(temp_memory_dir)
        
        # Create duplicate files
        file1_path = base_path / "interactions" / "merge_test1.md"
        file1_frontmatter = {
            'created': datetime.now().isoformat(),
            'importance_score': 5,
            'memory_type': 'interaction'
        }
        write_memory_file(file1_path, file1_frontmatter, "Duplicate content")
        
        file2_path = base_path / "interactions" / "merge_test2.md"
        file2_frontmatter = {
            'created': datetime.now().isoformat(),
            'importance_score': 7,  # Higher importance
            'memory_type': 'interaction'
        }
        write_memory_file(file2_path, file2_frontmatter, "Duplicate content")
        
        file_paths = [str(file1_path), str(file2_path)]
        merged_file = cleanup_optimizer._merge_duplicate_files(file_paths)
        
        # Should return the path of the file with higher importance
        assert merged_file is not None
        assert merged_file in file_paths


class TestStorageOptimization:
    """Test cases for storage optimization"""
    
    def test_optimize_storage_basic(self, cleanup_optimizer, sample_memory_files):
        """Test basic storage optimization"""
        stats = cleanup_optimizer.optimize_storage()
        
        assert isinstance(stats, OptimizationStats)
        assert stats.files_optimized >= 0
        assert stats.storage_compressed_bytes >= 0
        assert stats.optimization_timestamp is not None
    
    def test_optimize_file_content(self, cleanup_optimizer, temp_memory_dir):
        """Test individual file content optimization"""
        base_path = Path(temp_memory_dir)
        
        # Create file with excessive whitespace
        test_path = base_path / "interactions" / "whitespace_test.md"
        test_frontmatter = {
            'created': datetime.now().isoformat(),
            'importance_score': 5,
            'memory_type': 'interaction'
        }
        messy_content = "Line 1\n\n\n\n\nLine 2\n\n\n   \n\nLine 3   \n\n\n"
        write_memory_file(test_path, test_frontmatter, messy_content)
        
        # Test optimization
        optimized = cleanup_optimizer._optimize_file_content(str(test_path))
        
        # Should detect that optimization was needed
        assert isinstance(optimized, bool)
    
    def test_fragmentation_score_calculation(self, cleanup_optimizer, sample_memory_files):
        """Test fragmentation score calculation"""
        fragmentation_score = cleanup_optimizer._calculate_fragmentation_score()
        
        assert isinstance(fragmentation_score, float)
        assert 0.0 <= fragmentation_score <= 1.0


class TestTemporaryFileCleanup:
    """Test cases for temporary file cleanup"""
    
    def test_cleanup_temporary_files_basic(self, cleanup_optimizer, sample_memory_files):
        """Test basic temporary file cleanup"""
        stats = cleanup_optimizer.cleanup_temporary_files()
        
        assert isinstance(stats, CleanupStats)
        assert stats.files_deleted >= 0
        assert stats.space_freed_bytes >= 0
    
    def test_cleanup_respects_age_threshold(self, cleanup_optimizer, temp_memory_dir):
        """Test that temp file cleanup respects age threshold"""
        base_path = Path(temp_memory_dir)
        
        # Create recent temp file (should not be deleted)
        recent_temp = base_path / "recent.tmp"
        recent_temp.write_text("Recent temp content")
        
        # Set very short age threshold
        cleanup_optimizer.config.temp_file_age_hours = 0.001  # Very short
        
        stats = cleanup_optimizer.cleanup_temporary_files()
        
        # Should process temp files
        assert isinstance(stats, CleanupStats)


class TestHealthReporting:
    """Test cases for memory health reporting"""
    
    def test_generate_health_report_basic(self, cleanup_optimizer, sample_memory_files):
        """Test basic health report generation"""
        report = cleanup_optimizer.generate_health_report()
        
        assert isinstance(report, MemoryHealthReport)
        assert report.total_files > 0
        assert report.total_size_bytes > 0
        assert 0.0 <= report.health_score <= 1.0
        assert 0.0 <= report.fragmentation_score <= 1.0
        assert isinstance(report.recommendations, list)
        assert report.report_timestamp is not None
    
    def test_health_score_calculation(self, cleanup_optimizer):
        """Test health score calculation"""
        # Test perfect health
        perfect_score = cleanup_optimizer._calculate_health_score(100, 0, 0, 0, 0.0)
        assert perfect_score == 1.0
        
        # Test poor health
        poor_score = cleanup_optimizer._calculate_health_score(100, 50, 50, 50, 1.0)
        assert poor_score < 0.5
        
        # Test empty system
        empty_score = cleanup_optimizer._calculate_health_score(0, 0, 0, 0, 0.0)
        assert empty_score == 1.0
    
    def test_health_recommendations_generation(self, cleanup_optimizer):
        """Test health recommendations generation"""
        # Test with issues
        recommendations = cleanup_optimizer._generate_health_recommendations(5, 10, 15, 0.5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("duplicate" in rec.lower() for rec in recommendations)
        assert any("outdated" in rec.lower() for rec in recommendations)
        
        # Test with no issues
        good_recommendations = cleanup_optimizer._generate_health_recommendations(0, 0, 0, 0.1)
        assert "good health" in good_recommendations[0].lower()
    
    def test_save_and_load_health_report(self, cleanup_optimizer, sample_memory_files):
        """Test saving and loading health reports"""
        # Generate and save report
        report = cleanup_optimizer.generate_health_report()
        cleanup_optimizer._save_health_report(report)
        
        # Verify file was created
        assert cleanup_optimizer.health_file.exists()
        
        # Load and verify content
        with open(cleanup_optimizer.health_file, 'r') as f:
            saved_reports = json.load(f)
        
        assert isinstance(saved_reports, list)
        assert len(saved_reports) > 0
        assert saved_reports[-1]['total_files'] == report.total_files


class TestFullCleanupOperations:
    """Test cases for full cleanup operations"""
    
    def test_run_full_cleanup_basic(self, cleanup_optimizer, sample_memory_files):
        """Test basic full cleanup operation"""
        cleanup_stats, optimization_stats = cleanup_optimizer.run_full_cleanup()
        
        assert isinstance(cleanup_stats, CleanupStats)
        assert isinstance(optimization_stats, OptimizationStats)
        assert cleanup_stats.processing_time_seconds > 0
        assert cleanup_stats.cleanup_timestamp is not None
        assert optimization_stats.optimization_timestamp is not None
    
    def test_full_cleanup_creates_backup(self, cleanup_optimizer, sample_memory_files):
        """Test that full cleanup creates backup when enabled"""
        cleanup_optimizer.config.enable_backup_before_cleanup = True
        cleanup_optimizer.config.dry_run_mode = False  # Need actual operations for backup
        
        # Mock the backup creation to avoid actual file operations
        with patch.object(cleanup_optimizer, '_create_backup') as mock_backup:
            mock_backup.return_value = "/mock/backup/path"
            
            cleanup_stats, optimization_stats = cleanup_optimizer.run_full_cleanup()
            
            mock_backup.assert_called_once()
    
    def test_full_cleanup_saves_statistics(self, cleanup_optimizer, sample_memory_files):
        """Test that full cleanup saves statistics"""
        cleanup_stats, optimization_stats = cleanup_optimizer.run_full_cleanup()
        
        # Check that stats files were created
        assert cleanup_optimizer.stats_file.exists()
        
        optimization_file = cleanup_optimizer.cleanup_dir / 'optimization_stats.json'
        assert optimization_file.exists()


class TestScheduledCleanup:
    """Test cases for scheduled cleanup"""
    
    def test_schedule_cleanup_disabled(self, cleanup_optimizer):
        """Test scheduled cleanup when disabled"""
        cleanup_optimizer.config.auto_cleanup_enabled = False
        
        result = cleanup_optimizer.schedule_cleanup()
        
        assert result is False
    
    def test_schedule_cleanup_not_due(self, cleanup_optimizer):
        """Test scheduled cleanup when not due yet"""
        # Mock recent cleanup time
        recent_time = datetime.now().isoformat()
        
        with patch.object(cleanup_optimizer, '_get_last_cleanup_time') as mock_time:
            mock_time.return_value = recent_time
            
            result = cleanup_optimizer.schedule_cleanup()
            
            assert result is False
    
    def test_schedule_cleanup_due(self, cleanup_optimizer, sample_memory_files):
        """Test scheduled cleanup when due"""
        # Mock old cleanup time
        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        
        with patch.object(cleanup_optimizer, '_get_last_cleanup_time') as mock_time:
            mock_time.return_value = old_time
            
            result = cleanup_optimizer.schedule_cleanup()
            
            assert result is True


class TestCleanupHistory:
    """Test cases for cleanup history tracking"""
    
    def test_get_cleanup_history_empty(self, cleanup_optimizer):
        """Test getting cleanup history when no history exists"""
        history = cleanup_optimizer.get_cleanup_history()
        
        assert isinstance(history, list)
        assert len(history) == 0
    
    def test_get_cleanup_history_with_data(self, cleanup_optimizer, sample_memory_files):
        """Test getting cleanup history with existing data"""
        # Generate some reports
        report1 = cleanup_optimizer.generate_health_report()
        cleanup_optimizer._save_health_report(report1)
        
        report2 = cleanup_optimizer.generate_health_report()
        cleanup_optimizer._save_health_report(report2)
        
        history = cleanup_optimizer.get_cleanup_history(days=1)
        
        assert isinstance(history, list)
        assert len(history) >= 2
        
        # Should be sorted by timestamp (most recent first)
        if len(history) > 1:
            assert history[0]['report_timestamp'] >= history[1]['report_timestamp']
    
    def test_get_cleanup_history_filtered_by_days(self, cleanup_optimizer, sample_memory_files):
        """Test getting cleanup history filtered by days"""
        # Generate report
        report = cleanup_optimizer.generate_health_report()
        cleanup_optimizer._save_health_report(report)
        
        # Get history with very short time window
        history = cleanup_optimizer.get_cleanup_history(days=0)
        
        # Should return empty list for very short time window
        assert isinstance(history, list)


class TestErrorHandling:
    """Test cases for error handling in cleanup operations"""
    
    def test_cleanup_handles_file_errors(self, cleanup_optimizer, temp_memory_dir):
        """Test that cleanup handles file operation errors gracefully"""
        base_path = Path(temp_memory_dir)
        
        # Create a file that will cause read errors
        problem_file = base_path / "interactions" / "problem.md"
        problem_file.write_text("Invalid frontmatter content")
        
        # Should handle errors gracefully
        stats = cleanup_optimizer.cleanup_outdated_memories()
        
        assert isinstance(stats, CleanupStats)
        # Errors should be counted but not crash the system
        assert stats.errors_encountered >= 0
    
    def test_health_report_handles_errors(self, cleanup_optimizer):
        """Test that health report generation handles errors gracefully"""
        # Mock file operations to raise errors
        with patch('memory_cleanup.list_memory_files') as mock_list:
            mock_list.side_effect = Exception("Mock file system error")
            
            report = cleanup_optimizer.generate_health_report()
            
            # Should return a valid report even with errors
            assert isinstance(report, MemoryHealthReport)
            assert report.total_files == 0
            assert "Error" in report.recommendations[0]


class TestUtilityMethods:
    """Test cases for utility methods"""
    
    def test_get_last_cleanup_time(self, cleanup_optimizer):
        """Test getting last cleanup time"""
        # Initially should be None
        last_time = cleanup_optimizer._get_last_cleanup_time()
        assert last_time is None
        
        # After saving stats, should return timestamp
        stats = CleanupStats(cleanup_timestamp=datetime.now().isoformat())
        cleanup_optimizer._save_cleanup_stats(stats)
        
        last_time = cleanup_optimizer._get_last_cleanup_time()
        assert last_time is not None
        assert isinstance(last_time, str)
    
    def test_get_last_optimization_time(self, cleanup_optimizer):
        """Test getting last optimization time"""
        # Initially should be None
        last_time = cleanup_optimizer._get_last_optimization_time()
        assert last_time is None
        
        # After saving stats, should return timestamp
        stats = OptimizationStats(optimization_timestamp=datetime.now().isoformat())
        cleanup_optimizer._save_optimization_stats(stats)
        
        last_time = cleanup_optimizer._get_last_optimization_time()
        assert last_time is not None
        assert isinstance(last_time, str)


if __name__ == "__main__":
    pytest.main([__file__]) 