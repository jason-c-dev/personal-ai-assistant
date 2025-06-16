"""
Unit tests for Memory Analytics and Statistics Tracking System

Tests for comprehensive analytics functionality including metrics recording,
statistics generation, performance tracking, and report creation.
"""

import pytest
import tempfile
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Handle imports gracefully for both package and standalone execution
try:
    from .memory_analytics import (
        MemoryAnalytics, MetricType, TimeRange, MemoryUsageStats,
        PerformanceMetrics, UserBehaviorStats, SystemHealthTrends,
        AnalyticsReport, track_memory_operation, track_user_interaction
    )
    from .file_operations import write_memory_file
except ImportError:
    from memory_analytics import (
        MemoryAnalytics, MetricType, TimeRange, MemoryUsageStats,
        PerformanceMetrics, UserBehaviorStats, SystemHealthTrends,
        AnalyticsReport, track_memory_operation, track_user_interaction
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
    
    return str(memory_dir)


@pytest.fixture
def analytics_system(temp_memory_dir):
    """Create a MemoryAnalytics instance for testing"""
    return MemoryAnalytics(temp_memory_dir)


@pytest.fixture
def sample_memory_files(temp_memory_dir):
    """Create sample memory files for analytics testing"""
    base_path = Path(temp_memory_dir)
    files_created = {}
    
    # Recent high-importance file
    recent_path = base_path / "interactions" / "recent_important.md"
    recent_frontmatter = {
        'created': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'importance_score': 8,
        'memory_type': 'interaction',
        'category': 'work'
    }
    recent_content = "Recent important work discussion about project goals."
    write_memory_file(recent_path, recent_frontmatter, recent_content)
    files_created['recent'] = str(recent_path)
    
    # Medium importance file
    medium_path = base_path / "interactions" / "medium_importance.md"
    medium_frontmatter = {
        'created': (datetime.now() - timedelta(days=3)).isoformat(),
        'last_updated': (datetime.now() - timedelta(days=1)).isoformat(),
        'importance_score': 6,
        'memory_type': 'interaction',
        'category': 'personal'
    }
    medium_content = "Personal conversation about hobbies and interests."
    write_memory_file(medium_path, medium_frontmatter, medium_content)
    files_created['medium'] = str(medium_path)
    
    # Low importance file
    low_path = base_path / "interactions" / "low_importance.md"
    low_frontmatter = {
        'created': (datetime.now() - timedelta(days=7)).isoformat(),
        'last_updated': (datetime.now() - timedelta(days=5)).isoformat(),
        'importance_score': 3,
        'memory_type': 'interaction',
        'category': 'casual'
    }
    low_content = "Casual chat about weather and daily activities."
    write_memory_file(low_path, low_frontmatter, low_content)
    files_created['low'] = str(low_path)
    
    # User profile file
    profile_path = base_path / "core" / "user_profile.md"
    profile_frontmatter = {
        'created': (datetime.now() - timedelta(days=30)).isoformat(),
        'last_updated': datetime.now().isoformat(),
        'importance_score': 9,
        'memory_type': 'user_profile',
        'category': 'core'
    }
    profile_content = "User profile information including preferences and background."
    write_memory_file(profile_path, profile_frontmatter, profile_content)
    files_created['profile'] = str(profile_path)
    
    return files_created


class TestMemoryAnalyticsInitialization:
    """Test cases for MemoryAnalytics initialization"""
    
    def test_initialization_basic(self, temp_memory_dir):
        """Test basic initialization"""
        analytics = MemoryAnalytics(temp_memory_dir)
        
        assert analytics.base_path == Path(temp_memory_dir)
        assert analytics.analytics_dir.exists()
        assert analytics.reports_dir.exists()
        assert analytics.db_path.exists()
        assert analytics.importance_scorer is not None
        assert isinstance(analytics.metrics_cache, dict)
        assert isinstance(analytics.cache_expiry, dict)
    
    def test_database_initialization(self, analytics_system):
        """Test that database is properly initialized"""
        # Check that database file exists
        assert analytics_system.db_path.exists()
        
        # Check that tables are created
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            
            # Check metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'")
            assert cursor.fetchone() is not None
            
            # Check events table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
            assert cursor.fetchone() is not None
            
            # Check user_interactions table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_interactions'")
            assert cursor.fetchone() is not None
            
            # Check system_health table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_health'")
            assert cursor.fetchone() is not None
    
    def test_directory_structure_creation(self, temp_memory_dir):
        """Test that required directories are created"""
        analytics = MemoryAnalytics(temp_memory_dir)
        
        expected_analytics_dir = Path(temp_memory_dir) / 'system' / 'analytics'
        expected_reports_dir = expected_analytics_dir / 'reports'
        
        assert expected_analytics_dir.exists()
        assert expected_reports_dir.exists()


class TestMetricsRecording:
    """Test cases for metrics recording"""
    
    def test_record_metric_basic(self, analytics_system):
        """Test basic metric recording"""
        analytics_system.record_metric(
            MetricType.PERFORMANCE,
            "test_metric",
            42.5,
            {"source": "test"}
        )
        
        # Verify metric was recorded
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics WHERE metric_name = 'test_metric'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[2] == MetricType.PERFORMANCE.value  # metric_type
            assert result[3] == "test_metric"  # metric_name
            assert result[4] == 42.5  # value
            assert json.loads(result[5]) == {"source": "test"}  # metadata
    
    def test_record_metric_without_metadata(self, analytics_system):
        """Test metric recording without metadata"""
        analytics_system.record_metric(
            MetricType.USAGE,
            "simple_metric",
            100.0
        )
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics WHERE metric_name = 'simple_metric'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[4] == 100.0  # value
            assert result[5] is None  # metadata should be None
    
    def test_record_metric_with_custom_timestamp(self, analytics_system):
        """Test metric recording with custom timestamp"""
        custom_time = datetime.now() - timedelta(hours=1)
        
        analytics_system.record_metric(
            MetricType.HEALTH,
            "timestamped_metric",
            75.0,
            timestamp=custom_time
        )
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM metrics WHERE metric_name = 'timestamped_metric'")
            result = cursor.fetchone()
            
            assert result is not None
            recorded_time = datetime.fromisoformat(result[0])
            assert abs((recorded_time - custom_time).total_seconds()) < 1


class TestEventRecording:
    """Test cases for event recording"""
    
    def test_record_event_basic(self, analytics_system):
        """Test basic event recording"""
        analytics_system.record_event(
            "memory_operation",
            "read",
            {"file_path": "/test/file.md"},
            duration_ms=25.5,
            success=True
        )
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE event_name = 'read'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[2] == "memory_operation"  # event_type
            assert result[3] == "read"  # event_name
            assert json.loads(result[4]) == {"file_path": "/test/file.md"}  # details
            assert result[5] == 25.5  # duration_ms
            assert result[6] == 1  # success (True)
    
    def test_record_event_failure(self, analytics_system):
        """Test recording failed events"""
        analytics_system.record_event(
            "memory_operation",
            "write",
            {"error": "Permission denied"},
            success=False
        )
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT success FROM events WHERE event_name = 'write'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == 0  # success (False)
    
    def test_record_event_minimal(self, analytics_system):
        """Test recording event with minimal parameters"""
        analytics_system.record_event("user_action", "click")
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE event_name = 'click'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[2] == "user_action"  # event_type
            assert result[3] == "click"  # event_name
            assert result[4] is None  # details
            assert result[5] is None  # duration_ms
            assert result[6] == 1  # success (default True)


class TestUserInteractionRecording:
    """Test cases for user interaction recording"""
    
    def test_record_user_interaction_complete(self, analytics_system):
        """Test recording complete user interaction"""
        analytics_system.record_user_interaction(
            "file_access",
            "/test/file.md",
            "work",
            8,
            150.0
        )
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_interactions WHERE interaction_type = 'file_access'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[2] == "file_access"  # interaction_type
            assert result[3] == "/test/file.md"  # file_path
            assert result[4] == "work"  # category
            assert result[5] == 8  # importance_score
            assert result[6] == 150.0  # duration_ms
    
    def test_record_user_interaction_minimal(self, analytics_system):
        """Test recording minimal user interaction"""
        analytics_system.record_user_interaction("search")
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_interactions WHERE interaction_type = 'search'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[2] == "search"  # interaction_type
            assert result[3] is None  # file_path
            assert result[4] is None  # category
            assert result[5] is None  # importance_score
            assert result[6] is None  # duration_ms


class TestSystemHealthRecording:
    """Test cases for system health recording"""
    
    def test_record_system_health(self, analytics_system):
        """Test recording system health metrics"""
        analytics_system.record_system_health(
            health_score=0.85,
            total_files=150,
            total_size_bytes=1024000,
            fragmentation_score=0.25,
            cleanup_needed=False
        )
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_health ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[2] == 0.85  # health_score
            assert result[3] == 150  # total_files
            assert result[4] == 1024000  # total_size_bytes
            assert result[5] == 0.25  # fragmentation_score
            assert result[6] == 0  # cleanup_needed (False)
    
    def test_record_system_health_with_timestamp(self, analytics_system):
        """Test recording system health with custom timestamp"""
        custom_time = datetime.now() - timedelta(minutes=30)
        
        analytics_system.record_system_health(
            health_score=0.75,
            total_files=100,
            total_size_bytes=512000,
            fragmentation_score=0.4,
            cleanup_needed=True,
            timestamp=custom_time
        )
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM system_health ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            
            assert result is not None
            recorded_time = datetime.fromisoformat(result[0])
            assert abs((recorded_time - custom_time).total_seconds()) < 1


class TestUsageStatistics:
    """Test cases for usage statistics generation"""
    
    def test_get_usage_statistics_basic(self, analytics_system, sample_memory_files):
        """Test basic usage statistics generation"""
        stats = analytics_system.get_usage_statistics(TimeRange.WEEK)
        
        assert isinstance(stats, MemoryUsageStats)
        assert stats.total_files > 0
        assert stats.total_size_bytes > 0
        assert isinstance(stats.files_by_type, dict)
        assert isinstance(stats.files_by_category, dict)
        assert isinstance(stats.files_by_importance, dict)
        assert stats.average_file_size > 0
        assert stats.timestamp is not None
    
    def test_get_usage_statistics_empty_system(self, analytics_system):
        """Test usage statistics with no memory files"""
        stats = analytics_system.get_usage_statistics(TimeRange.DAY)
        
        assert stats.total_files == 0
        assert stats.total_size_bytes == 0
        assert stats.files_by_type == {}
        assert stats.files_by_category == {}
        assert stats.files_by_importance == {}
        assert stats.average_file_size == 0.0
        assert stats.largest_file_size == 0
        assert stats.smallest_file_size == 0
    
    def test_get_usage_statistics_different_time_ranges(self, analytics_system, sample_memory_files):
        """Test usage statistics with different time ranges"""
        day_stats = analytics_system.get_usage_statistics(TimeRange.DAY)
        week_stats = analytics_system.get_usage_statistics(TimeRange.WEEK)
        month_stats = analytics_system.get_usage_statistics(TimeRange.MONTH)
        
        # Week stats should include more files than day stats
        assert week_stats.total_files >= day_stats.total_files
        assert month_stats.total_files >= week_stats.total_files
    
    def test_usage_statistics_caching(self, analytics_system, sample_memory_files):
        """Test that usage statistics are cached"""
        # First call
        stats1 = analytics_system.get_usage_statistics(TimeRange.WEEK)
        
        # Second call should use cache
        stats2 = analytics_system.get_usage_statistics(TimeRange.WEEK)
        
        assert stats1.timestamp == stats2.timestamp
        assert stats1.total_files == stats2.total_files


class TestPerformanceMetrics:
    """Test cases for performance metrics"""
    
    def test_get_performance_metrics_basic(self, analytics_system):
        """Test basic performance metrics generation"""
        # Add some test events
        analytics_system.record_event("memory_operation", "read", duration_ms=25.0, success=True)
        analytics_system.record_event("memory_operation", "write", duration_ms=50.0, success=True)
        analytics_system.record_event("memory_operation", "read", duration_ms=30.0, success=False)
        
        metrics = analytics_system.get_performance_metrics(TimeRange.DAY)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.read_operations >= 2
        assert metrics.write_operations >= 1
        assert metrics.average_read_time_ms > 0
        assert metrics.average_write_time_ms > 0
        assert 0 <= metrics.error_rate <= 1
        assert metrics.timestamp is not None
    
    def test_get_performance_metrics_empty(self, analytics_system):
        """Test performance metrics with no data"""
        metrics = analytics_system.get_performance_metrics(TimeRange.HOUR)
        
        assert metrics.read_operations == 0
        assert metrics.write_operations == 0
        assert metrics.search_operations == 0
        assert metrics.cleanup_operations == 0
        assert metrics.average_read_time_ms == 0.0
        assert metrics.error_rate == 0.0
    
    def test_performance_metrics_error_rate_calculation(self, analytics_system):
        """Test error rate calculation in performance metrics"""
        # Add events with known success/failure ratio
        analytics_system.record_event("memory_operation", "test", success=True)
        analytics_system.record_event("memory_operation", "test", success=True)
        analytics_system.record_event("memory_operation", "test", success=False)
        analytics_system.record_event("memory_operation", "test", success=True)
        
        metrics = analytics_system.get_performance_metrics(TimeRange.DAY)
        
        # Should be 1 failure out of 4 operations = 0.25
        assert abs(metrics.error_rate - 0.25) < 0.01


class TestUserBehaviorStats:
    """Test cases for user behavior statistics"""
    
    def test_get_user_behavior_stats_basic(self, analytics_system):
        """Test basic user behavior statistics"""
        # Add some test interactions
        analytics_system.record_user_interaction("file_access", "/test1.md", "work", 8, 100.0)
        analytics_system.record_user_interaction("file_access", "/test2.md", "personal", 6, 150.0)
        analytics_system.record_user_interaction("search", category="work", duration_ms=50.0)
        
        stats = analytics_system.get_user_behavior_stats(TimeRange.DAY)
        
        assert isinstance(stats, UserBehaviorStats)
        assert stats.total_interactions >= 3
        assert stats.interactions_per_day > 0
        assert isinstance(stats.most_accessed_categories, list)
        assert isinstance(stats.most_accessed_files, list)
        assert isinstance(stats.search_patterns, list)
        assert isinstance(stats.peak_usage_hours, list)
        assert stats.session_duration_avg > 0
        assert stats.timestamp is not None
    
    def test_get_user_behavior_stats_empty(self, analytics_system):
        """Test user behavior statistics with no interactions"""
        stats = analytics_system.get_user_behavior_stats(TimeRange.WEEK)
        
        assert stats.total_interactions == 0
        assert stats.interactions_per_day == 0.0
        assert stats.most_accessed_categories == []
        assert stats.most_accessed_files == []
        assert stats.search_patterns == []
        assert stats.peak_usage_hours == []
        assert stats.session_duration_avg == 0.0
    
    def test_user_behavior_category_tracking(self, analytics_system):
        """Test category access tracking in user behavior"""
        # Add interactions with specific categories
        analytics_system.record_user_interaction("access", category="work")
        analytics_system.record_user_interaction("access", category="work")
        analytics_system.record_user_interaction("access", category="personal")
        
        stats = analytics_system.get_user_behavior_stats(TimeRange.DAY)
        
        # Work should be the most accessed category
        assert len(stats.most_accessed_categories) > 0
        assert stats.most_accessed_categories[0][0] == "work"
        assert stats.most_accessed_categories[0][1] == 2


class TestSystemHealthTrends:
    """Test cases for system health trends"""
    
    def test_get_system_health_trends_basic(self, analytics_system):
        """Test basic system health trends"""
        # Add some health records
        base_time = datetime.now() - timedelta(days=5)
        for i in range(5):
            analytics_system.record_system_health(
                health_score=0.8 + i * 0.02,
                total_files=100 + i * 10,
                total_size_bytes=1000000 + i * 100000,
                fragmentation_score=0.3 - i * 0.02,
                cleanup_needed=False,
                timestamp=base_time + timedelta(days=i)
            )
        
        trends = analytics_system.get_system_health_trends(TimeRange.WEEK)
        
        assert isinstance(trends, SystemHealthTrends)
        assert isinstance(trends.health_score_trend, list)
        assert isinstance(trends.storage_growth_trend, list)
        assert isinstance(trends.cleanup_frequency, list)
        assert isinstance(trends.error_frequency, list)
        assert isinstance(trends.performance_trend, list)
        assert isinstance(trends.fragmentation_trend, list)
        assert trends.timestamp is not None
    
    def test_get_system_health_trends_empty(self, analytics_system):
        """Test system health trends with no data"""
        trends = analytics_system.get_system_health_trends(TimeRange.MONTH)
        
        assert trends.health_score_trend == []
        assert trends.storage_growth_trend == []
        assert trends.cleanup_frequency == []
        assert trends.error_frequency == []
        assert trends.performance_trend == []
        assert trends.fragmentation_trend == []


class TestAnalyticsReports:
    """Test cases for analytics report generation"""
    
    def test_generate_analytics_report_basic(self, analytics_system, sample_memory_files):
        """Test basic analytics report generation"""
        # Add some test data
        analytics_system.record_event("memory_operation", "read", duration_ms=25.0)
        analytics_system.record_user_interaction("file_access", "/test.md", "work")
        analytics_system.record_system_health(0.85, 100, 1000000, 0.2, False)
        
        report = analytics_system.generate_analytics_report(TimeRange.WEEK, include_insights=True)
        
        assert isinstance(report, AnalyticsReport)
        assert report.report_id is not None
        assert report.time_range == TimeRange.WEEK.value
        assert isinstance(report.usage_stats, MemoryUsageStats)
        assert isinstance(report.performance_metrics, PerformanceMetrics)
        assert isinstance(report.user_behavior, UserBehaviorStats)
        assert isinstance(report.health_trends, SystemHealthTrends)
        assert isinstance(report.insights, list)
        assert isinstance(report.recommendations, list)
        assert report.generated_at is not None
    
    def test_generate_analytics_report_without_insights(self, analytics_system, sample_memory_files):
        """Test analytics report generation without insights"""
        report = analytics_system.generate_analytics_report(TimeRange.DAY, include_insights=False)
        
        assert isinstance(report, AnalyticsReport)
        assert report.insights == []
        assert report.recommendations == []
    
    def test_analytics_report_saving(self, analytics_system, sample_memory_files):
        """Test that analytics reports are saved to files"""
        report = analytics_system.generate_analytics_report(TimeRange.WEEK)
        
        # Check that report file was created
        report_file = analytics_system.reports_dir / f"{report.report_id}.json"
        assert report_file.exists()
        
        # Verify file content
        with open(report_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['report_id'] == report.report_id
        assert saved_data['time_range'] == report.time_range


class TestTopMetrics:
    """Test cases for top metrics retrieval"""
    
    def test_get_top_metrics_basic(self, analytics_system):
        """Test basic top metrics retrieval"""
        # Add some test metrics
        analytics_system.record_metric(MetricType.PERFORMANCE, "metric_a", 100.0)
        analytics_system.record_metric(MetricType.PERFORMANCE, "metric_b", 200.0)
        analytics_system.record_metric(MetricType.PERFORMANCE, "metric_c", 150.0)
        analytics_system.record_metric(MetricType.PERFORMANCE, "metric_a", 120.0)  # Second value for metric_a
        
        top_metrics = analytics_system.get_top_metrics(MetricType.PERFORMANCE, limit=5)
        
        assert isinstance(top_metrics, list)
        assert len(top_metrics) <= 5
        
        # Should be sorted by average value (descending)
        if len(top_metrics) >= 2:
            assert top_metrics[0][1] >= top_metrics[1][1]
    
    def test_get_top_metrics_empty(self, analytics_system):
        """Test top metrics with no data"""
        top_metrics = analytics_system.get_top_metrics(MetricType.USAGE, limit=10)
        
        assert top_metrics == []
    
    def test_get_top_metrics_limit(self, analytics_system):
        """Test top metrics limit functionality"""
        # Add more metrics than the limit
        for i in range(10):
            analytics_system.record_metric(MetricType.HEALTH, f"metric_{i}", float(i))
        
        top_metrics = analytics_system.get_top_metrics(MetricType.HEALTH, limit=3)
        
        assert len(top_metrics) == 3


class TestDataCleanup:
    """Test cases for analytics data cleanup"""
    
    def test_cleanup_old_data_basic(self, analytics_system):
        """Test basic old data cleanup"""
        # Add some old data
        old_time = datetime.now() - timedelta(days=100)
        recent_time = datetime.now() - timedelta(days=1)
        
        analytics_system.record_metric(MetricType.USAGE, "old_metric", 50.0, timestamp=old_time)
        analytics_system.record_metric(MetricType.USAGE, "recent_metric", 75.0, timestamp=recent_time)
        analytics_system.record_event("test", "old_event", timestamp=old_time)
        analytics_system.record_event("test", "recent_event", timestamp=recent_time)
        
        # Cleanup data older than 30 days
        deleted_count = analytics_system.cleanup_old_data(days_to_keep=30)
        
        assert deleted_count > 0
        
        # Verify old data was deleted and recent data remains
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM metrics WHERE metric_name = 'old_metric'")
            assert cursor.fetchone()[0] == 0
            
            cursor.execute("SELECT COUNT(*) FROM metrics WHERE metric_name = 'recent_metric'")
            assert cursor.fetchone()[0] == 1
    
    def test_cleanup_old_data_no_old_data(self, analytics_system):
        """Test cleanup when there's no old data"""
        # Add only recent data
        analytics_system.record_metric(MetricType.PERFORMANCE, "recent_metric", 100.0)
        
        deleted_count = analytics_system.cleanup_old_data(days_to_keep=30)
        
        assert deleted_count == 0


class TestConvenienceFunctions:
    """Test cases for convenience functions"""
    
    def test_track_memory_operation(self, analytics_system):
        """Test track_memory_operation convenience function"""
        track_memory_operation(
            analytics_system,
            "read",
            25.5,
            success=True,
            details={"file": "test.md"}
        )
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE event_name = 'read'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[2] == "memory_operation"  # event_type
            assert result[5] == 25.5  # duration_ms
            assert result[6] == 1  # success
    
    def test_track_user_interaction(self, analytics_system):
        """Test track_user_interaction convenience function"""
        track_user_interaction(
            analytics_system,
            "file_access",
            "/test/file.md",
            "work",
            duration_ms=100.0
        )
        
        with sqlite3.connect(analytics_system.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_interactions WHERE interaction_type = 'file_access'")
            result = cursor.fetchone()
            
            assert result is not None
            assert result[3] == "/test/file.md"  # file_path
            assert result[4] == "work"  # category
            assert result[6] == 100.0  # duration_ms


class TestCaching:
    """Test cases for analytics caching"""
    
    def test_cache_validity_check(self, analytics_system):
        """Test cache validity checking"""
        cache_key = "test_cache"
        
        # Initially cache should be invalid
        assert not analytics_system._is_cache_valid(cache_key)
        
        # Cache a result
        analytics_system._cache_result(cache_key, {"test": "data"})
        
        # Now cache should be valid
        assert analytics_system._is_cache_valid(cache_key)
        assert analytics_system.metrics_cache[cache_key] == {"test": "data"}
    
    def test_cache_expiry(self, analytics_system):
        """Test cache expiry functionality"""
        cache_key = "expiry_test"
        
        # Cache with very short duration
        analytics_system.cache_duration = timedelta(milliseconds=1)
        analytics_system._cache_result(cache_key, {"test": "data"})
        
        # Wait for cache to expire
        import time
        time.sleep(0.002)
        
        # Cache should now be invalid
        assert not analytics_system._is_cache_valid(cache_key)


class TestErrorHandling:
    """Test cases for error handling in analytics"""
    
    def test_record_metric_error_handling(self, analytics_system):
        """Test error handling in metric recording"""
        # Close the database to simulate an error
        analytics_system.db_path.unlink()
        
        # Should not raise an exception
        analytics_system.record_metric(MetricType.USAGE, "test_metric", 50.0)
    
    def test_get_usage_statistics_error_handling(self, analytics_system):
        """Test error handling in usage statistics"""
        # Corrupt the database
        analytics_system.db_path.unlink()
        
        # Should return empty stats without crashing
        stats = analytics_system.get_usage_statistics(TimeRange.DAY)
        
        assert isinstance(stats, MemoryUsageStats)
        assert stats.total_files == 0


class TestTimeRangeHandling:
    """Test cases for time range handling"""
    
    def test_get_cutoff_date_all_ranges(self, analytics_system):
        """Test cutoff date calculation for all time ranges"""
        now = datetime.now()
        
        hour_cutoff = analytics_system._get_cutoff_date(TimeRange.HOUR)
        day_cutoff = analytics_system._get_cutoff_date(TimeRange.DAY)
        week_cutoff = analytics_system._get_cutoff_date(TimeRange.WEEK)
        month_cutoff = analytics_system._get_cutoff_date(TimeRange.MONTH)
        year_cutoff = analytics_system._get_cutoff_date(TimeRange.YEAR)
        all_time_cutoff = analytics_system._get_cutoff_date(TimeRange.ALL_TIME)
        
        assert hour_cutoff < day_cutoff < week_cutoff < month_cutoff < year_cutoff
        assert all_time_cutoff is None
        
        # Check approximate correctness
        assert abs((now - hour_cutoff).total_seconds() - 3600) < 60  # Within 1 minute of 1 hour
        assert abs((now - day_cutoff).days - 1) == 0  # Exactly 1 day
    
    def test_get_days_in_range(self, analytics_system):
        """Test days calculation for time ranges"""
        assert analytics_system._get_days_in_range(TimeRange.HOUR) == 1
        assert analytics_system._get_days_in_range(TimeRange.DAY) == 1
        assert analytics_system._get_days_in_range(TimeRange.WEEK) == 7
        assert analytics_system._get_days_in_range(TimeRange.MONTH) == 30
        assert analytics_system._get_days_in_range(TimeRange.YEAR) == 365
        assert analytics_system._get_days_in_range(TimeRange.ALL_TIME) == 365


if __name__ == "__main__":
    pytest.main([__file__]) 