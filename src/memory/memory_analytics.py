"""
Memory Analytics and Statistics Tracking System

This module provides comprehensive analytics and statistics tracking for the memory system,
including usage patterns, performance metrics, user interaction statistics, and system
health trends over time.
"""

import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics
from enum import Enum

# Handle imports gracefully for both package and standalone execution
try:
    from .file_operations import MemoryFileOperations, read_memory_file
    from .importance_scoring import ImportanceScorer
    from .memory_cleanup import MemoryCleanupOptimizer
except ImportError:
    # Fallback for standalone execution
    from file_operations import MemoryFileOperations, read_memory_file
    from importance_scoring import ImportanceScorer
    from memory_cleanup import MemoryCleanupOptimizer

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked by the analytics system."""
    USAGE = "usage"
    PERFORMANCE = "performance"
    HEALTH = "health"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM = "system"


class TimeRange(Enum):
    """Time ranges for analytics queries."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ALL_TIME = "all_time"


@dataclass
class MemoryUsageStats:
    """Memory usage statistics."""
    total_files: int
    total_size_bytes: int
    files_by_type: Dict[str, int]
    files_by_category: Dict[str, int]
    files_by_importance: Dict[str, int]
    average_file_size: float
    largest_file_size: int
    smallest_file_size: int
    creation_rate_per_day: float
    update_rate_per_day: float
    timestamp: str


@dataclass
class PerformanceMetrics:
    """Performance metrics for memory operations."""
    read_operations: int
    write_operations: int
    search_operations: int
    cleanup_operations: int
    average_read_time_ms: float
    average_write_time_ms: float
    average_search_time_ms: float
    cache_hit_rate: float
    error_rate: float
    timestamp: str


@dataclass
class UserBehaviorStats:
    """User behavior and interaction statistics."""
    total_interactions: int
    interactions_per_day: float
    most_accessed_categories: List[Tuple[str, int]]
    most_accessed_files: List[Tuple[str, int]]
    search_patterns: List[Tuple[str, int]]
    peak_usage_hours: List[int]
    session_duration_avg: float
    memory_creation_patterns: Dict[str, int]
    timestamp: str


@dataclass
class SystemHealthTrends:
    """System health trends over time."""
    health_score_trend: List[Tuple[str, float]]
    storage_growth_trend: List[Tuple[str, int]]
    cleanup_frequency: List[Tuple[str, int]]
    error_frequency: List[Tuple[str, int]]
    performance_trend: List[Tuple[str, float]]
    fragmentation_trend: List[Tuple[str, float]]
    timestamp: str


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""
    report_id: str
    time_range: str
    usage_stats: MemoryUsageStats
    performance_metrics: PerformanceMetrics
    user_behavior: UserBehaviorStats
    health_trends: SystemHealthTrends
    insights: List[str]
    recommendations: List[str]
    generated_at: str


class MemoryAnalytics:
    """
    Comprehensive memory analytics and statistics tracking system.
    
    This class provides detailed analytics on memory usage patterns, performance
    metrics, user behavior, and system health trends to help optimize the
    memory system and understand usage patterns.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize the memory analytics system.
        
        Args:
            base_path: Base path for memory storage
        """
        self.base_path = Path(base_path)
        
        # Analytics storage
        self.analytics_dir = self.base_path / 'system' / 'analytics'
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for metrics storage
        self.db_path = self.analytics_dir / 'analytics.db'
        self.init_database()
        
        # JSON files for reports
        self.reports_dir = self.analytics_dir / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
        
        # Supporting systems
        self.importance_scorer = ImportanceScorer()
        
        # In-memory caches for performance
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=15)
        
    def init_database(self) -> None:
        """Initialize the analytics database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        event_name TEXT NOT NULL,
                        details TEXT,
                        duration_ms REAL,
                        success BOOLEAN DEFAULT TRUE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # User interactions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        interaction_type TEXT NOT NULL,
                        file_path TEXT,
                        category TEXT,
                        importance_score INTEGER,
                        duration_ms REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # System health table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_health (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        health_score REAL NOT NULL,
                        total_files INTEGER,
                        total_size_bytes INTEGER,
                        fragmentation_score REAL,
                        cleanup_needed BOOLEAN,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON user_interactions(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_timestamp ON system_health(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing analytics database: {e}")
    
    def record_metric(
        self,
        metric_type: MetricType,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            metric_type: Type of metric
            metric_name: Name of the metric
            value: Metric value
            metadata: Optional metadata
            timestamp: Optional timestamp (defaults to now)
        """
        try:
            ts = timestamp or datetime.now()
            metadata_json = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO metrics (timestamp, metric_type, metric_name, value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (ts.isoformat(), metric_type.value, metric_name, value, metadata_json))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")
    
    def record_event(
        self,
        event_type: str,
        event_name: str,
        details: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record an event.
        
        Args:
            event_type: Type of event
            event_name: Name of the event
            details: Optional event details
            duration_ms: Optional duration in milliseconds
            success: Whether the event was successful
            timestamp: Optional timestamp (defaults to now)
        """
        try:
            ts = timestamp or datetime.now()
            details_json = json.dumps(details) if details else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events (timestamp, event_type, event_name, details, duration_ms, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (ts.isoformat(), event_type, event_name, details_json, duration_ms, success))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording event {event_name}: {e}")
    
    def record_user_interaction(
        self,
        interaction_type: str,
        file_path: Optional[str] = None,
        category: Optional[str] = None,
        importance_score: Optional[int] = None,
        duration_ms: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a user interaction.
        
        Args:
            interaction_type: Type of interaction
            file_path: Optional file path involved
            category: Optional category
            importance_score: Optional importance score
            duration_ms: Optional duration in milliseconds
            timestamp: Optional timestamp (defaults to now)
        """
        try:
            ts = timestamp or datetime.now()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_interactions 
                    (timestamp, interaction_type, file_path, category, importance_score, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (ts.isoformat(), interaction_type, file_path, category, importance_score, duration_ms))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording user interaction: {e}")
    
    def record_system_health(
        self,
        health_score: float,
        total_files: int,
        total_size_bytes: int,
        fragmentation_score: float,
        cleanup_needed: bool,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record system health metrics.
        
        Args:
            health_score: Overall health score
            total_files: Total number of files
            total_size_bytes: Total size in bytes
            fragmentation_score: Fragmentation score
            cleanup_needed: Whether cleanup is needed
            timestamp: Optional timestamp (defaults to now)
        """
        try:
            ts = timestamp or datetime.now()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_health 
                    (timestamp, health_score, total_files, total_size_bytes, fragmentation_score, cleanup_needed)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (ts.isoformat(), health_score, total_files, total_size_bytes, fragmentation_score, cleanup_needed))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording system health: {e}")
    
    def get_usage_statistics(self, time_range: TimeRange = TimeRange.WEEK) -> MemoryUsageStats:
        """
        Get memory usage statistics.
        
        Args:
            time_range: Time range for statistics
            
        Returns:
            Memory usage statistics
        """
        cache_key = f"usage_stats_{time_range.value}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]
        
        try:
            # Get all memory files
            memory_files = self._get_all_memory_files()
            
            if not memory_files:
                return MemoryUsageStats(
                    total_files=0,
                    total_size_bytes=0,
                    files_by_type={},
                    files_by_category={},
                    files_by_importance={},
                    average_file_size=0.0,
                    largest_file_size=0,
                    smallest_file_size=0,
                    creation_rate_per_day=0.0,
                    update_rate_per_day=0.0,
                    timestamp=datetime.now().isoformat()
                )
            
            # Calculate statistics
            total_files = len(memory_files)
            total_size = 0
            files_by_type = Counter()
            files_by_category = Counter()
            files_by_importance = Counter()
            file_sizes = []
            creation_dates = []
            update_dates = []
            
            cutoff_date = self._get_cutoff_date(time_range)
            
            for file_info in memory_files:
                try:
                    file_path = file_info['path']
                    file_size = file_info['size']
                    
                    # Read file metadata
                    frontmatter, _ = read_memory_file(str(file_path))
                    
                    # Filter by time range if specified
                    created_str = frontmatter.get('created', '')
                    if created_str and cutoff_date:
                        try:
                            created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                            if created_date < cutoff_date:
                                continue
                        except ValueError:
                            pass
                    
                    total_size += file_size
                    file_sizes.append(file_size)
                    
                    # Categorize by type
                    memory_type = frontmatter.get('memory_type', 'unknown')
                    files_by_type[memory_type] += 1
                    
                    # Categorize by category
                    category = frontmatter.get('category', 'uncategorized')
                    files_by_category[category] += 1
                    
                    # Categorize by importance
                    importance = frontmatter.get('importance_score', 5)
                    importance_range = self._get_importance_range(importance)
                    files_by_importance[importance_range] += 1
                    
                    # Track creation and update dates
                    if created_str:
                        try:
                            creation_dates.append(datetime.fromisoformat(created_str.replace('Z', '+00:00')))
                        except ValueError:
                            pass
                    
                    updated_str = frontmatter.get('last_updated', '')
                    if updated_str:
                        try:
                            update_dates.append(datetime.fromisoformat(updated_str.replace('Z', '+00:00')))
                        except ValueError:
                            pass
                
                except Exception as e:
                    logger.warning(f"Error processing file {file_info.get('path', 'unknown')}: {e}")
            
            # Calculate rates
            creation_rate = self._calculate_rate_per_day(creation_dates, time_range)
            update_rate = self._calculate_rate_per_day(update_dates, time_range)
            
            stats = MemoryUsageStats(
                total_files=total_files,
                total_size_bytes=total_size,
                files_by_type=dict(files_by_type),
                files_by_category=dict(files_by_category),
                files_by_importance=dict(files_by_importance),
                average_file_size=statistics.mean(file_sizes) if file_sizes else 0.0,
                largest_file_size=max(file_sizes) if file_sizes else 0,
                smallest_file_size=min(file_sizes) if file_sizes else 0,
                creation_rate_per_day=creation_rate,
                update_rate_per_day=update_rate,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache the result
            self._cache_result(cache_key, stats)
            return stats
            
        except Exception as e:
            logger.error(f"Error getting usage statistics: {e}")
            return MemoryUsageStats(
                total_files=0,
                total_size_bytes=0,
                files_by_type={},
                files_by_category={},
                files_by_importance={},
                average_file_size=0.0,
                largest_file_size=0,
                smallest_file_size=0,
                creation_rate_per_day=0.0,
                update_rate_per_day=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    def get_performance_metrics(self, time_range: TimeRange = TimeRange.WEEK) -> PerformanceMetrics:
        """
        Get performance metrics.
        
        Args:
            time_range: Time range for metrics
            
        Returns:
            Performance metrics
        """
        cache_key = f"performance_metrics_{time_range.value}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]
        
        try:
            cutoff_date = self._get_cutoff_date(time_range)
            cutoff_str = cutoff_date.isoformat() if cutoff_date else '1970-01-01'
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get operation counts
                cursor.execute('''
                    SELECT event_name, COUNT(*) 
                    FROM events 
                    WHERE timestamp >= ? AND event_type = 'memory_operation'
                    GROUP BY event_name
                ''', (cutoff_str,))
                
                operation_counts = dict(cursor.fetchall())
                
                # Get average operation times
                cursor.execute('''
                    SELECT event_name, AVG(duration_ms) 
                    FROM events 
                    WHERE timestamp >= ? AND event_type = 'memory_operation' AND duration_ms IS NOT NULL
                    GROUP BY event_name
                ''', (cutoff_str,))
                
                operation_times = dict(cursor.fetchall())
                
                # Get error rate
                cursor.execute('''
                    SELECT 
                        COUNT(CASE WHEN success = 0 THEN 1 END) as errors,
                        COUNT(*) as total
                    FROM events 
                    WHERE timestamp >= ? AND event_type = 'memory_operation'
                ''', (cutoff_str,))
                
                error_data = cursor.fetchone()
                error_rate = (error_data[0] / error_data[1]) if error_data[1] > 0 else 0.0
                
                # Get cache hit rate (if available)
                cursor.execute('''
                    SELECT AVG(value) 
                    FROM metrics 
                    WHERE timestamp >= ? AND metric_name = 'cache_hit_rate'
                ''', (cutoff_str,))
                
                cache_hit_result = cursor.fetchone()
                cache_hit_rate = cache_hit_result[0] if cache_hit_result[0] is not None else 0.0
            
            metrics = PerformanceMetrics(
                read_operations=operation_counts.get('read', 0),
                write_operations=operation_counts.get('write', 0),
                search_operations=operation_counts.get('search', 0),
                cleanup_operations=operation_counts.get('cleanup', 0),
                average_read_time_ms=operation_times.get('read', 0.0),
                average_write_time_ms=operation_times.get('write', 0.0),
                average_search_time_ms=operation_times.get('search', 0.0),
                cache_hit_rate=cache_hit_rate,
                error_rate=error_rate,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache the result
            self._cache_result(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return PerformanceMetrics(
                read_operations=0,
                write_operations=0,
                search_operations=0,
                cleanup_operations=0,
                average_read_time_ms=0.0,
                average_write_time_ms=0.0,
                average_search_time_ms=0.0,
                cache_hit_rate=0.0,
                error_rate=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    def get_user_behavior_stats(self, time_range: TimeRange = TimeRange.WEEK) -> UserBehaviorStats:
        """
        Get user behavior statistics.
        
        Args:
            time_range: Time range for statistics
            
        Returns:
            User behavior statistics
        """
        cache_key = f"user_behavior_{time_range.value}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]
        
        try:
            cutoff_date = self._get_cutoff_date(time_range)
            cutoff_str = cutoff_date.isoformat() if cutoff_date else '1970-01-01'
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total interactions
                cursor.execute('''
                    SELECT COUNT(*) FROM user_interactions WHERE timestamp >= ?
                ''', (cutoff_str,))
                total_interactions = cursor.fetchone()[0]
                
                # Interactions per day
                days = self._get_days_in_range(time_range)
                interactions_per_day = total_interactions / days if days > 0 else 0
                
                # Most accessed categories
                cursor.execute('''
                    SELECT category, COUNT(*) 
                    FROM user_interactions 
                    WHERE timestamp >= ? AND category IS NOT NULL
                    GROUP BY category 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10
                ''', (cutoff_str,))
                most_accessed_categories = cursor.fetchall()
                
                # Most accessed files
                cursor.execute('''
                    SELECT file_path, COUNT(*) 
                    FROM user_interactions 
                    WHERE timestamp >= ? AND file_path IS NOT NULL
                    GROUP BY file_path 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10
                ''', (cutoff_str,))
                most_accessed_files = cursor.fetchall()
                
                # Search patterns (from events)
                cursor.execute('''
                    SELECT details, COUNT(*) 
                    FROM events 
                    WHERE timestamp >= ? AND event_name = 'search' AND details IS NOT NULL
                    GROUP BY details 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 10
                ''', (cutoff_str,))
                search_results = cursor.fetchall()
                search_patterns = []
                for details_json, count in search_results:
                    try:
                        details = json.loads(details_json)
                        query = details.get('query', 'unknown')
                        search_patterns.append((query, count))
                    except (json.JSONDecodeError, AttributeError):
                        pass
                
                # Peak usage hours
                cursor.execute('''
                    SELECT strftime('%H', timestamp) as hour, COUNT(*) 
                    FROM user_interactions 
                    WHERE timestamp >= ?
                    GROUP BY hour 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 5
                ''', (cutoff_str,))
                peak_hours_data = cursor.fetchall()
                peak_usage_hours = [int(hour) for hour, _ in peak_hours_data]
                
                # Average session duration
                cursor.execute('''
                    SELECT AVG(duration_ms) 
                    FROM user_interactions 
                    WHERE timestamp >= ? AND duration_ms IS NOT NULL
                ''', (cutoff_str,))
                avg_duration_result = cursor.fetchone()
                session_duration_avg = avg_duration_result[0] if avg_duration_result[0] is not None else 0.0
                
                # Memory creation patterns
                cursor.execute('''
                    SELECT interaction_type, COUNT(*) 
                    FROM user_interactions 
                    WHERE timestamp >= ? AND interaction_type LIKE '%create%'
                    GROUP BY interaction_type
                ''', (cutoff_str,))
                memory_creation_patterns = dict(cursor.fetchall())
            
            stats = UserBehaviorStats(
                total_interactions=total_interactions,
                interactions_per_day=interactions_per_day,
                most_accessed_categories=most_accessed_categories,
                most_accessed_files=most_accessed_files,
                search_patterns=search_patterns,
                peak_usage_hours=peak_usage_hours,
                session_duration_avg=session_duration_avg,
                memory_creation_patterns=memory_creation_patterns,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache the result
            self._cache_result(cache_key, stats)
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user behavior stats: {e}")
            return UserBehaviorStats(
                total_interactions=0,
                interactions_per_day=0.0,
                most_accessed_categories=[],
                most_accessed_files=[],
                search_patterns=[],
                peak_usage_hours=[],
                session_duration_avg=0.0,
                memory_creation_patterns={},
                timestamp=datetime.now().isoformat()
            )
    
    def get_system_health_trends(self, time_range: TimeRange = TimeRange.MONTH) -> SystemHealthTrends:
        """
        Get system health trends.
        
        Args:
            time_range: Time range for trends
            
        Returns:
            System health trends
        """
        cache_key = f"health_trends_{time_range.value}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.metrics_cache[cache_key]
        
        try:
            cutoff_date = self._get_cutoff_date(time_range)
            cutoff_str = cutoff_date.isoformat() if cutoff_date else '1970-01-01'
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Health score trend
                cursor.execute('''
                    SELECT DATE(timestamp) as date, AVG(health_score) 
                    FROM system_health 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp) 
                    ORDER BY date
                ''', (cutoff_str,))
                health_score_trend = cursor.fetchall()
                
                # Storage growth trend
                cursor.execute('''
                    SELECT DATE(timestamp) as date, AVG(total_size_bytes) 
                    FROM system_health 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp) 
                    ORDER BY date
                ''', (cutoff_str,))
                storage_growth_trend = [(date, int(size)) for date, size in cursor.fetchall()]
                
                # Cleanup frequency
                cursor.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) 
                    FROM events 
                    WHERE timestamp >= ? AND event_name = 'cleanup'
                    GROUP BY DATE(timestamp) 
                    ORDER BY date
                ''', (cutoff_str,))
                cleanup_frequency = cursor.fetchall()
                
                # Error frequency
                cursor.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) 
                    FROM events 
                    WHERE timestamp >= ? AND success = 0
                    GROUP BY DATE(timestamp) 
                    ORDER BY date
                ''', (cutoff_str,))
                error_frequency = cursor.fetchall()
                
                # Performance trend (average operation time)
                cursor.execute('''
                    SELECT DATE(timestamp) as date, AVG(duration_ms) 
                    FROM events 
                    WHERE timestamp >= ? AND duration_ms IS NOT NULL
                    GROUP BY DATE(timestamp) 
                    ORDER BY date
                ''', (cutoff_str,))
                performance_trend = cursor.fetchall()
                
                # Fragmentation trend
                cursor.execute('''
                    SELECT DATE(timestamp) as date, AVG(fragmentation_score) 
                    FROM system_health 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp) 
                    ORDER BY date
                ''', (cutoff_str,))
                fragmentation_trend = cursor.fetchall()
            
            trends = SystemHealthTrends(
                health_score_trend=health_score_trend,
                storage_growth_trend=storage_growth_trend,
                cleanup_frequency=cleanup_frequency,
                error_frequency=error_frequency,
                performance_trend=performance_trend,
                fragmentation_trend=fragmentation_trend,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache the result
            self._cache_result(cache_key, trends)
            return trends
            
        except Exception as e:
            logger.error(f"Error getting system health trends: {e}")
            return SystemHealthTrends(
                health_score_trend=[],
                storage_growth_trend=[],
                cleanup_frequency=[],
                error_frequency=[],
                performance_trend=[],
                fragmentation_trend=[],
                timestamp=datetime.now().isoformat()
            )
    
    def generate_analytics_report(
        self,
        time_range: TimeRange = TimeRange.WEEK,
        include_insights: bool = True
    ) -> AnalyticsReport:
        """
        Generate comprehensive analytics report.
        
        Args:
            time_range: Time range for the report
            include_insights: Whether to include insights and recommendations
            
        Returns:
            Comprehensive analytics report
        """
        try:
            report_id = f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Gather all statistics
            usage_stats = self.get_usage_statistics(time_range)
            performance_metrics = self.get_performance_metrics(time_range)
            user_behavior = self.get_user_behavior_stats(time_range)
            health_trends = self.get_system_health_trends(time_range)
            
            insights = []
            recommendations = []
            
            if include_insights:
                insights, recommendations = self._generate_insights_and_recommendations(
                    usage_stats, performance_metrics, user_behavior, health_trends
                )
            
            report = AnalyticsReport(
                report_id=report_id,
                time_range=time_range.value,
                usage_stats=usage_stats,
                performance_metrics=performance_metrics,
                user_behavior=user_behavior,
                health_trends=health_trends,
                insights=insights,
                recommendations=recommendations,
                generated_at=datetime.now().isoformat()
            )
            
            # Save report
            self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            raise
    
    def get_top_metrics(self, metric_type: MetricType, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get top metrics by type.
        
        Args:
            metric_type: Type of metrics to retrieve
            limit: Maximum number of results
            
        Returns:
            List of (metric_name, value) tuples
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT metric_name, AVG(value) as avg_value
                    FROM metrics 
                    WHERE metric_type = ?
                    GROUP BY metric_name 
                    ORDER BY avg_value DESC 
                    LIMIT ?
                ''', (metric_type.value, limit))
                
                return cursor.fetchall()
                
        except Exception as e:
            logger.error(f"Error getting top metrics: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old analytics data.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            deleted_count = 0
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old metrics
                cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_str,))
                deleted_count += cursor.rowcount
                
                # Delete old events
                cursor.execute('DELETE FROM events WHERE timestamp < ?', (cutoff_str,))
                deleted_count += cursor.rowcount
                
                # Delete old user interactions
                cursor.execute('DELETE FROM user_interactions WHERE timestamp < ?', (cutoff_str,))
                deleted_count += cursor.rowcount
                
                # Delete old system health records
                cursor.execute('DELETE FROM system_health WHERE timestamp < ?', (cutoff_str,))
                deleted_count += cursor.rowcount
                
                conn.commit()
            
            # Clean up old report files
            cutoff_timestamp = cutoff_date.timestamp()
            for report_file in self.reports_dir.glob('*.json'):
                if report_file.stat().st_mtime < cutoff_timestamp:
                    report_file.unlink()
            
            logger.info(f"Cleaned up {deleted_count} old analytics records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old analytics data: {e}")
            return 0
    
    # Helper methods
    
    def _get_all_memory_files(self) -> List[Dict[str, Any]]:
        """Get list of all memory files with metadata."""
        memory_files = []
        
        # Search in common memory directories
        search_dirs = ['interactions', 'core', 'archive']
        
        for dir_name in search_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                files = MemoryFileOperations.list_memory_files(dir_path, "*.md", include_metadata=True)
                memory_files.extend(files)
        
        return memory_files
    
    def _get_cutoff_date(self, time_range: TimeRange) -> Optional[datetime]:
        """Get cutoff date for time range."""
        now = datetime.now()
        
        if time_range == TimeRange.HOUR:
            return now - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            return now - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            return now - timedelta(weeks=1)
        elif time_range == TimeRange.MONTH:
            return now - timedelta(days=30)
        elif time_range == TimeRange.YEAR:
            return now - timedelta(days=365)
        else:  # ALL_TIME
            return None
    
    def _get_days_in_range(self, time_range: TimeRange) -> int:
        """Get number of days in time range."""
        if time_range == TimeRange.HOUR:
            return 1
        elif time_range == TimeRange.DAY:
            return 1
        elif time_range == TimeRange.WEEK:
            return 7
        elif time_range == TimeRange.MONTH:
            return 30
        elif time_range == TimeRange.YEAR:
            return 365
        else:  # ALL_TIME
            return 365  # Default to 1 year for rate calculations
    
    def _get_importance_range(self, importance: int) -> str:
        """Get importance range category."""
        if importance >= 8:
            return "high"
        elif importance >= 6:
            return "medium"
        elif importance >= 4:
            return "low"
        else:
            return "very_low"
    
    def _calculate_rate_per_day(self, dates: List[datetime], time_range: TimeRange) -> float:
        """Calculate rate per day for given dates."""
        if not dates:
            return 0.0
        
        days = self._get_days_in_range(time_range)
        return len(dates) / days if days > 0 else 0.0
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self.metrics_cache:
            return False
        
        if cache_key not in self.cache_expiry:
            return False
        
        return datetime.now() < self.cache_expiry[cache_key]
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a result with expiry."""
        self.metrics_cache[cache_key] = result
        self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
    
    def _generate_insights_and_recommendations(
        self,
        usage_stats: MemoryUsageStats,
        performance_metrics: PerformanceMetrics,
        user_behavior: UserBehaviorStats,
        health_trends: SystemHealthTrends
    ) -> Tuple[List[str], List[str]]:
        """Generate insights and recommendations from analytics data."""
        insights = []
        recommendations = []
        
        # Usage insights
        if usage_stats.total_files > 1000:
            insights.append(f"Large memory system with {usage_stats.total_files} files")
            recommendations.append("Consider implementing more aggressive cleanup policies")
        
        if usage_stats.creation_rate_per_day > 10:
            insights.append(f"High memory creation rate: {usage_stats.creation_rate_per_day:.1f} files/day")
            recommendations.append("Monitor storage growth and consider automated archiving")
        
        # Performance insights
        if performance_metrics.error_rate > 0.05:
            insights.append(f"High error rate detected: {performance_metrics.error_rate:.1%}")
            recommendations.append("Investigate and fix sources of memory operation errors")
        
        if performance_metrics.average_read_time_ms > 100:
            insights.append("Slow read operations detected")
            recommendations.append("Consider optimizing file access patterns or adding caching")
        
        # User behavior insights
        if user_behavior.total_interactions > 0:
            top_category = user_behavior.most_accessed_categories[0] if user_behavior.most_accessed_categories else None
            if top_category:
                insights.append(f"Most accessed category: {top_category[0]} ({top_category[1]} accesses)")
        
        if user_behavior.peak_usage_hours:
            peak_hour = user_behavior.peak_usage_hours[0]
            insights.append(f"Peak usage hour: {peak_hour}:00")
            recommendations.append(f"Schedule maintenance operations outside peak hours ({peak_hour}:00)")
        
        # Health trends insights
        if health_trends.health_score_trend:
            recent_health = health_trends.health_score_trend[-1][1] if health_trends.health_score_trend else 1.0
            if recent_health < 0.7:
                insights.append(f"System health declining: {recent_health:.2f}")
                recommendations.append("Run comprehensive cleanup and optimization")
        
        return insights, recommendations
    
    def _save_report(self, report: AnalyticsReport) -> None:
        """Save analytics report to file."""
        try:
            report_file = self.reports_dir / f"{report.report_id}.json"
            with open(report_file, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving analytics report: {e}")


# Convenience functions for common analytics operations
def track_memory_operation(
    analytics: MemoryAnalytics,
    operation_type: str,
    duration_ms: float,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Track a memory operation for analytics."""
    analytics.record_event(
        event_type="memory_operation",
        event_name=operation_type,
        duration_ms=duration_ms,
        success=success,
        details=details
    )


def track_user_interaction(
    analytics: MemoryAnalytics,
    interaction_type: str,
    file_path: Optional[str] = None,
    category: Optional[str] = None,
    duration_ms: Optional[float] = None
) -> None:
    """Track a user interaction for analytics."""
    analytics.record_user_interaction(
        interaction_type=interaction_type,
        file_path=file_path,
        category=category,
        duration_ms=duration_ms
    )


def update_system_health_metrics(
    analytics: MemoryAnalytics,
    cleanup_optimizer: MemoryCleanupOptimizer
) -> None:
    """Update system health metrics using cleanup optimizer."""
    try:
        health_report = cleanup_optimizer.generate_health_report()
        
        analytics.record_system_health(
            health_score=health_report.health_score,
            total_files=health_report.total_files,
            total_size_bytes=health_report.total_size_bytes,
            fragmentation_score=health_report.fragmentation_score,
            cleanup_needed=health_report.health_score < 0.8
        )
    except Exception as e:
        logger.error(f"Error updating system health metrics: {e}") 