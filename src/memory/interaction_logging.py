"""
Memory Interaction Logging System

This module provides comprehensive logging and monitoring of all memory operations,
including detailed audit trails, performance metrics, analytics, and debugging support.
"""

import json
import logging
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
import functools

# Handle imports gracefully for both package and standalone execution
try:
    from .importance_scoring import TimestampManager
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from importance_scoring import TimestampManager


class OperationType(Enum):
    """Types of memory operations that can be logged."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    VALIDATE = "validate"
    CONDENSE = "condense"
    BACKUP = "backup"
    RESTORE = "restore"
    REPAIR = "repair"
    IMPORT_SCORE = "importance_score"
    ACCESS = "access"
    SYSTEM = "system"


class LogLevel(Enum):
    """Log levels for memory operations."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class OperationStatus(Enum):
    """Status of memory operations."""
    STARTED = "started"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


@dataclass
class MemoryOperation:
    """Represents a single memory operation for logging."""
    operation_id: str
    operation_type: OperationType
    status: OperationStatus
    timestamp: str
    duration_ms: Optional[float] = None
    file_path: Optional[str] = None
    memory_type: Optional[str] = None
    importance_score: Optional[int] = None
    user_context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert operation to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryOperation':
        """Create operation from dictionary."""
        # Convert enum strings back to enums
        if isinstance(data.get('operation_type'), str):
            data['operation_type'] = OperationType(data['operation_type'])
        if isinstance(data.get('status'), str):
            data['status'] = OperationStatus(data['status'])
        
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Performance metrics for memory operations."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    operations_per_minute: float = 0.0
    memory_usage_mb: float = 0.0
    file_count: int = 0
    total_file_size_mb: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100
    
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        return 100.0 - self.success_rate()


class MemoryInteractionLogger:
    """Comprehensive memory interaction logging system."""
    
    def __init__(self, memory_base_path: Path, log_level: LogLevel = LogLevel.INFO):
        """
        Initialize the memory interaction logger.
        
        Args:
            memory_base_path: Base path to the memory directory
            log_level: Minimum log level to record
        """
        self.memory_base_path = Path(memory_base_path)
        self.log_level = log_level
        
        # Create logs directory
        self.logs_dir = self.memory_base_path / '.logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize logging components
        self._setup_logging()
        
        # Operation tracking
        self.operations: deque = deque(maxlen=10000)  # Keep last 10k operations
        self.active_operations: Dict[str, MemoryOperation] = {}
        self.operation_counter = 0
        self.lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.operation_stats = defaultdict(lambda: defaultdict(int))
        self.hourly_stats = defaultdict(lambda: defaultdict(int))
        
        # Analytics
        self.memory_type_stats = defaultdict(int)
        self.importance_distribution = defaultdict(int)
        self.error_patterns = defaultdict(int)
        
        self.logger.info("Memory interaction logger initialized")
    
    def _setup_logging(self):
        """Set up file-based logging."""
        # Create logger
        self.logger = logging.getLogger(f"memory_logger_{id(self)}")
        self.logger.setLevel(getattr(logging, self.log_level.value.upper()))
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler for detailed logs
        log_file = self.logs_dir / f"memory_operations_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        with self.lock:
            self.operation_counter += 1
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"op_{timestamp}_{self.operation_counter:06d}"
    
    @contextmanager
    def log_operation(
        self,
        operation_type: OperationType,
        file_path: Optional[Path] = None,
        memory_type: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for logging memory operations.
        
        Args:
            operation_type: Type of operation being performed
            file_path: Path to the memory file (if applicable)
            memory_type: Type of memory being operated on
            user_context: User context information
            metadata: Additional operation metadata
            
        Yields:
            operation_id: Unique identifier for this operation
        """
        operation_id = self.generate_operation_id()
        start_time = time.time()
        
        # Create operation record
        operation = MemoryOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            status=OperationStatus.STARTED,
            timestamp=TimestampManager.create_timestamp(),
            file_path=str(file_path) if file_path else None,
            memory_type=memory_type,
            user_context=user_context or {},
            metadata=metadata or {}
        )
        
        # Track active operation
        with self.lock:
            self.active_operations[operation_id] = operation
        
        # Log operation start
        self.logger.info(f"Started {operation_type.value} operation: {operation_id}")
        
        try:
            yield operation_id
            
            # Operation completed successfully
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            operation.status = OperationStatus.SUCCESS
            operation.duration_ms = duration_ms
            
            self._record_operation_completion(operation)
            self.logger.info(f"Completed {operation_type.value} operation: {operation_id} ({duration_ms:.2f}ms)")
            
        except Exception as e:
            # Operation failed
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            operation.status = OperationStatus.FAILED
            operation.duration_ms = duration_ms
            operation.error_message = str(e)
            
            self._record_operation_completion(operation)
            self.logger.error(f"Failed {operation_type.value} operation: {operation_id} - {str(e)}")
            
            # Track error patterns
            error_type = type(e).__name__
            with self.lock:
                self.error_patterns[error_type] += 1
            
            raise
        
        finally:
            # Remove from active operations
            with self.lock:
                self.active_operations.pop(operation_id, None)
    
    def _record_operation_completion(self, operation: MemoryOperation):
        """Record completed operation and update metrics."""
        with self.lock:
            # Add to operations history
            self.operations.append(operation)
            
            # Update performance metrics
            self.performance_metrics.total_operations += 1
            if operation.status == OperationStatus.SUCCESS:
                self.performance_metrics.successful_operations += 1
            else:
                self.performance_metrics.failed_operations += 1
            
            if operation.duration_ms:
                self.performance_metrics.total_duration_ms += operation.duration_ms
                self.performance_metrics.average_duration_ms = (
                    self.performance_metrics.total_duration_ms / 
                    self.performance_metrics.total_operations
                )
            
            # Update operation statistics
            op_type = operation.operation_type.value
            status = operation.status.value
            self.operation_stats[op_type][status] += 1
            
            # Update hourly statistics
            hour_key = datetime.now().strftime('%Y%m%d_%H')
            self.hourly_stats[hour_key][op_type] += 1
            
            # Update memory type statistics
            if operation.memory_type:
                self.memory_type_stats[operation.memory_type] += 1
            
            # Update importance distribution
            if operation.importance_score:
                score_range = f"{(operation.importance_score // 2) * 2}-{(operation.importance_score // 2) * 2 + 1}"
                self.importance_distribution[score_range] += 1
    
    def log_memory_access(
        self,
        file_path: Path,
        memory_type: str,
        importance_score: Optional[int] = None,
        access_type: str = "read",
        user_context: Optional[Dict[str, Any]] = None
    ):
        """
        Log memory file access.
        
        Args:
            file_path: Path to the accessed memory file
            memory_type: Type of memory accessed
            importance_score: Importance score of the memory
            access_type: Type of access (read, write, etc.)
            user_context: User context information
        """
        with self.log_operation(
            operation_type=OperationType.ACCESS,
            file_path=file_path,
            memory_type=memory_type,
            user_context=user_context,
            metadata={
                'access_type': access_type,
                'importance_score': importance_score
            }
        ) as operation_id:
            # Update access-specific metrics
            with self.lock:
                if importance_score:
                    self.importance_distribution[f"score_{importance_score}"] += 1
    
    def log_search_operation(
        self,
        query: str,
        results_count: int,
        search_type: str = "semantic",
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ):
        """
        Log memory search operation.
        
        Args:
            query: Search query
            results_count: Number of results returned
            search_type: Type of search performed
            filters: Search filters applied
            user_context: User context information
        """
        with self.log_operation(
            operation_type=OperationType.SEARCH,
            user_context=user_context,
            metadata={
                'query': query,
                'results_count': results_count,
                'search_type': search_type,
                'filters': filters or {}
            }
        ) as operation_id:
            pass
    
    def log_validation_operation(
        self,
        file_path: Optional[Path] = None,
        validation_type: str = "full_system",
        issues_found: int = 0,
        errors_count: int = 0,
        warnings_count: int = 0,
        auto_fix_enabled: bool = False
    ):
        """
        Log memory validation operation.
        
        Args:
            file_path: Path to validated file (if single file validation)
            validation_type: Type of validation performed
            issues_found: Total number of issues found
            errors_count: Number of errors found
            warnings_count: Number of warnings found
            auto_fix_enabled: Whether auto-fix was enabled
        """
        with self.log_operation(
            operation_type=OperationType.VALIDATE,
            file_path=file_path,
            metadata={
                'validation_type': validation_type,
                'issues_found': issues_found,
                'errors_count': errors_count,
                'warnings_count': warnings_count,
                'auto_fix_enabled': auto_fix_enabled
            }
        ) as operation_id:
            pass
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """
        Log performance metrics.
        
        Args:
            metrics: Performance metrics to log
        """
        with self.lock:
            self.performance_metrics.memory_usage_mb = metrics.get('memory_usage_mb', 0.0)
            self.performance_metrics.file_count = metrics.get('file_count', 0)
            self.performance_metrics.total_file_size_mb = metrics.get('total_file_size_mb', 0.0)
        
        self.logger.info(f"Performance metrics updated: {metrics}")
    
    def get_operation_history(
        self,
        operation_type: Optional[OperationType] = None,
        status: Optional[OperationStatus] = None,
        limit: int = 100
    ) -> List[MemoryOperation]:
        """
        Get operation history with optional filtering.
        
        Args:
            operation_type: Filter by operation type
            status: Filter by operation status
            limit: Maximum number of operations to return
            
        Returns:
            List of memory operations
        """
        with self.lock:
            operations = list(self.operations)
        
        # Apply filters
        if operation_type:
            operations = [op for op in operations if op.operation_type == operation_type]
        
        if status:
            operations = [op for op in operations if op.status == status]
        
        # Sort by timestamp (most recent first) and limit
        operations.sort(key=lambda op: op.timestamp, reverse=True)
        return operations[:limit]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance metrics and statistics
        """
        with self.lock:
            # Calculate operations per minute
            if self.operations:
                time_span_minutes = (
                    datetime.now() - 
                    TimestampManager.parse_timestamp(self.operations[0].timestamp)
                ).total_seconds() / 60
                
                if time_span_minutes > 0:
                    self.performance_metrics.operations_per_minute = (
                        self.performance_metrics.total_operations / time_span_minutes
                    )
            
            return {
                'performance_metrics': asdict(self.performance_metrics),
                'operation_stats': dict(self.operation_stats),
                'memory_type_stats': dict(self.memory_type_stats),
                'importance_distribution': dict(self.importance_distribution),
                'error_patterns': dict(self.error_patterns),
                'active_operations_count': len(self.active_operations),
                'total_operations_logged': len(self.operations)
            }
    
    def get_analytics_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate analytics report for the specified time period.
        
        Args:
            hours: Number of hours to include in the report
            
        Returns:
            Analytics report dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            # Filter operations within time period
            recent_ops = [
                op for op in self.operations
                if TimestampManager.parse_timestamp(op.timestamp) >= cutoff_time
            ]
        
        if not recent_ops:
            return {'message': 'No operations in the specified time period'}
        
        # Calculate analytics
        total_ops = len(recent_ops)
        successful_ops = sum(1 for op in recent_ops if op.status == OperationStatus.SUCCESS)
        failed_ops = total_ops - successful_ops
        
        # Operation type breakdown
        op_type_counts = defaultdict(int)
        for op in recent_ops:
            op_type_counts[op.operation_type.value] += 1
        
        # Average duration by operation type
        op_durations = defaultdict(list)
        for op in recent_ops:
            if op.duration_ms:
                op_durations[op.operation_type.value].append(op.duration_ms)
        
        avg_durations = {
            op_type: sum(durations) / len(durations)
            for op_type, durations in op_durations.items()
            if durations
        }
        
        # Memory type activity
        memory_type_activity = defaultdict(int)
        for op in recent_ops:
            if op.memory_type:
                memory_type_activity[op.memory_type] += 1
        
        # Error analysis
        error_analysis = defaultdict(int)
        for op in recent_ops:
            if op.status == OperationStatus.FAILED and op.error_message:
                error_type = op.error_message.split(':')[0] if ':' in op.error_message else 'Unknown'
                error_analysis[error_type] += 1
        
        return {
            'time_period_hours': hours,
            'summary': {
                'total_operations': total_ops,
                'successful_operations': successful_ops,
                'failed_operations': failed_ops,
                'success_rate': (successful_ops / total_ops * 100) if total_ops > 0 else 0,
                'operations_per_hour': total_ops / hours if hours > 0 else 0
            },
            'operation_breakdown': dict(op_type_counts),
            'average_durations_ms': avg_durations,
            'memory_type_activity': dict(memory_type_activity),
            'error_analysis': dict(error_analysis),
            'performance_trends': self._calculate_performance_trends(recent_ops)
        }
    
    def _calculate_performance_trends(self, operations: List[MemoryOperation]) -> Dict[str, Any]:
        """Calculate performance trends from operations."""
        if len(operations) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Group operations by hour
        hourly_performance = defaultdict(list)
        for op in operations:
            if op.duration_ms:
                hour_key = TimestampManager.parse_timestamp(op.timestamp).strftime('%Y%m%d_%H')
                hourly_performance[hour_key].append(op.duration_ms)
        
        # Calculate hourly averages
        hourly_averages = {
            hour: sum(durations) / len(durations)
            for hour, durations in hourly_performance.items()
        }
        
        if len(hourly_averages) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Calculate trend
        hours = sorted(hourly_averages.keys())
        values = [hourly_averages[hour] for hour in hours]
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        return {
            'hourly_averages': hourly_averages,
            'trend_slope': slope,
            'trend_direction': 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable',
            'latest_average_ms': values[-1] if values else 0,
            'best_hour': min(hourly_averages.items(), key=lambda x: x[1]),
            'worst_hour': max(hourly_averages.items(), key=lambda x: x[1])
        }
    
    def export_logs(self, output_path: Path, format: str = 'json') -> bool:
        """
        Export operation logs to file.
        
        Args:
            output_path: Path to export file
            format: Export format ('json' or 'csv')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with self.lock:
                operations_data = [op.to_dict() for op in self.operations]
            
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'export_timestamp': TimestampManager.create_timestamp(),
                        'total_operations': len(operations_data),
                        'operations': operations_data,
                        'performance_summary': self.get_performance_summary()
                    }, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                import csv
                
                if operations_data:
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=operations_data[0].keys())
                        writer.writeheader()
                        writer.writerows(operations_data)
            
            self.logger.info(f"Exported {len(operations_data)} operations to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            return False
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """
        Clean up old log files.
        
        Args:
            days_to_keep: Number of days to keep log files
            
        Returns:
            Number of log files cleaned up
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            cleaned_count = 0
            
            for log_file in self.logs_dir.glob("*.log"):
                if log_file.is_file():
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        log_file.unlink()
                        cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old log files")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
            return 0


def log_memory_operation(
    logger: MemoryInteractionLogger,
    operation_type: OperationType,
    **kwargs
):
    """
    Decorator for automatically logging memory operations.
    
    Args:
        logger: Memory interaction logger instance
        operation_type: Type of operation to log
        **kwargs: Additional arguments for log_operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            # Extract relevant information from function arguments
            file_path = func_kwargs.get('file_path') or (args[0] if args and isinstance(args[0], Path) else None)
            memory_type = func_kwargs.get('memory_type')
            
            with logger.log_operation(
                operation_type=operation_type,
                file_path=file_path,
                memory_type=memory_type,
                **kwargs
            ):
                return func(*args, **func_kwargs)
        
        return wrapper
    return decorator


# Global logger instance (can be initialized by the application)
_global_logger: Optional[MemoryInteractionLogger] = None


def initialize_global_logger(memory_base_path: Path, log_level: LogLevel = LogLevel.INFO):
    """Initialize the global memory interaction logger."""
    global _global_logger
    _global_logger = MemoryInteractionLogger(memory_base_path, log_level)


def get_global_logger() -> Optional[MemoryInteractionLogger]:
    """Get the global memory interaction logger."""
    return _global_logger


def log_operation(operation_type: OperationType, **kwargs):
    """Decorator using the global logger."""
    if _global_logger is None:
        raise RuntimeError("Global logger not initialized. Call initialize_global_logger() first.")
    
    return log_memory_operation(_global_logger, operation_type, **kwargs) 