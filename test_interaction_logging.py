#!/usr/bin/env python3
"""
Test Script for Memory Interaction Logging System

This script demonstrates the functionality of the memory interaction logging system
with real-world examples, performance monitoring, analytics, and comprehensive logging.
"""

import sys
import tempfile
import shutil
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from memory.interaction_logging import (
        OperationType,
        LogLevel,
        OperationStatus,
        MemoryInteractionLogger,
        log_memory_operation,
        initialize_global_logger,
        get_global_logger,
        log_operation
    )
    from memory.memory_initializer import MemoryInitializer
    from memory.file_operations import write_memory_file
    from memory.importance_scoring import TimestampManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the personal-ai-assistant directory")
    sys.exit(1)


def simulate_memory_operations(logger: MemoryInteractionLogger, memory_dir: Path):
    """Simulate various memory operations for demonstration."""
    
    print("🔄 Simulating memory operations...")
    
    interactions_dir = memory_dir / 'interactions'
    core_dir = memory_dir / 'core'
    
    # 1. Create memory files
    print("   Creating memory files...")
    for i in range(5):
        with logger.log_operation(
            operation_type=OperationType.CREATE,
            file_path=interactions_dir / f'interaction_{i}.md',
            memory_type='interaction',
            user_context={'session_id': f'session_{i}'},
            metadata={'batch': 'demo_batch_1'}
        ):
            frontmatter = {
                'created': TimestampManager.create_timestamp(),
                'memory_type': 'interaction',
                'importance_score': 5 + (i % 3),
                'categories': ['demo', 'test']
            }
            write_memory_file(
                interactions_dir / f'interaction_{i}.md',
                frontmatter,
                f"This is demo interaction {i} content."
            )
            time.sleep(0.01)  # Small delay to simulate processing time
    
    # 2. Read memory files
    print("   Reading memory files...")
    for i in range(3):
        logger.log_memory_access(
            file_path=interactions_dir / f'interaction_{i}.md',
            memory_type='interaction',
            importance_score=5 + (i % 3),
            access_type='read',
            user_context={'user_id': 'demo_user'}
        )
        time.sleep(0.005)
    
    # 3. Search operations
    print("   Performing search operations...")
    search_queries = [
        "demo interaction content",
        "test categories",
        "recent memories",
        "high importance"
    ]
    
    for query in search_queries:
        logger.log_search_operation(
            query=query,
            results_count=len(query.split()) + 2,  # Simulate varying results
            search_type='semantic',
            filters={'memory_type': 'interaction'},
            user_context={'search_session': 'demo_search'}
        )
        time.sleep(0.008)
    
    # 4. Update operations
    print("   Updating memory files...")
    for i in range(2):
        with logger.log_operation(
            operation_type=OperationType.UPDATE,
            file_path=interactions_dir / f'interaction_{i}.md',
            memory_type='interaction',
            metadata={'update_type': 'importance_score'}
        ):
            # Simulate update processing
            time.sleep(0.015)
    
    # 5. Validation operations
    print("   Running validation operations...")
    logger.log_validation_operation(
        validation_type='full_system',
        issues_found=3,
        errors_count=1,
        warnings_count=2,
        auto_fix_enabled=True
    )
    
    logger.log_validation_operation(
        file_path=interactions_dir / 'interaction_0.md',
        validation_type='single_file',
        issues_found=0,
        errors_count=0,
        warnings_count=0,
        auto_fix_enabled=False
    )
    
    # 6. Simulate some failures
    print("   Simulating operation failures...")
    try:
        with logger.log_operation(
            operation_type=OperationType.DELETE,
            file_path=Path('nonexistent.md'),
            metadata={'reason': 'cleanup'}
        ):
            raise FileNotFoundError("File not found for deletion")
    except FileNotFoundError:
        pass
    
    try:
        with logger.log_operation(
            operation_type=OperationType.READ,
            file_path=Path('corrupted.md')
        ):
            raise PermissionError("Permission denied")
    except PermissionError:
        pass
    
    # 7. Performance metrics update
    print("   Updating performance metrics...")
    logger.log_performance_metrics({
        'memory_usage_mb': 45.7,
        'file_count': 15,
        'total_file_size_mb': 2.3
    })
    
    print("✅ Completed memory operations simulation")


def demonstrate_analytics(logger: MemoryInteractionLogger):
    """Demonstrate analytics and reporting capabilities."""
    
    print("\n📊 ANALYTICS AND REPORTING DEMONSTRATION")
    print("-" * 50)
    
    # 1. Performance Summary
    print("\n1. Performance Summary:")
    summary = logger.get_performance_summary()
    
    metrics = summary['performance_metrics']
    print(f"   Total Operations: {metrics['total_operations']}")
    print(f"   Success Rate: {metrics['successful_operations']}/{metrics['total_operations']} ({(metrics['successful_operations']/metrics['total_operations']*100):.1f}%)")
    print(f"   Average Duration: {metrics['average_duration_ms']:.2f}ms")
    print(f"   Operations/Minute: {metrics['operations_per_minute']:.2f}")
    
    # 2. Operation Statistics
    print(f"\n2. Operation Type Breakdown:")
    op_stats = summary['operation_stats']
    for op_type, stats in op_stats.items():
        total = sum(stats.values())
        success = stats.get('success', 0)
        print(f"   {op_type.upper()}: {total} total, {success} successful ({success/total*100:.1f}%)")
    
    # 3. Memory Type Activity
    print(f"\n3. Memory Type Activity:")
    memory_stats = summary['memory_type_stats']
    for memory_type, count in memory_stats.items():
        print(f"   {memory_type}: {count} operations")
    
    # 4. Error Patterns
    print(f"\n4. Error Patterns:")
    error_patterns = summary['error_patterns']
    if error_patterns:
        for error_type, count in error_patterns.items():
            print(f"   {error_type}: {count} occurrences")
    else:
        print("   No errors detected")
    
    # 5. Analytics Report
    print(f"\n5. 24-Hour Analytics Report:")
    report = logger.get_analytics_report(hours=24)
    
    if 'message' not in report:
        report_summary = report['summary']
        print(f"   Operations in last 24h: {report_summary['total_operations']}")
        print(f"   Success Rate: {report_summary['success_rate']:.1f}%")
        print(f"   Operations/Hour: {report_summary['operations_per_hour']:.2f}")
        
        # Operation breakdown
        print(f"\n   Operation Breakdown:")
        for op_type, count in report['operation_breakdown'].items():
            print(f"     {op_type}: {count}")
        
        # Average durations
        print(f"\n   Average Durations (ms):")
        for op_type, duration in report['average_durations_ms'].items():
            print(f"     {op_type}: {duration:.2f}ms")
        
        # Performance trends
        trends = report['performance_trends']
        if 'message' not in trends:
            print(f"\n   Performance Trends:")
            print(f"     Direction: {trends['trend_direction']}")
            print(f"     Latest Average: {trends['latest_average_ms']:.2f}ms")
            if 'best_hour' in trends:
                best_hour, best_time = trends['best_hour']
                print(f"     Best Hour: {best_hour} ({best_time:.2f}ms)")
            if 'worst_hour' in trends:
                worst_hour, worst_time = trends['worst_hour']
                print(f"     Worst Hour: {worst_hour} ({worst_time:.2f}ms)")
    else:
        print(f"   {report['message']}")


def demonstrate_operation_history(logger: MemoryInteractionLogger):
    """Demonstrate operation history and filtering."""
    
    print("\n📋 OPERATION HISTORY DEMONSTRATION")
    print("-" * 50)
    
    # 1. Recent operations
    print("\n1. Recent Operations (last 10):")
    recent_ops = logger.get_operation_history(limit=10)
    
    for i, op in enumerate(recent_ops[:5], 1):  # Show first 5
        status_emoji = "✅" if op.status == OperationStatus.SUCCESS else "❌"
        print(f"   {i}. {status_emoji} {op.operation_type.value.upper()} - {op.duration_ms:.2f}ms")
        if op.file_path:
            print(f"      File: {Path(op.file_path).name}")
        if op.error_message:
            print(f"      Error: {op.error_message}")
    
    if len(recent_ops) > 5:
        print(f"   ... and {len(recent_ops) - 5} more operations")
    
    # 2. Failed operations
    print(f"\n2. Failed Operations:")
    failed_ops = logger.get_operation_history(status=OperationStatus.FAILED)
    
    if failed_ops:
        for i, op in enumerate(failed_ops, 1):
            print(f"   {i}. {op.operation_type.value.upper()} - {op.error_message}")
    else:
        print("   No failed operations found")
    
    # 3. Search operations
    print(f"\n3. Search Operations:")
    search_ops = logger.get_operation_history(operation_type=OperationType.SEARCH)
    
    for i, op in enumerate(search_ops[:3], 1):  # Show first 3
        query = op.metadata.get('query', 'Unknown')
        results = op.metadata.get('results_count', 0)
        print(f"   {i}. Query: '{query}' - {results} results ({op.duration_ms:.2f}ms)")


def demonstrate_export_functionality(logger: MemoryInteractionLogger, test_dir: Path):
    """Demonstrate log export functionality."""
    
    print("\n💾 EXPORT FUNCTIONALITY DEMONSTRATION")
    print("-" * 50)
    
    # 1. Export to JSON
    print("\n1. Exporting logs to JSON...")
    json_export_path = test_dir / 'memory_logs_export.json'
    
    success = logger.export_logs(json_export_path, format='json')
    if success:
        print(f"   ✅ Successfully exported to: {json_export_path}")
        
        # Show export file size
        file_size = json_export_path.stat().st_size
        print(f"   📁 Export file size: {file_size:,} bytes")
        
        # Show sample of exported data
        with open(json_export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   📊 Exported {data['total_operations']} operations")
        print(f"   🕒 Export timestamp: {data['export_timestamp']}")
    else:
        print("   ❌ Export failed")
    
    # 2. Export to CSV
    print("\n2. Exporting logs to CSV...")
    csv_export_path = test_dir / 'memory_logs_export.csv'
    
    success = logger.export_logs(csv_export_path, format='csv')
    if success:
        print(f"   ✅ Successfully exported to: {csv_export_path}")
        
        # Show CSV file info
        with open(csv_export_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"   📊 CSV contains {len(lines)} lines (including header)")
        print(f"   📋 Columns: {lines[0].strip()}")
    else:
        print("   ❌ CSV export failed")


def demonstrate_global_logger():
    """Demonstrate global logger functionality."""
    
    print("\n🌐 GLOBAL LOGGER DEMONSTRATION")
    print("-" * 50)
    
    # Create temporary directory for global logger
    global_test_dir = Path(tempfile.mkdtemp(prefix="global_logger_test_"))
    
    try:
        # Initialize global logger
        print("\n1. Initializing global logger...")
        initialize_global_logger(global_test_dir, LogLevel.INFO)
        
        global_logger = get_global_logger()
        print(f"   ✅ Global logger initialized at: {global_test_dir}")
        
        # Use global logger decorator
        print("\n2. Using global logger decorator...")
        
        @log_operation(OperationType.CREATE, metadata={'demo': 'global_decorator'})
        def create_demo_memory(name: str):
            time.sleep(0.01)  # Simulate work
            return f"Created memory: {name}"
        
        @log_operation(OperationType.READ)
        def read_demo_memory(name: str):
            time.sleep(0.005)  # Simulate work
            return f"Read memory: {name}"
        
        # Execute decorated functions
        result1 = create_demo_memory("global_demo_1")
        result2 = read_demo_memory("global_demo_1")
        
        print(f"   📝 {result1}")
        print(f"   📖 {result2}")
        
        # Show global logger statistics
        print("\n3. Global logger statistics:")
        summary = global_logger.get_performance_summary()
        metrics = summary['performance_metrics']
        
        print(f"   Total operations: {metrics['total_operations']}")
        print(f"   Success rate: {metrics['successful_operations']}/{metrics['total_operations']}")
        print(f"   Average duration: {metrics['average_duration_ms']:.2f}ms")
        
    finally:
        # Cleanup
        if global_test_dir.exists():
            shutil.rmtree(global_test_dir)


def demonstrate_logging_system():
    """Main demonstration function."""
    
    print("="*70)
    print("MEMORY INTERACTION LOGGING SYSTEM DEMONSTRATION")
    print("="*70)
    
    # Create temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="interaction_logging_test_"))
    print(f"\n📁 Using test directory: {test_dir}")
    
    try:
        # Initialize memory structure
        print("\n🏗️  Initializing memory structure...")
        initializer = MemoryInitializer(str(test_dir))
        initializer.initialize_memory_structure()
        print("✅ Memory structure initialized")
        
        # Initialize interaction logger
        print("\n📊 Initializing interaction logger...")
        logger = MemoryInteractionLogger(test_dir, LogLevel.DEBUG)
        print("✅ Interaction logger initialized")
        print(f"   Log directory: {logger.logs_dir}")
        
        # Simulate memory operations
        simulate_memory_operations(logger, test_dir)
        
        # Demonstrate analytics
        demonstrate_analytics(logger)
        
        # Demonstrate operation history
        demonstrate_operation_history(logger)
        
        # Demonstrate export functionality
        demonstrate_export_functionality(logger, test_dir)
        
        # Demonstrate cleanup
        print(f"\n🧹 CLEANUP DEMONSTRATION")
        print("-" * 50)
        
        # Create some old log files for cleanup demo
        logs_dir = logger.logs_dir
        old_log = logs_dir / 'old_test.log'
        old_log.touch()
        
        # Set old timestamp
        old_time = datetime.now() - timedelta(days=40)
        old_timestamp = old_time.timestamp()
        import os
        os.utime(old_log, (old_timestamp, old_timestamp))
        
        print(f"\n   Created old log file for cleanup demo")
        cleaned_count = logger.cleanup_old_logs(days_to_keep=30)
        print(f"   🗑️  Cleaned up {cleaned_count} old log files")
        
        # Final summary
        print(f"\n📈 FINAL SYSTEM SUMMARY")
        print("-" * 30)
        
        final_summary = logger.get_performance_summary()
        final_metrics = final_summary['performance_metrics']
        
        print(f"✅ Total operations logged: {final_metrics['total_operations']}")
        print(f"✅ Success rate: {final_metrics['successful_operations']}/{final_metrics['total_operations']} ({(final_metrics['successful_operations']/final_metrics['total_operations']*100):.1f}%)")
        print(f"✅ Average operation time: {final_metrics['average_duration_ms']:.2f}ms")
        print(f"✅ Operations per minute: {final_metrics['operations_per_minute']:.2f}")
        print(f"✅ Active operations: {final_summary['active_operations_count']}")
        print(f"✅ Log files created: {len(list(logger.logs_dir.glob('*.log')))}")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test directory...")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("✅ Cleanup complete")
    
    # Demonstrate global logger
    demonstrate_global_logger()
    
    print(f"\n🎉 Memory interaction logging system demonstration complete!")
    print("="*70)


if __name__ == "__main__":
    demonstrate_logging_system() 