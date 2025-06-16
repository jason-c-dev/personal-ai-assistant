"""
Memory Cleanup and Optimization System

This module provides comprehensive cleanup and optimization routines for the memory system,
including automated cleanup, duplicate detection, storage optimization, and memory health
maintenance to ensure optimal performance over time.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import shutil
import re

# Handle imports gracefully for both package and standalone execution
try:
    from .file_operations import read_memory_file, write_memory_file, MemoryFileOperations
    from .importance_scoring import ImportanceScorer
    from .memory_prioritization import IntelligentMemoryPrioritizer
    from .memory_reasoning import MemoryReasoningEngine, ReasoningType
except ImportError:
    # Fallback for standalone execution
    from file_operations import read_memory_file, write_memory_file, MemoryFileOperations
    from importance_scoring import ImportanceScorer
    from memory_prioritization import IntelligentMemoryPrioritizer
    from memory_reasoning import MemoryReasoningEngine, ReasoningType

logger = logging.getLogger(__name__)


@dataclass
class CleanupStats:
    """Statistics from cleanup operations."""
    files_processed: int = 0
    files_deleted: int = 0
    files_archived: int = 0
    files_merged: int = 0
    duplicates_removed: int = 0
    space_freed_bytes: int = 0
    errors_encountered: int = 0
    processing_time_seconds: float = 0.0
    cleanup_timestamp: str = ""


@dataclass
class OptimizationStats:
    """Statistics from optimization operations."""
    files_optimized: int = 0
    storage_compressed_bytes: int = 0
    index_rebuilt: bool = False
    fragmentation_reduced: float = 0.0
    performance_improvement: float = 0.0
    optimization_timestamp: str = ""


@dataclass
class MemoryHealthReport:
    """Comprehensive memory system health report."""
    total_files: int
    total_size_bytes: int
    duplicate_files: int
    outdated_files: int
    low_importance_files: int
    fragmentation_score: float
    health_score: float
    recommendations: List[str]
    last_cleanup: Optional[str]
    last_optimization: Optional[str]
    report_timestamp: str


@dataclass
class CleanupConfig:
    """Configuration for memory cleanup operations."""
    # Age-based cleanup
    max_age_days: int = 365
    archive_age_days: int = 180
    temp_file_age_hours: int = 24
    
    # Importance-based cleanup
    min_importance_threshold: float = 2.0
    low_importance_age_days: int = 30
    
    # Duplicate detection
    enable_duplicate_detection: bool = True
    similarity_threshold: float = 0.85
    content_hash_similarity: bool = True
    
    # Storage optimization
    enable_compression: bool = True
    max_file_size_kb: int = 1024
    enable_fragmentation_cleanup: bool = True
    
    # Safety settings
    enable_backup_before_cleanup: bool = True
    max_files_per_cleanup: int = 1000
    dry_run_mode: bool = False
    
    # Scheduling
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24
    optimization_interval_hours: int = 168  # Weekly


class MemoryCleanupOptimizer:
    """
    Comprehensive memory cleanup and optimization system.
    
    This class provides automated cleanup routines, duplicate detection,
    storage optimization, and memory health monitoring to maintain
    optimal memory system performance over time.
    """
    
    def __init__(
        self,
        base_path: str,
        config: Optional[CleanupConfig] = None
    ):
        """
        Initialize the memory cleanup optimizer.
        
        Args:
            base_path: Base path for memory storage
            config: Optional cleanup configuration
        """
        self.base_path = Path(base_path)
        self.config = config or CleanupConfig()
        
        # Initialize supporting systems
        self.importance_scorer = ImportanceScorer()
        self.prioritizer = IntelligentMemoryPrioritizer(str(base_path))
        self.reasoning_engine = MemoryReasoningEngine(str(base_path))
        
        # Cleanup tracking
        self.cleanup_dir = self.base_path / 'system' / 'cleanup'
        self.cleanup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup directory
        self.backup_dir = self.base_path / 'system' / 'backups'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats_file = self.cleanup_dir / 'cleanup_stats.json'
        self.health_file = self.cleanup_dir / 'health_reports.json'
        
        # Content hash cache for duplicate detection
        self.content_hash_cache: Dict[str, str] = {}
    
    def _list_all_memory_files(self) -> List[str]:
        """Get list of all memory files in the memory system."""
        memory_files = []
        
        # Search in common memory directories
        search_dirs = ['interactions', 'core', 'archive']
        
        for dir_name in search_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                files = MemoryFileOperations.list_memory_files(dir_path, "*.md")
                memory_files.extend([f['path'] for f in files])
        
        return memory_files
        
    def run_full_cleanup(self) -> Tuple[CleanupStats, OptimizationStats]:
        """
        Run comprehensive cleanup and optimization.
        
        Returns:
            Tuple of cleanup and optimization statistics
        """
        start_time = datetime.now()
        logger.info("Starting full memory cleanup and optimization")
        
        try:
            # Create backup if enabled
            if self.config.enable_backup_before_cleanup:
                self._create_backup()
            
            # Run cleanup operations
            cleanup_stats = self._run_cleanup_operations()
            
            # Run optimization operations
            optimization_stats = self._run_optimization_operations()
            
            # Update statistics
            total_time = (datetime.now() - start_time).total_seconds()
            cleanup_stats.processing_time_seconds = total_time
            cleanup_stats.cleanup_timestamp = datetime.now().isoformat()
            optimization_stats.optimization_timestamp = datetime.now().isoformat()
            
            # Save statistics
            self._save_cleanup_stats(cleanup_stats)
            self._save_optimization_stats(optimization_stats)
            
            # Generate health report
            health_report = self.generate_health_report()
            self._save_health_report(health_report)
            
            logger.info(f"Full cleanup completed in {total_time:.2f} seconds")
            return cleanup_stats, optimization_stats
            
        except Exception as e:
            logger.error(f"Error during full cleanup: {e}")
            raise
    
    def cleanup_outdated_memories(self) -> CleanupStats:
        """
        Clean up outdated and low-value memories.
        
        Returns:
            Cleanup statistics
        """
        stats = CleanupStats()
        cutoff_date = datetime.now() - timedelta(days=self.config.max_age_days)
        archive_date = datetime.now() - timedelta(days=self.config.archive_age_days)
        
        try:
            memory_files = self._list_all_memory_files()
            stats.files_processed = len(memory_files)
            
            for file_path in memory_files:
                try:
                    if stats.files_deleted + stats.files_archived >= self.config.max_files_per_cleanup:
                        break
                    
                    frontmatter, content = read_memory_file(file_path)
                    
                    # Parse creation date
                    created_str = frontmatter.get('created', '')
                    if not created_str:
                        continue
                    
                    try:
                        created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                    
                    # Check if file should be deleted
                    if created_date < cutoff_date:
                        importance = frontmatter.get('importance_score', 5)
                        
                        if importance < self.config.min_importance_threshold:
                            # Delete low importance old files
                            if not self.config.dry_run_mode:
                                file_size = Path(file_path).stat().st_size
                                os.remove(file_path)
                                stats.space_freed_bytes += file_size
                            
                            stats.files_deleted += 1
                            logger.debug(f"Deleted outdated low-importance file: {file_path}")
                        
                        elif created_date < archive_date:
                            # Archive old but important files
                            if not self.config.dry_run_mode:
                                self._archive_file(file_path)
                            
                            stats.files_archived += 1
                            logger.debug(f"Archived outdated file: {file_path}")
                
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {e}")
                    stats.errors_encountered += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during outdated memory cleanup: {e}")
            stats.errors_encountered += 1
            return stats
    
    def detect_and_remove_duplicates(self) -> CleanupStats:
        """
        Detect and remove duplicate memories.
        
        Returns:
            Cleanup statistics
        """
        stats = CleanupStats()
        
        if not self.config.enable_duplicate_detection:
            return stats
        
        try:
            memory_files = self._list_all_memory_files()
            stats.files_processed = len(memory_files)
            
            # Build content hash map
            content_hashes: Dict[str, List[str]] = defaultdict(list)
            similarity_groups: List[List[str]] = []
            
            for file_path in memory_files:
                try:
                    frontmatter, content = read_memory_file(file_path)
                    
                    # Generate content hash
                    content_hash = self._generate_content_hash(content)
                    content_hashes[content_hash].append(file_path)
                    
                except Exception as e:
                    logger.warning(f"Error hashing file {file_path}: {e}")
                    stats.errors_encountered += 1
            
            # Find exact duplicates
            for hash_value, file_list in content_hashes.items():
                if len(file_list) > 1:
                    similarity_groups.append(file_list)
            
            # Find similar content (if enabled)
            if self.config.content_hash_similarity:
                similarity_groups.extend(self._find_similar_content_groups(memory_files))
            
            # Process duplicate groups
            for group in similarity_groups:
                if len(group) > 1:
                    merged_file = self._merge_duplicate_files(group)
                    if merged_file:
                        stats.files_merged += 1
                        stats.duplicates_removed += len(group) - 1
                        
                        # Remove original files
                        for file_path in group:
                            if file_path != merged_file and not self.config.dry_run_mode:
                                try:
                                    file_size = Path(file_path).stat().st_size
                                    os.remove(file_path)
                                    stats.space_freed_bytes += file_size
                                except Exception as e:
                                    logger.warning(f"Error removing duplicate {file_path}: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during duplicate detection: {e}")
            stats.errors_encountered += 1
            return stats
    
    def optimize_storage(self) -> OptimizationStats:
        """
        Optimize memory storage for better performance.
        
        Returns:
            Optimization statistics
        """
        stats = OptimizationStats()
        
        try:
            memory_files = self._list_all_memory_files()
            
            for file_path in memory_files:
                try:
                    original_size = Path(file_path).stat().st_size
                    
                    # Optimize file content
                    if self._optimize_file_content(file_path):
                        new_size = Path(file_path).stat().st_size
                        stats.storage_compressed_bytes += original_size - new_size
                        stats.files_optimized += 1
                
                except Exception as e:
                    logger.warning(f"Error optimizing file {file_path}: {e}")
            
            # Rebuild indexes if needed
            if self.config.enable_fragmentation_cleanup:
                stats.index_rebuilt = self._rebuild_memory_indexes()
                stats.fragmentation_reduced = self._calculate_fragmentation_reduction()
            
            stats.optimization_timestamp = datetime.now().isoformat()
            return stats
            
        except Exception as e:
            logger.error(f"Error during storage optimization: {e}")
            return stats
    
    def cleanup_temporary_files(self) -> CleanupStats:
        """
        Clean up temporary and system files.
        
        Returns:
            Cleanup statistics
        """
        stats = CleanupStats()
        cutoff_time = datetime.now() - timedelta(hours=self.config.temp_file_age_hours)
        
        try:
            # Clean up temporary files
            temp_patterns = ['*.tmp', '*.temp', '*.bak', '*~', '.DS_Store']
            
            for pattern in temp_patterns:
                for temp_file in self.base_path.rglob(pattern):
                    try:
                        if temp_file.stat().st_mtime < cutoff_time.timestamp():
                            if not self.config.dry_run_mode:
                                file_size = temp_file.stat().st_size
                                temp_file.unlink()
                                stats.space_freed_bytes += file_size
                            
                            stats.files_deleted += 1
                    
                    except Exception as e:
                        logger.warning(f"Error removing temp file {temp_file}: {e}")
                        stats.errors_encountered += 1
            
            # Clean up old reasoning files
            reasoning_dir = self.base_path / 'system' / 'reasoning'
            if reasoning_dir.exists():
                for reasoning_file in reasoning_dir.glob('*.json'):
                    try:
                        if reasoning_file.stat().st_mtime < cutoff_time.timestamp():
                            if not self.config.dry_run_mode:
                                file_size = reasoning_file.stat().st_size
                                reasoning_file.unlink()
                                stats.space_freed_bytes += file_size
                            
                            stats.files_deleted += 1
                    
                    except Exception as e:
                        logger.warning(f"Error removing reasoning file {reasoning_file}: {e}")
                        stats.errors_encountered += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during temporary file cleanup: {e}")
            stats.errors_encountered += 1
            return stats
    
    def generate_health_report(self) -> MemoryHealthReport:
        """
        Generate comprehensive memory system health report.
        
        Returns:
            Memory health report
        """
        try:
            memory_files = self._list_all_memory_files()
            total_size = 0
            duplicate_count = 0
            outdated_count = 0
            low_importance_count = 0
            
            cutoff_date = datetime.now() - timedelta(days=self.config.max_age_days)
            
            # Analyze files
            for file_path in memory_files:
                try:
                    file_size = Path(file_path).stat().st_size
                    total_size += file_size
                    
                    frontmatter, content = read_memory_file(file_path)
                    
                    # Check if outdated
                    created_str = frontmatter.get('created', '')
                    if created_str:
                        try:
                            created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                            if created_date < cutoff_date:
                                outdated_count += 1
                        except ValueError:
                            pass
                    
                    # Check importance
                    importance = frontmatter.get('importance_score', 5)
                    if importance < self.config.min_importance_threshold:
                        low_importance_count += 1
                
                except Exception as e:
                    logger.warning(f"Error analyzing file {file_path}: {e}")
            
            # Calculate health metrics
            fragmentation_score = self._calculate_fragmentation_score()
            health_score = self._calculate_health_score(
                len(memory_files), duplicate_count, outdated_count, 
                low_importance_count, fragmentation_score
            )
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(
                duplicate_count, outdated_count, low_importance_count, fragmentation_score
            )
            
            # Get last cleanup/optimization times
            last_cleanup = self._get_last_cleanup_time()
            last_optimization = self._get_last_optimization_time()
            
            return MemoryHealthReport(
                total_files=len(memory_files),
                total_size_bytes=total_size,
                duplicate_files=duplicate_count,
                outdated_files=outdated_count,
                low_importance_files=low_importance_count,
                fragmentation_score=fragmentation_score,
                health_score=health_score,
                recommendations=recommendations,
                last_cleanup=last_cleanup,
                last_optimization=last_optimization,
                report_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return MemoryHealthReport(
                total_files=0,
                total_size_bytes=0,
                duplicate_files=0,
                outdated_files=0,
                low_importance_files=0,
                fragmentation_score=0.0,
                health_score=0.0,
                recommendations=["Error generating report"],
                last_cleanup=None,
                last_optimization=None,
                report_timestamp=datetime.now().isoformat()
            )
    
    def schedule_cleanup(self) -> bool:
        """
        Check if scheduled cleanup should run and execute if needed.
        
        Returns:
            True if cleanup was executed, False otherwise
        """
        if not self.config.auto_cleanup_enabled:
            return False
        
        try:
            last_cleanup_time = self._get_last_cleanup_time()
            
            if last_cleanup_time:
                last_cleanup = datetime.fromisoformat(last_cleanup_time)
                next_cleanup = last_cleanup + timedelta(hours=self.config.cleanup_interval_hours)
                
                if datetime.now() < next_cleanup:
                    return False
            
            # Run scheduled cleanup
            logger.info("Running scheduled memory cleanup")
            cleanup_stats, optimization_stats = self.run_full_cleanup()
            
            logger.info(f"Scheduled cleanup completed: {cleanup_stats.files_deleted} deleted, "
                       f"{cleanup_stats.files_archived} archived, {cleanup_stats.duplicates_removed} duplicates removed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during scheduled cleanup: {e}")
            return False
    
    # Helper methods
    
    def _run_cleanup_operations(self) -> CleanupStats:
        """Run all cleanup operations and combine statistics."""
        combined_stats = CleanupStats()
        
        # Cleanup outdated memories
        outdated_stats = self.cleanup_outdated_memories()
        combined_stats.files_processed += outdated_stats.files_processed
        combined_stats.files_deleted += outdated_stats.files_deleted
        combined_stats.files_archived += outdated_stats.files_archived
        combined_stats.space_freed_bytes += outdated_stats.space_freed_bytes
        combined_stats.errors_encountered += outdated_stats.errors_encountered
        
        # Remove duplicates
        duplicate_stats = self.detect_and_remove_duplicates()
        combined_stats.files_processed += duplicate_stats.files_processed
        combined_stats.files_merged += duplicate_stats.files_merged
        combined_stats.duplicates_removed += duplicate_stats.duplicates_removed
        combined_stats.space_freed_bytes += duplicate_stats.space_freed_bytes
        combined_stats.errors_encountered += duplicate_stats.errors_encountered
        
        # Cleanup temporary files
        temp_stats = self.cleanup_temporary_files()
        combined_stats.files_deleted += temp_stats.files_deleted
        combined_stats.space_freed_bytes += temp_stats.space_freed_bytes
        combined_stats.errors_encountered += temp_stats.errors_encountered
        
        return combined_stats
    
    def _run_optimization_operations(self) -> OptimizationStats:
        """Run all optimization operations."""
        return self.optimize_storage()
    
    def _create_backup(self) -> str:
        """Create backup of memory system before cleanup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"memory_backup_{timestamp}"
        
        try:
            shutil.copytree(self.base_path, backup_path, ignore=shutil.ignore_patterns('backups', 'system'))
            logger.info(f"Created backup at {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    def _archive_file(self, file_path: str) -> str:
        """Archive a memory file."""
        archive_dir = self.base_path / 'archive'
        archive_dir.mkdir(exist_ok=True)
        
        source_path = Path(file_path)
        archive_path = archive_dir / source_path.name
        
        # Handle name conflicts
        counter = 1
        while archive_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            archive_path = archive_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.move(str(source_path), str(archive_path))
        return str(archive_path)
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content comparison."""
        # Normalize content for comparison
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _find_similar_content_groups(self, file_paths: List[str]) -> List[List[str]]:
        """Find groups of files with similar content."""
        # Simplified similarity detection
        # In a real implementation, this would use more sophisticated NLP techniques
        
        similarity_groups = []
        processed_files = set()
        
        for i, file_path1 in enumerate(file_paths):
            if file_path1 in processed_files:
                continue
            
            try:
                _, content1 = read_memory_file(file_path1)
                similar_files = [file_path1]
                
                for j, file_path2 in enumerate(file_paths[i+1:], i+1):
                    if file_path2 in processed_files:
                        continue
                    
                    try:
                        _, content2 = read_memory_file(file_path2)
                        
                        # Simple similarity check
                        similarity = self._calculate_content_similarity(content1, content2)
                        if similarity >= self.config.similarity_threshold:
                            similar_files.append(file_path2)
                            processed_files.add(file_path2)
                    
                    except Exception as e:
                        logger.warning(f"Error comparing files {file_path1} and {file_path2}: {e}")
                
                if len(similar_files) > 1:
                    similarity_groups.append(similar_files)
                    processed_files.update(similar_files)
            
            except Exception as e:
                logger.warning(f"Error processing file {file_path1}: {e}")
        
        return similarity_groups
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _merge_duplicate_files(self, file_paths: List[str]) -> Optional[str]:
        """Merge duplicate files into a single file."""
        if len(file_paths) < 2:
            return None
        
        try:
            # Read all files and find the best one to keep
            file_data = []
            for file_path in file_paths:
                frontmatter, content = read_memory_file(file_path)
                file_data.append((file_path, frontmatter, content))
            
            # Choose the file with highest importance or most recent
            best_file = max(file_data, key=lambda x: (
                x[1].get('importance_score', 0),
                x[1].get('created', ''),
                len(x[2])
            ))
            
            best_path, best_frontmatter, best_content = best_file
            
            # Merge metadata from all files
            merged_frontmatter = best_frontmatter.copy()
            merged_frontmatter['merged_from'] = [fp for fp, _, _ in file_data if fp != best_path]
            merged_frontmatter['merge_timestamp'] = datetime.now().isoformat()
            
            # Write merged file
            if not self.config.dry_run_mode:
                write_memory_file(best_path, merged_frontmatter, best_content)
            
            return best_path
            
        except Exception as e:
            logger.error(f"Error merging duplicate files: {e}")
            return None
    
    def _optimize_file_content(self, file_path: str) -> bool:
        """Optimize individual file content."""
        try:
            frontmatter, content = read_memory_file(file_path)
            
            # Remove excessive whitespace
            optimized_content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            optimized_content = optimized_content.strip()
            
            # Update file if content changed
            if optimized_content != content:
                if not self.config.dry_run_mode:
                    write_memory_file(file_path, frontmatter, optimized_content)
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error optimizing file {file_path}: {e}")
            return False
    
    def _rebuild_memory_indexes(self) -> bool:
        """Rebuild memory system indexes."""
        # Placeholder for index rebuilding logic
        # In a real implementation, this would rebuild search indexes, etc.
        return True
    
    def _calculate_fragmentation_reduction(self) -> float:
        """Calculate fragmentation reduction after optimization."""
        # Placeholder calculation
        return 0.1  # 10% reduction
    
    def _calculate_fragmentation_score(self) -> float:
        """Calculate current fragmentation score."""
        # Simplified fragmentation calculation
        try:
            memory_files = self._list_all_memory_files()
            if not memory_files:
                return 0.0
            
            # Calculate based on file distribution and sizes
            total_files = len(memory_files)
            total_size = sum(Path(f).stat().st_size for f in memory_files)
            avg_size = total_size / total_files if total_files > 0 else 0
            
            # Simple fragmentation metric
            size_variance = sum((Path(f).stat().st_size - avg_size) ** 2 for f in memory_files)
            size_variance /= total_files if total_files > 0 else 1
            
            # Normalize to 0-1 scale
            fragmentation = min(size_variance / (avg_size ** 2) if avg_size > 0 else 0, 1.0)
            return fragmentation
            
        except Exception as e:
            logger.warning(f"Error calculating fragmentation score: {e}")
            return 0.0
    
    def _calculate_health_score(
        self,
        total_files: int,
        duplicates: int,
        outdated: int,
        low_importance: int,
        fragmentation: float
    ) -> float:
        """Calculate overall memory system health score."""
        if total_files == 0:
            return 1.0
        
        # Calculate component scores
        duplicate_score = 1.0 - (duplicates / total_files)
        outdated_score = 1.0 - (outdated / total_files)
        importance_score = 1.0 - (low_importance / total_files)
        fragmentation_score = 1.0 - fragmentation
        
        # Weighted average
        health_score = (
            duplicate_score * 0.25 +
            outdated_score * 0.25 +
            importance_score * 0.25 +
            fragmentation_score * 0.25
        )
        
        return max(0.0, min(1.0, health_score))
    
    def _generate_health_recommendations(
        self,
        duplicates: int,
        outdated: int,
        low_importance: int,
        fragmentation: float
    ) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if duplicates > 0:
            recommendations.append(f"Remove {duplicates} duplicate files to save space")
        
        if outdated > 0:
            recommendations.append(f"Archive or delete {outdated} outdated files")
        
        if low_importance > 0:
            recommendations.append(f"Review {low_importance} low-importance files for cleanup")
        
        if fragmentation > 0.3:
            recommendations.append("Run storage optimization to reduce fragmentation")
        
        if not recommendations:
            recommendations.append("Memory system is in good health")
        
        return recommendations
    
    def _get_last_cleanup_time(self) -> Optional[str]:
        """Get timestamp of last cleanup operation."""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    stats_data = json.load(f)
                    return stats_data.get('cleanup_timestamp')
        except Exception as e:
            logger.warning(f"Error reading cleanup stats: {e}")
        return None
    
    def _get_last_optimization_time(self) -> Optional[str]:
        """Get timestamp of last optimization operation."""
        try:
            optimization_file = self.cleanup_dir / 'optimization_stats.json'
            if optimization_file.exists():
                with open(optimization_file, 'r') as f:
                    stats_data = json.load(f)
                    return stats_data.get('optimization_timestamp')
        except Exception as e:
            logger.warning(f"Error reading optimization stats: {e}")
        return None
    
    def _save_cleanup_stats(self, stats: CleanupStats) -> None:
        """Save cleanup statistics to file."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(asdict(stats), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cleanup stats: {e}")
    
    def _save_optimization_stats(self, stats: OptimizationStats) -> None:
        """Save optimization statistics to file."""
        try:
            optimization_file = self.cleanup_dir / 'optimization_stats.json'
            with open(optimization_file, 'w') as f:
                json.dump(asdict(stats), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving optimization stats: {e}")
    
    def _save_health_report(self, report: MemoryHealthReport) -> None:
        """Save health report to file."""
        try:
            # Load existing reports
            reports = []
            if self.health_file.exists():
                with open(self.health_file, 'r') as f:
                    reports = json.load(f)
            
            # Add new report
            reports.append(asdict(report))
            
            # Keep only last 30 reports
            reports = reports[-30:]
            
            # Save updated reports
            with open(self.health_file, 'w') as f:
                json.dump(reports, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving health report: {e}")
    
    def get_cleanup_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get cleanup history for the specified number of days."""
        try:
            if not self.health_file.exists():
                return []
            
            with open(self.health_file, 'r') as f:
                reports = json.load(f)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            filtered_reports = []
            for report in reports:
                report_date = datetime.fromisoformat(report['report_timestamp'])
                if report_date >= cutoff_date:
                    filtered_reports.append(report)
            
            return sorted(filtered_reports, key=lambda x: x['report_timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting cleanup history: {e}")
            return [] 