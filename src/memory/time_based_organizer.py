"""
Time-Based Memory Organizer

This module implements intelligent time-based memory organization for the Personal AI Assistant.
It automatically categorizes memories into Recent (0-30 days), Medium (30-180 days), and 
Archive (180+ days) windows, managing condensation and importance-based retention.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Handle imports gracefully for both package and standalone execution
try:
    from .file_operations import MemoryFileOperations, read_memory_file, write_memory_file
    from .memory_manager import MemoryEntry, SearchResult, MemoryType, ImportanceLevel
except ImportError:
    # Fallback for standalone execution
    from file_operations import MemoryFileOperations, read_memory_file, write_memory_file
    from memory_manager import MemoryEntry, SearchResult, MemoryType, ImportanceLevel

logger = logging.getLogger(__name__)


class TimeWindow(Enum):
    """Time windows for memory organization."""
    RECENT = "recent"      # 0-30 days: Full detail retention
    MEDIUM = "medium"      # 30-180 days: Summarized key points
    ARCHIVE = "archive"    # 180+ days: Essential facts only


@dataclass
class MemoryTimeMetrics:
    """Metrics for memory time-based analysis."""
    total_memories: int
    recent_count: int
    medium_count: int
    archive_count: int
    average_age_days: float
    oldest_memory_days: int
    memory_size_mb: float
    condensation_candidates: int


@dataclass
class CondensationCandidate:
    """Represents a memory candidate for condensation."""
    file_path: str
    age_days: int
    importance_score: int
    size_kb: float
    category: str
    interaction_count: int
    last_accessed: Optional[datetime] = None
    condensation_priority: float = 0.0


class TimeBasedOrganizer:
    """
    Manages time-based organization and condensation of memories.
    
    This class implements the intelligent memory management strategy that:
    1. Automatically categorizes memories by age
    2. Applies different retention policies by time window
    3. Manages condensation and summarization
    4. Maintains importance-based retention
    """
    
    def __init__(self, base_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the time-based organizer.
        
        Args:
            base_path: Base path for memory storage
            config: Optional configuration overrides
        """
        self.base_path = Path(base_path)
        self.config = self._load_config(config)
        
        # Time windows (in days)
        self.recent_days = self.config.get('recent_days', 30)
        self.medium_days = self.config.get('medium_days', 180)
        
        # Importance thresholds for retention
        self.critical_threshold = self.config.get('critical_threshold', 9)
        self.high_threshold = self.config.get('high_threshold', 7)
        self.medium_threshold = self.config.get('medium_threshold', 4)
        
        # Condensation settings
        self.condensation_batch_size = self.config.get('condensation_batch_size', 10)
        self.min_interactions_for_condensation = self.config.get('min_interactions_for_condensation', 3)
        self.condensation_ratio = self.config.get('condensation_ratio', 0.3)  # Target size reduction
        
        # Directories
        self.interactions_dir = self.base_path / 'interactions'
        self.condensed_dir = self.base_path / 'condensed'
        
        # Ensure condensed directories exist
        self._ensure_condensed_structure()
    
    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration with defaults."""
        default_config = {
            'recent_days': 30,
            'medium_days': 180,
            'critical_threshold': 9,
            'high_threshold': 7,
            'medium_threshold': 4,
            'condensation_batch_size': 10,
            'min_interactions_for_condensation': 3,
            'condensation_ratio': 0.3,
            'max_memory_size_mb': 100,
            'cleanup_enabled': True,
            'backup_before_condensation': True
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _ensure_condensed_structure(self) -> None:
        """Ensure condensed memory directory structure exists."""
        condensed_dirs = [
            self.condensed_dir / 'recent',
            self.condensed_dir / 'medium', 
            self.condensed_dir / 'archive'
        ]
        
        for dir_path in condensed_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def classify_memory_by_age(self, memory_date: datetime) -> TimeWindow:
        """
        Classify a memory into a time window based on its age.
        
        Args:
            memory_date: Date when the memory was created
            
        Returns:
            TimeWindow classification
        """
        age_days = (datetime.now() - memory_date).days
        
        if age_days <= self.recent_days:
            return TimeWindow.RECENT
        elif age_days <= self.medium_days:
            return TimeWindow.MEDIUM
        else:
            return TimeWindow.ARCHIVE
    
    def get_memory_age_distribution(self) -> Dict[str, MemoryTimeMetrics]:
        """
        Analyze the age distribution of memories across time windows.
        
        Returns:
            Dict mapping time windows to their metrics
        """
        distribution = {
            TimeWindow.RECENT.value: {'files': [], 'total_size': 0},
            TimeWindow.MEDIUM.value: {'files': [], 'total_size': 0},
            TimeWindow.ARCHIVE.value: {'files': [], 'total_size': 0}
        }
        
        # Analyze interaction memories
        if self.interactions_dir.exists():
            for memory_file in self.interactions_dir.rglob('*.md'):
                try:
                    frontmatter, content = read_memory_file(memory_file)
                    created_str = frontmatter.get('created', '')
                    
                    if created_str:
                        created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                        time_window = self.classify_memory_by_age(created_date)
                        
                        file_size = memory_file.stat().st_size
                        distribution[time_window.value]['files'].append({
                            'path': str(memory_file),
                            'size': file_size,
                            'age_days': (datetime.now() - created_date).days,
                            'importance': frontmatter.get('importance_score', 5),
                            'category': frontmatter.get('category', 'unknown')
                        })
                        distribution[time_window.value]['total_size'] += file_size
                        
                except Exception as e:
                    logger.warning(f"Error analyzing memory file {memory_file}: {e}")
        
        # Convert to metrics objects
        metrics = {}
        total_memories = 0
        total_age_days = 0
        
        for window, data in distribution.items():
            file_count = len(data['files'])
            total_memories += file_count
            
            if file_count > 0:
                avg_age = sum(f['age_days'] for f in data['files']) / file_count
                oldest_age = max(f['age_days'] for f in data['files'])
                total_age_days += sum(f['age_days'] for f in data['files'])
            else:
                avg_age = 0
                oldest_age = 0
            
            metrics[window] = MemoryTimeMetrics(
                total_memories=file_count,
                recent_count=len(distribution[TimeWindow.RECENT.value]['files']) if window == TimeWindow.RECENT.value else 0,
                medium_count=len(distribution[TimeWindow.MEDIUM.value]['files']) if window == TimeWindow.MEDIUM.value else 0,
                archive_count=len(distribution[TimeWindow.ARCHIVE.value]['files']) if window == TimeWindow.ARCHIVE.value else 0,
                average_age_days=avg_age,
                oldest_memory_days=oldest_age,
                memory_size_mb=data['total_size'] / (1024 * 1024),
                condensation_candidates=len([f for f in data['files'] 
                                           if f['importance'] < self.high_threshold 
                                           and f['age_days'] > self.recent_days])
            )
        
        return metrics
    
    def identify_condensation_candidates(
        self, 
        time_window: Optional[TimeWindow] = None,
        min_importance: Optional[int] = None,
        max_candidates: int = 50
    ) -> List[CondensationCandidate]:
        """
        Identify memories that are candidates for condensation.
        
        Args:
            time_window: Optional time window to focus on
            min_importance: Minimum importance score to consider
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of condensation candidates, prioritized by need
        """
        candidates = []
        
        if not self.interactions_dir.exists():
            return candidates
        
        for memory_file in self.interactions_dir.rglob('*.md'):
            try:
                frontmatter, content = read_memory_file(memory_file)
                created_str = frontmatter.get('created', '')
                
                if not created_str:
                    continue
                
                created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                age_days = (datetime.now() - created_date).days
                file_time_window = self.classify_memory_by_age(created_date)
                
                # Filter by time window if specified
                if time_window and file_time_window != time_window:
                    continue
                
                importance_score = frontmatter.get('importance_score', 5)
                
                # Filter by minimum importance
                if min_importance and importance_score >= min_importance:
                    continue
                
                # Don't condense critical or very recent memories
                if importance_score >= self.critical_threshold or age_days <= 7:
                    continue
                
                file_size_kb = memory_file.stat().st_size / 1024
                category = frontmatter.get('category', 'unknown')
                
                # Count interactions in the same time period for batching
                interaction_count = len(list(memory_file.parent.glob('*.md')))
                
                # Calculate condensation priority (higher = more urgent)
                priority = self._calculate_condensation_priority(
                    age_days, importance_score, file_size_kb, interaction_count
                )
                
                candidate = CondensationCandidate(
                    file_path=str(memory_file),
                    age_days=age_days,
                    importance_score=importance_score,
                    size_kb=file_size_kb,
                    category=category,
                    interaction_count=interaction_count,
                    condensation_priority=priority
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                logger.warning(f"Error analyzing condensation candidate {memory_file}: {e}")
        
        # Sort by priority (highest first) and limit results
        candidates.sort(key=lambda x: x.condensation_priority, reverse=True)
        return candidates[:max_candidates]
    
    def _calculate_condensation_priority(
        self, 
        age_days: int, 
        importance_score: int, 
        size_kb: float, 
        interaction_count: int
    ) -> float:
        """
        Calculate priority score for condensation (higher = more urgent).
        
        Args:
            age_days: Age of the memory in days
            importance_score: Importance score (1-10)
            size_kb: File size in KB
            interaction_count: Number of related interactions
            
        Returns:
            Priority score (0-100)
        """
        # Base priority increases with age
        age_factor = min(age_days / 365.0, 2.0)  # Cap at 2 years
        
        # Decrease priority for high importance
        importance_factor = max(0.1, (10 - importance_score) / 10.0)
        
        # Increase priority for large files
        size_factor = min(size_kb / 1024.0, 2.0)  # Cap at 1MB
        
        # Increase priority for many related interactions (better condensation potential)
        batch_factor = min(interaction_count / 20.0, 1.5)
        
        priority = (age_factor * 40) + (importance_factor * 30) + (size_factor * 20) + (batch_factor * 10)
        
        return min(priority, 100.0)
    
    def organize_memories_by_time(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Organize memories into time-based condensed structure.
        
        Args:
            dry_run: If True, only analyze without making changes
            
        Returns:
            Summary of organization actions taken or planned
        """
        summary = {
            'analyzed_files': 0,
            'moved_to_recent': 0,
            'moved_to_medium': 0,
            'moved_to_archive': 0,
            'condensation_candidates': 0,
            'errors': [],
            'dry_run': dry_run
        }
        
        if not self.interactions_dir.exists():
            return summary
        
        candidates = self.identify_condensation_candidates()
        summary['condensation_candidates'] = len(candidates)
        
        # Group candidates by time window
        window_groups = {
            TimeWindow.RECENT: [],
            TimeWindow.MEDIUM: [],
            TimeWindow.ARCHIVE: []
        }
        
        for candidate in candidates:
            try:
                memory_file = Path(candidate.file_path)
                frontmatter, content = read_memory_file(memory_file)
                created_str = frontmatter.get('created', '')
                
                if created_str:
                    created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    time_window = self.classify_memory_by_age(created_date)
                    window_groups[time_window].append(candidate)
                    summary['analyzed_files'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {candidate.file_path}: {e}"
                summary['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Process each time window
        for time_window, group_candidates in window_groups.items():
            if not group_candidates:
                continue
            
            condensed_path = self.condensed_dir / time_window.value
            
            if not dry_run:
                # Create condensed summaries for each group
                self._create_condensed_summaries(group_candidates, condensed_path, time_window)
            
            # Update summary counts
            if time_window == TimeWindow.RECENT:
                summary['moved_to_recent'] = len(group_candidates)
            elif time_window == TimeWindow.MEDIUM:
                summary['moved_to_medium'] = len(group_candidates)
            else:
                summary['moved_to_archive'] = len(group_candidates)
        
        return summary
    
    def _create_condensed_summaries(
        self, 
        candidates: List[CondensationCandidate], 
        output_path: Path, 
        time_window: TimeWindow
    ) -> None:
        """
        Create condensed summaries for a group of memory candidates.
        
        Args:
            candidates: List of condensation candidates
            output_path: Path to store condensed summaries
            time_window: Time window being processed
        """
        # Group candidates by category and time period
        groups = self._group_candidates_for_condensation(candidates)
        
        for group_key, group_candidates in groups.items():
            try:
                condensed_content = self._condense_memory_group(group_candidates, time_window)
                
                if condensed_content:
                    # Create condensed file
                    condensed_filename = f"{group_key}_{datetime.now().strftime('%Y%m%d')}.md"
                    condensed_file_path = output_path / condensed_filename
                    
                    # Write condensed memory
                    condensed_frontmatter = {
                        'created': datetime.now().isoformat(),
                        'memory_type': 'condensed',
                        'time_window': time_window.value,
                        'source_count': len(group_candidates),
                        'importance_score': max(c.importance_score for c in group_candidates),
                        'category': group_key.split('_')[0],
                        'condensation_date': datetime.now().isoformat(),
                        'source_files': [c.file_path for c in group_candidates]
                    }
                    
                    write_memory_file(condensed_file_path, condensed_frontmatter, condensed_content)
                    logger.info(f"Created condensed memory: {condensed_file_path}")
                    
            except Exception as e:
                logger.error(f"Error creating condensed summary for group {group_key}: {e}")
    
    def _group_candidates_for_condensation(
        self, 
        candidates: List[CondensationCandidate]
    ) -> Dict[str, List[CondensationCandidate]]:
        """
        Group candidates by category and time period for efficient condensation.
        
        Args:
            candidates: List of candidates to group
            
        Returns:
            Dict mapping group keys to candidate lists
        """
        groups = {}
        
        for candidate in candidates:
            # Create group key based on category and rough time period
            file_path = Path(candidate.file_path)
            date_part = file_path.parent.name  # Usually YYYY-MM format
            group_key = f"{candidate.category}_{date_part}"
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(candidate)
        
        # Only return groups with multiple candidates for condensation
        return {k: v for k, v in groups.items() if len(v) >= self.min_interactions_for_condensation}
    
    def _condense_memory_group(
        self, 
        candidates: List[CondensationCandidate], 
        time_window: TimeWindow
    ) -> str:
        """
        Condense a group of related memories into a summary.
        
        Args:
            candidates: List of memory candidates to condense
            time_window: Time window for condensation strategy
            
        Returns:
            Condensed content string
        """
        if not candidates:
            return ""
        
        # Collect all content from candidates
        all_content = []
        for candidate in candidates:
            try:
                frontmatter, content = read_memory_file(candidate.file_path)
                all_content.append({
                    'content': content,
                    'importance': candidate.importance_score,
                    'date': frontmatter.get('created', ''),
                    'category': candidate.category
                })
            except Exception as e:
                logger.warning(f"Error reading candidate {candidate.file_path}: {e}")
        
        if not all_content:
            return ""
        
        # Sort by importance and date
        all_content.sort(key=lambda x: (x['importance'], x['date']), reverse=True)
        
        # Apply condensation strategy based on time window
        if time_window == TimeWindow.RECENT:
            # Recent: Keep most important details, light summarization
            return self._create_recent_summary(all_content)
        elif time_window == TimeWindow.MEDIUM:
            # Medium: Summarize key points, keep important facts
            return self._create_medium_summary(all_content)
        else:
            # Archive: Keep only essential facts
            return self._create_archive_summary(all_content)
    
    def _create_recent_summary(self, content_list: List[Dict[str, Any]]) -> str:
        """Create summary for recent time window (light condensation)."""
        summary_parts = [
            "# Recent Memory Summary",
            f"*Condensed from {len(content_list)} interactions*",
            ""
        ]
        
        # Keep high-importance content mostly intact
        for item in content_list:
            if item['importance'] >= self.high_threshold:
                summary_parts.append(f"## Important: {item['date']}")
                summary_parts.append(item['content'][:500] + "..." if len(item['content']) > 500 else item['content'])
                summary_parts.append("")
        
        # Summarize medium importance content
        medium_items = [item for item in content_list if self.medium_threshold <= item['importance'] < self.high_threshold]
        if medium_items:
            summary_parts.append("## Key Points")
            for item in medium_items[:5]:  # Limit to top 5
                summary_parts.append(f"- {item['content'][:200]}...")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _create_medium_summary(self, content_list: List[Dict[str, Any]]) -> str:
        """Create summary for medium time window (moderate condensation)."""
        summary_parts = [
            "# Medium-Term Memory Summary",
            f"*Condensed from {len(content_list)} interactions*",
            ""
        ]
        
        # Extract key themes and important facts
        important_items = [item for item in content_list if item['importance'] >= self.high_threshold]
        medium_items = [item for item in content_list if self.medium_threshold <= item['importance'] < self.high_threshold]
        
        if important_items:
            summary_parts.append("## Important Facts")
            for item in important_items[:3]:
                summary_parts.append(f"- {item['content'][:300]}")
            summary_parts.append("")
        
        if medium_items:
            summary_parts.append("## Key Themes")
            # Simple keyword extraction and grouping
            themes = {}
            for item in medium_items:
                words = item['content'].lower().split()
                for word in words:
                    if len(word) > 4:  # Simple filter for meaningful words
                        themes[word] = themes.get(word, 0) + 1
            
            top_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]
            for theme, count in top_themes:
                summary_parts.append(f"- {theme.title()} (mentioned {count} times)")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _create_archive_summary(self, content_list: List[Dict[str, Any]]) -> str:
        """Create summary for archive time window (heavy condensation)."""
        summary_parts = [
            "# Archived Memory Summary",
            f"*Condensed from {len(content_list)} interactions*",
            ""
        ]
        
        # Keep only critical facts
        critical_items = [item for item in content_list if item['importance'] >= self.critical_threshold]
        important_items = [item for item in content_list if item['importance'] >= self.high_threshold]
        
        if critical_items:
            summary_parts.append("## Critical Information")
            for item in critical_items[:2]:
                summary_parts.append(f"- {item['content'][:150]}")
            summary_parts.append("")
        
        if important_items:
            summary_parts.append("## Key Facts")
            summary_parts.append(f"- Period covered: {len(content_list)} interactions")
            summary_parts.append(f"- Main categories: {', '.join(set(item['category'] for item in content_list))}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def get_condensation_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the current state of memory condensation.
        
        Returns:
            Dict with condensation metrics and recommendations
        """
        age_distribution = self.get_memory_age_distribution()
        candidates = self.identify_condensation_candidates()
        
        total_size_mb = sum(metrics.memory_size_mb for metrics in age_distribution.values())
        total_memories = sum(metrics.total_memories for metrics in age_distribution.values())
        
        # Calculate potential savings
        potential_savings_mb = sum(c.size_kb for c in candidates) / 1024 * self.condensation_ratio
        
        return {
            'current_state': {
                'total_memories': total_memories,
                'total_size_mb': round(total_size_mb, 2),
                'age_distribution': {k: {
                    'count': v.total_memories,
                    'size_mb': round(v.memory_size_mb, 2),
                    'avg_age_days': round(v.average_age_days, 1)
                } for k, v in age_distribution.items()}
            },
            'condensation_opportunities': {
                'total_candidates': len(candidates),
                'potential_savings_mb': round(potential_savings_mb, 2),
                'high_priority_candidates': len([c for c in candidates if c.condensation_priority > 70]),
                'recommended_action': self._get_condensation_recommendation(total_size_mb, len(candidates))
            }
        }
    
    def _get_condensation_recommendation(self, total_size_mb: float, candidate_count: int) -> str:
        """Get recommendation for condensation action."""
        max_size = self.config.get('max_memory_size_mb', 100)
        
        if total_size_mb > max_size:
            return "URGENT: Memory size exceeds limits, immediate condensation recommended"
        elif candidate_count > 50:
            return "HIGH: Many condensation candidates available, condensation recommended"
        elif candidate_count > 20:
            return "MEDIUM: Some condensation opportunities, consider condensation"
        else:
            return "LOW: Few condensation opportunities, no immediate action needed" 