"""
Automated Memory Condensation and Summarization System

This module implements intelligent memory condensation using AI-powered summarization,
chain-of-thought reasoning for memory decisions, and automated condensation workflows.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

# Handle imports gracefully for both package and standalone execution
try:
    from .time_based_organizer import TimeBasedOrganizer, TimeWindow, CondensationCandidate
    from .file_operations import read_memory_file, write_memory_file
    from .memory_manager import MemoryEntry, ImportanceLevel
except ImportError:
    # Fallback for standalone execution
    from time_based_organizer import TimeBasedOrganizer, TimeWindow, CondensationCandidate
    from file_operations import read_memory_file, write_memory_file
    from memory_manager import MemoryEntry, ImportanceLevel

logger = logging.getLogger(__name__)


class CondensationStrategy(Enum):
    """Strategy for memory condensation."""
    PRESERVE_IMPORTANT = "preserve_important"  # Keep high-importance memories intact
    EXTRACT_THEMES = "extract_themes"         # Extract key themes and patterns
    FACTUAL_SUMMARY = "factual_summary"       # Create factual summaries only
    ADAPTIVE = "adaptive"                     # Choose strategy based on content


class CondensationTrigger(Enum):
    """Triggers for automated condensation."""
    SIZE_THRESHOLD = "size_threshold"         # Memory size exceeds threshold
    TIME_BASED = "time_based"                # Scheduled time-based condensation
    MANUAL = "manual"                        # User-initiated condensation
    IMPORTANCE_DECAY = "importance_decay"     # Importance-based aging


@dataclass
class CondensationReasoning:
    """Chain-of-thought reasoning for condensation decisions."""
    timestamp: str
    decision: str
    reasoning_steps: List[str]
    factors_considered: Dict[str, Any]
    confidence_score: float
    alternative_approaches: List[str]
    expected_outcome: str
    preservation_rationale: str


@dataclass
class CondensationResult:
    """Result of a memory condensation operation."""
    source_files: List[str]
    condensed_file: str
    original_size_kb: float
    condensed_size_kb: float
    compression_ratio: float
    memories_processed: int
    reasoning: CondensationReasoning
    quality_metrics: Dict[str, float]
    timestamp: str
    strategy_used: CondensationStrategy


class MemoryCondensationSystem:
    """
    Automated memory condensation and summarization system.
    
    This system provides intelligent memory condensation with:
    - AI-powered summarization strategies
    - Chain-of-thought reasoning for decisions
    - Automated condensation workflows
    - Quality assessment and validation
    """
    
    def __init__(
        self, 
        base_path: str, 
        config: Optional[Dict[str, Any]] = None,
        ai_client: Optional[Any] = None
    ):
        """
        Initialize the memory condensation system.
        
        Args:
            base_path: Base path for memory storage
            config: Optional configuration overrides
            ai_client: Optional AI client for summarization
        """
        self.base_path = Path(base_path)
        self.config = self._load_config(config)
        self.ai_client = ai_client
        
        # Initialize time-based organizer
        self.time_organizer = TimeBasedOrganizer(str(self.base_path), self.config)
        
        # Condensation settings
        self.size_threshold_mb = self.config.get('size_threshold_mb', 50)
        self.condensation_schedule_hours = self.config.get('condensation_schedule_hours', 24)
        self.min_batch_size = self.config.get('min_batch_size', 3)
        self.max_batch_size = self.config.get('max_batch_size', 20)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        
        # Reasoning and decision-making
        self.reasoning_enabled = self.config.get('reasoning_enabled', True)
        self.preserve_critical_threshold = self.config.get('preserve_critical_threshold', 9)
        self.adaptive_strategy = self.config.get('adaptive_strategy', True)
        
        # Directories
        self.condensation_log_dir = self.base_path / 'system' / 'condensation_logs'
        self.condensation_log_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration with defaults."""
        default_config = {
            'size_threshold_mb': 50,
            'condensation_schedule_hours': 24,
            'min_batch_size': 3,
            'max_batch_size': 20,
            'quality_threshold': 0.7,
            'reasoning_enabled': True,
            'preserve_critical_threshold': 9,
            'adaptive_strategy': True,
            'backup_before_condensation': True,
            'validate_condensed_memories': True,
            'max_summary_length': 1000,
            'preserve_metadata': True
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def should_trigger_condensation(self) -> Tuple[bool, List[CondensationTrigger], Dict[str, Any]]:
        """
        Determine if condensation should be triggered.
        
        Returns:
            Tuple of (should_trigger, triggers, analysis_data)
        """
        triggers = []
        analysis = {}
        
        # Check size threshold
        metrics = self.time_organizer.get_condensation_metrics()
        current_size_mb = metrics['current_state']['total_size_mb']
        
        if current_size_mb > self.size_threshold_mb:
            triggers.append(CondensationTrigger.SIZE_THRESHOLD)
            analysis['size_trigger'] = {
                'current_size_mb': current_size_mb,
                'threshold_mb': self.size_threshold_mb,
                'excess_mb': current_size_mb - self.size_threshold_mb
            }
        
        # Check time-based trigger
        last_condensation = self._get_last_condensation_time()
        hours_since_last = (datetime.now() - last_condensation).total_seconds() / 3600
        
        if hours_since_last >= self.condensation_schedule_hours:
            triggers.append(CondensationTrigger.TIME_BASED)
            analysis['time_trigger'] = {
                'hours_since_last': hours_since_last,
                'schedule_hours': self.condensation_schedule_hours,
                'last_condensation': last_condensation.isoformat()
            }
        
        # Check importance decay trigger
        candidates = self.time_organizer.identify_condensation_candidates()
        high_priority_candidates = [c for c in candidates if c.condensation_priority > 80]
        
        if len(high_priority_candidates) > 10:
            triggers.append(CondensationTrigger.IMPORTANCE_DECAY)
            analysis['importance_trigger'] = {
                'high_priority_count': len(high_priority_candidates),
                'total_candidates': len(candidates),
                'priority_threshold': 80
            }
        
        should_trigger = len(triggers) > 0
        return should_trigger, triggers, analysis
    
    def _get_last_condensation_time(self) -> datetime:
        """Get the timestamp of the last condensation operation."""
        log_files = list(self.condensation_log_dir.glob('condensation_*.json'))
        if not log_files:
            return datetime.now() - timedelta(days=30)  # Default to 30 days ago
        
        latest_log_file = max(log_files, key=lambda f: f.stat().st_mtime)
        try:
            with open(latest_log_file, 'r') as f:
                log_data = json.load(f)
                return datetime.fromisoformat(log_data.get('timestamp', ''))
        except Exception as e:
            logger.warning(f"Error reading condensation log {latest_log_file}: {e}")
            return datetime.now() - timedelta(days=30)
    
    async def run_automated_condensation(
        self, 
        dry_run: bool = False,
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run automated memory condensation process.
        
        Args:
            dry_run: If True, analyze but don't make changes
            max_batches: Maximum number of batches to process
            
        Returns:
            Summary of condensation operations
        """
        logger.info("Starting automated memory condensation process")
        
        # Check if condensation should be triggered
        should_trigger, triggers, trigger_analysis = self.should_trigger_condensation()
        
        if not should_trigger and not dry_run:
            return {
                'status': 'skipped',
                'reason': 'No condensation triggers activated',
                'analysis': trigger_analysis
            }
        
        # Get condensation candidates
        candidates = self.time_organizer.identify_condensation_candidates()
        if not candidates:
            return {
                'status': 'completed',
                'reason': 'No condensation candidates found',
                'triggers': [t.value for t in triggers],
                'analysis': trigger_analysis
            }
        
        # Group candidates into batches
        batches = self._create_condensation_batches(candidates)
        if max_batches:
            batches = batches[:max_batches]
        
        # Process each batch
        results = []
        total_processed = 0
        total_savings_mb = 0
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing condensation batch {i+1}/{len(batches)}")
            
            if not dry_run:
                batch_result = await self._process_condensation_batch(batch)
                results.append(batch_result)
                total_processed += batch_result.memories_processed
                total_savings_mb += (batch_result.original_size_kb - batch_result.condensed_size_kb) / 1024
            else:
                # Dry run analysis
                batch_size = sum(c.size_kb for c in batch) / 1024
                estimated_savings = batch_size * 0.7  # Estimate 70% compression
                results.append({
                    'batch_id': i,
                    'candidates': len(batch),
                    'estimated_size_mb': batch_size,
                    'estimated_savings_mb': estimated_savings,
                    'dry_run': True
                })
                total_processed += len(batch)
                total_savings_mb += estimated_savings
        
        # Create summary
        summary = {
            'status': 'completed',
            'dry_run': dry_run,
            'triggers': [t.value for t in triggers],
            'trigger_analysis': trigger_analysis,
            'batches_processed': len(batches),
            'memories_processed': total_processed,
            'estimated_savings_mb': round(total_savings_mb, 2),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the condensation operation
        if not dry_run:
            self._log_condensation_operation(summary)
        
        logger.info(f"Automated condensation completed: {total_processed} memories processed")
        return summary
    
    def _create_condensation_batches(
        self, 
        candidates: List[CondensationCandidate]
    ) -> List[List[CondensationCandidate]]:
        """
        Group condensation candidates into optimal batches.
        
        Args:
            candidates: List of condensation candidates
            
        Returns:
            List of candidate batches
        """
        # Sort candidates by priority (highest first)
        sorted_candidates = sorted(candidates, key=lambda c: c.condensation_priority, reverse=True)
        
        # Group by category and time period for better condensation
        groups = {}
        for candidate in sorted_candidates:
            # Create group key based on category and rough time period
            file_path = Path(candidate.file_path)
            date_part = file_path.parent.name if file_path.parent.name else 'unknown'
            group_key = f"{candidate.category}_{date_part}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(candidate)
        
        # Create batches from groups
        batches = []
        for group_candidates in groups.values():
            if len(group_candidates) >= self.min_batch_size:
                # Split larger groups into multiple batches
                for i in range(0, len(group_candidates), self.max_batch_size):
                    batch = group_candidates[i:i + self.max_batch_size]
                    if len(batch) >= self.min_batch_size:
                        batches.append(batch)
        
        return batches
    
    async def _process_condensation_batch(
        self, 
        batch: List[CondensationCandidate]
    ) -> CondensationResult:
        """
        Process a batch of condensation candidates.
        
        Args:
            batch: List of candidates to condense
            
        Returns:
            Condensation result
        """
        # Analyze batch characteristics
        batch_analysis = self._analyze_batch(batch)
        
        # Generate reasoning for condensation decisions
        reasoning = await self._generate_condensation_reasoning(batch, batch_analysis)
        
        # Choose condensation strategy
        strategy = self._choose_condensation_strategy(batch_analysis, reasoning)
        
        # Read all source memories
        source_memories = []
        original_size_kb = 0
        
        for candidate in batch:
            try:
                frontmatter, content = read_memory_file(candidate.file_path)
                source_memories.append({
                    'path': candidate.file_path,
                    'frontmatter': frontmatter,
                    'content': content,
                    'candidate': candidate
                })
                original_size_kb += candidate.size_kb
            except Exception as e:
                logger.error(f"Error reading memory file {candidate.file_path}: {e}")
        
        # Perform condensation
        condensed_content, condensed_frontmatter = await self._condense_memories(
            source_memories, strategy, reasoning
        )
        
        # Create condensed file
        condensed_file_path = self._create_condensed_file_path(batch, strategy)
        write_memory_file(condensed_file_path, condensed_frontmatter, condensed_content)
        
        # Calculate metrics
        condensed_size_kb = condensed_file_path.stat().st_size / 1024
        compression_ratio = condensed_size_kb / original_size_kb if original_size_kb > 0 else 0
        
        # Assess quality
        quality_metrics = await self._assess_condensation_quality(
            source_memories, condensed_content, strategy
        )
        
        # Create result
        result = CondensationResult(
            source_files=[m['path'] for m in source_memories],
            condensed_file=str(condensed_file_path),
            original_size_kb=original_size_kb,
            condensed_size_kb=condensed_size_kb,
            compression_ratio=compression_ratio,
            memories_processed=len(batch),
            reasoning=reasoning,
            quality_metrics=quality_metrics,
            timestamp=datetime.now().isoformat(),
            strategy_used=strategy
        )
        
        # Archive original files if quality is acceptable
        if quality_metrics.get('overall_quality', 0) >= self.quality_threshold:
            await self._archive_source_files(source_memories)
        
        return result
    
    def _analyze_batch(self, batch: List[CondensationCandidate]) -> Dict[str, Any]:
        """Analyze characteristics of a condensation batch."""
        if not batch:
            return {}
        
        # Calculate statistics
        total_size = sum(c.size_kb for c in batch)
        avg_importance = sum(c.importance_score for c in batch) / len(batch)
        avg_age = sum(c.age_days for c in batch) / len(batch)
        categories = list(set(c.category for c in batch))
        
        # Determine time window
        time_windows = []
        for candidate in batch:
            file_path = Path(candidate.file_path)
            try:
                frontmatter, _ = read_memory_file(file_path)
                created_str = frontmatter.get('created', '')
                if created_str:
                    created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    window = self.time_organizer.classify_memory_by_age(created_date)
                    time_windows.append(window)
            except Exception:
                continue
        
        primary_time_window = max(set(time_windows), key=time_windows.count) if time_windows else TimeWindow.MEDIUM
        
        return {
            'batch_size': len(batch),
            'total_size_kb': total_size,
            'avg_importance': avg_importance,
            'avg_age_days': avg_age,
            'categories': categories,
            'primary_category': max(set(c.category for c in batch), key=[c.category for c in batch].count),
            'primary_time_window': primary_time_window,
            'high_importance_count': len([c for c in batch if c.importance_score >= 7]),
            'critical_importance_count': len([c for c in batch if c.importance_score >= 9])
        }
    
    async def _generate_condensation_reasoning(
        self, 
        batch: List[CondensationCandidate],
        batch_analysis: Dict[str, Any]
    ) -> CondensationReasoning:
        """Generate chain-of-thought reasoning for condensation decisions."""
        
        reasoning_steps = [
            f"Analyzing batch of {len(batch)} memory candidates for condensation",
            f"Total size: {batch_analysis.get('total_size_kb', 0):.1f} KB",
            f"Average importance: {batch_analysis.get('avg_importance', 0):.1f}/10",
            f"Average age: {batch_analysis.get('avg_age_days', 0):.0f} days",
            f"Primary category: {batch_analysis.get('primary_category', 'unknown')}",
            f"Primary time window: {batch_analysis.get('primary_time_window', TimeWindow.MEDIUM).value}"
        ]
        
        # Analyze preservation needs
        high_importance_count = batch_analysis.get('high_importance_count', 0)
        critical_count = batch_analysis.get('critical_importance_count', 0)
        
        if critical_count > 0:
            reasoning_steps.append(f"Found {critical_count} critical importance memories - will preserve key details")
        if high_importance_count > 0:
            reasoning_steps.append(f"Found {high_importance_count} high importance memories - will maintain essential information")
        
        # Determine condensation approach
        primary_window = batch_analysis.get('primary_time_window', TimeWindow.MEDIUM)
        if primary_window == TimeWindow.RECENT:
            reasoning_steps.append("Recent memories - applying light condensation strategy")
        elif primary_window == TimeWindow.MEDIUM:
            reasoning_steps.append("Medium-age memories - applying moderate condensation with theme extraction")
        else:
            reasoning_steps.append("Archive memories - applying aggressive condensation, preserving only essential facts")
        
        # Calculate confidence
        confidence_factors = {
            'batch_coherence': 0.8 if len(batch_analysis.get('categories', [])) <= 2 else 0.6,
            'importance_clarity': 0.9 if batch_analysis.get('avg_importance', 0) >= 5 else 0.7,
            'age_consistency': 0.8 if batch_analysis.get('avg_age_days', 0) > 30 else 0.6,
        }
        confidence_score = sum(confidence_factors.values()) / len(confidence_factors)
        
        # Alternative approaches
        alternatives = [
            "Individual file condensation instead of batch processing",
            "Delay condensation until more memories accumulate",
            "Apply different time window strategy"
        ]
        
        # Expected outcome
        expected_compression = 0.7 if primary_window == TimeWindow.ARCHIVE else 0.5
        expected_outcome = f"Expected {expected_compression*100:.0f}% size reduction while preserving key information"
        
        # Preservation rationale
        preservation_rationale = "Preserving high-importance memories and maintaining information coherence across the batch"
        
        return CondensationReasoning(
            timestamp=datetime.now().isoformat(),
            decision=f"Condense batch using {primary_window.value} strategy",
            reasoning_steps=reasoning_steps,
            factors_considered=confidence_factors,
            confidence_score=confidence_score,
            alternative_approaches=alternatives,
            expected_outcome=expected_outcome,
            preservation_rationale=preservation_rationale
        )
    
    def _choose_condensation_strategy(
        self, 
        batch_analysis: Dict[str, Any],
        reasoning: CondensationReasoning
    ) -> CondensationStrategy:
        """Choose the optimal condensation strategy for a batch."""
        
        if not self.adaptive_strategy:
            return CondensationStrategy.EXTRACT_THEMES
        
        # Check for critical memories
        if batch_analysis.get('critical_importance_count', 0) > 0:
            return CondensationStrategy.PRESERVE_IMPORTANT
        
        # Choose based on time window
        primary_window = batch_analysis.get('primary_time_window', TimeWindow.MEDIUM)
        
        if primary_window == TimeWindow.RECENT:
            return CondensationStrategy.PRESERVE_IMPORTANT
        elif primary_window == TimeWindow.MEDIUM:
            return CondensationStrategy.EXTRACT_THEMES
        else:
            return CondensationStrategy.FACTUAL_SUMMARY
    
    async def _condense_memories(
        self,
        source_memories: List[Dict[str, Any]],
        strategy: CondensationStrategy,
        reasoning: CondensationReasoning
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Condense memories using the specified strategy.
        
        Args:
            source_memories: List of source memory data
            strategy: Condensation strategy to use
            reasoning: Reasoning for the condensation
            
        Returns:
            Tuple of (condensed_content, condensed_frontmatter)
        """
        # Combine all source content
        all_content = []
        all_frontmatter = []
        
        for memory in source_memories:
            all_content.append(memory['content'])
            all_frontmatter.append(memory['frontmatter'])
        
        # Apply condensation strategy
        if strategy == CondensationStrategy.PRESERVE_IMPORTANT:
            condensed_content = self._preserve_important_condensation(all_content, all_frontmatter)
        elif strategy == CondensationStrategy.EXTRACT_THEMES:
            condensed_content = self._extract_themes_condensation(all_content, all_frontmatter)
        elif strategy == CondensationStrategy.FACTUAL_SUMMARY:
            condensed_content = self._factual_summary_condensation(all_content, all_frontmatter)
        else:
            condensed_content = self._extract_themes_condensation(all_content, all_frontmatter)
        
        # Create condensed frontmatter
        condensed_frontmatter = {
            'created': datetime.now().isoformat(),
            'memory_type': 'condensed',
            'condensation_strategy': strategy.value,
            'source_count': len(source_memories),
            'source_files': [m['path'] for m in source_memories],
            'reasoning': asdict(reasoning),
            'importance_score': max(fm.get('importance_score', 5) for fm in all_frontmatter),
            'categories': list(set(fm.get('category', 'unknown') for fm in all_frontmatter)),
            'condensation_timestamp': datetime.now().isoformat()
        }
        
        return condensed_content, condensed_frontmatter
    
    def _preserve_important_condensation(
        self, 
        content_list: List[str], 
        frontmatter_list: List[Dict[str, Any]]
    ) -> str:
        """Condensation that preserves important information."""
        sections = []
        sections.append("# Condensed Memory - Important Information Preserved")
        sections.append(f"*Condensed from {len(content_list)} memories*")
        sections.append("")
        
        # Group by importance
        high_importance = []
        medium_importance = []
        low_importance = []
        
        for i, (content, frontmatter) in enumerate(zip(content_list, frontmatter_list)):
            importance = frontmatter.get('importance_score', 5)
            if importance >= 8:
                high_importance.append((content, frontmatter))
            elif importance >= 6:
                medium_importance.append((content, frontmatter))
            else:
                low_importance.append((content, frontmatter))
        
        # Add high importance content (mostly preserved)
        if high_importance:
            sections.append("## High Importance Information")
            for content, frontmatter in high_importance:
                date = frontmatter.get('created', 'Unknown date')
                sections.append(f"### {date}")
                # Keep high importance content mostly intact
                sections.append(content[:800] + "..." if len(content) > 800 else content)
                sections.append("")
        
        # Add medium importance content (summarized)
        if medium_importance:
            sections.append("## Key Information")
            for content, frontmatter in medium_importance:
                sections.append(f"- {content[:300]}...")
                sections.append("")
        
        # Add low importance content (brief mentions)
        if low_importance:
            sections.append("## Additional Context")
            brief_mentions = [content[:100] + "..." for content, _ in low_importance[:3]]
            sections.extend([f"- {mention}" for mention in brief_mentions])
            sections.append("")
        
        return "\n".join(sections)
    
    def _extract_themes_condensation(
        self, 
        content_list: List[str], 
        frontmatter_list: List[Dict[str, Any]]
    ) -> str:
        """Condensation that extracts key themes and patterns."""
        sections = []
        sections.append("# Condensed Memory - Theme-Based Summary")
        sections.append(f"*Condensed from {len(content_list)} memories*")
        sections.append("")
        
        # Simple theme extraction (keyword frequency)
        all_text = " ".join(content_list).lower()
        words = all_text.split()
        
        # Count significant words (> 4 characters)
        word_freq = {}
        for word in words:
            if len(word) > 4 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top themes
        top_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_themes:
            sections.append("## Key Themes")
            for theme, count in top_themes:
                sections.append(f"- **{theme.title()}** (mentioned {count} times)")
            sections.append("")
        
        # Add timeline of important events
        dated_content = []
        for content, frontmatter in zip(content_list, frontmatter_list):
            date = frontmatter.get('created', '')
            importance = frontmatter.get('importance_score', 5)
            if date and importance >= 6:
                dated_content.append((date, content[:200], importance))
        
        if dated_content:
            sections.append("## Timeline of Key Events")
            dated_content.sort(key=lambda x: x[0])
            for date, content, importance in dated_content:
                sections.append(f"- **{date[:10]}**: {content}... (importance: {importance})")
            sections.append("")
        
        return "\n".join(sections)
    
    def _factual_summary_condensation(
        self, 
        content_list: List[str], 
        frontmatter_list: List[Dict[str, Any]]
    ) -> str:
        """Condensation that creates concise factual summaries."""
        sections = []
        sections.append("# Condensed Memory - Factual Summary")
        sections.append(f"*Condensed from {len(content_list)} memories*")
        sections.append("")
        
        # Extract key facts (sentences with high information density)
        all_sentences = []
        for content in content_list:
            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Meaningful sentences
                    all_sentences.append(sentence)
        
        # Simple scoring: prefer sentences with numbers, proper nouns, specific terms
        scored_sentences = []
        for sentence in all_sentences:
            score = 0
            if any(char.isdigit() for char in sentence):
                score += 2  # Contains numbers
            if any(word[0].isupper() for word in sentence.split()):
                score += 1  # Contains proper nouns
            if len(sentence.split()) > 5:
                score += 1  # Reasonable length
            scored_sentences.append((sentence, score))
        
        # Get top facts
        top_facts = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:15]
        
        if top_facts:
            sections.append("## Key Facts")
            for sentence, score in top_facts:
                sections.append(f"- {sentence}.")
            sections.append("")
        
        # Add summary statistics
        sections.append("## Summary Statistics")
        sections.append(f"- Period covered: {len(content_list)} interactions")
        categories = list(set(fm.get('category', 'unknown') for fm in frontmatter_list))
        sections.append(f"- Categories: {', '.join(categories)}")
        avg_importance = sum(fm.get('importance_score', 5) for fm in frontmatter_list) / len(frontmatter_list)
        sections.append(f"- Average importance: {avg_importance:.1f}/10")
        sections.append("")
        
        return "\n".join(sections)
    
    def _create_condensed_file_path(
        self, 
        batch: List[CondensationCandidate], 
        strategy: CondensationStrategy
    ) -> Path:
        """Create path for condensed memory file."""
        # Determine time window
        batch_analysis = self._analyze_batch(batch)
        time_window = batch_analysis.get('primary_time_window', TimeWindow.MEDIUM)
        category = batch_analysis.get('primary_category', 'mixed')
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"condensed_{category}_{strategy.value}_{timestamp}.md"
        
        # Determine directory
        condensed_dir = self.base_path / 'condensed' / time_window.value
        condensed_dir.mkdir(parents=True, exist_ok=True)
        
        return condensed_dir / filename
    
    async def _assess_condensation_quality(
        self,
        source_memories: List[Dict[str, Any]],
        condensed_content: str,
        strategy: CondensationStrategy
    ) -> Dict[str, float]:
        """Assess the quality of condensation."""
        metrics = {}
        
        # Calculate compression ratio
        original_length = sum(len(m['content']) for m in source_memories)
        condensed_length = len(condensed_content)
        compression_ratio = condensed_length / original_length if original_length > 0 else 0
        metrics['compression_ratio'] = compression_ratio
        
        # Information preservation score (heuristic)
        # Check if key terms from original are preserved
        original_text = " ".join(m['content'] for m in source_memories).lower()
        original_words = set(word for word in original_text.split() if len(word) > 4)
        condensed_words = set(word for word in condensed_content.lower().split() if len(word) > 4)
        
        if original_words:
            preserved_ratio = len(original_words & condensed_words) / len(original_words)
            metrics['information_preservation'] = preserved_ratio
        else:
            metrics['information_preservation'] = 0.5
        
        # Structure quality (has proper sections and formatting)
        structure_score = 0.0
        if "##" in condensed_content:
            structure_score += 0.3
        if "- " in condensed_content:
            structure_score += 0.3
        if len(condensed_content.split('\n')) > 5:
            structure_score += 0.4
        metrics['structure_quality'] = min(structure_score, 1.0)
        
        # Strategy effectiveness
        strategy_score = 0.8  # Base score
        if strategy == CondensationStrategy.PRESERVE_IMPORTANT and "High Importance" in condensed_content:
            strategy_score = 0.9
        elif strategy == CondensationStrategy.EXTRACT_THEMES and "Key Themes" in condensed_content:
            strategy_score = 0.9
        elif strategy == CondensationStrategy.FACTUAL_SUMMARY and "Key Facts" in condensed_content:
            strategy_score = 0.9
        metrics['strategy_effectiveness'] = strategy_score
        
        # Overall quality score
        metrics['overall_quality'] = (
            metrics['information_preservation'] * 0.4 +
            metrics['structure_quality'] * 0.3 +
            metrics['strategy_effectiveness'] * 0.3
        )
        
        return metrics
    
    async def _archive_source_files(self, source_memories: List[Dict[str, Any]]) -> None:
        """Archive original source files after successful condensation."""
        if not self.config.get('backup_before_condensation', True):
            return
        
        archive_dir = self.base_path / 'system' / 'archived_memories'
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for memory in source_memories:
            source_path = Path(memory['path'])
            if source_path.exists():
                # Create archive filename
                archive_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
                archive_path = archive_dir / archive_filename
                
                # Move file to archive
                source_path.rename(archive_path)
                logger.info(f"Archived {source_path} to {archive_path}")
    
    def _log_condensation_operation(self, summary: Dict[str, Any]) -> None:
        """Log condensation operation for tracking and analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"condensation_{timestamp}.json"
        log_path = self.condensation_log_dir / log_filename
        
        with open(log_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Condensation operation logged to {log_path}")
    
    def get_condensation_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get condensation operation history."""
        cutoff_date = datetime.now() - timedelta(days=days)
        history = []
        
        for log_file in self.condensation_log_dir.glob('condensation_*.json'):
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                    log_date = datetime.fromisoformat(log_data.get('timestamp', ''))
                    if log_date >= cutoff_date:
                        history.append(log_data)
            except Exception as e:
                logger.warning(f"Error reading condensation log {log_file}: {e}")
        
        return sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True) 