"""
Intelligent Memory Prioritization System

This module provides advanced memory prioritization algorithms that work with
the importance scoring system to intelligently manage memory lifecycle,
adaptive learning from user patterns, and integration with condensation workflows.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import math
import statistics

# Handle imports gracefully for both package and standalone execution
try:
    from .importance_scoring import ImportanceScorer, ImportanceScore, ImportanceFactors
    from .time_based_organizer import TimeBasedOrganizer, TimeWindow, CondensationCandidate
    from .memory_condensation import MemoryCondensationSystem
    from .file_operations import read_memory_file, write_memory_file
except ImportError:
    # Fallback for standalone execution
    from importance_scoring import ImportanceScorer, ImportanceScore, ImportanceFactors
    from time_based_organizer import TimeBasedOrganizer, TimeWindow, CondensationCandidate
    from memory_condensation import MemoryCondensationSystem
    from file_operations import read_memory_file, write_memory_file

logger = logging.getLogger(__name__)


class PriorityLevel(Enum):
    """Priority levels for memory management."""
    CRITICAL = "critical"      # Never condense, always preserve
    HIGH = "high"             # Preserve with minimal condensation
    MEDIUM = "medium"         # Standard condensation rules
    LOW = "low"              # Aggressive condensation
    ARCHIVAL = "archival"    # Heavy condensation, fact extraction only


class PriorityFactor(Enum):
    """Factors that influence memory priority."""
    RECENCY = "recency"                    # How recent the memory is
    FREQUENCY = "frequency"                # How often it's referenced
    UNIQUENESS = "uniqueness"              # How unique the information is
    USER_PREFERENCE = "user_preference"    # Explicit user preferences
    CONTEXT_RELEVANCE = "context_relevance" # Relevance to current context
    EMOTIONAL_SIGNIFICANCE = "emotional_significance"  # Emotional importance
    DECISION_IMPACT = "decision_impact"     # Impact on future decisions
    RELATIONSHIP_BUILDING = "relationship_building"   # Relationship context


@dataclass
class MemoryPriority:
    """Represents priority analysis for a memory."""
    file_path: str
    overall_priority: PriorityLevel
    priority_score: float  # 0-100
    factor_scores: Dict[str, float]
    importance_score: ImportanceScore
    preservation_reason: str
    condensation_eligibility: bool
    access_patterns: Dict[str, Any]
    timestamp: str
    confidence: float


@dataclass
class UserPattern:
    """Represents learned user interaction patterns."""
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    observation_count: int
    last_updated: str
    examples: List[str]


@dataclass
class PrioritizationConfig:
    """Configuration for memory prioritization."""
    critical_threshold: float = 95.0
    high_threshold: float = 80.0
    medium_threshold: float = 60.0
    low_threshold: float = 40.0
    
    # Factor weights
    recency_weight: float = 0.15
    frequency_weight: float = 0.20
    uniqueness_weight: float = 0.15
    user_preference_weight: float = 0.25
    context_relevance_weight: float = 0.10
    emotional_significance_weight: float = 0.10
    decision_impact_weight: float = 0.05
    
    # Adaptive learning settings
    learning_enabled: bool = True
    pattern_confidence_threshold: float = 0.7
    min_observations_for_pattern: int = 5
    pattern_decay_days: int = 30


class IntelligentMemoryPrioritizer:
    """
    Advanced memory prioritization system with adaptive learning.
    
    This system provides intelligent memory prioritization that:
    1. Learns from user interaction patterns
    2. Adapts priority algorithms based on user behavior
    3. Integrates with condensation and time-based organization
    4. Provides contextual priority adjustments
    5. Maintains preservation policies for critical memories
    """
    
    def __init__(
        self, 
        base_path: str,
        config: Optional[PrioritizationConfig] = None
    ):
        """
        Initialize the intelligent memory prioritizer.
        
        Args:
            base_path: Base path for memory storage
            config: Optional prioritization configuration
        """
        self.base_path = Path(base_path)
        self.config = config or PrioritizationConfig()
        
        # Initialize components
        self.importance_scorer = ImportanceScorer()
        self.time_organizer = TimeBasedOrganizer(str(self.base_path))
        
        # Pattern learning
        self.user_patterns: Dict[str, UserPattern] = {}
        self.access_log: List[Dict[str, Any]] = []
        
        # Directories
        self.patterns_dir = self.base_path / 'system' / 'user_patterns'
        self.access_log_dir = self.base_path / 'system' / 'access_logs'
        self.priorities_dir = self.base_path / 'system' / 'priorities'
        
        for dir_path in [self.patterns_dir, self.access_log_dir, self.priorities_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing patterns
        self._load_user_patterns()
    
    def prioritize_memory(
        self,
        file_path: str,
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> MemoryPriority:
        """
        Calculate comprehensive priority for a memory file.
        
        Args:
            file_path: Path to the memory file
            context: Current context for relevance scoring
            user_preferences: User-specific preferences
            
        Returns:
            MemoryPriority with detailed analysis
        """
        try:
            # Read memory file
            frontmatter, content = read_memory_file(file_path)
            
            # Get importance score
            memory_type = frontmatter.get('memory_type', 'interaction')
            importance_score = self.importance_scorer.calculate_importance(
                content, memory_type, frontmatter, context
            )
            
            # Calculate priority factors
            factor_scores = {}
            
            # Recency factor
            factor_scores[PriorityFactor.RECENCY.value] = self._calculate_recency_score(frontmatter)
            
            # Frequency factor (access patterns)
            factor_scores[PriorityFactor.FREQUENCY.value] = self._calculate_frequency_score(file_path)
            
            # Uniqueness factor
            factor_scores[PriorityFactor.UNIQUENESS.value] = self._calculate_uniqueness_score(
                content, frontmatter
            )
            
            # User preference factor
            factor_scores[PriorityFactor.USER_PREFERENCE.value] = self._calculate_user_preference_score(
                content, frontmatter, user_preferences
            )
            
            # Context relevance factor
            factor_scores[PriorityFactor.CONTEXT_RELEVANCE.value] = self._calculate_context_relevance_score(
                content, frontmatter, context
            )
            
            # Emotional significance factor
            factor_scores[PriorityFactor.EMOTIONAL_SIGNIFICANCE.value] = self._calculate_emotional_significance_score(
                importance_score
            )
            
            # Decision impact factor
            factor_scores[PriorityFactor.DECISION_IMPACT.value] = self._calculate_decision_impact_score(
                importance_score
            )
            
            # Relationship building factor
            factor_scores[PriorityFactor.RELATIONSHIP_BUILDING.value] = self._calculate_relationship_building_score(
                content, frontmatter
            )
            
            # Calculate overall priority score
            priority_score = self._calculate_weighted_priority_score(factor_scores, importance_score)
            
            # Determine priority level
            priority_level = self._determine_priority_level(priority_score)
            
            # Generate preservation reason
            preservation_reason = self._generate_preservation_reason(
                factor_scores, importance_score, priority_level
            )
            
            # Determine condensation eligibility
            condensation_eligibility = self._determine_condensation_eligibility(
                priority_level, importance_score, frontmatter
            )
            
            # Get access patterns
            access_patterns = self._get_access_patterns(file_path)
            
            # Calculate confidence
            confidence = self._calculate_priority_confidence(factor_scores, importance_score)
            
            return MemoryPriority(
                file_path=file_path,
                overall_priority=priority_level,
                priority_score=priority_score,
                factor_scores=factor_scores,
                importance_score=importance_score,
                preservation_reason=preservation_reason,
                condensation_eligibility=condensation_eligibility,
                access_patterns=access_patterns,
                timestamp=datetime.now().isoformat(),
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error prioritizing memory {file_path}: {e}")
            # Return default low priority for error cases
            return MemoryPriority(
                file_path=file_path,
                overall_priority=PriorityLevel.LOW,
                priority_score=30.0,
                factor_scores={},
                importance_score=ImportanceScore(0.3, 3, {}, 0.5, ["Error in analysis"]),
                preservation_reason="Error in analysis - default low priority",
                condensation_eligibility=True,
                access_patterns={},
                timestamp=datetime.now().isoformat(),
                confidence=0.1
            )
    
    def _calculate_recency_score(self, frontmatter: Dict[str, Any]) -> float:
        """Calculate recency-based priority score."""
        created_str = frontmatter.get('created', '')
        if not created_str:
            return 0.5  # Default middle score for unknown dates
        
        try:
            created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
            age_days = (datetime.now() - created_date).days
            
            # Exponential decay with configurable half-life
            half_life_days = 30  # Memories lose half importance after 30 days
            recency_score = math.exp(-0.693 * age_days / half_life_days)
            
            return max(0.0, min(1.0, recency_score))
        except Exception:
            return 0.5
    
    def _calculate_frequency_score(self, file_path: str) -> float:
        """Calculate frequency-based priority score."""
        access_count = 0
        recent_access_count = 0
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for access_entry in self.access_log:
            if access_entry.get('file_path') == file_path:
                access_count += 1
                access_date = datetime.fromisoformat(access_entry.get('timestamp', ''))
                if access_date >= cutoff_date:
                    recent_access_count += 1
        
        # Combine total access count with recent activity
        total_score = min(access_count / 20.0, 1.0)  # Cap at 20 accesses
        recent_score = min(recent_access_count / 10.0, 1.0)  # Cap at 10 recent accesses
        
        return (total_score * 0.6) + (recent_score * 0.4)
    
    def _calculate_uniqueness_score(self, content: str, frontmatter: Dict[str, Any]) -> float:
        """Calculate uniqueness-based priority score."""
        # Simple uniqueness heuristics
        score = 0.0
        
        # Length-based uniqueness (longer content often more unique)
        length_score = min(len(content) / 1000.0, 0.3)  # Up to 0.3 for length
        score += length_score
        
        # Specific terms that indicate uniqueness
        unique_indicators = [
            'specific', 'particular', 'unique', 'special', 'custom', 'personal',
            'first time', 'never', 'always', 'only', 'exclusive', 'private'
        ]
        
        content_lower = content.lower()
        unique_term_count = sum(1 for term in unique_indicators if term in content_lower)
        unique_term_score = min(unique_term_count * 0.1, 0.4)  # Up to 0.4 for unique terms
        score += unique_term_score
        
        # Category-based uniqueness
        category = frontmatter.get('category', '')
        if category in ['personal', 'private', 'confidential', 'secret']:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_user_preference_score(
        self,
        content: str,
        frontmatter: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate user preference-based priority score."""
        if not user_preferences:
            user_preferences = self._get_inferred_user_preferences()
        
        score = 0.0
        
        # Check content against user preferences
        if 'preferred_topics' in user_preferences:
            content_lower = content.lower()
            for topic in user_preferences['preferred_topics']:
                if topic.lower() in content_lower:
                    score += 0.2
        
        # Check category preferences
        if 'preferred_categories' in user_preferences:
            category = frontmatter.get('category', '')
            if category in user_preferences['preferred_categories']:
                score += 0.3
        
        # Check for explicitly marked important content
        if frontmatter.get('user_marked_important', False):
            score += 0.5
        
        # Apply learned patterns
        pattern_score = self._apply_learned_patterns(content, frontmatter)
        score += pattern_score * 0.4
        
        return min(score, 1.0)
    
    def _calculate_context_relevance_score(
        self,
        content: str,
        frontmatter: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate context relevance-based priority score."""
        if not context:
            return 0.5  # Default middle score
        
        score = 0.0
        
        # Current topic relevance
        if 'current_topic' in context:
            current_topic = context['current_topic'].lower()
            content_lower = content.lower()
            
            # Simple keyword matching
            topic_words = current_topic.split()
            matching_words = sum(1 for word in topic_words if word in content_lower)
            if topic_words:
                topic_relevance = matching_words / len(topic_words)
                score += topic_relevance * 0.6
        
        # Temporal context relevance
        if 'time_context' in context:
            time_context = context['time_context']
            created_str = frontmatter.get('created', '')
            
            if created_str and time_context == 'recent':
                try:
                    created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    if (datetime.now() - created_date).days <= 7:
                        score += 0.3
                except Exception:
                    pass
        
        # User activity context
        if 'user_activity' in context:
            activity = context['user_activity']
            category = frontmatter.get('category', '')
            
            activity_category_map = {
                'learning': ['education', 'study', 'research'],
                'work': ['professional', 'career', 'project'],
                'personal': ['personal', 'family', 'relationship']
            }
            
            if activity in activity_category_map:
                if category in activity_category_map[activity]:
                    score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_emotional_significance_score(self, importance_score: ImportanceScore) -> float:
        """Calculate emotional significance-based priority score."""
        # Extract emotional factors from importance score
        emotional_factor = importance_score.factor_scores.get(
            ImportanceFactors.EMOTIONAL_CONTENT.value, 0.0
        )
        
        # Boost for high emotional content
        if emotional_factor > 0.6:
            return emotional_factor * 1.2  # Boost emotional content
        else:
            return emotional_factor
    
    def _calculate_decision_impact_score(self, importance_score: ImportanceScore) -> float:
        """Calculate decision impact-based priority score."""
        # Extract decision factors from importance score
        decision_factor = importance_score.factor_scores.get(
            ImportanceFactors.DECISION_POINTS.value, 0.0
        )
        
        return decision_factor
    
    def _calculate_relationship_building_score(
        self,
        content: str,
        frontmatter: Dict[str, Any]
    ) -> float:
        """Calculate relationship building-based priority score."""
        score = 0.0
        content_lower = content.lower()
        
        # Keywords that indicate relationship building
        relationship_keywords = [
            'relationship', 'trust', 'friendship', 'connection', 'bond',
            'understand', 'empathy', 'support', 'help', 'share',
            'personal', 'intimate', 'close', 'care', 'appreciate'
        ]
        
        keyword_count = sum(1 for keyword in relationship_keywords if keyword in content_lower)
        keyword_score = min(keyword_count * 0.1, 0.6)
        score += keyword_score
        
        # Memory type bonus for relationship-building types
        memory_type = frontmatter.get('memory_type', '')
        if memory_type in ['user_profile', 'relationship_evolution', 'preferences_patterns']:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_weighted_priority_score(
        self,
        factor_scores: Dict[str, float],
        importance_score: ImportanceScore
    ) -> float:
        """Calculate weighted overall priority score."""
        # Base score from importance
        base_score = importance_score.normalized_score * 10  # Convert to 0-100 scale
        
        # Weighted factor scores
        weighted_sum = 0.0
        total_weight = 0.0
        
        factor_weights = {
            PriorityFactor.RECENCY.value: self.config.recency_weight,
            PriorityFactor.FREQUENCY.value: self.config.frequency_weight,
            PriorityFactor.UNIQUENESS.value: self.config.uniqueness_weight,
            PriorityFactor.USER_PREFERENCE.value: self.config.user_preference_weight,
            PriorityFactor.CONTEXT_RELEVANCE.value: self.config.context_relevance_weight,
            PriorityFactor.EMOTIONAL_SIGNIFICANCE.value: self.config.emotional_significance_weight,
            PriorityFactor.DECISION_IMPACT.value: self.config.decision_impact_weight,
            PriorityFactor.RELATIONSHIP_BUILDING.value: 0.0  # Not in main config weights
        }
        
        for factor, score in factor_scores.items():
            weight = factor_weights.get(factor, 0.0)
            weighted_sum += score * weight * 100  # Convert to 0-100 scale
            total_weight += weight
        
        # Combine base score with weighted factors
        if total_weight > 0:
            factor_contribution = weighted_sum / total_weight
            # 70% from importance, 30% from priority factors
            final_score = (base_score * 0.7) + (factor_contribution * 0.3)
        else:
            final_score = base_score
        
        return max(0.0, min(100.0, final_score))
    
    def _determine_priority_level(self, priority_score: float) -> PriorityLevel:
        """Determine priority level from numeric score."""
        if priority_score >= self.config.critical_threshold:
            return PriorityLevel.CRITICAL
        elif priority_score >= self.config.high_threshold:
            return PriorityLevel.HIGH
        elif priority_score >= self.config.medium_threshold:
            return PriorityLevel.MEDIUM
        elif priority_score >= self.config.low_threshold:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.ARCHIVAL
    
    def _generate_preservation_reason(
        self,
        factor_scores: Dict[str, float],
        importance_score: ImportanceScore,
        priority_level: PriorityLevel
    ) -> str:
        """Generate human-readable preservation reason."""
        reasons = []
        
        # High-scoring factors
        for factor, score in factor_scores.items():
            if score > 0.6:
                if factor == PriorityFactor.RECENCY.value:
                    reasons.append("recent memory")
                elif factor == PriorityFactor.FREQUENCY.value:
                    reasons.append("frequently accessed")
                elif factor == PriorityFactor.UNIQUENESS.value:
                    reasons.append("unique information")
                elif factor == PriorityFactor.USER_PREFERENCE.value:
                    reasons.append("matches user preferences")
                elif factor == PriorityFactor.EMOTIONAL_SIGNIFICANCE.value:
                    reasons.append("emotionally significant")
                elif factor == PriorityFactor.RELATIONSHIP_BUILDING.value:
                    reasons.append("important for relationship building")
        
        # Importance factors
        if importance_score.normalized_score >= 8:
            reasons.append("high importance score")
        
        if not reasons:
            if priority_level == PriorityLevel.CRITICAL:
                reasons.append("critical memory type")
            elif priority_level == PriorityLevel.HIGH:
                reasons.append("high overall priority")
            else:
                reasons.append("standard priority assessment")
        
        return f"Priority {priority_level.value}: {', '.join(reasons[:3])}"
    
    def _determine_condensation_eligibility(
        self,
        priority_level: PriorityLevel,
        importance_score: ImportanceScore,
        frontmatter: Dict[str, Any]
    ) -> bool:
        """Determine if memory is eligible for condensation."""
        # Never condense critical memories
        if priority_level == PriorityLevel.CRITICAL:
            return False
        
        # Never condense very high importance scores
        if importance_score.normalized_score >= 9:
            return False
        
        # Never condense if explicitly marked as important
        if frontmatter.get('user_marked_important', False):
            return False
        
        # Check memory type restrictions
        memory_type = frontmatter.get('memory_type', '')
        protected_types = ['user_profile', 'relationship_evolution']
        if memory_type in protected_types:
            return False
        
        return True
    
    def _get_access_patterns(self, file_path: str) -> Dict[str, Any]:
        """Get access patterns for a memory file."""
        patterns = {
            'total_accesses': 0,
            'recent_accesses': 0,
            'last_access': None,
            'access_frequency': 0.0,
            'access_trend': 'stable'
        }
        
        accesses = [entry for entry in self.access_log if entry.get('file_path') == file_path]
        patterns['total_accesses'] = len(accesses)
        
        if accesses:
            # Sort by timestamp
            accesses.sort(key=lambda x: x.get('timestamp', ''))
            patterns['last_access'] = accesses[-1].get('timestamp')
            
            # Recent accesses (last 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            recent_accesses = [
                a for a in accesses 
                if datetime.fromisoformat(a.get('timestamp', '')) >= cutoff_date
            ]
            patterns['recent_accesses'] = len(recent_accesses)
            
            # Calculate frequency (accesses per day)
            if len(accesses) > 1:
                first_access = datetime.fromisoformat(accesses[0].get('timestamp', ''))
                last_access = datetime.fromisoformat(accesses[-1].get('timestamp', ''))
                days_span = max((last_access - first_access).days, 1)
                patterns['access_frequency'] = len(accesses) / days_span
            
            # Determine trend
            if len(accesses) >= 6:
                mid_point = len(accesses) // 2
                first_half = accesses[:mid_point]
                second_half = accesses[mid_point:]
                
                if len(second_half) > len(first_half) * 1.2:
                    patterns['access_trend'] = 'increasing'
                elif len(second_half) < len(first_half) * 0.8:
                    patterns['access_trend'] = 'decreasing'
        
        return patterns
    
    def _calculate_priority_confidence(
        self,
        factor_scores: Dict[str, float],
        importance_score: ImportanceScore
    ) -> float:
        """Calculate confidence in priority assessment."""
        # Base confidence from importance score
        base_confidence = importance_score.confidence
        
        # Confidence from factor score consistency
        if factor_scores:
            factor_values = list(factor_scores.values())
            factor_std = statistics.stdev(factor_values) if len(factor_values) > 1 else 0
            consistency_score = 1.0 - min(factor_std, 1.0)  # Lower std = higher consistency
        else:
            consistency_score = 0.5
        
        # Combined confidence
        combined_confidence = (base_confidence * 0.7) + (consistency_score * 0.3)
        
        return max(0.0, min(1.0, combined_confidence))
    
    def _get_inferred_user_preferences(self) -> Dict[str, Any]:
        """Infer user preferences from patterns and access history."""
        preferences = {
            'preferred_topics': [],
            'preferred_categories': [],
            'importance_bias': 'balanced'
        }
        
        # Analyze access patterns to infer preferences
        category_counts = {}
        for entry in self.access_log[-100:]:  # Last 100 accesses
            category = entry.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Top categories become preferred
        if category_counts:
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            preferences['preferred_categories'] = [cat for cat, count in sorted_categories[:5]]
        
        return preferences
    
    def _apply_learned_patterns(self, content: str, frontmatter: Dict[str, Any]) -> float:
        """Apply learned user patterns to calculate preference score."""
        if not self.config.learning_enabled:
            return 0.5
        
        total_score = 0.0
        pattern_count = 0
        
        for pattern_name, pattern in self.user_patterns.items():
            if pattern.confidence >= self.config.pattern_confidence_threshold:
                pattern_score = self._evaluate_pattern_match(content, frontmatter, pattern)
                total_score += pattern_score
                pattern_count += 1
        
        return total_score / pattern_count if pattern_count > 0 else 0.5
    
    def _evaluate_pattern_match(
        self,
        content: str,
        frontmatter: Dict[str, Any],
        pattern: UserPattern
    ) -> float:
        """Evaluate how well content matches a learned pattern."""
        score = 0.0
        
        if pattern.pattern_type == 'category_preference':
            category = frontmatter.get('category', '')
            preferred_categories = pattern.pattern_data.get('categories', [])
            if category in preferred_categories:
                score = 0.8
        
        elif pattern.pattern_type == 'keyword_preference':
            content_lower = content.lower()
            preferred_keywords = pattern.pattern_data.get('keywords', [])
            matches = sum(1 for keyword in preferred_keywords if keyword in content_lower)
            score = min(matches / len(preferred_keywords), 1.0) if preferred_keywords else 0.0
        
        elif pattern.pattern_type == 'time_preference':
            created_str = frontmatter.get('created', '')
            if created_str:
                try:
                    created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    preferred_hours = pattern.pattern_data.get('hours', [])
                    if created_date.hour in preferred_hours:
                        score = 0.6
                except Exception:
                    pass
        
        return score * pattern.confidence
    
    def record_access(
        self,
        file_path: str,
        access_type: str = "read",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record access to a memory file for pattern learning."""
        access_entry = {
            'file_path': file_path,
            'access_type': access_type,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        
        # Add to access log
        self.access_log.append(access_entry)
        
        # Limit access log size
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]  # Keep last 5000 entries
        
        # Save access log periodically
        if len(self.access_log) % 50 == 0:
            self._save_access_log()
        
        # Update patterns if learning is enabled
        if self.config.learning_enabled:
            self._update_patterns_from_access(access_entry)
    
    def _update_patterns_from_access(self, access_entry: Dict[str, Any]) -> None:
        """Update learned patterns based on access behavior."""
        try:
            file_path = access_entry['file_path']
            # Ensure we have a proper Path object
            file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
            
            if file_path.exists():
                frontmatter, content = read_memory_file(str(file_path))
                
                # Update category preference pattern
                self._update_category_preference_pattern(frontmatter, access_entry)
                
                # Update keyword preference pattern
                self._update_keyword_preference_pattern(content, access_entry)
                
                # Update time preference pattern
                self._update_time_preference_pattern(access_entry)
                
        except Exception as e:
            logger.warning(f"Error updating patterns from access: {e}")
            # Don't let pattern learning errors break the main functionality
    
    def _update_category_preference_pattern(
        self,
        frontmatter: Dict[str, Any],
        access_entry: Dict[str, Any]
    ) -> None:
        """Update category preference learning pattern."""
        category = frontmatter.get('category', 'unknown')
        pattern_name = 'category_preference'
        
        if pattern_name not in self.user_patterns:
            self.user_patterns[pattern_name] = UserPattern(
                pattern_type=pattern_name,
                pattern_data={'categories': {}},
                confidence=0.0,
                observation_count=0,
                last_updated=datetime.now().isoformat(),
                examples=[]
            )
        
        pattern = self.user_patterns[pattern_name]
        categories = pattern.pattern_data['categories']
        categories[category] = categories.get(category, 0) + 1
        
        pattern.observation_count += 1
        pattern.last_updated = datetime.now().isoformat()
        
        # Update confidence based on consistency
        if pattern.observation_count >= self.config.min_observations_for_pattern:
            total_accesses = sum(categories.values())
            top_category_count = max(categories.values())
            pattern.confidence = min(top_category_count / total_accesses, 1.0)
        
        # Keep top categories
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        pattern.pattern_data['categories'] = [cat for cat, count in sorted_categories[:10]]
    
    def _update_keyword_preference_pattern(
        self,
        content: str,
        access_entry: Dict[str, Any]
    ) -> None:
        """Update keyword preference learning pattern."""
        # Extract significant words from content
        words = content.lower().split()
        significant_words = [word for word in words if len(word) > 4 and word.isalpha()]
        
        pattern_name = 'keyword_preference'
        
        if pattern_name not in self.user_patterns:
            self.user_patterns[pattern_name] = UserPattern(
                pattern_type=pattern_name,
                pattern_data={'keywords': {}},
                confidence=0.0,
                observation_count=0,
                last_updated=datetime.now().isoformat(),
                examples=[]
            )
        
        pattern = self.user_patterns[pattern_name]
        keywords = pattern.pattern_data['keywords']
        
        for word in significant_words[:5]:  # Top 5 words per access
            keywords[word] = keywords.get(word, 0) + 1
        
        pattern.observation_count += 1
        pattern.last_updated = datetime.now().isoformat()
        
        # Update confidence
        if pattern.observation_count >= self.config.min_observations_for_pattern:
            pattern.confidence = min(pattern.observation_count / 50.0, 0.9)
        
        # Keep top keywords
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        pattern.pattern_data['keywords'] = [kw for kw, count in sorted_keywords[:20]]
    
    def _update_time_preference_pattern(self, access_entry: Dict[str, Any]) -> None:
        """Update time preference learning pattern."""
        access_time = datetime.fromisoformat(access_entry['timestamp'])
        hour = access_time.hour
        
        pattern_name = 'time_preference'
        
        if pattern_name not in self.user_patterns:
            self.user_patterns[pattern_name] = UserPattern(
                pattern_type=pattern_name,
                pattern_data={'hours': {}},
                confidence=0.0,
                observation_count=0,
                last_updated=datetime.now().isoformat(),
                examples=[]
            )
        
        pattern = self.user_patterns[pattern_name]
        hours = pattern.pattern_data['hours']
        hours[hour] = hours.get(hour, 0) + 1
        
        pattern.observation_count += 1
        pattern.last_updated = datetime.now().isoformat()
        
        # Update confidence
        if pattern.observation_count >= self.config.min_observations_for_pattern:
            total_accesses = sum(hours.values())
            max_hour_count = max(hours.values())
            pattern.confidence = min(max_hour_count / total_accesses, 0.8)
        
        # Keep preferred hours
        sorted_hours = sorted(hours.items(), key=lambda x: x[1], reverse=True)
        pattern.pattern_data['hours'] = [h for h, count in sorted_hours[:8]]  # Top 8 hours
    
    def get_prioritized_memories(
        self,
        limit: int = 100,
        priority_filter: Optional[List[PriorityLevel]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[MemoryPriority]:
        """
        Get prioritized list of memories.
        
        Args:
            limit: Maximum number of memories to return
            priority_filter: Filter by specific priority levels
            context: Context for relevance scoring
            
        Returns:
            List of MemoryPriority objects sorted by priority
        """
        memories = []
        
        # Find all memory files
        memory_files = []
        for pattern in ['**/*.md']:
            memory_files.extend(self.base_path.glob(pattern))
        
        # Filter out system files
        memory_files = [
            f for f in memory_files 
            if not str(f).startswith(str(self.base_path / 'system'))
        ]
        
        # Prioritize each memory
        for memory_file in memory_files:
            try:
                priority = self.prioritize_memory(str(memory_file), context)
                
                # Apply priority filter
                if priority_filter and priority.overall_priority not in priority_filter:
                    continue
                
                memories.append(priority)
                
            except Exception as e:
                logger.warning(f"Error prioritizing {memory_file}: {e}")
        
        # Sort by priority score (highest first)
        memories.sort(key=lambda x: x.priority_score, reverse=True)
        
        return memories[:limit]
    
    def get_condensation_recommendations(self) -> List[Tuple[str, str]]:
        """
        Get recommendations for memory condensation based on priorities.
        
        Returns:
            List of (file_path, recommendation) tuples
        """
        recommendations = []
        
        # Get all memories
        all_memories = self.get_prioritized_memories(limit=1000)
        
        # Group by priority level
        priority_groups = {}
        for memory in all_memories:
            level = memory.overall_priority
            if level not in priority_groups:
                priority_groups[level] = []
            priority_groups[level].append(memory)
        
        # Generate recommendations
        for level, memories in priority_groups.items():
            if level == PriorityLevel.CRITICAL:
                for memory in memories:
                    recommendations.append((
                        memory.file_path,
                        "PRESERVE: Never condense - critical priority"
                    ))
            
            elif level == PriorityLevel.HIGH:
                for memory in memories:
                    if memory.condensation_eligibility:
                        recommendations.append((
                            memory.file_path,
                            "LIGHT_CONDENSATION: Preserve key details, minimal compression"
                        ))
                    else:
                        recommendations.append((
                            memory.file_path,
                            "PRESERVE: High priority, preserve as-is"
                        ))
            
            elif level == PriorityLevel.MEDIUM:
                for memory in memories:
                    if memory.condensation_eligibility:
                        recommendations.append((
                            memory.file_path,
                            "STANDARD_CONDENSATION: Apply theme extraction and summarization"
                        ))
                    else:
                        recommendations.append((
                            memory.file_path,
                            "PRESERVE: Medium priority, minimal changes only"
                        ))
            
            elif level == PriorityLevel.LOW:
                for memory in memories:
                    recommendations.append((
                        memory.file_path,
                        "AGGRESSIVE_CONDENSATION: Extract key facts, heavy compression"
                    ))
            
            elif level == PriorityLevel.ARCHIVAL:
                for memory in memories:
                    recommendations.append((
                        memory.file_path,
                        "ARCHIVE: Condense to essential facts only or consider deletion"
                    ))
        
        return recommendations
    
    def _load_user_patterns(self) -> None:
        """Load user patterns from storage."""
        try:
            patterns_file = self.patterns_dir / 'learned_patterns.json'
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    
                for name, data in patterns_data.items():
                    self.user_patterns[name] = UserPattern(**data)
            
            # Load access log
            access_log_file = self.access_log_dir / 'access_log.json'
            if access_log_file.exists():
                with open(access_log_file, 'r') as f:
                    self.access_log = json.load(f)
                    
        except Exception as e:
            logger.warning(f"Error loading user patterns: {e}")
    
    def _save_user_patterns(self) -> None:
        """Save user patterns to storage."""
        try:
            patterns_data = {}
            for name, pattern in self.user_patterns.items():
                patterns_data[name] = asdict(pattern)
            
            patterns_file = self.patterns_dir / 'learned_patterns.json'
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving user patterns: {e}")
    
    def _save_access_log(self) -> None:
        """Save access log to storage."""
        try:
            access_log_file = self.access_log_dir / 'access_log.json'
            with open(access_log_file, 'w') as f:
                json.dump(self.access_log[-1000:], f, indent=2)  # Save last 1000 entries
                
        except Exception as e:
            logger.error(f"Error saving access log: {e}")
    
    def get_prioritization_stats(self) -> Dict[str, Any]:
        """Get statistics about memory prioritization."""
        all_memories = self.get_prioritized_memories(limit=1000)
        
        # Count by priority level
        priority_counts = {level.value: 0 for level in PriorityLevel}
        total_score = 0.0
        condensation_eligible = 0
        
        for memory in all_memories:
            priority_counts[memory.overall_priority.value] += 1
            total_score += memory.priority_score
            if memory.condensation_eligibility:
                condensation_eligible += 1
        
        stats = {
            'total_memories': len(all_memories),
            'priority_distribution': priority_counts,
            'average_priority_score': total_score / len(all_memories) if all_memories else 0,
            'condensation_eligible_count': condensation_eligible,
            'condensation_eligible_percentage': (condensation_eligible / len(all_memories) * 100) if all_memories else 0,
            'learned_patterns_count': len(self.user_patterns),
            'access_log_size': len(self.access_log),
            'learning_enabled': self.config.learning_enabled
        }
        
        return stats 