"""
Importance Scoring System

This module provides intelligent importance scoring for memory entries based on
multiple factors including content analysis, user engagement, temporal factors,
and memory type-specific criteria.
"""

import re
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Handle imports gracefully for both package and standalone execution
try:
    from ..utils.config import (
        get_memory_high_importance_threshold,
        get_memory_medium_importance_threshold
    )
except ImportError:
    # Fallback for standalone execution
    import os
    def get_memory_high_importance_threshold() -> int:
        return int(os.getenv('MEMORY_HIGH_IMPORTANCE_THRESHOLD', '7'))
    def get_memory_medium_importance_threshold() -> int:
        return int(os.getenv('MEMORY_MEDIUM_IMPORTANCE_THRESHOLD', '4'))


class ImportanceFactors(Enum):
    """Factors that contribute to importance scoring."""
    CONTENT_KEYWORDS = "content_keywords"
    USER_ENGAGEMENT = "user_engagement"
    TEMPORAL_RECENCY = "temporal_recency"
    MEMORY_TYPE = "memory_type"
    INTERACTION_FREQUENCY = "interaction_frequency"
    EMOTIONAL_CONTENT = "emotional_content"
    DECISION_POINTS = "decision_points"
    PERSONAL_INFORMATION = "personal_information"


@dataclass
class ImportanceScore:
    """Represents an importance score with breakdown."""
    total_score: float
    normalized_score: int  # 1-10 scale
    factor_scores: Dict[str, float]
    confidence: float
    reasoning: List[str]
    
    def __post_init__(self):
        # Ensure normalized score is within bounds
        self.normalized_score = max(1, min(10, self.normalized_score))


@dataclass
class ContentAnalysis:
    """Results of content analysis for importance scoring."""
    word_count: int
    sentence_count: int
    keyword_matches: List[str]
    emotional_indicators: List[str]
    decision_indicators: List[str]
    personal_info_indicators: List[str]
    technical_terms: List[str]
    question_count: int
    exclamation_count: int


class ImportanceScorer:
    """Calculates importance scores for memory entries."""
    
    # High-importance keywords by category
    HIGH_IMPORTANCE_KEYWORDS = {
        'personal': [
            'name', 'age', 'birthday', 'family', 'relationship', 'married', 'children',
            'home', 'address', 'phone', 'email', 'job', 'career', 'work', 'company',
            'health', 'medical', 'doctor', 'hospital', 'medication', 'allergy',
            'goal', 'dream', 'aspiration', 'plan', 'future', 'important', 'priority'
        ],
        'emotional': [
            'love', 'hate', 'fear', 'worry', 'excited', 'happy', 'sad', 'angry',
            'frustrated', 'disappointed', 'proud', 'grateful', 'anxious', 'stressed',
            'confident', 'insecure', 'passionate', 'motivated', 'depressed'
        ],
        'decisions': [
            'decide', 'decision', 'choose', 'choice', 'option', 'alternative',
            'consider', 'evaluate', 'compare', 'pros', 'cons', 'advantage',
            'disadvantage', 'risk', 'benefit', 'consequence', 'impact'
        ],
        'preferences': [
            'prefer', 'like', 'dislike', 'enjoy', 'hate', 'favorite', 'best',
            'worst', 'always', 'never', 'usually', 'typically', 'tend to',
            'style', 'approach', 'method', 'way'
        ],
        'learning': [
            'learn', 'study', 'understand', 'knowledge', 'skill', 'experience',
            'practice', 'improve', 'develop', 'grow', 'progress', 'achievement',
            'success', 'failure', 'mistake', 'lesson', 'insight', 'discovery'
        ]
    }
    
    # Memory type base scores
    MEMORY_TYPE_SCORES = {
        'user_profile': 8.0,
        'relationship_evolution': 7.0,
        'preferences_patterns': 6.0,
        'life_context': 7.5,
        'active_context': 5.0,
        'interaction': 4.0,
        'condensed': 6.0,
        'system': 3.0
    }
    
    def __init__(self):
        """Initialize the importance scorer."""
        self.high_threshold = get_memory_high_importance_threshold()
        self.medium_threshold = get_memory_medium_importance_threshold()
    
    def calculate_importance(
        self,
        content: str,
        memory_type: str = "interaction",
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ImportanceScore:
        """
        Calculate importance score for content.
        
        Args:
            content: The content to analyze
            memory_type: Type of memory (user_profile, interaction, etc.)
            metadata: Additional metadata about the content
            context: Contextual information for scoring
            
        Returns:
            ImportanceScore with detailed breakdown
        """
        if metadata is None:
            metadata = {}
        if context is None:
            context = {}
        
        # Analyze content
        analysis = self._analyze_content(content)
        
        # Calculate factor scores
        factor_scores = {}
        reasoning = []
        
        # Content keyword scoring
        keyword_score = self._score_keywords(analysis, content)
        factor_scores[ImportanceFactors.CONTENT_KEYWORDS.value] = keyword_score
        if keyword_score > 0.5:
            reasoning.append(f"High-importance keywords detected (score: {keyword_score:.2f})")
        
        # Memory type scoring
        type_score = self._score_memory_type(memory_type)
        factor_scores[ImportanceFactors.MEMORY_TYPE.value] = type_score
        reasoning.append(f"Memory type '{memory_type}' base score: {type_score:.2f}")
        
        # User engagement scoring
        engagement_score = self._score_user_engagement(analysis, metadata, context)
        factor_scores[ImportanceFactors.USER_ENGAGEMENT.value] = engagement_score
        if engagement_score > 0.3:
            reasoning.append(f"High user engagement detected (score: {engagement_score:.2f})")
        
        # Temporal recency scoring
        recency_score = self._score_temporal_recency(metadata)
        factor_scores[ImportanceFactors.TEMPORAL_RECENCY.value] = recency_score
        
        # Emotional content scoring
        emotional_score = self._score_emotional_content(analysis)
        factor_scores[ImportanceFactors.EMOTIONAL_CONTENT.value] = emotional_score
        if emotional_score > 0.4:
            reasoning.append(f"Emotional content detected (score: {emotional_score:.2f})")
        
        # Decision points scoring
        decision_score = self._score_decision_points(analysis)
        factor_scores[ImportanceFactors.DECISION_POINTS.value] = decision_score
        if decision_score > 0.3:
            reasoning.append(f"Decision-making content detected (score: {decision_score:.2f})")
        
        # Personal information scoring
        personal_score = self._score_personal_information(analysis)
        factor_scores[ImportanceFactors.PERSONAL_INFORMATION.value] = personal_score
        if personal_score > 0.4:
            reasoning.append(f"Personal information detected (score: {personal_score:.2f})")
        
        # Calculate total score with weights
        total_score = self._calculate_weighted_score(factor_scores, memory_type)
        
        # Normalize to 1-10 scale
        normalized_score = self._normalize_score(total_score)
        
        # Calculate confidence based on content length and factor agreement
        confidence = self._calculate_confidence(analysis, factor_scores)
        
        return ImportanceScore(
            total_score=total_score,
            normalized_score=normalized_score,
            factor_scores=factor_scores,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _analyze_content(self, content: str) -> ContentAnalysis:
        """Analyze content for various importance indicators."""
        # Basic metrics
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        # Find keyword matches
        content_lower = content.lower()
        keyword_matches = []
        for category, keywords in self.HIGH_IMPORTANCE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    keyword_matches.append(f"{category}:{keyword}")
        
        # Emotional indicators
        emotional_indicators = []
        for keyword in self.HIGH_IMPORTANCE_KEYWORDS['emotional']:
            if keyword in content_lower:
                emotional_indicators.append(keyword)
        
        # Decision indicators
        decision_indicators = []
        for keyword in self.HIGH_IMPORTANCE_KEYWORDS['decisions']:
            if keyword in content_lower:
                decision_indicators.append(keyword)
        
        # Personal information indicators
        personal_info_indicators = []
        for keyword in self.HIGH_IMPORTANCE_KEYWORDS['personal']:
            if keyword in content_lower:
                personal_info_indicators.append(keyword)
        
        # Technical terms (simple heuristic)
        technical_terms = []
        tech_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\.\w+\b',   # Dotted notation (e.g., file.txt, api.method)
            r'\b\w+_\w+\b',    # Underscore notation
            r'\b\d+\.\d+\b'    # Version numbers
        ]
        for pattern in tech_patterns:
            matches = re.findall(pattern, content)
            technical_terms.extend(matches)
        
        # Count questions and exclamations
        question_count = content.count('?')
        exclamation_count = content.count('!')
        
        return ContentAnalysis(
            word_count=len(words),
            sentence_count=len([s for s in sentences if s.strip()]),
            keyword_matches=keyword_matches,
            emotional_indicators=emotional_indicators,
            decision_indicators=decision_indicators,
            personal_info_indicators=personal_info_indicators,
            technical_terms=list(set(technical_terms)),
            question_count=question_count,
            exclamation_count=exclamation_count
        )
    
    def _score_keywords(self, analysis: ContentAnalysis, content: str) -> float:
        """Score based on keyword presence and density."""
        if analysis.word_count == 0:
            return 0.0
        
        # Calculate keyword density
        keyword_count = len(analysis.keyword_matches)
        keyword_density = keyword_count / analysis.word_count
        
        # Weight by keyword categories
        category_weights = {
            'personal': 1.0,
            'emotional': 0.8,
            'decisions': 0.9,
            'preferences': 0.7,
            'learning': 0.6
        }
        
        weighted_score = 0.0
        for match in analysis.keyword_matches:
            category = match.split(':')[0]
            weight = category_weights.get(category, 0.5)
            weighted_score += weight
        
        # Normalize and cap
        score = min(1.0, weighted_score * keyword_density * 10)
        return score
    
    def _score_memory_type(self, memory_type: str) -> float:
        """Score based on memory type importance."""
        base_score = self.MEMORY_TYPE_SCORES.get(memory_type, 4.0)
        return base_score / 10.0  # Normalize to 0-1 scale
    
    def _score_user_engagement(
        self, 
        analysis: ContentAnalysis, 
        metadata: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> float:
        """Score based on user engagement indicators."""
        score = 0.0
        
        # Length indicates engagement
        if analysis.word_count > 50:
            score += 0.2
        if analysis.word_count > 200:
            score += 0.2
        
        # Questions indicate engagement
        if analysis.question_count > 0:
            score += min(0.3, analysis.question_count * 0.1)
        
        # Exclamations indicate emotion/engagement
        if analysis.exclamation_count > 0:
            score += min(0.2, analysis.exclamation_count * 0.05)
        
        # Multiple sentences indicate thoughtfulness
        if analysis.sentence_count > 3:
            score += 0.1
        
        # Technical terms might indicate specific interests
        if len(analysis.technical_terms) > 2:
            score += 0.1
        
        # Context-based scoring
        if context.get('conversation_length', 0) > 5:
            score += 0.1
        
        if context.get('follow_up_questions', 0) > 0:
            score += 0.2
        
        return min(1.0, score)
    
    def _score_temporal_recency(self, metadata: Dict[str, Any]) -> float:
        """Score based on temporal recency."""
        created_str = metadata.get('created', '')
        if not created_str:
            return 0.5  # Neutral score if no timestamp
        
        try:
            created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
            now = datetime.now(created_date.tzinfo) if created_date.tzinfo else datetime.now()
            
            # Calculate age in days
            age_days = (now - created_date).days
            
            # Recency scoring (exponential decay)
            if age_days <= 1:
                return 1.0
            elif age_days <= 7:
                return 0.8
            elif age_days <= 30:
                return 0.6
            elif age_days <= 90:
                return 0.4
            else:
                return 0.2
        
        except (ValueError, TypeError):
            return 0.5
    
    def _score_emotional_content(self, analysis: ContentAnalysis) -> float:
        """Score based on emotional content."""
        if not analysis.emotional_indicators:
            return 0.0
        
        # Count unique emotional indicators
        unique_emotions = len(set(analysis.emotional_indicators))
        
        # Score based on emotional diversity and intensity
        score = min(1.0, unique_emotions * 0.2)
        
        # Boost for high-intensity emotions
        high_intensity = ['love', 'hate', 'fear', 'excited', 'passionate', 'depressed']
        for emotion in analysis.emotional_indicators:
            if emotion in high_intensity:
                score += 0.1
        
        return min(1.0, score)
    
    def _score_decision_points(self, analysis: ContentAnalysis) -> float:
        """Score based on decision-making content."""
        if not analysis.decision_indicators:
            return 0.0
        
        # Count decision-related terms
        decision_count = len(analysis.decision_indicators)
        
        # Score based on decision complexity
        score = min(1.0, decision_count * 0.15)
        
        # Boost for complex decision terms
        complex_terms = ['evaluate', 'compare', 'consequence', 'impact', 'risk', 'benefit']
        for term in analysis.decision_indicators:
            if term in complex_terms:
                score += 0.1
        
        return min(1.0, score)
    
    def _score_personal_information(self, analysis: ContentAnalysis) -> float:
        """Score based on personal information content."""
        if not analysis.personal_info_indicators:
            return 0.0
        
        # Count personal information indicators
        personal_count = len(set(analysis.personal_info_indicators))
        
        # Score based on personal information density
        score = min(1.0, personal_count * 0.2)
        
        # Boost for highly personal information
        highly_personal = ['name', 'family', 'health', 'goal', 'dream', 'important']
        for info in analysis.personal_info_indicators:
            if info in highly_personal:
                score += 0.15
        
        return min(1.0, score)
    
    def _calculate_weighted_score(self, factor_scores: Dict[str, float], memory_type: str) -> float:
        """Calculate weighted total score based on memory type."""
        # Base weights
        weights = {
            ImportanceFactors.CONTENT_KEYWORDS.value: 0.25,
            ImportanceFactors.MEMORY_TYPE.value: 0.20,
            ImportanceFactors.USER_ENGAGEMENT.value: 0.15,
            ImportanceFactors.TEMPORAL_RECENCY.value: 0.10,
            ImportanceFactors.EMOTIONAL_CONTENT.value: 0.10,
            ImportanceFactors.DECISION_POINTS.value: 0.10,
            ImportanceFactors.PERSONAL_INFORMATION.value: 0.10
        }
        
        # Adjust weights based on memory type
        if memory_type in ['user_profile', 'relationship_evolution']:
            weights[ImportanceFactors.PERSONAL_INFORMATION.value] *= 1.5
            weights[ImportanceFactors.EMOTIONAL_CONTENT.value] *= 1.3
        elif memory_type == 'preferences_patterns':
            weights[ImportanceFactors.DECISION_POINTS.value] *= 1.4
            weights[ImportanceFactors.USER_ENGAGEMENT.value] *= 1.2
        elif memory_type == 'active_context':
            weights[ImportanceFactors.TEMPORAL_RECENCY.value] *= 1.5
            weights[ImportanceFactors.USER_ENGAGEMENT.value] *= 1.3
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        total_score = 0.0
        for factor, score in factor_scores.items():
            weight = normalized_weights.get(factor, 0.0)
            total_score += score * weight
        
        return total_score
    
    def _normalize_score(self, total_score: float) -> int:
        """Normalize score to 1-10 integer scale."""
        # Apply sigmoid-like transformation for better distribution
        normalized = 1 + 9 * (1 / (1 + math.exp(-10 * (total_score - 0.5))))
        return round(normalized)
    
    def _calculate_confidence(self, analysis: ContentAnalysis, factor_scores: Dict[str, float]) -> float:
        """Calculate confidence in the importance score."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence with more content
        if analysis.word_count > 20:
            confidence += 0.1
        if analysis.word_count > 100:
            confidence += 0.1
        
        # Higher confidence with multiple factors
        active_factors = sum(1 for score in factor_scores.values() if score > 0.1)
        confidence += min(0.3, active_factors * 0.05)
        
        # Higher confidence with clear indicators
        if len(analysis.keyword_matches) > 3:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def get_importance_category(self, score: int) -> str:
        """Get importance category for a score."""
        if score >= self.high_threshold:
            return "high"
        elif score >= self.medium_threshold:
            return "medium"
        else:
            return "low"
    
    def suggest_score_adjustment(
        self, 
        current_score: int, 
        user_feedback: str,
        interaction_count: int = 0
    ) -> Tuple[int, str]:
        """
        Suggest score adjustment based on user feedback and interaction patterns.
        
        Args:
            current_score: Current importance score
            user_feedback: User feedback about the memory
            interaction_count: Number of times this memory has been referenced
            
        Returns:
            Tuple of (suggested_score, reasoning)
        """
        suggested_score = current_score
        reasoning = []
        
        # Analyze user feedback
        feedback_lower = user_feedback.lower()
        
        if any(word in feedback_lower for word in ['important', 'crucial', 'critical', 'essential']):
            suggested_score = min(10, current_score + 2)
            reasoning.append("User indicated high importance")
        elif any(word in feedback_lower for word in ['not important', 'irrelevant', 'forget']):
            suggested_score = max(1, current_score - 2)
            reasoning.append("User indicated low importance")
        elif any(word in feedback_lower for word in ['somewhat', 'maybe', 'possibly']):
            # Slight adjustment toward medium
            if current_score < 5:
                suggested_score = min(6, current_score + 1)
            elif current_score > 6:
                suggested_score = max(5, current_score - 1)
            reasoning.append("User indicated moderate importance")
        
        # Adjust based on interaction frequency
        if interaction_count > 5:
            suggested_score = min(10, suggested_score + 1)
            reasoning.append(f"High interaction frequency ({interaction_count} references)")
        elif interaction_count > 10:
            suggested_score = min(10, suggested_score + 2)
            reasoning.append(f"Very high interaction frequency ({interaction_count} references)")
        
        reasoning_text = "; ".join(reasoning) if reasoning else "No adjustment needed"
        return suggested_score, reasoning_text


class TimestampManager:
    """Manages timestamps for memory entries."""
    
    @staticmethod
    def create_timestamp(dt: Optional[datetime] = None) -> str:
        """
        Create ISO format timestamp.
        
        Args:
            dt: Optional datetime object, defaults to now
            
        Returns:
            ISO format timestamp string
        """
        if dt is None:
            dt = datetime.now()
        return dt.isoformat()
    
    @staticmethod
    def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
        """
        Parse timestamp string to datetime object.
        
        Args:
            timestamp_str: ISO format timestamp string
            
        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Handle various ISO format variations
            timestamp_str = timestamp_str.replace('Z', '+00:00')
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def get_age_in_days(timestamp_str: str) -> Optional[int]:
        """
        Get age of timestamp in days.
        
        Args:
            timestamp_str: ISO format timestamp string
            
        Returns:
            Age in days or None if parsing fails
        """
        dt = TimestampManager.parse_timestamp(timestamp_str)
        if dt is None:
            return None
        
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        return (now - dt).days
    
    @staticmethod
    def is_recent(timestamp_str: str, days: int = 7) -> bool:
        """
        Check if timestamp is within recent days.
        
        Args:
            timestamp_str: ISO format timestamp string
            days: Number of days to consider recent
            
        Returns:
            True if recent, False otherwise
        """
        age = TimestampManager.get_age_in_days(timestamp_str)
        return age is not None and age <= days
    
    @staticmethod
    def format_relative_time(timestamp_str: str) -> str:
        """
        Format timestamp as relative time (e.g., "2 days ago").
        
        Args:
            timestamp_str: ISO format timestamp string
            
        Returns:
            Relative time string
        """
        dt = TimestampManager.parse_timestamp(timestamp_str)
        if dt is None:
            return "unknown time"
        
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            if diff.days == 1:
                return "1 day ago"
            elif diff.days < 7:
                return f"{diff.days} days ago"
            elif diff.days < 30:
                weeks = diff.days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            elif diff.days < 365:
                months = diff.days // 30
                return f"{months} month{'s' if months > 1 else ''} ago"
            else:
                years = diff.days // 365
                return f"{years} year{'s' if years > 1 else ''} ago"
        else:
            hours = diff.seconds // 3600
            if hours > 0:
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
            else:
                minutes = diff.seconds // 60
                if minutes > 0:
                    return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                else:
                    return "just now"
    
    @staticmethod
    def update_timestamps(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update timestamps in frontmatter.
        
        Args:
            frontmatter: Memory file frontmatter
            
        Returns:
            Updated frontmatter with current timestamps
        """
        now = TimestampManager.create_timestamp()
        
        # Set created timestamp if not exists
        if 'created' not in frontmatter:
            frontmatter['created'] = now
        
        # Always update last_updated
        frontmatter['last_updated'] = now
        
        # Update access tracking
        if 'last_accessed' not in frontmatter:
            frontmatter['last_accessed'] = now
        
        if 'access_count' not in frontmatter:
            frontmatter['access_count'] = 1
        else:
            frontmatter['access_count'] += 1
        
        return frontmatter 