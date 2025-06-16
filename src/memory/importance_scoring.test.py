"""
Unit tests for Importance Scoring System

Tests all functionality of the importance scoring and timestamping system
including content analysis, scoring algorithms, and temporal management.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Use absolute imports to avoid import issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

try:
    from src.memory.importance_scoring import (
        ImportanceFactors,
        ImportanceScore,
        ContentAnalysis,
        ImportanceScorer,
        TimestampManager
    )
except ImportError:
    # Fallback for direct execution
    from importance_scoring import (
        ImportanceFactors,
        ImportanceScore,
        ContentAnalysis,
        ImportanceScorer,
        TimestampManager
    )


class TestImportanceScore:
    """Test ImportanceScore dataclass."""
    
    def test_importance_score_creation(self):
        """Test creating an ImportanceScore."""
        factor_scores = {
            'content_keywords': 0.7,
            'memory_type': 0.8,
            'user_engagement': 0.5
        }
        reasoning = ["High-importance keywords detected", "Personal information found"]
        
        score = ImportanceScore(
            total_score=0.65,
            normalized_score=7,
            factor_scores=factor_scores,
            confidence=0.8,
            reasoning=reasoning
        )
        
        assert score.total_score == 0.65
        assert score.normalized_score == 7
        assert score.factor_scores == factor_scores
        assert score.confidence == 0.8
        assert score.reasoning == reasoning
    
    def test_importance_score_bounds_enforcement(self):
        """Test that normalized score is kept within bounds."""
        # Test upper bound
        score = ImportanceScore(
            total_score=1.0,
            normalized_score=15,  # Too high
            factor_scores={},
            confidence=0.8,
            reasoning=[]
        )
        assert score.normalized_score == 10
        
        # Test lower bound
        score = ImportanceScore(
            total_score=0.0,
            normalized_score=-5,  # Too low
            factor_scores={},
            confidence=0.8,
            reasoning=[]
        )
        assert score.normalized_score == 1


class TestContentAnalysis:
    """Test ContentAnalysis dataclass."""
    
    def test_content_analysis_creation(self):
        """Test creating a ContentAnalysis."""
        analysis = ContentAnalysis(
            word_count=50,
            sentence_count=5,
            keyword_matches=['personal:name', 'emotional:happy'],
            emotional_indicators=['happy', 'excited'],
            decision_indicators=['decide', 'choice'],
            personal_info_indicators=['name', 'job'],
            technical_terms=['API', 'HTTP'],
            question_count=2,
            exclamation_count=1
        )
        
        assert analysis.word_count == 50
        assert analysis.sentence_count == 5
        assert len(analysis.keyword_matches) == 2
        assert len(analysis.emotional_indicators) == 2
        assert len(analysis.decision_indicators) == 2
        assert len(analysis.personal_info_indicators) == 2
        assert len(analysis.technical_terms) == 2
        assert analysis.question_count == 2
        assert analysis.exclamation_count == 1


class TestImportanceScorer:
    """Test suite for ImportanceScorer."""
    
    def setup_method(self):
        """Set up test environment."""
        self.scorer = ImportanceScorer()
    
    def test_scorer_initialization(self):
        """Test scorer initialization."""
        assert self.scorer.high_threshold >= 1
        assert self.scorer.medium_threshold >= 1
        assert self.scorer.high_threshold > self.scorer.medium_threshold
    
    def test_analyze_content_basic(self):
        """Test basic content analysis."""
        content = "Hello! My name is John and I'm happy to meet you. What do you think?"
        analysis = self.scorer._analyze_content(content)
        
        assert analysis.word_count == 14
        assert analysis.sentence_count >= 2
        assert analysis.question_count == 1
        assert analysis.exclamation_count == 1
        assert 'personal:name' in analysis.keyword_matches
        assert 'emotional:happy' in analysis.keyword_matches
        assert 'happy' in analysis.emotional_indicators
        assert 'name' in analysis.personal_info_indicators
    
    def test_analyze_content_technical(self):
        """Test content analysis with technical terms."""
        content = "I'm working with the REST API and HTTP requests. The version is 2.1.0."
        analysis = self.scorer._analyze_content(content)
        
        assert 'API' in analysis.technical_terms
        assert 'HTTP' in analysis.technical_terms
        assert '2.1.0' in analysis.technical_terms
    
    def test_analyze_content_emotional(self):
        """Test content analysis with emotional content."""
        content = "I'm really excited about this project but also worried about the deadline. I love programming!"
        analysis = self.scorer._analyze_content(content)
        
        assert 'excited' in analysis.emotional_indicators
        assert 'worried' in analysis.emotional_indicators
        assert 'love' in analysis.emotional_indicators
        assert analysis.exclamation_count == 1
    
    def test_analyze_content_decisions(self):
        """Test content analysis with decision content."""
        content = "I need to decide between two options. Let me evaluate the pros and cons of each choice."
        analysis = self.scorer._analyze_content(content)
        
        assert 'decide' in analysis.decision_indicators
        assert 'evaluate' in analysis.decision_indicators
        assert 'choice' in analysis.decision_indicators
    
    def test_score_keywords_high_importance(self):
        """Test keyword scoring with high-importance content."""
        content = "My name is Sarah and I work as a doctor. My family is very important to me."
        analysis = self.scorer._analyze_content(content)
        
        score = self.scorer._score_keywords(analysis, content)
        assert score > 0.3  # Should be relatively high due to personal keywords
    
    def test_score_keywords_low_importance(self):
        """Test keyword scoring with low-importance content."""
        content = "The weather is nice today. I saw a bird in the park."
        analysis = self.scorer._analyze_content(content)
        
        score = self.scorer._score_keywords(analysis, content)
        assert score < 0.2  # Should be low due to lack of important keywords
    
    def test_score_memory_type(self):
        """Test memory type scoring."""
        # High-importance memory types
        assert self.scorer._score_memory_type('user_profile') > 0.7
        assert self.scorer._score_memory_type('life_context') > 0.7
        
        # Medium-importance memory types
        assert 0.4 < self.scorer._score_memory_type('preferences_patterns') < 0.7
        assert 0.4 < self.scorer._score_memory_type('active_context') < 0.7
        
        # Lower-importance memory types
        assert self.scorer._score_memory_type('system') < 0.4
    
    def test_score_user_engagement_high(self):
        """Test user engagement scoring with high engagement."""
        content = "This is a very detailed explanation of my thoughts and feelings about this important topic. " \
                 "I have many questions about how this works? What are the implications? How can I improve? " \
                 "I'm really excited about learning more!"
        
        analysis = self.scorer._analyze_content(content)
        metadata = {}
        context = {
            'conversation_length': 10,
            'follow_up_questions': 3
        }
        
        score = self.scorer._score_user_engagement(analysis, metadata, context)
        assert score > 0.5  # Should be high due to length, questions, and context
    
    def test_score_user_engagement_low(self):
        """Test user engagement scoring with low engagement."""
        content = "Yes."
        
        analysis = self.scorer._analyze_content(content)
        metadata = {}
        context = {}
        
        score = self.scorer._score_user_engagement(analysis, metadata, context)
        assert score < 0.3  # Should be low due to minimal content
    
    def test_score_temporal_recency(self):
        """Test temporal recency scoring."""
        now = datetime.now()
        
        # Recent content (today)
        recent_metadata = {'created': now.isoformat()}
        recent_score = self.scorer._score_temporal_recency(recent_metadata)
        assert recent_score >= 0.8
        
        # Week-old content
        week_old = now - timedelta(days=7)
        week_metadata = {'created': week_old.isoformat()}
        week_score = self.scorer._score_temporal_recency(week_metadata)
        assert 0.6 <= week_score < 0.9
        
        # Month-old content
        month_old = now - timedelta(days=30)
        month_metadata = {'created': month_old.isoformat()}
        month_score = self.scorer._score_temporal_recency(month_metadata)
        assert 0.4 <= month_score < 0.7
        
        # Very old content
        old = now - timedelta(days=365)
        old_metadata = {'created': old.isoformat()}
        old_score = self.scorer._score_temporal_recency(old_metadata)
        assert old_score <= 0.3
    
    def test_score_emotional_content(self):
        """Test emotional content scoring."""
        # High emotional content
        high_emotional = ContentAnalysis(
            word_count=20, sentence_count=3,
            keyword_matches=[], emotional_indicators=['love', 'hate', 'excited', 'passionate'],
            decision_indicators=[], personal_info_indicators=[], technical_terms=[],
            question_count=0, exclamation_count=2
        )
        high_score = self.scorer._score_emotional_content(high_emotional)
        assert high_score > 0.5
        
        # No emotional content
        no_emotional = ContentAnalysis(
            word_count=20, sentence_count=3,
            keyword_matches=[], emotional_indicators=[],
            decision_indicators=[], personal_info_indicators=[], technical_terms=[],
            question_count=0, exclamation_count=0
        )
        no_score = self.scorer._score_emotional_content(no_emotional)
        assert no_score == 0.0
    
    def test_score_decision_points(self):
        """Test decision points scoring."""
        # High decision content
        high_decision = ContentAnalysis(
            word_count=30, sentence_count=4,
            keyword_matches=[], emotional_indicators=[],
            decision_indicators=['decide', 'evaluate', 'compare', 'consequence'],
            personal_info_indicators=[], technical_terms=[],
            question_count=0, exclamation_count=0
        )
        high_score = self.scorer._score_decision_points(high_decision)
        assert high_score > 0.4
        
        # No decision content
        no_decision = ContentAnalysis(
            word_count=30, sentence_count=4,
            keyword_matches=[], emotional_indicators=[], decision_indicators=[],
            personal_info_indicators=[], technical_terms=[],
            question_count=0, exclamation_count=0
        )
        no_score = self.scorer._score_decision_points(no_decision)
        assert no_score == 0.0
    
    def test_score_personal_information(self):
        """Test personal information scoring."""
        # High personal content
        high_personal = ContentAnalysis(
            word_count=25, sentence_count=3,
            keyword_matches=[], emotional_indicators=[],
            decision_indicators=[], personal_info_indicators=['name', 'family', 'health', 'goal'],
            technical_terms=[], question_count=0, exclamation_count=0
        )
        high_score = self.scorer._score_personal_information(high_personal)
        assert high_score > 0.5
        
        # No personal content
        no_personal = ContentAnalysis(
            word_count=25, sentence_count=3,
            keyword_matches=[], emotional_indicators=[], decision_indicators=[],
            personal_info_indicators=[], technical_terms=[],
            question_count=0, exclamation_count=0
        )
        no_score = self.scorer._score_personal_information(no_personal)
        assert no_score == 0.0
    
    def test_calculate_importance_comprehensive(self):
        """Test comprehensive importance calculation."""
        content = "My name is Alice and I'm a software engineer. I'm really excited about my new job at the tech company! " \
                 "I need to decide whether to accept the promotion offer. What do you think I should consider?"
        
        metadata = {'created': datetime.now().isoformat()}
        context = {'conversation_length': 8, 'follow_up_questions': 1}
        
        score = self.scorer.calculate_importance(
            content=content,
            memory_type='user_profile',
            metadata=metadata,
            context=context
        )
        
        assert isinstance(score, ImportanceScore)
        assert 1 <= score.normalized_score <= 10
        assert 0.0 <= score.total_score <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert len(score.factor_scores) > 0
        assert len(score.reasoning) > 0
        
        # Should be relatively high due to personal info, emotions, and decisions
        assert score.normalized_score >= 6
    
    def test_calculate_importance_minimal_content(self):
        """Test importance calculation with minimal content."""
        content = "OK"
        
        score = self.scorer.calculate_importance(content=content)
        
        assert isinstance(score, ImportanceScore)
        assert 1 <= score.normalized_score <= 10
        assert score.normalized_score <= 5  # Should be low for minimal content
    
    def test_get_importance_category(self):
        """Test importance category classification."""
        # Test with default thresholds (assuming 7 high, 4 medium)
        assert self.scorer.get_importance_category(9) == "high"
        assert self.scorer.get_importance_category(7) == "high"
        assert self.scorer.get_importance_category(5) == "medium"
        assert self.scorer.get_importance_category(4) == "medium"
        assert self.scorer.get_importance_category(3) == "low"
        assert self.scorer.get_importance_category(1) == "low"
    
    def test_suggest_score_adjustment_user_feedback(self):
        """Test score adjustment based on user feedback."""
        # High importance feedback
        new_score, reasoning = self.scorer.suggest_score_adjustment(
            current_score=5,
            user_feedback="This is really important and crucial for my work",
            interaction_count=2
        )
        assert new_score > 5
        assert "high importance" in reasoning.lower()
        
        # Low importance feedback
        new_score, reasoning = self.scorer.suggest_score_adjustment(
            current_score=7,
            user_feedback="This is not important and irrelevant",
            interaction_count=1
        )
        assert new_score < 7
        assert "low importance" in reasoning.lower()
        
        # Moderate feedback
        new_score, reasoning = self.scorer.suggest_score_adjustment(
            current_score=3,
            user_feedback="This is somewhat relevant maybe",
            interaction_count=0
        )
        assert new_score >= 3
        assert "moderate" in reasoning.lower()
    
    def test_suggest_score_adjustment_interaction_frequency(self):
        """Test score adjustment based on interaction frequency."""
        # High interaction frequency
        new_score, reasoning = self.scorer.suggest_score_adjustment(
            current_score=5,
            user_feedback="",
            interaction_count=8
        )
        assert new_score > 5
        assert "interaction frequency" in reasoning.lower()
        
        # Very high interaction frequency
        new_score, reasoning = self.scorer.suggest_score_adjustment(
            current_score=5,
            user_feedback="",
            interaction_count=15
        )
        assert new_score > 6
        assert "very high interaction frequency" in reasoning.lower()


class TestTimestampManager:
    """Test suite for TimestampManager."""
    
    def test_create_timestamp_default(self):
        """Test creating timestamp with default (now)."""
        timestamp = TimestampManager.create_timestamp()
        
        assert isinstance(timestamp, str)
        assert 'T' in timestamp  # ISO format should have T separator
        
        # Should be parseable back to datetime
        dt = TimestampManager.parse_timestamp(timestamp)
        assert dt is not None
        assert isinstance(dt, datetime)
    
    def test_create_timestamp_specific(self):
        """Test creating timestamp with specific datetime."""
        specific_dt = datetime(2024, 1, 15, 14, 30, 45)
        timestamp = TimestampManager.create_timestamp(specific_dt)
        
        assert '2024-01-15T14:30:45' in timestamp
    
    def test_parse_timestamp_valid(self):
        """Test parsing valid timestamp strings."""
        # Standard ISO format
        dt1 = TimestampManager.parse_timestamp('2024-01-15T14:30:45')
        assert dt1 is not None
        assert dt1.year == 2024
        assert dt1.month == 1
        assert dt1.day == 15
        
        # ISO format with Z timezone
        dt2 = TimestampManager.parse_timestamp('2024-01-15T14:30:45Z')
        assert dt2 is not None
        
        # ISO format with timezone offset
        dt3 = TimestampManager.parse_timestamp('2024-01-15T14:30:45+00:00')
        assert dt3 is not None
    
    def test_parse_timestamp_invalid(self):
        """Test parsing invalid timestamp strings."""
        assert TimestampManager.parse_timestamp('invalid') is None
        assert TimestampManager.parse_timestamp('') is None
        assert TimestampManager.parse_timestamp('2024-13-45') is None
    
    def test_get_age_in_days(self):
        """Test getting age in days."""
        now = datetime.now()
        
        # Today
        today_timestamp = now.isoformat()
        age = TimestampManager.get_age_in_days(today_timestamp)
        assert age == 0
        
        # Yesterday
        yesterday = now - timedelta(days=1)
        yesterday_timestamp = yesterday.isoformat()
        age = TimestampManager.get_age_in_days(yesterday_timestamp)
        assert age == 1
        
        # Week ago
        week_ago = now - timedelta(days=7)
        week_timestamp = week_ago.isoformat()
        age = TimestampManager.get_age_in_days(week_timestamp)
        assert age == 7
        
        # Invalid timestamp
        age = TimestampManager.get_age_in_days('invalid')
        assert age is None
    
    def test_is_recent(self):
        """Test checking if timestamp is recent."""
        now = datetime.now()
        
        # Today (should be recent)
        today_timestamp = now.isoformat()
        assert TimestampManager.is_recent(today_timestamp, days=7) is True
        
        # 5 days ago (should be recent within 7 days)
        five_days_ago = now - timedelta(days=5)
        five_days_timestamp = five_days_ago.isoformat()
        assert TimestampManager.is_recent(five_days_timestamp, days=7) is True
        
        # 10 days ago (should not be recent within 7 days)
        ten_days_ago = now - timedelta(days=10)
        ten_days_timestamp = ten_days_ago.isoformat()
        assert TimestampManager.is_recent(ten_days_timestamp, days=7) is False
        
        # Invalid timestamp
        assert TimestampManager.is_recent('invalid', days=7) is False
    
    def test_format_relative_time(self):
        """Test formatting relative time."""
        now = datetime.now()
        
        # Just now
        now_timestamp = now.isoformat()
        relative = TimestampManager.format_relative_time(now_timestamp)
        assert 'just now' in relative or 'minute' in relative
        
        # Hours ago
        hours_ago = now - timedelta(hours=3)
        hours_timestamp = hours_ago.isoformat()
        relative = TimestampManager.format_relative_time(hours_timestamp)
        assert 'hour' in relative
        
        # Days ago
        days_ago = now - timedelta(days=3)
        days_timestamp = days_ago.isoformat()
        relative = TimestampManager.format_relative_time(days_timestamp)
        assert 'day' in relative
        
        # Weeks ago
        weeks_ago = now - timedelta(days=14)
        weeks_timestamp = weeks_ago.isoformat()
        relative = TimestampManager.format_relative_time(weeks_timestamp)
        assert 'week' in relative
        
        # Months ago
        months_ago = now - timedelta(days=60)
        months_timestamp = months_ago.isoformat()
        relative = TimestampManager.format_relative_time(months_timestamp)
        assert 'month' in relative
        
        # Years ago
        years_ago = now - timedelta(days=400)
        years_timestamp = years_ago.isoformat()
        relative = TimestampManager.format_relative_time(years_timestamp)
        assert 'year' in relative
        
        # Invalid timestamp
        relative = TimestampManager.format_relative_time('invalid')
        assert relative == 'unknown time'
    
    def test_update_timestamps_new_entry(self):
        """Test updating timestamps for new entry."""
        frontmatter = {}
        
        updated = TimestampManager.update_timestamps(frontmatter)
        
        assert 'created' in updated
        assert 'last_updated' in updated
        assert 'last_accessed' in updated
        assert 'access_count' in updated
        assert updated['access_count'] == 1
        
        # All timestamps should be recent
        assert TimestampManager.is_recent(updated['created'])
        assert TimestampManager.is_recent(updated['last_updated'])
        assert TimestampManager.is_recent(updated['last_accessed'])
    
    def test_update_timestamps_existing_entry(self):
        """Test updating timestamps for existing entry."""
        old_time = (datetime.now() - timedelta(days=5)).isoformat()
        frontmatter = {
            'created': old_time,
            'last_updated': old_time,
            'last_accessed': old_time,
            'access_count': 3
        }
        
        updated = TimestampManager.update_timestamps(frontmatter)
        
        # Created should remain unchanged
        assert updated['created'] == old_time
        
        # Other timestamps should be updated
        assert updated['last_updated'] != old_time
        assert updated['last_accessed'] != old_time
        assert TimestampManager.is_recent(updated['last_updated'])
        assert TimestampManager.is_recent(updated['last_accessed'])
        
        # Access count should be incremented
        assert updated['access_count'] == 4


class TestIntegration:
    """Integration tests for importance scoring and timestamping."""
    
    def test_full_workflow(self):
        """Test complete workflow of scoring and timestamping."""
        scorer = ImportanceScorer()
        
        # Create content with various importance factors
        content = "Hi! My name is John and I'm a software engineer. I'm really excited about my new project " \
                 "and need to decide whether to use Python or JavaScript. What are your thoughts on this important decision?"
        
        # Create metadata with timestamp
        metadata = {
            'created': TimestampManager.create_timestamp(),
            'category': 'technical_discussion'
        }
        
        # Add context
        context = {
            'conversation_length': 5,
            'follow_up_questions': 2
        }
        
        # Calculate importance
        importance = scorer.calculate_importance(
            content=content,
            memory_type='interaction',
            metadata=metadata,
            context=context
        )
        
        # Verify results
        assert isinstance(importance, ImportanceScore)
        assert importance.normalized_score >= 5  # Should be medium-high importance
        assert importance.confidence > 0.5
        assert len(importance.reasoning) > 0
        
        # Test timestamp operations
        age = TimestampManager.get_age_in_days(metadata['created'])
        assert age == 0  # Should be today
        
        relative_time = TimestampManager.format_relative_time(metadata['created'])
        assert 'just now' in relative_time or 'minute' in relative_time
        
        # Update timestamps
        updated_metadata = TimestampManager.update_timestamps(metadata.copy())
        assert updated_metadata['access_count'] == 1
        assert 'last_accessed' in updated_metadata


if __name__ == "__main__":
    pytest.main([__file__]) 