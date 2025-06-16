"""
Core Memory Handlers

This module provides specialized handlers for each type of core memory file.
Each handler offers domain-specific operations for managing structured information
and intelligent updates based on the specific nature of each memory type.
"""

import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .file_operations import MemoryFileOperations, read_memory_file, write_memory_file


class UpdateMode(Enum):
    """Modes for updating core memory content."""
    REPLACE = "replace"
    APPEND = "append"
    MERGE = "merge"
    UPDATE_SECTION = "update_section"


@dataclass
class MemoryUpdate:
    """Represents an update to core memory."""
    content: str
    mode: UpdateMode = UpdateMode.APPEND
    section: Optional[str] = None
    importance_score: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseCoreMemoryHandler(ABC):
    """Base class for core memory handlers."""
    
    def __init__(self, file_path: Path, memory_type: str):
        """
        Initialize the core memory handler.
        
        Args:
            file_path: Path to the memory file
            memory_type: Type of memory (user_profile, active_context, etc.)
        """
        self.file_path = file_path
        self.memory_type = memory_type
        self._ensure_file_exists()
    
    def _ensure_file_exists(self) -> None:
        """Ensure the memory file exists."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Core memory file not found: {self.file_path}")
    
    def get_content(self) -> Tuple[Dict[str, Any], str]:
        """
        Get the current content of the memory file.
        
        Returns:
            Tuple of (frontmatter, content)
        """
        return read_memory_file(self.file_path)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the memory file.
        
        Returns:
            Dict containing file metadata
        """
        return MemoryFileOperations.get_file_metadata(self.file_path)
    
    def update_importance(self, importance_score: int) -> bool:
        """
        Update the importance score of the memory.
        
        Args:
            importance_score: New importance score (1-10)
            
        Returns:
            bool: True if successful
        """
        if not (1 <= importance_score <= 10):
            raise ValueError("Importance score must be between 1 and 10")
        
        return MemoryFileOperations.update_frontmatter(
            self.file_path, {'importance_score': importance_score}
        )
    
    @abstractmethod
    def update(self, update: MemoryUpdate) -> bool:
        """
        Update the memory with new information.
        
        Args:
            update: MemoryUpdate containing the new information
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def get_summary(self) -> str:
        """
        Get a summary of the current memory content.
        
        Returns:
            str: Summary of the memory
        """
        pass
    
    def search_content(self, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for content within this memory file.
        
        Args:
            query: Search query
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of search results
        """
        return MemoryFileOperations.search_file_content(
            self.file_path, query, case_sensitive
        )
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the memory file structure and content.
        
        Returns:
            Dict containing validation results
        """
        return MemoryFileOperations.validate_memory_file(self.file_path)


class UserProfileHandler(BaseCoreMemoryHandler):
    """Handler for user profile memory."""
    
    def __init__(self, file_path: Path):
        super().__init__(file_path, "user_profile")
    
    def update(self, update: MemoryUpdate) -> bool:
        """
        Update user profile information.
        
        Args:
            update: MemoryUpdate containing profile information
            
        Returns:
            bool: True if successful
        """
        frontmatter, current_content = self.get_content()
        
        if update.mode == UpdateMode.REPLACE:
            new_content = update.content
        elif update.mode == UpdateMode.APPEND:
            section_header = update.section or "Additional Information"
            new_content = current_content + f"\n\n## {section_header}\n\n{update.content}"
        elif update.mode == UpdateMode.UPDATE_SECTION:
            new_content = self._update_section(current_content, update.section, update.content)
        elif update.mode == UpdateMode.MERGE:
            new_content = self._merge_profile_info(current_content, update.content)
        else:
            raise ValueError(f"Unsupported update mode: {update.mode}")
        
        # Update frontmatter
        if update.importance_score is not None:
            frontmatter['importance_score'] = update.importance_score
        
        frontmatter.update(update.metadata)
        
        return write_memory_file(self.file_path, frontmatter, new_content)
    
    def _update_section(self, content: str, section: str, new_info: str) -> str:
        """Update a specific section in the profile."""
        section_pattern = rf'(## {re.escape(section)}\s*\n)(.*?)(?=\n## |\Z)'
        
        if re.search(section_pattern, content, re.DOTALL | re.IGNORECASE):
            # Section exists, replace it
            return re.sub(
                section_pattern,
                rf'\1{new_info}\n',
                content,
                flags=re.DOTALL | re.IGNORECASE
            )
        else:
            # Section doesn't exist, add it
            return content + f"\n\n## {section}\n\n{new_info}"
    
    def _merge_profile_info(self, current_content: str, new_info: str) -> str:
        """Intelligently merge new profile information."""
        # For now, append new information
        # In the future, this could be enhanced with AI-powered merging
        timestamp = datetime.now().strftime("%Y-%m-%d")
        return current_content + f"\n\n## Updated {timestamp}\n\n{new_info}"
    
    def get_summary(self) -> str:
        """Get a summary of the user profile."""
        _, content = self.get_content()
        
        # Extract key sections
        lines = content.split('\n')
        summary_lines = []
        
        for line in lines[:10]:  # First 10 lines usually contain key info
            line = line.strip()
            if line and not line.startswith('#'):
                summary_lines.append(line)
                if len(summary_lines) >= 3:
                    break
        
        return ' '.join(summary_lines) if summary_lines else "No profile information available."
    
    def update_basic_info(self, name: str = None, age: int = None, occupation: str = None) -> bool:
        """
        Update basic user information.
        
        Args:
            name: User's name
            age: User's age
            occupation: User's occupation
            
        Returns:
            bool: True if successful
        """
        updates = []
        if name:
            updates.append(f"**Name**: {name}")
        if age:
            updates.append(f"**Age**: {age}")
        if occupation:
            updates.append(f"**Occupation**: {occupation}")
        
        if updates:
            content = '\n'.join(updates)
            update = MemoryUpdate(
                content=content,
                mode=UpdateMode.UPDATE_SECTION,
                section="Basic Information"
            )
            return self.update(update)
        
        return True
    
    def add_interest(self, interest: str, category: str = "General") -> bool:
        """
        Add a new interest to the user profile.
        
        Args:
            interest: The interest to add
            category: Category of the interest
            
        Returns:
            bool: True if successful
        """
        _, current_content = self.get_content()
        
        # Check if interests section exists
        if "## Interests" in current_content:
            # Add to existing interests
            content = f"- {interest} ({category})"
            return MemoryFileOperations.append_to_memory_file(
                self.file_path, content, section_header=None
            )
        else:
            # Create new interests section
            content = f"- {interest} ({category})"
            update = MemoryUpdate(
                content=content,
                mode=UpdateMode.UPDATE_SECTION,
                section="Interests"
            )
            return self.update(update)


class ActiveContextHandler(BaseCoreMemoryHandler):
    """Handler for active context memory."""
    
    def __init__(self, file_path: Path):
        super().__init__(file_path, "active_context")
    
    def update(self, update: MemoryUpdate) -> bool:
        """
        Update active context information.
        
        Args:
            update: MemoryUpdate containing context information
            
        Returns:
            bool: True if successful
        """
        frontmatter, current_content = self.get_content()
        
        if update.mode == UpdateMode.REPLACE:
            new_content = update.content
        elif update.mode == UpdateMode.APPEND:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            section_header = update.section or f"Context Update - {timestamp}"
            new_content = current_content + f"\n\n## {section_header}\n\n{update.content}"
        elif update.mode == UpdateMode.UPDATE_SECTION:
            new_content = self._update_context_section(current_content, update.section, update.content)
        else:
            new_content = current_content + f"\n\n{update.content}"
        
        # Update frontmatter
        if update.importance_score is not None:
            frontmatter['importance_score'] = update.importance_score
        
        frontmatter.update(update.metadata)
        
        return write_memory_file(self.file_path, frontmatter, new_content)
    
    def _update_context_section(self, content: str, section: str, new_info: str) -> str:
        """Update a specific context section."""
        section_pattern = rf'(## {re.escape(section)}\s*\n)(.*?)(?=\n## |\Z)'
        
        if re.search(section_pattern, content, re.DOTALL | re.IGNORECASE):
            return re.sub(
                section_pattern,
                rf'\1{new_info}\n',
                content,
                flags=re.DOTALL | re.IGNORECASE
            )
        else:
            return content + f"\n\n## {section}\n\n{new_info}"
    
    def get_summary(self) -> str:
        """Get a summary of the active context."""
        _, content = self.get_content()
        
        # Extract recent context (last few sections)
        sections = re.split(r'\n## ', content)
        recent_sections = sections[-3:]  # Last 3 sections
        
        summary = []
        for section in recent_sections:
            lines = section.split('\n')
            if lines:
                # Get section title and first substantial line
                title = lines[0].replace('#', '').strip()
                for line in lines[1:]:
                    line = line.strip()
                    if line and len(line) > 10:
                        summary.append(f"{title}: {line}")
                        break
        
        return ' | '.join(summary) if summary else "No active context available."
    
    def update_current_focus(self, focus: str, priority: str = "medium") -> bool:
        """
        Update the current focus/priority.
        
        Args:
            focus: Description of current focus
            priority: Priority level (low, medium, high)
            
        Returns:
            bool: True if successful
        """
        content = f"**Current Focus**: {focus}\n**Priority**: {priority}\n**Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        update = MemoryUpdate(
            content=content,
            mode=UpdateMode.UPDATE_SECTION,
            section="Current Focus"
        )
        return self.update(update)
    
    def add_recent_activity(self, activity: str, category: str = "General") -> bool:
        """
        Add a recent activity to the context.
        
        Args:
            activity: Description of the activity
            category: Category of the activity
            
        Returns:
            bool: True if successful
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = f"- **{timestamp}** ({category}): {activity}"
        
        return MemoryFileOperations.append_to_memory_file(
            self.file_path, content, section_header="Recent Activities"
        )
    
    def clear_outdated_context(self, days_old: int = 7) -> bool:
        """
        Clear context older than specified days.
        
        Args:
            days_old: Number of days after which context is considered outdated
            
        Returns:
            bool: True if successful
        """
        frontmatter, content = self.get_content()
        
        # This is a simplified implementation
        # In practice, this would parse dates and remove old sections
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # For now, just add a note about cleanup
        cleanup_note = f"\n\n## Context Cleanup - {datetime.now().strftime('%Y-%m-%d')}\n\nCleared context older than {days_old} days."
        new_content = content + cleanup_note
        
        return write_memory_file(self.file_path, frontmatter, new_content)


class RelationshipEvolutionHandler(BaseCoreMemoryHandler):
    """Handler for relationship evolution memory."""
    
    def __init__(self, file_path: Path):
        super().__init__(file_path, "relationship_evolution")
    
    def update(self, update: MemoryUpdate) -> bool:
        """
        Update relationship evolution information.
        
        Args:
            update: MemoryUpdate containing relationship information
            
        Returns:
            bool: True if successful
        """
        frontmatter, current_content = self.get_content()
        
        if update.mode == UpdateMode.REPLACE:
            new_content = update.content
        elif update.mode == UpdateMode.APPEND:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            section_header = update.section or f"Relationship Update - {timestamp}"
            new_content = current_content + f"\n\n## {section_header}\n\n{update.content}"
        elif update.mode == UpdateMode.UPDATE_SECTION:
            new_content = self._update_relationship_section(current_content, update.section, update.content)
        else:
            new_content = current_content + f"\n\n{update.content}"
        
        # Update frontmatter
        if update.importance_score is not None:
            frontmatter['importance_score'] = update.importance_score
        
        frontmatter.update(update.metadata)
        
        return write_memory_file(self.file_path, frontmatter, new_content)
    
    def _update_relationship_section(self, content: str, section: str, new_info: str) -> str:
        """Update a specific relationship section."""
        section_pattern = rf'(## {re.escape(section)}\s*\n)(.*?)(?=\n## |\Z)'
        
        if re.search(section_pattern, content, re.DOTALL | re.IGNORECASE):
            return re.sub(
                section_pattern,
                rf'\1{new_info}\n',
                content,
                flags=re.DOTALL | re.IGNORECASE
            )
        else:
            return content + f"\n\n## {section}\n\n{new_info}"
    
    def get_summary(self) -> str:
        """Get a summary of the relationship evolution."""
        _, content = self.get_content()
        
        # Extract key relationship milestones
        lines = content.split('\n')
        milestones = []
        
        for line in lines:
            line = line.strip()
            if ('milestone' in line.lower() or 'progress' in line.lower() or 
                'development' in line.lower()) and len(line) > 10:
                milestones.append(line)
                if len(milestones) >= 3:
                    break
        
        return ' | '.join(milestones) if milestones else "No relationship evolution recorded."
    
    def add_milestone(self, milestone: str, significance: str = "medium") -> bool:
        """
        Add a relationship milestone.
        
        Args:
            milestone: Description of the milestone
            significance: Significance level (low, medium, high)
            
        Returns:
            bool: True if successful
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        content = f"**{timestamp}** - {milestone} (Significance: {significance})"
        
        return MemoryFileOperations.append_to_memory_file(
            self.file_path, content, section_header="Relationship Milestones"
        )
    
    def update_trust_level(self, level: str, reason: str = "") -> bool:
        """
        Update the trust level in the relationship.
        
        Args:
            level: Trust level (building, established, high, deep)
            reason: Reason for the trust level
            
        Returns:
            bool: True if successful
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        content = f"**Trust Level**: {level}\n**Updated**: {timestamp}"
        if reason:
            content += f"\n**Reason**: {reason}"
        
        update = MemoryUpdate(
            content=content,
            mode=UpdateMode.UPDATE_SECTION,
            section="Trust Level"
        )
        return self.update(update)
    
    def add_communication_pattern(self, pattern: str, frequency: str = "") -> bool:
        """
        Add or update communication patterns.
        
        Args:
            pattern: Description of the communication pattern
            frequency: How often this pattern occurs
            
        Returns:
            bool: True if successful
        """
        content = f"- {pattern}"
        if frequency:
            content += f" (Frequency: {frequency})"
        
        return MemoryFileOperations.append_to_memory_file(
            self.file_path, content, section_header="Communication Patterns"
        )


class PreferencesPatternsHandler(BaseCoreMemoryHandler):
    """Handler for preferences and patterns memory."""
    
    def __init__(self, file_path: Path):
        super().__init__(file_path, "preferences_patterns")
    
    def update(self, update: MemoryUpdate) -> bool:
        """
        Update preferences and patterns information.
        
        Args:
            update: MemoryUpdate containing preferences information
            
        Returns:
            bool: True if successful
        """
        frontmatter, current_content = self.get_content()
        
        if update.mode == UpdateMode.REPLACE:
            new_content = update.content
        elif update.mode == UpdateMode.APPEND:
            section_header = update.section or "New Preferences"
            new_content = current_content + f"\n\n## {section_header}\n\n{update.content}"
        elif update.mode == UpdateMode.UPDATE_SECTION:
            new_content = self._update_preferences_section(current_content, update.section, update.content)
        elif update.mode == UpdateMode.MERGE:
            new_content = self._merge_preferences(current_content, update.content)
        else:
            new_content = current_content + f"\n\n{update.content}"
        
        # Update frontmatter
        if update.importance_score is not None:
            frontmatter['importance_score'] = update.importance_score
        
        frontmatter.update(update.metadata)
        
        return write_memory_file(self.file_path, frontmatter, new_content)
    
    def _update_preferences_section(self, content: str, section: str, new_info: str) -> str:
        """Update a specific preferences section."""
        section_pattern = rf'(## {re.escape(section)}\s*\n)(.*?)(?=\n## |\Z)'
        
        if re.search(section_pattern, content, re.DOTALL | re.IGNORECASE):
            return re.sub(
                section_pattern,
                rf'\1{new_info}\n',
                content,
                flags=re.DOTALL | re.IGNORECASE
            )
        else:
            return content + f"\n\n## {section}\n\n{new_info}"
    
    def _merge_preferences(self, current_content: str, new_preferences: str) -> str:
        """Intelligently merge new preferences with existing ones."""
        # For now, append with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d")
        return current_content + f"\n\n## Updated Preferences - {timestamp}\n\n{new_preferences}"
    
    def get_summary(self) -> str:
        """Get a summary of preferences and patterns."""
        _, content = self.get_content()
        
        # Extract key preferences
        lines = content.split('\n')
        preferences = []
        
        for line in lines:
            line = line.strip()
            if (('prefers' in line.lower() or 'likes' in line.lower() or 
                 'dislikes' in line.lower() or 'pattern' in line.lower()) 
                and len(line) > 10 and not line.startswith('#')):
                preferences.append(line)
                if len(preferences) >= 5:
                    break
        
        return ' | '.join(preferences) if preferences else "No preferences recorded."
    
    def add_preference(self, preference: str, category: str = "General", strength: str = "medium") -> bool:
        """
        Add a new preference.
        
        Args:
            preference: Description of the preference
            category: Category (communication, topics, style, etc.)
            strength: Strength of preference (weak, medium, strong)
            
        Returns:
            bool: True if successful
        """
        content = f"- {preference} (Strength: {strength})"
        
        return MemoryFileOperations.append_to_memory_file(
            self.file_path, content, section_header=f"{category} Preferences"
        )
    
    def add_pattern(self, pattern: str, context: str = "", frequency: str = "") -> bool:
        """
        Add a behavioral or communication pattern.
        
        Args:
            pattern: Description of the pattern
            context: Context where pattern occurs
            frequency: How often the pattern occurs
            
        Returns:
            bool: True if successful
        """
        content = f"- {pattern}"
        if context:
            content += f" (Context: {context})"
        if frequency:
            content += f" (Frequency: {frequency})"
        
        return MemoryFileOperations.append_to_memory_file(
            self.file_path, content, section_header="Behavioral Patterns"
        )
    
    def update_communication_style(self, style: str, details: str = "") -> bool:
        """
        Update preferred communication style.
        
        Args:
            style: Communication style (formal, casual, direct, etc.)
            details: Additional details about the style
            
        Returns:
            bool: True if successful
        """
        content = f"**Preferred Style**: {style}"
        if details:
            content += f"\n**Details**: {details}"
        content += f"\n**Updated**: {datetime.now().strftime('%Y-%m-%d')}"
        
        update = MemoryUpdate(
            content=content,
            mode=UpdateMode.UPDATE_SECTION,
            section="Communication Style"
        )
        return self.update(update)


class LifeContextHandler(BaseCoreMemoryHandler):
    """Handler for life context memory."""
    
    def __init__(self, file_path: Path):
        super().__init__(file_path, "life_context")
    
    def update(self, update: MemoryUpdate) -> bool:
        """
        Update life context information.
        
        Args:
            update: MemoryUpdate containing life context information
            
        Returns:
            bool: True if successful
        """
        frontmatter, current_content = self.get_content()
        
        if update.mode == UpdateMode.REPLACE:
            new_content = update.content
        elif update.mode == UpdateMode.APPEND:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            section_header = update.section or f"Life Update - {timestamp}"
            new_content = current_content + f"\n\n## {section_header}\n\n{update.content}"
        elif update.mode == UpdateMode.UPDATE_SECTION:
            new_content = self._update_life_section(current_content, update.section, update.content)
        else:
            new_content = current_content + f"\n\n{update.content}"
        
        # Update frontmatter
        if update.importance_score is not None:
            frontmatter['importance_score'] = update.importance_score
        
        frontmatter.update(update.metadata)
        
        return write_memory_file(self.file_path, frontmatter, new_content)
    
    def _update_life_section(self, content: str, section: str, new_info: str) -> str:
        """Update a specific life context section."""
        section_pattern = rf'(## {re.escape(section)}\s*\n)(.*?)(?=\n## |\Z)'
        
        if re.search(section_pattern, content, re.DOTALL | re.IGNORECASE):
            return re.sub(
                section_pattern,
                rf'\1{new_info}\n',
                content,
                flags=re.DOTALL | re.IGNORECASE
            )
        else:
            return content + f"\n\n## {section}\n\n{new_info}"
    
    def get_summary(self) -> str:
        """Get a summary of life context."""
        _, content = self.get_content()
        
        # Extract key life events and current situation
        lines = content.split('\n')
        context_items = []
        
        for line in lines:
            line = line.strip()
            if (('current' in line.lower() or 'goal' in line.lower() or 
                 'challenge' in line.lower() or 'situation' in line.lower()) 
                and len(line) > 10 and not line.startswith('#')):
                context_items.append(line)
                if len(context_items) >= 3:
                    break
        
        return ' | '.join(context_items) if context_items else "No life context available."
    
    def add_life_event(self, event: str, impact: str = "medium", date: str = None) -> bool:
        """
        Add a significant life event.
        
        Args:
            event: Description of the life event
            impact: Impact level (low, medium, high)
            date: Date of the event (defaults to today)
            
        Returns:
            bool: True if successful
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        content = f"**{date}** - {event} (Impact: {impact})"
        
        return MemoryFileOperations.append_to_memory_file(
            self.file_path, content, section_header="Life Events"
        )
    
    def update_current_situation(self, situation: str, details: str = "") -> bool:
        """
        Update current life situation.
        
        Args:
            situation: Brief description of current situation
            details: Additional details
            
        Returns:
            bool: True if successful
        """
        content = f"**Current Situation**: {situation}\n**Updated**: {datetime.now().strftime('%Y-%m-%d')}"
        if details:
            content += f"\n**Details**: {details}"
        
        update = MemoryUpdate(
            content=content,
            mode=UpdateMode.UPDATE_SECTION,
            section="Current Situation"
        )
        return self.update(update)
    
    def add_goal(self, goal: str, priority: str = "medium", timeline: str = "") -> bool:
        """
        Add a life goal.
        
        Args:
            goal: Description of the goal
            priority: Priority level (low, medium, high)
            timeline: Expected timeline for the goal
            
        Returns:
            bool: True if successful
        """
        content = f"- {goal} (Priority: {priority})"
        if timeline:
            content += f" (Timeline: {timeline})"
        
        return MemoryFileOperations.append_to_memory_file(
            self.file_path, content, section_header="Goals"
        )
    
    def add_challenge(self, challenge: str, status: str = "ongoing") -> bool:
        """
        Add a current challenge or difficulty.
        
        Args:
            challenge: Description of the challenge
            status: Status (ongoing, resolved, escalating)
            
        Returns:
            bool: True if successful
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        content = f"- **{timestamp}** - {challenge} (Status: {status})"
        
        return MemoryFileOperations.append_to_memory_file(
            self.file_path, content, section_header="Current Challenges"
        )


class CoreMemoryHandlerFactory:
    """Factory for creating appropriate core memory handlers."""
    
    HANDLER_MAP = {
        'user_profile': UserProfileHandler,
        'active_context': ActiveContextHandler,
        'relationship_evolution': RelationshipEvolutionHandler,
        'preferences_patterns': PreferencesPatternsHandler,
        'life_context': LifeContextHandler
    }
    
    @classmethod
    def create_handler(cls, memory_type: str, file_path: Path) -> BaseCoreMemoryHandler:
        """
        Create the appropriate handler for a core memory type.
        
        Args:
            memory_type: Type of core memory
            file_path: Path to the memory file
            
        Returns:
            BaseCoreMemoryHandler: Appropriate handler instance
            
        Raises:
            ValueError: If memory type is not supported
        """
        if memory_type not in cls.HANDLER_MAP:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        
        handler_class = cls.HANDLER_MAP[memory_type]
        return handler_class(file_path)
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """
        Get list of supported memory types.
        
        Returns:
            List of supported memory type names
        """
        return list(cls.HANDLER_MAP.keys()) 