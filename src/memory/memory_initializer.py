"""
Memory System Initializer

This module handles the initialization of the memory directory structure
for the Personal AI Assistant, creating all necessary folders and core memory files.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import yaml

# Use absolute import or handle the import gracefully
try:
    from utils.config import get_memory_base_path
except ImportError:
    # Fallback if running standalone
    def get_memory_base_path() -> str:
        return os.getenv('MEMORY_BASE_PATH', '~/.assistant_memory')


class MemoryInitializer:
    """Handles initialization of the memory directory structure."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the memory initializer.
        
        Args:
            base_path: Optional override for memory base path
        """
        self.base_path = Path(base_path or get_memory_base_path()).expanduser()
        self.core_files = {
            'user_profile.md': 'Basic user information and preferences',
            'active_context.md': 'Current conversation topics and ongoing projects',
            'relationship_evolution.md': 'How the relationship has developed over time',
            'preferences_patterns.md': 'Communication style and interaction preferences',
            'life_context.md': 'Work, family, interests, and life situations'
        }
    
    def initialize_memory_structure(self) -> bool:
        """
        Initialize the complete memory directory structure.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Create main directory structure
            self._create_directory_structure()
            
            # Initialize core memory files
            self._initialize_core_files()
            
            # Create system configuration
            self._initialize_system_config()
            
            # Create monthly interactions folder
            self._create_current_month_folder()
            
            return True
            
        except Exception as e:
            print(f"Error initializing memory structure: {e}")
            return False
    
    def _create_directory_structure(self) -> None:
        """Create the memory directory structure."""
        directories = [
            self.base_path,
            self.base_path / 'core',
            self.base_path / 'interactions',
            self.base_path / 'condensed' / 'recent',
            self.base_path / 'condensed' / 'medium',
            self.base_path / 'condensed' / 'archive',
            self.base_path / 'system',
            self.base_path / 'system' / 'embeddings'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def _initialize_core_files(self) -> None:
        """Initialize core memory files with templates."""
        core_path = self.base_path / 'core'
        
        for filename, description in self.core_files.items():
            file_path = core_path / filename
            
            if not file_path.exists():
                content = self._create_core_file_template(filename, description)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"Created core file: {file_path}")
    
    def _create_core_file_template(self, filename: str, description: str) -> str:
        """
        Create a template for a core memory file.
        
        Args:
            filename: Name of the memory file
            description: Description of the file's purpose
            
        Returns:
            str: Template content with YAML frontmatter
        """
        timestamp = datetime.now().isoformat()
        
        # YAML frontmatter
        frontmatter = {
            'created': timestamp,
            'last_updated': timestamp,
            'memory_type': self._get_memory_type_from_filename(filename),
            'description': description,
            'importance_score': 10,  # Core files are always high importance
            'category': self._get_category_from_filename(filename)
        }
        
        # Convert frontmatter to YAML
        yaml_content = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        
        # File content based on type
        content_template = self._get_content_template(filename)
        
        return f"---\n{yaml_content}---\n\n{content_template}"
    
    def _get_memory_type_from_filename(self, filename: str) -> str:
        """Get memory_type based on filename."""
        memory_type_map = {
            'user_profile.md': 'user_profile',
            'active_context.md': 'active_context',
            'relationship_evolution.md': 'relationship_evolution',
            'preferences_patterns.md': 'preferences_patterns',
            'life_context.md': 'life_context'
        }
        return memory_type_map.get(filename, 'core')
    
    def _get_category_from_filename(self, filename: str) -> str:
        """Get category based on filename."""
        category_map = {
            'user_profile.md': 'personal',
            'active_context.md': 'context',
            'relationship_evolution.md': 'relationship',
            'preferences_patterns.md': 'preferences',
            'life_context.md': 'personal'
        }
        return category_map.get(filename, 'general')
    
    def _get_content_template(self, filename: str) -> str:
        """Get content template based on filename."""
        templates = {
            'user_profile.md': self._user_profile_template(),
            'active_context.md': self._active_context_template(),
            'relationship_evolution.md': self._relationship_evolution_template(),
            'preferences_patterns.md': self._preferences_patterns_template(),
            'life_context.md': self._life_context_template()
        }
        return templates.get(filename, "# Memory File\n\n*This file will be populated during conversations.*")
    
    def _user_profile_template(self) -> str:
        """Template for user profile file."""
        return """# User Profile

## Basic Information
- **Name**: *Not yet provided*
- **Preferred Name**: *Not yet provided*
- **Age/Age Range**: *Not yet provided*
- **Location**: *Not yet provided*

## Professional Information
- **Occupation**: *Not yet provided*
- **Industry**: *Not yet provided*
- **Skills**: *Not yet provided*
- **Current Role**: *Not yet provided*

## Personal Information
- **Interests**: *Not yet provided*
- **Hobbies**: *Not yet provided*
- **Goals**: *Not yet provided*
- **Values**: *Not yet provided*

## Communication Preferences
- **Preferred Communication Style**: *Not yet determined*
- **Preferred Response Length**: *Not yet determined*
- **Topics of Interest**: *Not yet provided*

---
*This profile will be updated as I learn more about you through our conversations.*
"""
    
    def _active_context_template(self) -> str:
        """Template for active context file."""
        return """# Active Context

## Current Conversations
*No active conversations yet*

## Ongoing Projects
*No ongoing projects yet*

## Recent Topics
*No recent topics yet*

## Current Focus Areas
*No current focus areas yet*

## Pending Questions
*No pending questions yet*

---
*This file tracks our ongoing conversations and will be updated with each interaction.*
"""
    
    def _relationship_evolution_template(self) -> str:
        """Template for relationship evolution file."""
        return """# Relationship Evolution

## Relationship Timeline
*Relationship just beginning*

## Communication Patterns
*No patterns established yet*

## Trust Level
*Building initial trust*

## Shared Experiences
*No shared experiences yet*

## Milestone Conversations
*No milestone conversations yet*

---
*This file tracks how our relationship develops over time.*
"""
    
    def _preferences_patterns_template(self) -> str:
        """Template for preferences and patterns file."""
        return """# Preferences and Patterns

## Communication Style Preferences
*Not yet determined*

## Response Format Preferences
*Not yet determined*

## Topic Preferences
*Not yet determined*

## Interaction Patterns
*No patterns established yet*

## Feedback History
*No feedback yet*

---
*This file captures your preferences and interaction patterns as they emerge.*
"""
    
    def _life_context_template(self) -> str:
        """Template for life context file."""
        return """# Life Context

## Work Life
*Not yet discussed*

## Family & Relationships
*Not yet discussed*

## Health & Wellness
*Not yet discussed*

## Living Situation
*Not yet discussed*

## Current Life Phase
*Not yet discussed*

## Major Life Events
*None yet shared*

## Stress Factors
*Not yet identified*

## Support Systems
*Not yet discussed*

---
*This file captures the broader context of your life situation.*
"""
    
    def _initialize_system_config(self) -> None:
        """Initialize system configuration files."""
        system_path = self.base_path / 'system'
        
        # Create memory system configuration
        config = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'memory_settings': {
                'recent_days': 30,
                'medium_days': 180,
                'archive_days': 365,
                'max_file_size_mb': 5,
                'backup_enabled': True
            },
            'importance_thresholds': {
                'high': 7,
                'medium': 4,
                'low': 1
            },
            'condensation_settings': {
                'auto_condense': True,
                'condensation_ratio': 0.3,
                'preserve_high_importance': True
            }
        }
        
        config_path = system_path / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"Created system config: {config_path}")
    
    def _create_current_month_folder(self) -> None:
        """Create folder for current month's interactions."""
        current_month = datetime.now().strftime('%Y-%m')
        month_path = self.base_path / 'interactions' / current_month
        month_path.mkdir(parents=True, exist_ok=True)
        
        # Create a README file for the month
        readme_content = f"""# Interactions for {current_month}

This folder contains conversation logs and interaction records for {current_month}.

Files in this folder:
- Individual conversation files with timestamps
- Daily interaction summaries
- Context updates and memory captures

---
*Files are automatically created during conversations.*
"""
        
        readme_path = month_path / 'README.md'
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
        
        print(f"Created month folder: {month_path}")
    
    def is_initialized(self) -> bool:
        """
        Check if memory structure is already initialized.
        
        Returns:
            bool: True if structure exists, False otherwise
        """
        required_paths = [
            self.base_path / 'core',
            self.base_path / 'interactions',
            self.base_path / 'condensed',
            self.base_path / 'system'
        ]
        
        return all(path.exists() for path in required_paths)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get information about the memory system.
        
        Returns:
            Dict containing memory system information
        """
        if not self.is_initialized():
            return {'initialized': False}
        
        info = {
            'initialized': True,
            'base_path': str(self.base_path),
            'core_files': [],
            'interaction_months': [],
            'total_size_mb': 0
        }
        
        # Get core files info
        core_path = self.base_path / 'core'
        if core_path.exists():
            for file_path in core_path.glob('*.md'):
                stat = file_path.stat()
                info['core_files'].append({
                    'name': file_path.name,
                    'size_bytes': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Get interaction months
        interactions_path = self.base_path / 'interactions'
        if interactions_path.exists():
            info['interaction_months'] = [
                folder.name for folder in interactions_path.iterdir() 
                if folder.is_dir() and folder.name != '.DS_Store'
            ]
        
        # Calculate total size (rough estimate)
        try:
            total_size = sum(
                f.stat().st_size for f in self.base_path.rglob('*') 
                if f.is_file()
            )
            info['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        except Exception:
            info['total_size_mb'] = 0
        
        return info


def initialize_memory_system(base_path: Optional[str] = None) -> bool:
    """
    Convenience function to initialize the memory system.
    
    Args:
        base_path: Optional override for memory base path
        
    Returns:
        bool: True if initialization successful
    """
    initializer = MemoryInitializer(base_path)
    return initializer.initialize_memory_structure()


if __name__ == "__main__":
    # For direct testing
    success = initialize_memory_system()
    if success:
        print("Memory system initialized successfully!")
    else:
        print("Failed to initialize memory system.") 