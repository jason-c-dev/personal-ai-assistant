"""
CLI Configuration and Customization System

Handles CLI-specific settings, user preferences, themes, and behavior customization
for the Personal AI Assistant.
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum

from rich.console import Console
from rich.theme import Theme
from rich.style import Style

console = Console()


class ResponseStyle(Enum):
    """Available response styles."""
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    DETAILED = "detailed"
    CONCISE = "concise"
    TECHNICAL = "technical"
    CREATIVE = "creative"


class DisplayMode(Enum):
    """Available display modes."""
    NORMAL = "normal"
    COMPACT = "compact"
    VERBOSE = "verbose"
    MINIMAL = "minimal"


class ColorTheme(Enum):
    """Available color themes."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    BLUE = "blue"
    GREEN = "green"
    PURPLE = "purple"
    MONOCHROME = "monochrome"


@dataclass
class UIThemeConfig:
    """UI theme configuration."""
    name: str = "default"
    primary_color: str = "cyan"
    secondary_color: str = "blue"
    success_color: str = "green"
    warning_color: str = "yellow"
    error_color: str = "red"
    info_color: str = "blue"
    dim_color: str = "dim"
    
    # Text styling
    bold_headers: bool = True
    use_borders: bool = True
    use_icons: bool = True
    
    # Panel styles
    panel_border_style: str = "rounded"
    table_style: str = "rounded"
    
    def to_rich_theme(self) -> Theme:
        """Convert to Rich theme object."""
        styles = {
            "primary": self.primary_color,
            "secondary": self.secondary_color,
            "success": self.success_color,
            "warning": self.warning_color,
            "error": self.error_color,
            "info": self.info_color,
            "dim": self.dim_color,
        }
        return Theme(styles)


@dataclass
class DisplayConfig:
    """Display and formatting configuration."""
    mode: DisplayMode = DisplayMode.NORMAL
    max_history_display: int = 20
    max_search_results: int = 15
    show_timestamps: bool = True
    show_session_stats: bool = True
    show_memory_stats: bool = True
    
    # Streaming settings
    enable_streaming: bool = True
    stream_chunk_delay: float = 0.01
    
    # Progress indicators
    show_progress_bars: bool = True
    use_spinners: bool = True
    
    # Auto-scroll and paging
    auto_scroll: bool = True
    page_size: int = 10
    
    # Response formatting
    wrap_responses: bool = True
    syntax_highlighting: bool = True
    markdown_rendering: bool = True


@dataclass
class BehaviorConfig:
    """Behavior and interaction configuration."""
    response_style: ResponseStyle = ResponseStyle.CONVERSATIONAL
    auto_save_frequency: int = 30  # seconds
    confirmation_prompts: bool = True
    auto_greetings: bool = True
    
    # Memory preferences
    memory_verbosity: str = "normal"  # minimal, normal, detailed
    show_memory_reasoning: bool = False
    auto_tag_sessions: bool = True
    
    # Error handling
    debug_mode_default: bool = False
    error_recovery_attempts: int = 3
    show_error_details: bool = False
    
    # Session management
    auto_create_sessions: bool = True
    session_title_generation: bool = True
    bookmark_important_sessions: bool = True


@dataclass
class ShortcutsConfig:
    """Keyboard shortcuts and command aliases."""
    aliases: Dict[str, str] = field(default_factory=dict)
    quick_commands: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        # Default aliases
        if not self.aliases:
            self.aliases = {
                "q": "/quit",
                "h": "/help",
                "s": "/status",
                "m": "/memory",
                "hist": "/history",
                "save": "/save",
                "load": "/load",
                "new": "/new",
                "clear": "/clear",
                "debug": "/debug",
                "config": "/configure",
                "theme": "/theme",
                "prefs": "/preferences"
            }
        
        # Default quick commands
        if not self.quick_commands:
            self.quick_commands = {
                "whatsup": "What's my current status and any important updates?",
                "remember": "What are the key things you remember about me?",
                "summary": "Give me a summary of our recent conversations",
                "goals": "What are my current goals and projects?",
                "help-me": "I need help with something. What information do you need?"
            }


@dataclass
class CLIConfig:
    """Complete CLI configuration."""
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration sections
    ui_theme: UIThemeConfig = field(default_factory=UIThemeConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    shortcuts: ShortcutsConfig = field(default_factory=ShortcutsConfig)
    
    # User preferences
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.last_modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CLIConfig':
        """Create from dictionary."""
        # Handle nested dataclasses
        if 'ui_theme' in data:
            data['ui_theme'] = UIThemeConfig(**data['ui_theme'])
        if 'display' in data:
            data['display'] = DisplayConfig(**data['display'])
        if 'behavior' in data:
            data['behavior'] = BehaviorConfig(**data['behavior'])
        if 'shortcuts' in data:
            data['shortcuts'] = ShortcutsConfig(**data['shortcuts'])
        
        return cls(**data)


class CLIConfigManager:
    """Manages CLI configuration and user preferences."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config/cli")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "cli_config.json"
        self.preferences_file = self.config_dir / "user_preferences.json"
        self.themes_file = self.config_dir / "themes.json"
        
        self._config: Optional[CLIConfig] = None
        self._themes: Dict[str, UIThemeConfig] = {}
        
    async def initialize(self):
        """Initialize the configuration manager."""
        try:
            await self.load_config()
            await self.load_themes()
            console.print("[dim]⚙️  CLI configuration loaded[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not load CLI config: {e}[/yellow]")
            self._config = CLIConfig()
            await self.save_config()
    
    async def load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            self._config = CLIConfig.from_dict(data)
        else:
            self._config = CLIConfig()
            await self.save_config()
    
    async def save_config(self):
        """Save configuration to file."""
        if self._config:
            self._config.update_timestamp()
            with open(self.config_file, 'w') as f:
                json.dump(self._config.to_dict(), f, indent=2)
    
    async def load_themes(self):
        """Load available themes."""
        if self.themes_file.exists():
            with open(self.themes_file, 'r') as f:
                themes_data = json.load(f)
            
            for name, theme_data in themes_data.items():
                self._themes[name] = UIThemeConfig(**theme_data)
        else:
            # Create default themes
            await self._create_default_themes()
    
    async def _create_default_themes(self):
        """Create default themes."""
        self._themes = {
            "default": UIThemeConfig(
                name="default",
                primary_color="cyan",
                secondary_color="blue",
                success_color="green",
                warning_color="yellow",
                error_color="red"
            ),
            "dark": UIThemeConfig(
                name="dark",
                primary_color="bright_cyan",
                secondary_color="bright_blue",
                success_color="bright_green",
                warning_color="bright_yellow",
                error_color="bright_red",
                dim_color="bright_black"
            ),
            "light": UIThemeConfig(
                name="light",
                primary_color="blue",
                secondary_color="cyan",
                success_color="green",
                warning_color="orange3",
                error_color="red",
                dim_color="grey50"
            ),
            "blue": UIThemeConfig(
                name="blue",
                primary_color="dodger_blue1",
                secondary_color="steel_blue",
                success_color="spring_green1",
                warning_color="gold1",
                error_color="red1"
            ),
            "green": UIThemeConfig(
                name="green",
                primary_color="spring_green1",
                secondary_color="green",
                success_color="lime",
                warning_color="yellow",
                error_color="red"
            ),
            "purple": UIThemeConfig(
                name="purple",
                primary_color="magenta",
                secondary_color="purple",
                success_color="green",
                warning_color="yellow",
                error_color="red"
            ),
            "monochrome": UIThemeConfig(
                name="monochrome",
                primary_color="white",
                secondary_color="bright_white",
                success_color="white",
                warning_color="white",
                error_color="white",
                dim_color="grey50"
            )
        }
        
        await self.save_themes()
    
    async def save_themes(self):
        """Save themes to file."""
        themes_data = {}
        for name, theme in self._themes.items():
            themes_data[name] = asdict(theme)
        
        with open(self.themes_file, 'w') as f:
            json.dump(themes_data, f, indent=2)
    
    @property
    def config(self) -> CLIConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = CLIConfig()
        return self._config
    
    def get_theme(self, theme_name: str) -> Optional[UIThemeConfig]:
        """Get a specific theme."""
        return self._themes.get(theme_name)
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names."""
        return list(self._themes.keys())
    
    async def set_theme(self, theme_name: str) -> bool:
        """Set the active theme."""
        if theme_name in self._themes:
            self.config.ui_theme = self._themes[theme_name]
            await self.save_config()
            return True
        return False
    
    async def update_preference(self, key: str, value: Any):
        """Update a user preference."""
        self.config.user_preferences[key] = value
        await self.save_config()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.config.user_preferences.get(key, default)
    
    async def set_response_style(self, style: ResponseStyle):
        """Set the response style."""
        self.config.behavior.response_style = style
        await self.save_config()
    
    async def set_display_mode(self, mode: DisplayMode):
        """Set the display mode."""
        self.config.display.mode = mode
        await self.save_config()
    
    async def add_alias(self, alias: str, command: str):
        """Add a command alias."""
        self.config.shortcuts.aliases[alias] = command
        await self.save_config()
    
    async def remove_alias(self, alias: str) -> bool:
        """Remove a command alias."""
        if alias in self.config.shortcuts.aliases:
            del self.config.shortcuts.aliases[alias]
            await self.save_config()
            return True
        return False
    
    async def add_quick_command(self, name: str, command: str):
        """Add a quick command."""
        self.config.shortcuts.quick_commands[name] = command
        await self.save_config()
    
    async def remove_quick_command(self, name: str) -> bool:
        """Remove a quick command."""
        if name in self.config.shortcuts.quick_commands:
            del self.config.shortcuts.quick_commands[name]
            await self.save_config()
            return True
        return False
    
    def resolve_alias(self, input_text: str) -> str:
        """Resolve command aliases."""
        command = input_text.strip()
        
        # Check direct aliases
        if command in self.config.shortcuts.aliases:
            return self.config.shortcuts.aliases[command]
        
        # Check if it starts with an alias (for commands with arguments)
        for alias, full_command in self.config.shortcuts.aliases.items():
            if command.startswith(alias + " "):
                return command.replace(alias, full_command, 1)
        
        return input_text
    
    def resolve_quick_command(self, input_text: str) -> Optional[str]:
        """Resolve quick commands."""
        command = input_text.strip()
        return self.config.shortcuts.quick_commands.get(command)
    
    async def export_config(self, export_path: Path):
        """Export configuration to a file."""
        config_data = {
            'cli_config': self.config.to_dict(),
            'themes': {name: asdict(theme) for name, theme in self._themes.items()},
            'export_timestamp': datetime.now().isoformat(),
            'version': self.config.version
        }
        
        with open(export_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    async def import_config(self, import_path: Path) -> bool:
        """Import configuration from a file."""
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            # Import CLI config
            if 'cli_config' in data:
                self._config = CLIConfig.from_dict(data['cli_config'])
            
            # Import themes
            if 'themes' in data:
                for name, theme_data in data['themes'].items():
                    self._themes[name] = UIThemeConfig(**theme_data)
            
            await self.save_config()
            await self.save_themes()
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to import config: {e}[/red]")
            return False
    
    async def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self._config = CLIConfig()
        await self._create_default_themes()
        await self.save_config()
        await self.save_themes()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            'theme': self.config.ui_theme.name,
            'response_style': self.config.behavior.response_style.value,
            'display_mode': self.config.display.mode.value,
            'streaming_enabled': self.config.display.enable_streaming,
            'auto_save_frequency': self.config.behavior.auto_save_frequency,
            'aliases_count': len(self.config.shortcuts.aliases),
            'quick_commands_count': len(self.config.shortcuts.quick_commands),
            'user_preferences_count': len(self.config.user_preferences),
            'last_modified': self.config.last_modified
        } 