"""
CLI Testing Framework for Task 5.8

Comprehensive testing framework for the Personal AI Assistant CLI,
providing mocks, fixtures, and utilities for testing CLI functionality.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import pytest

from rich.console import Console
from rich.text import Text


@dataclass
class CLITestContext:
    """Test context for CLI operations."""
    temp_dir: Path
    config_dir: Path
    memory_dir: Path
    console: Console
    captured_output: List[str]
    user_inputs: List[str]
    input_index: int = 0
    
    def add_user_input(self, input_text: str):
        """Add simulated user input."""
        self.user_inputs.append(input_text)
    
    def get_next_input(self) -> str:
        """Get next simulated user input."""
        if self.input_index < len(self.user_inputs):
            result = self.user_inputs[self.input_index]
            self.input_index += 1
            return result
        return ""
    
    def reset_inputs(self):
        """Reset input simulation."""
        self.input_index = 0
        self.user_inputs.clear()
    
    def capture_output(self, text: str):
        """Capture console output."""
        self.captured_output.append(text)
    
    def get_captured_output(self) -> str:
        """Get all captured output as string."""
        return "\n".join(self.captured_output)
    
    def clear_output(self):
        """Clear captured output."""
        self.captured_output.clear()


class MockPrompt:
    """Mock Rich Prompt for testing."""
    
    def __init__(self, test_context: CLITestContext):
        self.test_context = test_context
    
    def ask(self, question: str, **kwargs) -> str:
        """Simulate prompt asking."""
        response = self.test_context.get_next_input()
        self.test_context.capture_output(f"PROMPT: {question} -> {response}")
        return response


class MockConfirm:
    """Mock Rich Confirm for testing."""
    
    def __init__(self, test_context: CLITestContext):
        self.test_context = test_context
    
    def ask(self, question: str, default: bool = True) -> bool:
        """Simulate confirmation asking."""
        response = self.test_context.get_next_input()
        bool_response = response.lower() in ['y', 'yes', 'true', '1'] if response else default
        self.test_context.capture_output(f"CONFIRM: {question} -> {bool_response}")
        return bool_response


class MockConsole:
    """Mock Rich Console for testing."""
    
    def __init__(self, test_context: CLITestContext):
        self.test_context = test_context
    
    def print(self, *args, **kwargs):
        """Capture print output."""
        text = " ".join(str(arg) for arg in args)
        self.test_context.capture_output(f"PRINT: {text}")
    
    def rule(self, title: str = "", **kwargs):
        """Mock rule output."""
        self.test_context.capture_output(f"RULE: {title}")


class MockAgentConfig:
    """Mock agent configuration for testing."""
    
    def __init__(self):
        self.agent_name = "Test Assistant"
        self.agent_version = "1.0.0"
        self.model = MockModelConfig()
        self.memory = MockMemoryConfig()
        self.mcp = MockMCPConfig()
        self.system_prompt = "Test system prompt"
        self.enable_conversation_logging = True
        self.max_conversation_history = 50
        self.enable_builtin_tools = True
        self.custom_tools = []


class MockModelConfig:
    """Mock model configuration."""
    
    def __init__(self):
        self.provider = "test_provider"
        self.model_id = "test_model"
        self.temperature = 0.7
        self.max_tokens = 4000
        self.streaming = True
        self.region_name = "us-west-2"


class MockMemoryConfig:
    """Mock memory configuration."""
    
    def __init__(self):
        self.enabled = True
        self.memory_base_path = Path("test_memory")
        self.max_context_memories = 10
        self.importance_threshold = 0.3
        self.enable_semantic_search = True


class MockMCPConfig:
    """Mock MCP configuration."""
    
    def __init__(self):
        self.enabled = True
        self.filesystem_server_url = "http://localhost:8001/mcp"
        self.memory_server_url = "http://localhost:8002/mcp"
        self.connection_timeout = 30
        self.retry_attempts = 3


class MockPersonalAssistantAgent:
    """Mock Personal Assistant Agent for testing."""
    
    def __init__(self, config: MockAgentConfig):
        self.config = config
        self.is_initialized = False
        self.memory_manager = None
        self.mcp_client = None
        self.strands_mcp_tools = None
        self.conversation_history = []
        self.session_context = {}
    
    async def initialize(self) -> bool:
        """Mock initialization."""
        self.is_initialized = True
        return True
    
    async def process_message(self, message: str, user_id: str = "test_user") -> str:
        """Mock message processing."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "user_id": user_id
        })
        
        # Simulate AI response
        if "error" in message.lower():
            raise Exception("Simulated error")
        elif "hello" in message.lower():
            return "Hello! How can I help you today?"
        elif "goodbye" in message.lower():
            return "Goodbye! Have a great day!"
        else:
            return f"I received your message: {message}"
    
    async def stream_response(self, message: str, user_id: str = "test_user") -> AsyncGenerator[str, None]:
        """Mock streaming response."""
        response = await self.process_message(message, user_id)
        words = response.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)  # Simulate streaming delay
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Mock agent status."""
        return {
            "agent_name": self.config.agent_name,
            "initialized": self.is_initialized,
            "memory_enabled": self.config.memory.enabled,
            "mcp_enabled": self.config.mcp.enabled,
            "conversation_turns": len(self.conversation_history)
        }
    
    async def shutdown(self):
        """Mock shutdown."""
        self.is_initialized = False
        self.conversation_history.clear()


class MockSessionManager:
    """Mock Session Manager for testing."""
    
    def __init__(self):
        self.current_session = None
        self.sessions = {}
        self.initialized = False
    
    async def initialize(self):
        """Mock initialization."""
        self.initialized = True
    
    def create_new_session(self, title: Optional[str] = None):
        """Mock session creation."""
        from agent.session_manager import SessionState
        session_id = f"test_session_{len(self.sessions) + 1}"
        session = SessionState(
            session_id=session_id,
            title=title or f"Test Session {len(self.sessions) + 1}"
        )
        self.sessions[session_id] = session
        self.current_session = session
        return session
    
    async def save_session(self, session_id: Optional[str] = None, force: bool = False):
        """Mock session saving."""
        return True
    
    async def load_session(self, session_id: str) -> bool:
        """Mock session loading."""
        if session_id in self.sessions:
            self.current_session = self.sessions[session_id]
            return True
        return False
    
    def get_recent_sessions(self, limit: int = 15):
        """Mock recent sessions."""
        return list(self.sessions.values())[:limit]
    
    def search_sessions(self, query: str) -> List:
        """Mock session search."""
        results = []
        for session in self.sessions.values():
            if query.lower() in session.title.lower():
                results.append({
                    'session': session,
                    'relevance_score': 0.8
                })
        return results
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Mock session statistics."""
        return {
            "total_sessions": len(self.sessions),
            "total_conversations": sum(len(s.conversation_turns) for s in self.sessions.values()),
            "total_messages": sum(s.total_messages for s in self.sessions.values()),
            "average_session_length": 5.0
        }


class MockCLIConfigManager:
    """Mock CLI Config Manager for testing."""
    
    def __init__(self):
        from agent.cli_config import CLIConfig
        self.config = CLIConfig()
        self.initialized = False
        self.themes = {
            "default": {"name": "default", "primary_color": "cyan"},
            "dark": {"name": "dark", "primary_color": "bright_cyan"},
            "light": {"name": "light", "primary_color": "blue"}
        }
    
    async def initialize(self):
        """Mock initialization."""
        self.initialized = True
    
    async def save_config(self):
        """Mock config saving."""
        return True
    
    async def load_config(self):
        """Mock config loading."""
        return True
    
    def get_available_themes(self) -> List[str]:
        """Mock theme list."""
        return list(self.themes.keys())
    
    async def set_theme(self, theme_name: str) -> bool:
        """Mock theme setting."""
        if theme_name in self.themes:
            self.config.ui_theme.name = theme_name
            return True
        return False
    
    def resolve_alias(self, input_text: str) -> str:
        """Mock alias resolution."""
        aliases = self.config.shortcuts.aliases
        if input_text in aliases:
            return aliases[input_text]
        return input_text
    
    def resolve_quick_command(self, input_text: str) -> Optional[str]:
        """Mock quick command resolution."""
        quick_commands = self.config.shortcuts.quick_commands
        return quick_commands.get(input_text)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Mock config summary."""
        return {
            "theme": self.config.ui_theme.name,
            "response_style": self.config.behavior.response_style.value,
            "display_mode": self.config.display.mode.value,
            "streaming_enabled": self.config.display.enable_streaming,
            "auto_save_frequency": self.config.behavior.auto_save_frequency,
            "aliases_count": len(self.config.shortcuts.aliases),
            "quick_commands_count": len(self.config.shortcuts.quick_commands),
            "user_preferences_count": len(self.config.user_preferences),
            "last_modified": self.config.last_modified
        }


class CLITestFramework:
    """Main CLI testing framework."""
    
    def __init__(self):
        self.test_context: Optional[CLITestContext] = None
        self.mock_agent: Optional[MockPersonalAssistantAgent] = None
        self.mock_session_manager: Optional[MockSessionManager] = None
        self.mock_cli_config: Optional[MockCLIConfigManager] = None
        self.patches = []
    
    @asynccontextmanager
    async def setup_cli_test(self, test_name: str = "cli_test"):
        """Setup complete CLI test environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_dir = temp_path / "config"
            memory_dir = temp_path / "memory"
            
            config_dir.mkdir(parents=True)
            memory_dir.mkdir(parents=True)
            
            # Create test context
            console = Console()
            self.test_context = CLITestContext(
                temp_dir=temp_path,
                config_dir=config_dir,
                memory_dir=memory_dir,
                console=console,
                captured_output=[],
                user_inputs=[]
            )
            
            # Setup mocks
            self.mock_agent = MockPersonalAssistantAgent(MockAgentConfig())
            self.mock_session_manager = MockSessionManager()
            self.mock_cli_config = MockCLIConfigManager()
            
            # Initialize mocks
            await self.mock_agent.initialize()
            await self.mock_session_manager.initialize()
            await self.mock_cli_config.initialize()
            
            try:
                yield self.test_context
            finally:
                # Cleanup
                await self.cleanup()
    
    async def cleanup(self):
        """Cleanup test environment."""
        if self.mock_agent:
            await self.mock_agent.shutdown()
        
        # Remove all patches
        for patcher in self.patches:
            patcher.stop()
        self.patches.clear()
    
    def create_mock_cli(self, test_context: CLITestContext):
        """Create a mock CLI instance for testing."""
        from agent.cli import AgentCLI
        
        # Create real CLI instance
        cli = AgentCLI()
        
        # Replace with mocks
        cli.agent = self.mock_agent
        cli.session_manager = self.mock_session_manager
        cli.cli_config = self.mock_cli_config
        cli.is_running = False
        cli.conversation_history.current_session_turns.clear()
        cli.total_messages = 0
        cli.session_start_time = datetime.now()
        
        return cli
    
    def patch_console_methods(self, test_context: CLITestContext):
        """Patch console methods for testing."""
        mock_prompt = MockPrompt(test_context)
        mock_confirm = MockConfirm(test_context)
        mock_console = MockConsole(test_context)
        
        # Patch Rich components
        prompt_patch = patch('rich.prompt.Prompt.ask', side_effect=mock_prompt.ask)
        confirm_patch = patch('rich.prompt.Confirm.ask', side_effect=mock_confirm.ask)
        console_patch = patch('agent.cli.console', mock_console)
        
        self.patches.extend([prompt_patch, confirm_patch, console_patch])
        
        for patcher in [prompt_patch, confirm_patch, console_patch]:
            patcher.start()
        
        return mock_prompt, mock_confirm, mock_console
    
    def simulate_user_session(self, test_context: CLITestContext, commands: List[str]):
        """Simulate a complete user session with commands."""
        test_context.reset_inputs()
        for command in commands:
            test_context.add_user_input(command)
    
    async def run_command_test(self, test_context: CLITestContext, command: str, expected_outputs: List[str] = None) -> str:
        """Run a single command test."""
        cli = self.create_mock_cli(test_context)
        self.patch_console_methods(test_context)
        
        test_context.clear_output()
        test_context.add_user_input(command)
        
        # Execute command
        if command.startswith('/'):
            result = await cli._handle_enhanced_commands(command)
        else:
            await cli._process_message_with_enhanced_streaming(command)
            result = True
        
        output = test_context.get_captured_output()
        
        # Check expected outputs
        if expected_outputs:
            for expected in expected_outputs:
                assert expected in output, f"Expected '{expected}' not found in output: {output}"
        
        return output
    
    def assert_session_state(self, expected_messages: int = None, 
                           expected_session_title: str = None):
        """Assert session state matches expectations."""
        if expected_messages is not None:
            actual_messages = len(self.mock_agent.conversation_history)
            assert actual_messages == expected_messages, \
                f"Expected {expected_messages} messages, got {actual_messages}"
        
        if expected_session_title is not None and self.mock_session_manager.current_session:
            actual_title = self.mock_session_manager.current_session.title
            assert actual_title == expected_session_title, \
                f"Expected session title '{expected_session_title}', got '{actual_title}'"
    
    def assert_configuration_changed(self, setting_path: str, expected_value: Any):
        """Assert configuration setting has changed."""
        config = self.mock_cli_config.config
        parts = setting_path.split('.')
        current = config
        for part in parts[:-1]:
            current = getattr(current, part)
        
        actual_value = getattr(current, parts[-1])
        assert actual_value == expected_value, \
            f"Expected {setting_path}={expected_value}, got {actual_value}"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history from mock agent."""
        return self.mock_agent.conversation_history.copy()
    
    def get_session_data(self) -> Dict[str, Any]:
        """Get current session data."""
        if self.mock_session_manager.current_session:
            return {
                "session_id": self.mock_session_manager.current_session.session_id,
                "title": self.mock_session_manager.current_session.title,
                "conversation_count": len(self.mock_session_manager.current_session.conversation_turns)
            }
        return {}


# Pytest fixtures for easy test setup
@pytest.fixture
async def cli_test_framework():
    """Pytest fixture for CLI test framework."""
    framework = CLITestFramework()
    try:
        yield framework
    finally:
        await framework.cleanup()


@pytest.fixture
async def cli_test_context(cli_test_framework):
    """Pytest fixture for CLI test context."""
    async with cli_test_framework.setup_cli_test() as context:
        yield context


def create_test_memory_files(test_dir: Path):
    """Create test memory files for testing."""
    memory_dir = test_dir / "memory"
    memory_dir.mkdir(exist_ok=True)
    
    # Create core memory files
    core_dir = memory_dir / "core"
    core_dir.mkdir(exist_ok=True)
    
    test_files = {
        "user_profile.md": "# User Profile\n\nName: Test User\nPreferences: Testing",
        "active_context.md": "# Active Context\n\nCurrent testing session",
        "relationship_evolution.md": "# Relationship Evolution\n\nBuilding test relationship"
    }
    
    for filename, content in test_files.items():
        (core_dir / filename).write_text(content)
    
    return memory_dir


def create_test_session_files(test_dir: Path):
    """Create test session files for testing."""
    sessions_dir = test_dir / "memory" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample session
    session_data = {
        "session_id": "test_session_1",
        "title": "Test Session",
        "created_at": datetime.now().isoformat(),
        "conversation_turns": [
            {
                "timestamp": datetime.now().isoformat(),
                "user_message": "Hello",
                "assistant_response": "Hi there!"
            }
        ]
    }
    
    with open(sessions_dir / "test_session_1.json", 'w') as f:
        json.dump(session_data, f, indent=2)
    
    return sessions_dir


# Utility functions for test assertions
def assert_output_contains(output: str, expected: List[str]):
    """Assert output contains all expected strings."""
    for expected_str in expected:
        assert expected_str in output, f"Expected '{expected_str}' not found in output"


def assert_output_not_contains(output: str, unexpected: List[str]):
    """Assert output does not contain unexpected strings."""
    for unexpected_str in unexpected:
        assert unexpected_str not in output, f"Unexpected '{unexpected_str}' found in output"


def extract_command_responses(output: str) -> List[str]:
    """Extract command responses from captured output."""
    lines = output.split('\n')
    responses = []
    for line in lines:
        if line.startswith('PRINT: ') and not line.startswith('PRINT: PROMPT:'):
            responses.append(line[7:])  # Remove 'PRINT: ' prefix
    return responses 