"""
Command Line Interface for Personal AI Assistant

Provides an interactive CLI for communicating with the agent,
including support for streaming responses and agent management.
"""

import asyncio
import sys
import os
import traceback
import signal
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.columns import Columns
from rich.align import Align
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.rule import Rule
from rich.markdown import Markdown
from rich.syntax import Syntax

from .core_agent import PersonalAssistantAgent
from .agent_config import AgentConfig
from .session_manager import SessionManager, SessionState
from .cli_config import CLIConfigManager, ResponseStyle, DisplayMode


console = Console()


class CLIError(Exception):
    """Base exception for CLI-specific errors."""
    pass


class AgentNotInitializedError(CLIError):
    """Raised when attempting operations without initialized agent."""
    pass


class MemorySystemUnavailableError(CLIError):
    """Raised when memory system is not available."""
    pass


class UserInterruptError(CLIError):
    """Raised when user interrupts operation."""
    pass


class ErrorHandler:
    """Centralized error handling and recovery for CLI operations."""
    
    def __init__(self, console: Console):
        self.console = console
        self.error_count = 0
        self.last_error_time = None
        self.recovery_attempted = False
        
    def handle_error(self, error: Exception, operation: str, context: Dict[str, Any] = None) -> bool:
        """
        Handle an error with appropriate recovery strategies.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            context: Additional context for recovery
            
        Returns:
            bool: True if operation should be retried, False otherwise
        """
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        # Determine error type and appropriate response
        if isinstance(error, KeyboardInterrupt):
            return self._handle_user_interrupt(operation)
        elif isinstance(error, EOFError):
            return self._handle_eof_error(operation)
        elif isinstance(error, AgentNotInitializedError):
            return self._handle_agent_not_initialized(operation, context)
        elif isinstance(error, MemorySystemUnavailableError):
            return self._handle_memory_unavailable(operation, context)
        elif isinstance(error, ConnectionError):
            return self._handle_connection_error(error, operation, context)
        elif isinstance(error, FileNotFoundError):
            return self._handle_file_not_found(error, operation, context)
        elif isinstance(error, PermissionError):
            return self._handle_permission_error(error, operation, context)
        elif isinstance(error, ValueError):
            return self._handle_value_error(error, operation, context)
        else:
            return self._handle_generic_error(error, operation, context)
    
    def _handle_user_interrupt(self, operation: str) -> bool:
        """Handle user interruption (Ctrl+C)."""
        self.console.print(f"\n[yellow]⚠️  Operation '{operation}' interrupted by user[/yellow]")
        
        if Confirm.ask("🤔 Would you like to continue with the CLI?", default=True):
            self.console.print("[green]✅ Continuing...[/green]")
            return False  # Don't retry the operation
        else:
            self.console.print("[yellow]👋 Exiting CLI...[/yellow]")
            sys.exit(0)
    
    def _handle_eof_error(self, operation: str) -> bool:
        """Handle EOF error (input stream closed)."""
        self.console.print(f"\n[yellow]⚠️  Input stream closed during '{operation}'[/yellow]")
        self.console.print("[yellow]👋 Exiting CLI...[/yellow]")
        sys.exit(0)
    
    def _handle_agent_not_initialized(self, operation: str, context: Dict[str, Any]) -> bool:
        """Handle agent not initialized error."""
        self.console.print(f"[red]❌ Agent not initialized for operation '{operation}'[/red]")
        
        if not self.recovery_attempted and context and 'agent_cli' in context:
            self.console.print("[yellow]🔄 Attempting to initialize agent...[/yellow]")
            self.recovery_attempted = True
            
            # Attempt to reinitialize
            try:
                agent_cli = context['agent_cli']
                success = asyncio.create_task(agent_cli.initialize_agent())
                if success:
                    self.console.print("[green]✅ Agent reinitialized successfully[/green]")
                    return True  # Retry the operation
                else:
                    self.console.print("[red]❌ Agent reinitialization failed[/red]")
            except Exception as e:
                self.console.print(f"[red]❌ Reinitialization error: {e}[/red]")
        
        self.console.print("[yellow]💡 Suggestion: Restart the CLI to reinitialize the agent[/yellow]")
        return False
    
    def _handle_memory_unavailable(self, operation: str, context: Dict[str, Any]) -> bool:
        """Handle memory system unavailable error."""
        self.console.print(f"[red]❌ Memory system unavailable for operation '{operation}'[/red]")
        self.console.print("[yellow]💡 Some features may be limited without memory access[/yellow]")
        
        # Offer degraded functionality
        if Confirm.ask("🤔 Continue with limited functionality?", default=True):
            self.console.print("[yellow]⚠️  Continuing with reduced capabilities[/yellow]")
            return False  # Don't retry, continue with limitations
        else:
            return False
    
    def _handle_connection_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Handle connection-related errors."""
        self.console.print(f"[red]❌ Connection error during '{operation}': {error}[/red]")
        
        if not self.recovery_attempted:
            self.console.print("[yellow]🔄 Attempting to reconnect...[/yellow]")
            self.recovery_attempted = True
            
            # Wait a moment and suggest retry
            import time
            time.sleep(2)
            
            if Confirm.ask("🔄 Retry the operation?", default=True):
                return True  # Retry
        
        self.console.print("[yellow]💡 Suggestion: Check network connectivity and restart if needed[/yellow]")
        return False
    
    def _handle_file_not_found(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Handle file not found errors."""
        self.console.print(f"[red]❌ File not found during '{operation}': {error}[/red]")
        
        # Extract filename if possible
        filename = str(error).split("'")[1] if "'" in str(error) else "unknown file"
        self.console.print(f"[yellow]💡 Missing file: {filename}[/yellow]")
        
        # Offer to create missing directories or files if applicable
        if "memory" in operation.lower() and not self.recovery_attempted:
            self.recovery_attempted = True
            if Confirm.ask("🔧 Attempt to create missing memory directories?", default=True):
                try:
                    # This would need to be implemented based on the specific error
                    self.console.print("[yellow]🔄 Creating missing directories...[/yellow]")
                    # Implementation would go here
                    self.console.print("[green]✅ Directories created[/green]")
                    return True  # Retry
                except Exception as e:
                    self.console.print(f"[red]❌ Failed to create directories: {e}[/red]")
        
        return False
    
    def _handle_permission_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Handle permission errors."""
        self.console.print(f"[red]❌ Permission denied during '{operation}': {error}[/red]")
        self.console.print("[yellow]💡 Suggestion: Check file/directory permissions or run with appropriate privileges[/yellow]")
        
        # Extract path if possible
        path = str(error).split("'")[1] if "'" in str(error) else "unknown path"
        self.console.print(f"[yellow]📁 Path: {path}[/yellow]")
        
        return False
    
    def _handle_value_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Handle value errors (invalid input, etc.)."""
        self.console.print(f"[red]❌ Invalid input during '{operation}': {error}[/red]")
        self.console.print("[yellow]💡 Please check your input and try again[/yellow]")
        
        # For value errors, we can usually retry with corrected input
        return False
    
    def _handle_generic_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Handle unexpected errors."""
        self.console.print(f"[red]❌ Unexpected error during '{operation}': {error}[/red]")
        
        # Show detailed error in debug mode or if requested
        if context and context.get('debug', False):
            self.console.print("\n[dim]Full error traceback:[/dim]")
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        elif Confirm.ask("🔍 Show detailed error information?", default=False):
            self.console.print("\n[dim]Full error traceback:[/dim]")
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
        
        # Offer retry for generic errors
        if not self.recovery_attempted and self.error_count < 3:
            self.recovery_attempted = True
            if Confirm.ask("🔄 Retry the operation?", default=False):
                return True
        
        self.console.print("[yellow]💡 If the problem persists, please restart the CLI[/yellow]")
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        return {
            'total_errors': self.error_count,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'recovery_attempted': self.recovery_attempted
        }


def handle_cli_operation(error_handler: ErrorHandler):
    """Decorator for handling CLI operations with error recovery."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            operation_name = func.__name__.replace('_', ' ').title()
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Reset recovery flag for each attempt
                    if retry_count > 0:
                        error_handler.recovery_attempted = False
                    
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    retry_count += 1
                    
                    # Prepare context for error handler
                    context = {
                        'retry_count': retry_count,
                        'max_retries': max_retries,
                        'function_name': func.__name__
                    }
                    
                    # Add agent_cli to context if available
                    if args and hasattr(args[0], 'agent'):
                        context['agent_cli'] = args[0]
                    
                    # Handle the error
                    should_retry = error_handler.handle_error(e, operation_name, context)
                    
                    if not should_retry or retry_count >= max_retries:
                        if retry_count >= max_retries:
                            console.print(f"[red]❌ Maximum retries ({max_retries}) exceeded for '{operation_name}'[/red]")
                        break
                    else:
                        console.print(f"[yellow]🔄 Retrying '{operation_name}' (attempt {retry_count + 1}/{max_retries})[/yellow]")
                        await asyncio.sleep(1)  # Brief delay before retry
            
            return None
        return wrapper
    return decorator


class ConversationHistory:
    """Manages conversation history display and formatting."""
    
    def __init__(self, max_display_turns: int = 20):
        self.max_display_turns = max_display_turns
        self.current_session_turns = []
    
    def add_turn(self, user_message: str, assistant_response: str, timestamp: Optional[str] = None):
        """Add a conversation turn to the current session history."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.current_session_turns.append({
            'timestamp': timestamp,
            'user_message': user_message,
            'assistant_response': assistant_response
        })
    
    def get_formatted_history(self, num_turns: Optional[int] = None) -> Panel:
        """Get formatted conversation history as a Rich Panel."""
        if not self.current_session_turns:
            return Panel(
                Text("No conversation history in this session.", style="dim italic"),
                title="📜 Conversation History",
                border_style="blue"
            )
        
        display_turns = num_turns or self.max_display_turns
        recent_turns = self.current_session_turns[-display_turns:]
        
        history_content = []
        
        for i, turn in enumerate(recent_turns, 1):
            timestamp = datetime.fromisoformat(turn['timestamp']).strftime("%H:%M:%S")
            
            # User message
            user_text = Text()
            user_text.append(f"[{timestamp}] ", style="dim")
            user_text.append("You: ", style="bold cyan")
            user_text.append(turn['user_message'][:200] + ("..." if len(turn['user_message']) > 200 else ""))
            
            # Assistant response
            assistant_text = Text()
            assistant_text.append(" " * 11)  # Indent to align with user message
            assistant_text.append("Assistant: ", style="bold green")
            assistant_text.append(turn['assistant_response'][:200] + ("..." if len(turn['assistant_response']) > 200 else ""))
            
            history_content.extend([user_text, assistant_text, Text()])  # Empty line between turns
        
        # Remove last empty line
        if history_content:
            history_content.pop()
        
        content = Text()
        for item in history_content:
            content.append(item)
            content.append("\n")
        
        return Panel(
            content,
            title=f"📜 Conversation History (Last {len(recent_turns)} turns)",
            border_style="blue"
        )


class AgentCLI:
    """Enhanced command-line interface for the Personal AI Assistant."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the CLI.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.agent: Optional[PersonalAssistantAgent] = None
        self.config_path = config_path
        self.is_running = False
        self.conversation_history = ConversationHistory()
        self.session_start_time = datetime.now()
        self.total_messages = 0
        self.error_handler = ErrorHandler(console)
        self.debug_mode = False
        self.graceful_shutdown = False
        
        # Session management
        self.session_manager: Optional[SessionManager] = None
        self.current_session_id: Optional[str] = None
        self.session_persistence_enabled = True
        
        # CLI configuration
        self.cli_config: Optional[CLIConfigManager] = None
        
    async def initialize_agent(self) -> bool:
        """Initialize the agent with configuration and enhanced progress display."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create initialization progress
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    
                    # Load configuration
                    config_task = progress.add_task("Loading configuration...", total=None)
                    await asyncio.sleep(0.1)  # Small delay for visual effect
                    
                    if self.config_path and self.config_path.exists():
                        config = AgentConfig.from_file(self.config_path)
                        progress.update(config_task, description=f"✓ Loaded config from {self.config_path}")
                    else:
                        config = AgentConfig.from_env()
                        progress.update(config_task, description="✓ Using default configuration")
                    
                    await asyncio.sleep(0.2)
                    
                                    # Initialize CLI configuration
                config_task = progress.add_task("Loading CLI configuration...", total=None)
                self.cli_config = CLIConfigManager()
                await self.cli_config.initialize()
                progress.update(config_task, description="✓ CLI configuration loaded")
                await asyncio.sleep(0.2)
                
                    # Initialize session manager
                session_task = progress.add_task("Setting up session management...", total=None)
                self.session_manager = SessionManager()
                await self.session_manager.initialize()
                progress.update(session_task, description="✓ Session manager ready")
                await asyncio.sleep(0.2)
                
                # Initialize agent
                agent_task = progress.add_task("Initializing AI agent...", total=None)
                self.agent = PersonalAssistantAgent(config)
                
                # Initialize with progress updates
                progress.update(agent_task, description="Connecting to AI model...")
                await asyncio.sleep(0.3)
                
                progress.update(agent_task, description="Loading memory system...")
                success = await self.agent.initialize()
                
                if success:
                    progress.update(agent_task, description="✓ Agent initialized successfully!")
                    await asyncio.sleep(0.5)
                else:
                    progress.update(agent_task, description="✗ Agent initialization failed")
                    await asyncio.sleep(0.5)
                    raise AgentNotInitializedError("Agent initialization returned False")
                
                # Display success message
                console.print()
                console.print("🎉 [bold green]Personal AI Assistant is ready![/bold green]")
                console.print()
                return True
                    
            except Exception as e:
                retry_count += 1
                
                # Prepare context for error handler
                context = {
                    'retry_count': retry_count,
                    'max_retries': max_retries,
                    'agent_cli': self,
                    'debug': self.debug_mode
                }
                
                # Handle the error
                should_retry = self.error_handler.handle_error(e, "Agent Initialization", context)
                
                if not should_retry or retry_count >= max_retries:
                    if retry_count >= max_retries:
                        console.print(f"[red]❌ Maximum initialization attempts ({max_retries}) exceeded[/red]")
                    console.print("[yellow]💡 Try restarting with a different configuration or check your environment setup[/yellow]")
                    return False
                else:
                    console.print(f"[yellow]🔄 Retrying initialization (attempt {retry_count + 1}/{max_retries})[/yellow]")
                    await asyncio.sleep(2)  # Brief delay before retry
        
        return False
    
    async def run_interactive_session(self) -> None:
        """Run an enhanced interactive chat session with the agent."""
        if not self.agent:
            raise AgentNotInitializedError("Agent must be initialized before starting interactive session")
        
        self.is_running = True
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            self.graceful_shutdown = True
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Display enhanced welcome
        self._display_enhanced_welcome()
        
        # Initialize a new session if session management is enabled
        if self.session_persistence_enabled and self.session_manager:
            self.session_manager.create_new_session()
        
        try:
            while self.is_running and not self.graceful_shutdown:
                try:
                    # Display session info periodically
                    if self.total_messages > 0 and self.total_messages % 10 == 0:
                        self._display_session_stats()
                    
                    # Get user input with enhanced prompt
                    console.print()
                    user_input = Prompt.ask(
                        "[bold cyan]💬 You",
                        console=console,
                        show_default=False
                    )
                    
                    if not user_input.strip():
                        continue
                    
                    # Process aliases and quick commands if CLI config is available
                    processed_input = user_input.strip()
                    if self.cli_config:
                        # Check for quick commands first
                        quick_command = self.cli_config.resolve_quick_command(processed_input)
                        if quick_command:
                            console.print(f"[dim]📝 Quick command: {processed_input} → {quick_command}[/dim]")
                            processed_input = quick_command
                        else:
                            # Check for aliases
                            processed_input = self.cli_config.resolve_alias(processed_input)
                    
                    # Handle special commands
                    if await self._handle_enhanced_commands(processed_input):
                        continue
                    
                    # Process message with enhanced streaming
                    await self._process_message_with_enhanced_streaming(processed_input)
                    self.total_messages += 1
                    
                except (KeyboardInterrupt, EOFError) as e:
                    # Handle user interruption gracefully
                    context = {'graceful_shutdown': True}
                    should_continue = not self.error_handler.handle_error(e, "Interactive Session", context)
                    if not should_continue:
                        break
                
                except Exception as e:
                    # Handle unexpected errors during session
                    context = {
                        'agent_cli': self,
                        'debug': self.debug_mode,
                        'session_running': True
                    }
                    
                    should_retry = self.error_handler.handle_error(e, "Interactive Session", context)
                    
                    if not should_retry:
                        console.print("[yellow]⚠️  Continuing session despite error...[/yellow]")
                        continue
                
        except Exception as e:
            # Final catch-all for severe errors
            console.print(f"[red]💥 Critical error in interactive session: {e}[/red]")
            if self.debug_mode:
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
        finally:
            await self._cleanup()
    
    def _display_enhanced_welcome(self) -> None:
        """Display enhanced welcome message with better formatting."""
        # Create main welcome panel
        welcome_text = Text()
        welcome_text.append("🤖 Personal AI Assistant", style="bold blue")
        welcome_text.append("\n\nWelcome! I'm your AI assistant with persistent memory capabilities.")
        welcome_text.append("\nI remember our conversations and learn about your preferences over time.")
        
        # Create commands table
        commands_table = Table(show_header=True, header_style="bold magenta", box=None)
        commands_table.add_column("Command", style="cyan", width=12)
        commands_table.add_column("Description", style="white")
        
        commands_table.add_row("/status", "Show detailed agent and memory status")
        commands_table.add_row("/history", "Display conversation history")
        commands_table.add_row("/memory", "Browse and search memory files")
        commands_table.add_row("/stats", "Show session statistics")
        commands_table.add_row("/clear", "Clear current session history")
        commands_table.add_row("/debug", "Toggle debug mode on/off")
        commands_table.add_row("/errors", "Show error summary and system health")
        commands_table.add_row("/recover", "Attempt system recovery")
        commands_table.add_row("/reset", "Reset and reinitialize agent")
        commands_table.add_row("/sessions", "Browse and manage conversation sessions")
        commands_table.add_row("/save", "Save current session")
        commands_table.add_row("/load", "Load a previous session")
        commands_table.add_row("/new", "Start a new session")
        commands_table.add_row("/configure", "Configure CLI settings and preferences")
        commands_table.add_row("/theme", "Change color theme")
        commands_table.add_row("/preferences", "Manage user preferences")
        commands_table.add_row("/aliases", "Manage command aliases")
        commands_table.add_row("/quick", "Show quick commands")
        commands_table.add_row("/help", "Show this help message")
        commands_table.add_row("/quit", "Exit the assistant")
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(Panel(welcome_text, title="🎯 Welcome", border_style="blue"), size=6),
            Layout(Panel(commands_table, title="⚡ Available Commands", border_style="green"), size=10)
        )
        
        console.print(layout)
        console.print()
        console.print("💡 [dim]Tip: Just start typing to begin our conversation![/dim]")
    
    def _display_session_stats(self) -> None:
        """Display current session statistics."""
        session_duration = datetime.now() - self.session_start_time
        hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        stats_text = Text()
        stats_text.append(f"📊 Session: {hours:02d}:{minutes:02d}:{seconds:02d} | ", style="dim")
        stats_text.append(f"Messages: {self.total_messages} | ", style="dim")
        stats_text.append(f"History: {len(self.conversation_history.current_session_turns)} turns", style="dim")
        
        console.print(Rule(stats_text, style="dim"))
    
    async def _handle_enhanced_commands(self, command: str) -> bool:
        """
        Handle enhanced CLI commands with better formatting.
        
        Args:
            command: User input command
            
        Returns:
            True if command was handled, False otherwise
        """
        cmd = command.lower()
        
        if cmd in ['/quit', '/exit', '/q']:
            if Confirm.ask("🤔 Are you sure you want to exit?", default=True):
                self.is_running = False
                console.print("[yellow]👋 Goodbye! Thanks for chatting![/yellow]")
            return True
        
        elif cmd == '/status':
            await self._show_enhanced_status()
            return True
        
        elif cmd == '/history':
            self._show_enhanced_history()
            return True
        
        elif cmd == '/memory':
            await self._show_memory_browser()
            return True
        
        elif cmd == '/stats':
            self._show_session_statistics()
            return True
        
        elif cmd == '/clear':
            if Confirm.ask("🗑️  Clear conversation history for this session?", default=False):
                self._clear_session_history()
            return True
        
        elif cmd == '/debug':
            self._toggle_debug_mode()
            return True
        
        elif cmd == '/errors':
            self._show_error_summary()
            return True
        
        elif cmd == '/recover':
            await self._attempt_system_recovery()
            return True
        
        elif cmd == '/reset':
            if Confirm.ask("🔄 Reset agent and reinitialize? This may resolve persistent issues.", default=False):
                await self._reset_agent()
            return True
        
        elif cmd == '/sessions':
            await self._show_session_browser()
            return True
        
        elif cmd == '/save':
            await self._save_current_session()
            return True
        
        elif cmd == '/load':
            await self._load_session_interactive()
            return True
        
        elif cmd == '/new':
            await self._create_new_session()
            return True
        
        elif cmd == '/configure' or cmd == '/config':
            await self._show_configuration_menu()
            return True
        
        elif cmd == '/theme':
            await self._show_theme_selector()
            return True
        
        elif cmd == '/preferences' or cmd == '/prefs':
            await self._show_preferences_menu()
            return True
        
        elif cmd == '/aliases':
            await self._show_aliases_menu()
            return True
        
        elif cmd == '/quick':
            await self._show_quick_commands()
            return True
        
        elif cmd == '/help':
            self._display_enhanced_welcome()
            return True
        
        return False
    
    async def _process_message_with_enhanced_streaming(self, message: str) -> None:
        """Process a message with enhanced streaming response display."""
        if not self.agent:
            raise AgentNotInitializedError("Agent not initialized for message processing")
        
        max_retries = 2
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Show thinking indicator
                with console.status("[bold blue]🤔 Thinking...", spinner="dots"):
                    await asyncio.sleep(0.5)  # Brief pause for effect
                
                console.print("[bold green]🤖 Assistant:[/bold green] ", end="")
                
                response_text = ""
                start_time = datetime.now()
                
                # Check agent availability before streaming
                if not hasattr(self.agent, 'stream_response'):
                    raise MemorySystemUnavailableError("Agent streaming not available")
                
                async for chunk in self.agent.stream_response(message):
                    response_text += chunk
                    console.print(chunk, end="", highlight=False)
                
                console.print()  # New line after response
                
                # Add to conversation history
                self.conversation_history.add_turn(message, response_text)
                
                # Store in session if session management is enabled
                if self.session_persistence_enabled and self.session_manager and self.session_manager.current_session:
                    self.session_manager.current_session.add_conversation_turn(message, response_text)
                    
                    # Auto-save session periodically
                    await self.session_manager.auto_save_check()
                
                # Show response time
                response_time = (datetime.now() - start_time).total_seconds()
                console.print(f"[dim]⏱️  Response time: {response_time:.1f}s[/dim]")
                
                # Store interaction in memory system if available
                try:
                    if hasattr(self.agent, '_store_interaction') and self.agent.memory_manager:
                        await self.agent._store_interaction(message, response_text, "default")
                except Exception as memory_error:
                    console.print(f"[yellow]⚠️  Memory storage failed: {memory_error}[/yellow]")
                
                return  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                
                # Prepare context for error handler
                context = {
                    'retry_count': retry_count,
                    'max_retries': max_retries,
                    'agent_cli': self,
                    'debug': self.debug_mode,
                    'message': message
                }
                
                # Handle the error
                should_retry = self.error_handler.handle_error(e, "Message Processing", context)
                
                if not should_retry or retry_count >= max_retries:
                    if retry_count >= max_retries:
                        console.print(f"[red]❌ Failed to process message after {max_retries} attempts[/red]")
                    
                    # Provide fallback response
                    console.print("[yellow]📝 Message recorded but response failed. Please try again.[/yellow]")
                    self.conversation_history.add_turn(message, "[Error: Response failed]")
                    break
                else:
                    console.print(f"[yellow]🔄 Retrying message processing (attempt {retry_count + 1}/{max_retries})[/yellow]")
                    await asyncio.sleep(1)  # Brief delay before retry
    
    async def _show_enhanced_status(self) -> None:
        """Display enhanced agent status information."""
        if not self.agent:
            raise AgentNotInitializedError("Agent not initialized for status display")
        
        try:
            with console.status("[bold blue]📊 Gathering status information...", spinner="dots"):
                status = await self.agent.get_agent_status()
            
            # Create status table
            status_table = Table(show_header=False, box=None, padding=(0, 2))
            status_table.add_column("Property", style="cyan", width=20)
            status_table.add_column("Value", style="white")
            
            # Basic info
            status_table.add_row("🤖 Agent", f"{status.get('agent_name', 'Unknown')} v{status.get('agent_version', 'Unknown')}")
            status_table.add_row("🔧 Initialized", "✅ Yes" if status.get('initialized', False) else "❌ No")
            status_table.add_row("🧠 Memory", "✅ Enabled" if status.get('memory_enabled', False) else "❌ Disabled")
            status_table.add_row("🔗 MCP", "✅ Enabled" if status.get('mcp_enabled', False) else "❌ Disabled")
            status_table.add_row("💬 Conversations", str(status.get('conversation_turns', 0)))
            
            # Error summary
            error_summary = self.error_handler.get_error_summary()
            if error_summary['total_errors'] > 0:
                status_table.add_row("", "")  # Separator
                status_table.add_row("⚠️  Errors", str(error_summary['total_errors']))
                if error_summary['last_error_time']:
                    last_error = datetime.fromisoformat(error_summary['last_error_time'])
                    status_table.add_row("🕒 Last Error", last_error.strftime("%H:%M:%S"))
            
            # Memory stats
            if 'memory_stats' in status:
                mem_stats = status['memory_stats']
                status_table.add_row("", "")  # Separator
                status_table.add_row("📚 Total Memories", str(mem_stats.get('total_memories', 'N/A')))
                status_table.add_row("🔄 Interactions", str(mem_stats.get('total_interactions', 'N/A')))
            
            # MCP status
            if 'mcp_status' in status:
                mcp_status = status['mcp_status']
                status_table.add_row("", "")  # Separator
                status_table.add_row("🌐 MCP Status", mcp_status.get('overall_status', 'unknown'))
                status_table.add_row("🔌 Servers", f"{mcp_status.get('connected_servers', 0)}/{mcp_status.get('total_servers', 0)}")
                status_table.add_row("🛠️  Tools", str(mcp_status.get('available_tools', 0)))
            
            # Session info
            session_duration = datetime.now() - self.session_start_time
            status_table.add_row("", "")  # Separator
            status_table.add_row("⏱️  Session Time", str(session_duration).split('.')[0])  # Remove microseconds
            status_table.add_row("🗣️  Messages", str(self.total_messages))
            status_table.add_row("🐞 Debug Mode", "✅ Enabled" if self.debug_mode else "❌ Disabled")
            
            console.print(Panel(status_table, title="📊 Agent Status", border_style="green"))
            
        except Exception as e:
            # Prepare context for error handler
            context = {
                'agent_cli': self,
                'debug': self.debug_mode,
                'operation': 'status_display'
            }
            
            # Handle the error
            should_retry = self.error_handler.handle_error(e, "Status Display", context)
            
            if not should_retry:
                # Provide fallback status display
                console.print(Panel(
                    Text("❌ Unable to retrieve full status information\n"
                         f"Agent: {'✅ Initialized' if self.agent else '❌ Not Initialized'}\n"
                         f"Session: {datetime.now() - self.session_start_time}\n"
                         f"Messages: {self.total_messages}\n"
                         f"Errors: {self.error_handler.error_count}",
                         style="yellow"),
                    title="📊 Basic Status (Limited)",
                    border_style="yellow"
                ))
    
    def _show_enhanced_history(self) -> None:
        """Display enhanced conversation history."""
        if not self.conversation_history.current_session_turns:
            console.print(Panel(
                Text("No conversation history in this session yet.", style="dim italic"),
                title="📜 Conversation History",
                border_style="yellow"
            ))
            return
        
        # Ask how many turns to show
        max_turns = len(self.conversation_history.current_session_turns)
        if max_turns > 10:
            num_turns = Prompt.ask(
                f"How many recent turns to show? (1-{max_turns})",
                default="10",
                console=console
            )
            try:
                num_turns = min(int(num_turns), max_turns)
            except ValueError:
                num_turns = 10
        else:
            num_turns = max_turns
        
        history_panel = self.conversation_history.get_formatted_history(num_turns)
        console.print(history_panel)
    
    async def _show_memory_browser(self) -> None:
        """Display comprehensive memory browser interface."""
        if not self.agent or not self.agent.memory_manager:
            console.print(Panel(
                Text("❌ Memory system not available", style="red"),
                title="🔍 Memory Browser",
                border_style="red"
            ))
            return
        
        while True:
            # Display memory browser menu
            console.print()
            menu_table = Table(show_header=True, header_style="bold magenta", box=None)
            menu_table.add_column("Option", style="cyan", width=8)
            menu_table.add_column("Description", style="white")
            
            menu_table.add_row("1", "Browse all core memories")
            menu_table.add_row("2", "Search memories")
            menu_table.add_row("3", "View recent interactions")
            menu_table.add_row("4", "Memory statistics")
            menu_table.add_row("5", "Memory health check")
            menu_table.add_row("6", "Core memory details")
            menu_table.add_row("7", "Interaction analysis")
            menu_table.add_row("q", "Return to main chat")
            
            console.print(Panel(menu_table, title="🧠 Memory Browser", border_style="blue"))
            
            choice = Prompt.ask(
                "[bold cyan]Select option",
                choices=["1", "2", "3", "4", "5", "6", "7", "q"],
                default="q"
            )
            
            if choice == "q":
                break
            elif choice == "1":
                await self._browse_core_memories()
            elif choice == "2":
                await self._search_memories_interactive()
            elif choice == "3":
                await self._view_recent_interactions()
            elif choice == "4":
                await self._show_memory_statistics()
            elif choice == "5":
                await self._show_memory_health()
            elif choice == "6":
                await self._show_core_memory_details()
            elif choice == "7":
                await self._show_interaction_analysis()
    
    async def _browse_core_memories(self) -> None:
        """Browse all core memory files."""
        try:
            console.print("\n📚 [bold blue]Core Memories[/bold blue]")
            
            core_memories = self.agent.memory_manager.get_all_core_memories()
            
            if not core_memories:
                console.print(Panel(
                    Text("No core memories found.", style="yellow"),
                    title="📚 Core Memories",
                    border_style="yellow"
                ))
                return
            
            for memory_type, (frontmatter, content) in core_memories.items():
                # Create memory display
                memory_info = Table(show_header=False, box=None, padding=(0, 1))
                memory_info.add_column("Field", style="cyan", width=15)
                memory_info.add_column("Value", style="white")
                
                # Add metadata
                memory_info.add_row("Type", memory_type.replace('_', ' ').title())
                memory_info.add_row("Importance", str(frontmatter.get('importance_score', 'N/A')))
                memory_info.add_row("Created", frontmatter.get('created', 'N/A'))
                memory_info.add_row("Updated", frontmatter.get('updated', 'N/A'))
                memory_info.add_row("Size", f"{len(content)} characters")
                
                # Preview content
                preview = content[:300] + "..." if len(content) > 300 else content
                
                memory_display = Layout()
                memory_display.split_column(
                    Layout(Panel(memory_info, border_style="green"), size=6),
                    Layout(Panel(Text(preview), title="Content Preview", border_style="blue"), size=8)
                )
                
                console.print(memory_display)
                console.print()
                
        except Exception as e:
            console.print(f"[red]❌ Error browsing core memories: {e}[/red]")
    
    async def _search_memories_interactive(self) -> None:
        """Interactive memory search interface."""
        try:
            console.print("\n🔍 [bold blue]Memory Search[/bold blue]")
            
            # Get search query
            query = Prompt.ask("Enter search query")
            if not query.strip():
                return
            
            # Search options
            console.print("\n🔧 Search Options:")
            limit = Prompt.ask("Maximum results", default="10")
            try:
                limit = int(limit)
            except ValueError:
                limit = 10
            
            # Importance filter
            importance_filter = Prompt.ask(
                "Minimum importance (1-10, or 'none')",
                default="none"
            )
            importance_threshold = None
            if importance_filter.lower() != "none":
                try:
                    importance_threshold = int(importance_filter)
                except ValueError:
                    importance_threshold = None
            
            # Perform search
            with console.status("[bold blue]🔍 Searching memories...", spinner="dots"):
                results = self.agent.memory_manager.search_memories(
                    query=query,
                    limit=limit,
                    importance_threshold=importance_threshold
                )
            
            # Display results
            if not results:
                console.print(Panel(
                    Text(f"No memories found for query: '{query}'", style="yellow"),
                    title="🔍 Search Results",
                    border_style="yellow"
                ))
                return
            
            console.print(f"\n📊 Found {len(results)} results for '{query}':")
            
            for i, result in enumerate(results, 1):
                result_table = Table(show_header=False, box=None, padding=(0, 1))
                result_table.add_column("Field", style="cyan", width=15)
                result_table.add_column("Value", style="white")
                
                result_table.add_row("File", result.file_path)
                result_table.add_row("Category", result.category)
                result_table.add_row("Importance", str(result.importance_score))
                result_table.add_row("Relevance", f"{result.relevance_score:.2f}")
                result_table.add_row("Created", result.created)
                
                # Content snippet
                snippet_panel = Panel(
                    Text(result.content_snippet),
                    title="Content",
                    border_style="green"
                )
                
                result_layout = Layout()
                result_layout.split_column(
                    Layout(Panel(result_table, title=f"Result #{i}", border_style="blue"), size=6),
                    Layout(snippet_panel, size=4)
                )
                
                console.print(result_layout)
                console.print()
                
        except Exception as e:
            console.print(f"[red]❌ Error searching memories: {e}[/red]")
    
    async def _view_recent_interactions(self) -> None:
        """View recent interactions with filtering options."""
        try:
            console.print("\n💬 [bold blue]Recent Interactions[/bold blue]")
            
            # Get options
            days = Prompt.ask("Days to look back", default="7")
            try:
                days = int(days)
            except ValueError:
                days = 7
            
            limit = Prompt.ask("Maximum interactions", default="20")
            try:
                limit = int(limit)
            except ValueError:
                limit = 20
            
            # Get recent interactions
            with console.status("[bold blue]📚 Loading interactions...", spinner="dots"):
                interactions = self.agent.memory_manager.get_recent_interactions(
                    days=days, limit=limit
                )
            
            if not interactions:
                console.print(Panel(
                    Text(f"No interactions found in the last {days} days.", style="yellow"),
                    title="💬 Recent Interactions",
                    border_style="yellow"
                ))
                return
            
            console.print(f"\n📊 Found {len(interactions)} interactions from the last {days} days:")
            
            for i, interaction in enumerate(interactions, 1):
                interaction_table = Table(show_header=False, box=None, padding=(0, 1))
                interaction_table.add_column("Field", style="cyan", width=15)
                interaction_table.add_column("Value", style="white")
                
                interaction_table.add_row("Created", interaction.get('created', 'N/A'))
                interaction_table.add_row("Category", interaction.get('category', 'N/A'))
                interaction_table.add_row("Importance", str(interaction.get('importance_score', 'N/A')))
                interaction_table.add_row("File", interaction.get('file_path', 'N/A'))
                
                # Content preview
                content = interaction.get('content', '')
                preview = content[:400] + "..." if len(content) > 400 else content
                
                content_panel = Panel(
                    Text(preview),
                    title="Interaction Content",
                    border_style="green"
                )
                
                interaction_layout = Layout()
                interaction_layout.split_column(
                    Layout(Panel(interaction_table, title=f"Interaction #{i}", border_style="blue"), size=5),
                    Layout(content_panel, size=8)
                )
                
                console.print(interaction_layout)
                console.print()
                
        except Exception as e:
            console.print(f"[red]❌ Error viewing recent interactions: {e}[/red]")
    
    async def _show_memory_statistics(self) -> None:
        """Display comprehensive memory statistics."""
        try:
            console.print("\n📊 [bold blue]Memory Statistics[/bold blue]")
            
            with console.status("[bold blue]📊 Calculating statistics...", spinner="dots"):
                stats = self.agent.memory_manager.get_memory_statistics()
            
            # Main statistics table
            main_stats = Table(show_header=True, header_style="bold magenta")
            main_stats.add_column("Metric", style="cyan", width=25)
            main_stats.add_column("Value", style="white", width=15)
            main_stats.add_column("Details", style="dim")
            
            main_stats.add_row("Total Memories", str(stats.get('total_memories', 0)), "All memory files")
            main_stats.add_row("Core Memories", str(stats.get('core_memories', 0)), "User profile, context, etc.")
            main_stats.add_row("Interactions", str(stats.get('total_interactions', 0)), "Conversation logs")
            main_stats.add_row("Condensed Memories", str(stats.get('condensed_memories', 0)), "Summarized memories")
            
            # File statistics
            if 'file_stats' in stats:
                file_stats = stats['file_stats']
                main_stats.add_row("", "", "")  # Separator
                main_stats.add_row("Total Files", str(file_stats.get('total_files', 0)), "All memory files")
                main_stats.add_row("Total Size", f"{file_stats.get('total_size_mb', 0):.2f} MB", "Disk usage")
                main_stats.add_row("Avg File Size", f"{file_stats.get('average_file_size_kb', 0):.1f} KB", "Per file")
            
            console.print(Panel(main_stats, title="📊 Memory Statistics", border_style="green"))
            
            # Importance distribution
            if 'importance_distribution' in stats:
                importance_dist = stats['importance_distribution']
                
                importance_table = Table(show_header=True, header_style="bold magenta")
                importance_table.add_column("Importance Level", style="cyan")
                importance_table.add_column("Count", style="white")
                importance_table.add_column("Percentage", style="green")
                
                total = sum(importance_dist.values())
                for level, count in sorted(importance_dist.items()):
                    percentage = (count / total * 100) if total > 0 else 0
                    importance_table.add_row(f"Level {level}", str(count), f"{percentage:.1f}%")
                
                console.print(Panel(importance_table, title="📈 Importance Distribution", border_style="blue"))
            
            # Time-based statistics
            if 'time_distribution' in stats:
                time_dist = stats['time_distribution']
                
                time_table = Table(show_header=True, header_style="bold magenta")
                time_table.add_column("Time Period", style="cyan")
                time_table.add_column("Memories", style="white")
                time_table.add_column("Percentage", style="green")
                
                total = sum(time_dist.values())
                for period, count in time_dist.items():
                    percentage = (count / total * 100) if total > 0 else 0
                    time_table.add_row(period.title(), str(count), f"{percentage:.1f}%")
                
                console.print(Panel(time_table, title="⏰ Time Distribution", border_style="magenta"))
                
        except Exception as e:
            console.print(f"[red]❌ Error getting memory statistics: {e}[/red]")
    
    async def _show_memory_health(self) -> None:
        """Display memory system health check."""
        try:
            console.print("\n🏥 [bold blue]Memory Health Check[/bold blue]")
            
            with console.status("[bold blue]🏥 Running health check...", spinner="dots"):
                health = self.agent.memory_manager.validate_memory_system()
            
            # Overall health status
            overall_health = health.get('overall_health', False)
            health_color = "green" if overall_health else "red"
            health_status = "✅ Healthy" if overall_health else "❌ Issues Found"
            
            console.print(f"\n🏥 Overall Health: [{health_color}]{health_status}[/{health_color}]")
            
            # Detailed health information
            health_table = Table(show_header=True, header_style="bold magenta")
            health_table.add_column("Component", style="cyan", width=20)
            health_table.add_column("Status", style="white", width=15)
            health_table.add_column("Details", style="dim")
            
            # Directory structure
            structure_ok = health.get('directory_structure', {}).get('valid', False)
            health_table.add_row(
                "Directory Structure",
                "✅ Valid" if structure_ok else "❌ Invalid",
                "Memory directories exist"
            )
            
            # Core files
            core_files = health.get('core_files', {})
            missing_files = core_files.get('missing_files', [])
            health_table.add_row(
                "Core Files",
                "✅ Complete" if not missing_files else f"❌ Missing {len(missing_files)}",
                f"Essential memory files"
            )
            
            # File integrity
            file_integrity = health.get('file_integrity', {})
            corrupted_files = file_integrity.get('corrupted_files', [])
            health_table.add_row(
                "File Integrity",
                "✅ Good" if not corrupted_files else f"❌ {len(corrupted_files)} corrupted",
                "YAML frontmatter validation"
            )
            
            # Permissions
            permissions = health.get('permissions', {})
            permission_issues = permissions.get('issues', [])
            health_table.add_row(
                "Permissions",
                "✅ Good" if not permission_issues else f"❌ {len(permission_issues)} issues",
                "Read/write access"
            )
            
            console.print(Panel(health_table, title="🏥 Health Details", border_style=health_color))
            
            # Show issues if any
            if not overall_health:
                issues = []
                if missing_files:
                    issues.extend([f"Missing: {f}" for f in missing_files])
                if corrupted_files:
                    issues.extend([f"Corrupted: {f}" for f in corrupted_files])
                if permission_issues:
                    issues.extend([f"Permission: {f}" for f in permission_issues])
                
                if issues:
                    console.print("\n⚠️  [bold yellow]Issues Found:[/bold yellow]")
                    for issue in issues[:10]:  # Show first 10 issues
                        console.print(f"  • {issue}")
                    if len(issues) > 10:
                        console.print(f"  ... and {len(issues) - 10} more")
                        
        except Exception as e:
            console.print(f"[red]❌ Error checking memory health: {e}[/red]")
    
    async def _show_core_memory_details(self) -> None:
        """Show detailed view of a specific core memory file."""
        try:
            console.print("\n📋 [bold blue]Core Memory Details[/bold blue]")
            
            # Get available core memory types
            core_memories = self.agent.memory_manager.get_all_core_memories()
            
            if not core_memories:
                console.print(Panel(
                    Text("No core memories available.", style="yellow"),
                    title="📋 Core Memory Details",
                    border_style="yellow"
                ))
                return
            
            # Let user choose which core memory to view
            memory_choices = list(core_memories.keys())
            memory_table = Table(show_header=True, header_style="bold magenta", box=None)
            memory_table.add_column("Option", style="cyan", width=5)
            memory_table.add_column("Memory Type", style="white")
            memory_table.add_column("Size", style="green")
            
            for i, memory_type in enumerate(memory_choices, 1):
                _, content = core_memories[memory_type]
                size = f"{len(content)} chars"
                display_name = memory_type.replace('_', ' ').title()
                memory_table.add_row(str(i), display_name, size)
            
            console.print(Panel(memory_table, title="📋 Available Core Memories", border_style="blue"))
            
            choice = Prompt.ask(
                "Select memory to view",
                choices=[str(i) for i in range(1, len(memory_choices) + 1)] + ["q"],
                default="q"
            )
            
            if choice == "q":
                return
            
            # Show selected memory in detail
            selected_type = memory_choices[int(choice) - 1]
            frontmatter, content = core_memories[selected_type]
            
            # Metadata table
            metadata_table = Table(show_header=False, box=None, padding=(0, 1))
            metadata_table.add_column("Field", style="cyan", width=15)
            metadata_table.add_column("Value", style="white")
            
            metadata_table.add_row("Type", selected_type.replace('_', ' ').title())
            metadata_table.add_row("Importance", str(frontmatter.get('importance_score', 'N/A')))
            metadata_table.add_row("Created", frontmatter.get('created', 'N/A'))
            metadata_table.add_row("Updated", frontmatter.get('updated', 'N/A'))
            metadata_table.add_row("Category", frontmatter.get('category', 'N/A'))
            metadata_table.add_row("Size", f"{len(content)} characters")
            
            # Tags if available
            tags = frontmatter.get('tags', [])
            if tags:
                metadata_table.add_row("Tags", ", ".join(tags))
            
            # Content display
            content_panel = Panel(
                Text(content),
                title="Memory Content",
                border_style="green"
            )
            
            detail_layout = Layout()
            detail_layout.split_column(
                Layout(Panel(metadata_table, title=f"📋 {selected_type.replace('_', ' ').title()}", border_style="blue"), size=8),
                Layout(content_panel)
            )
            
            console.print(detail_layout)
            
        except Exception as e:
            console.print(f"[red]❌ Error viewing core memory details: {e}[/red]")
    
    async def _show_interaction_analysis(self) -> None:
        """Show analysis of interaction patterns and trends."""
        try:
            console.print("\n📈 [bold blue]Interaction Analysis[/bold blue]")
            
            with console.status("[bold blue]📈 Analyzing interactions...", spinner="dots"):
                # Get recent interactions for analysis
                interactions = self.agent.memory_manager.get_recent_interactions(days=30, limit=100)
                stats = self.agent.memory_manager.get_memory_statistics()
            
            if not interactions:
                console.print(Panel(
                    Text("No interactions found for analysis.", style="yellow"),
                    title="📈 Interaction Analysis",
                    border_style="yellow"
                ))
                return
            
            # Basic interaction statistics
            analysis_table = Table(show_header=True, header_style="bold magenta")
            analysis_table.add_column("Metric", style="cyan", width=25)
            analysis_table.add_column("Value", style="white", width=15)
            analysis_table.add_column("Details", style="dim")
            
            analysis_table.add_row("Total Interactions", str(len(interactions)), "Last 30 days")
            
            # Average interaction length
            total_chars = sum(len(i.get('content', '')) for i in interactions)
            avg_length = total_chars / len(interactions) if interactions else 0
            analysis_table.add_row("Avg. Length", f"{avg_length:.0f} chars", "Per interaction")
            
            # Importance distribution in recent interactions
            importance_counts = {}
            for interaction in interactions:
                importance = interaction.get('importance_score', 5)
                importance_counts[importance] = importance_counts.get(importance, 0) + 1
            
            if importance_counts:
                avg_importance = sum(k * v for k, v in importance_counts.items()) / len(interactions)
                analysis_table.add_row("Avg. Importance", f"{avg_importance:.1f}", "Recent interactions")
            
            # Most active days (if we have date info)
            date_counts = {}
            for interaction in interactions:
                created = interaction.get('created', '')
                if created:
                    # Extract date part
                    date_part = created.split('T')[0] if 'T' in created else created[:10]
                    date_counts[date_part] = date_counts.get(date_part, 0) + 1
            
            if date_counts:
                most_active_date = max(date_counts.items(), key=lambda x: x[1])
                analysis_table.add_row("Most Active Day", most_active_date[0], f"{most_active_date[1]} interactions")
            
            console.print(Panel(analysis_table, title="📈 Interaction Analysis", border_style="green"))
            
            # Show recent interaction pattern
            if date_counts:
                pattern_table = Table(show_header=True, header_style="bold magenta")
                pattern_table.add_column("Date", style="cyan")
                pattern_table.add_column("Interactions", style="white")
                pattern_table.add_column("Activity", style="green")
                
                # Show last 7 days with activity
                sorted_dates = sorted(date_counts.items(), key=lambda x: x[0], reverse=True)[:7]
                max_count = max(date_counts.values()) if date_counts else 1
                
                for date, count in sorted_dates:
                    activity_bar = "█" * min(int(count / max_count * 20), 20)
                    pattern_table.add_row(date, str(count), activity_bar)
                
                console.print(Panel(pattern_table, title="📅 Recent Activity Pattern", border_style="blue"))
                
        except Exception as e:
            console.print(f"[red]❌ Error analyzing interactions: {e}[/red]")
    
    def _show_session_statistics(self) -> None:
        """Display current session statistics."""
        session_duration = datetime.now() - self.session_start_time
        hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Metric", style="cyan", width=20)
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("🕐 Session Duration", f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        stats_table.add_row("💬 Messages Sent", str(self.total_messages))
        stats_table.add_row("🔄 Conversation Turns", str(len(self.conversation_history.current_session_turns)))
        stats_table.add_row("⏰ Started At", self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        if self.total_messages > 0:
            avg_time = session_duration.total_seconds() / self.total_messages
            stats_table.add_row("⚡ Avg. Time/Message", f"{avg_time:.1f}s")
        
        console.print(Panel(stats_table, title="📈 Session Statistics", border_style="magenta"))
    
    def _clear_session_history(self) -> None:
        """Clear current session conversation history."""
        self.conversation_history.current_session_turns.clear()
        console.print("[green]✅ Session conversation history cleared[/green]")
    
    def _toggle_debug_mode(self) -> None:
        """Toggle debug mode on/off."""
        self.debug_mode = not self.debug_mode
        status = "enabled" if self.debug_mode else "disabled"
        emoji = "🐞" if self.debug_mode else "🚫"
        console.print(f"[{'green' if self.debug_mode else 'yellow'}]{emoji} Debug mode {status}[/{'green' if self.debug_mode else 'yellow'}]")
        
        if self.debug_mode:
            console.print("[dim]Debug mode will show detailed error information and verbose logging.[/dim]")
        else:
            console.print("[dim]Debug mode disabled. Errors will show basic information only.[/dim]")
    
    def _show_error_summary(self) -> None:
        """Display comprehensive error summary and statistics."""
        error_summary = self.error_handler.get_error_summary()
        
        # Create error summary table
        error_table = Table(show_header=False, box=None, padding=(0, 2))
        error_table.add_column("Metric", style="cyan", width=20)
        error_table.add_column("Value", style="white")
        
        error_table.add_row("⚠️  Total Errors", str(error_summary['total_errors']))
        error_table.add_row("🔄 Recovery Attempted", "Yes" if error_summary['recovery_attempted'] else "No")
        
        if error_summary['last_error_time']:
            last_error = datetime.fromisoformat(error_summary['last_error_time'])
            error_table.add_row("🕒 Last Error", last_error.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            error_table.add_row("🕒 Last Error", "None")
        
        # Session health
        session_duration = datetime.now() - self.session_start_time
        error_rate = error_summary['total_errors'] / max(self.total_messages, 1)
        
        error_table.add_row("", "")  # Separator
        error_table.add_row("📊 Error Rate", f"{error_rate:.2%}")
        error_table.add_row("⏱️  Session Time", str(session_duration).split('.')[0])
        error_table.add_row("💬 Total Messages", str(self.total_messages))
        
        # Health assessment
        if error_summary['total_errors'] == 0:
            health_status = "🟢 Excellent"
        elif error_rate < 0.1:
            health_status = "🟡 Good"
        elif error_rate < 0.3:
            health_status = "🟠 Fair"
        else:
            health_status = "🔴 Poor"
        
        error_table.add_row("🏥 System Health", health_status)
        
        console.print(Panel(error_table, title="📋 Error Summary", border_style="yellow"))
        
        # Show recommendations if there are errors
        if error_summary['total_errors'] > 0:
            recommendations = []
            
            if error_rate > 0.2:
                recommendations.append("Consider restarting the CLI (/reset)")
            if error_summary['total_errors'] > 5:
                recommendations.append("Check system resources and network connectivity")
            if not error_summary['recovery_attempted']:
                recommendations.append("Try the recovery command (/recover)")
            
            if recommendations:
                rec_text = "\n".join(f"• {rec}" for rec in recommendations)
                console.print(Panel(
                    Text(rec_text, style="yellow"),
                    title="💡 Recommendations",
                    border_style="blue"
                ))
    
    async def _attempt_system_recovery(self) -> None:
        """Attempt to recover from system errors and restore functionality."""
        console.print("[yellow]🔄 Attempting system recovery...[/yellow]")
        
        recovery_steps = [
            ("Checking agent status", self._recovery_check_agent),
            ("Validating memory system", self._recovery_check_memory),
            ("Testing MCP connections", self._recovery_check_mcp),
            ("Clearing error state", self._recovery_clear_errors)
        ]
        
        success_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            for step_name, step_func in recovery_steps:
                task = progress.add_task(f"🔧 {step_name}...", total=None)
                
                try:
                    result = await step_func()
                    if result:
                        progress.update(task, description=f"✅ {step_name} - OK")
                        success_count += 1
                    else:
                        progress.update(task, description=f"⚠️  {step_name} - Issues detected")
                    
                    await asyncio.sleep(0.5)  # Visual delay
                    
                except Exception as e:
                    progress.update(task, description=f"❌ {step_name} - Failed: {e}")
                    
                    if self.debug_mode:
                        console.print(f"[dim]Recovery step error: {e}[/dim]")
        
        # Recovery summary
        if success_count == len(recovery_steps):
            console.print("[green]✅ System recovery completed successfully![/green]")
            console.print("[green]All components are functioning normally.[/green]")
        elif success_count >= len(recovery_steps) // 2:
            console.print("[yellow]⚠️  Partial recovery completed.[/yellow]")
            console.print(f"[yellow]{success_count}/{len(recovery_steps)} checks passed.[/yellow]")
            console.print("[yellow]Some issues may persist. Consider a full reset (/reset).[/yellow]")
        else:
            console.print("[red]❌ Recovery failed.[/red]")
            console.print("[red]Multiple system components have issues.[/red]")
            console.print("[red]A full restart may be required.[/red]")
    
    async def _recovery_check_agent(self) -> bool:
        """Check agent status during recovery."""
        try:
            if not self.agent:
                return False
            
            status = await self.agent.get_agent_status()
            return status.get('initialized', False)
        except Exception:
            return False
    
    async def _recovery_check_memory(self) -> bool:
        """Check memory system during recovery."""
        try:
            if not self.agent or not hasattr(self.agent, 'memory_manager'):
                return False
            
            # Try a basic memory operation
            self.agent.memory_manager.get_all_core_memories()
            return True
        except Exception:
            return False
    
    async def _recovery_check_mcp(self) -> bool:
        """Check MCP connections during recovery."""
        try:
            if not self.agent:
                return False
            
            status = await self.agent.get_agent_status()
            return status.get('mcp_enabled', False)
        except Exception:
            return False
    
    async def _recovery_clear_errors(self) -> bool:
        """Clear error state during recovery."""
        try:
            # Reset error handler state
            self.error_handler.error_count = 0
            self.error_handler.last_error_time = None
            self.error_handler.recovery_attempted = False
            return True
        except Exception:
            return False
    
    async def _reset_agent(self) -> None:
        """Reset and reinitialize the agent system."""
        console.print("[yellow]🔄 Resetting agent system...[/yellow]")
        
        # Clear current state
        self.agent = None
        self.error_handler = ErrorHandler(console)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            reset_task = progress.add_task("🔄 Resetting system...", total=None)
            await asyncio.sleep(1)
            
            # Attempt reinitialization
            progress.update(reset_task, description="🚀 Reinitializing agent...")
            success = await self.initialize_agent()
            
            if success:
                progress.update(reset_task, description="✅ Agent reset successfully!")
                await asyncio.sleep(1)
                console.print("[green]✅ Agent system reset and reinitialized successfully![/green]")
                console.print("[green]You can now continue with normal operations.[/green]")
            else:
                progress.update(reset_task, description="❌ Reset failed")
                await asyncio.sleep(1)
                console.print("[red]❌ Agent reset failed.[/red]")
                console.print("[red]Please restart the CLI manually.[/red]")
    
    async def _show_session_browser(self) -> None:
        """Display comprehensive session browser interface."""
        if not self.session_manager:
            console.print(Panel(
                Text("❌ Session management not available", style="red"),
                title="🗂️  Session Browser",
                border_style="red"
            ))
            return
        
        while True:
            # Display session browser menu
            console.print()
            menu_table = Table(show_header=True, header_style="bold magenta", box=None)
            menu_table.add_column("Option", style="cyan", width=8)
            menu_table.add_column("Description", style="white")
            
            menu_table.add_row("1", "View recent sessions")
            menu_table.add_row("2", "Search sessions")
            menu_table.add_row("3", "Browse all sessions")
            menu_table.add_row("4", "View bookmarked sessions")
            menu_table.add_row("5", "Session statistics")
            menu_table.add_row("6", "Load session")
            menu_table.add_row("7", "Delete session")
            menu_table.add_row("8", "Manage current session")
            menu_table.add_row("q", "Return to main chat")
            
            console.print(Panel(menu_table, title="🗂️  Session Browser", border_style="blue"))
            
            choice = Prompt.ask(
                "[bold cyan]Select option",
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "q"],
                default="q"
            )
            
            if choice == "q":
                break
            elif choice == "1":
                await self._show_recent_sessions()
            elif choice == "2":
                await self._search_sessions_interactive()
            elif choice == "3":
                await self._browse_all_sessions()
            elif choice == "4":
                await self._show_bookmarked_sessions()
            elif choice == "5":
                await self._show_session_statistics()
            elif choice == "6":
                await self._load_session_interactive()
            elif choice == "7":
                await self._delete_session_interactive()
            elif choice == "8":
                await self._manage_current_session()
    
    async def _show_recent_sessions(self) -> None:
        """Show recent sessions."""
        recent_sessions = self.session_manager.get_recent_sessions(limit=15)
        
        if not recent_sessions:
            console.print(Panel(
                Text("No sessions found.", style="yellow"),
                title="📅 Recent Sessions",
                border_style="yellow"
            ))
            return
        
        sessions_table = Table(show_header=True, header_style="bold magenta")
        sessions_table.add_column("ID", style="cyan", width=10)
        sessions_table.add_column("Title", style="white", width=30)
        sessions_table.add_column("Last Access", style="dim", width=12)
        sessions_table.add_column("Turns", style="green", width=8)
        sessions_table.add_column("Duration", style="blue", width=10)
        sessions_table.add_column("Tags", style="yellow", width=15)
        
        for session in recent_sessions:
            last_access = datetime.fromisoformat(session['last_accessed'])
            time_ago = self._format_time_ago(last_access)
            
            # Truncate title if too long
            title = session['title'][:27] + "..." if len(session['title']) > 30 else session['title']
            
            # Format tags
            tags_str = ", ".join(session.get('tags', [])[:2])
            if len(session.get('tags', [])) > 2:
                tags_str += "..."
            
            bookmark_indicator = "📌 " if session.get('bookmarked', False) else ""
            
            sessions_table.add_row(
                session['session_id'][:8],
                f"{bookmark_indicator}{title}",
                time_ago,
                str(session['total_turns']),
                f"{session['duration_minutes']}m",
                tags_str
            )
        
        console.print(Panel(sessions_table, title="📅 Recent Sessions", border_style="green"))
    
    async def _search_sessions_interactive(self) -> None:
        """Interactive session search."""
        console.print("\n🔍 [bold blue]Session Search[/bold blue]")
        
        query = Prompt.ask("Enter search query")
        if not query.strip():
            return
        
        matches = self.session_manager.search_sessions(query, limit=20)
        
        if not matches:
            console.print(Panel(
                Text(f"No sessions found for '{query}'", style="yellow"),
                title="🔍 Search Results",
                border_style="yellow"
            ))
            return
        
        console.print(f"\n📋 Found {len(matches)} sessions:")
        
        sessions_table = Table(show_header=True, header_style="bold magenta")
        sessions_table.add_column("ID", style="cyan", width=10)
        sessions_table.add_column("Title", style="white", width=35)
        sessions_table.add_column("Turns", style="green", width=8)
        sessions_table.add_column("Tags", style="yellow")
        
        for session in matches:
            title = session['title'][:32] + "..." if len(session['title']) > 35 else session['title']
            tags_str = ", ".join(session.get('tags', []))
            bookmark_indicator = "📌 " if session.get('bookmarked', False) else ""
            
            sessions_table.add_row(
                session['session_id'][:8],
                f"{bookmark_indicator}{title}",
                str(session['total_turns']),
                tags_str
            )
        
        console.print(Panel(sessions_table, title=f"🔍 Search Results for '{query}'", border_style="green"))
    
    async def _browse_all_sessions(self) -> None:
        """Browse all sessions with pagination."""
        all_sessions = list(self.session_manager._session_index.values())
        all_sessions.sort(key=lambda x: x['last_accessed'], reverse=True)
        
        if not all_sessions:
            console.print(Panel(
                Text("No sessions found.", style="yellow"),
                title="📚 All Sessions",
                border_style="yellow"
            ))
            return
        
        # Simple pagination
        page_size = 10
        total_pages = (len(all_sessions) + page_size - 1) // page_size
        
        for page in range(total_pages):
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, len(all_sessions))
            page_sessions = all_sessions[start_idx:end_idx]
            
            sessions_table = Table(show_header=True, header_style="bold magenta")
            sessions_table.add_column("ID", style="cyan", width=10)
            sessions_table.add_column("Title", style="white", width=30)
            sessions_table.add_column("Created", style="dim", width=12)
            sessions_table.add_column("Turns", style="green", width=8)
            sessions_table.add_column("Duration", style="blue", width=10)
            
            for session in page_sessions:
                created = datetime.fromisoformat(session['created_at'])
                created_str = created.strftime("%m/%d %H:%M")
                title = session['title'][:27] + "..." if len(session['title']) > 30 else session['title']
                bookmark_indicator = "📌 " if session.get('bookmarked', False) else ""
                
                sessions_table.add_row(
                    session['session_id'][:8],
                    f"{bookmark_indicator}{title}",
                    created_str,
                    str(session['total_turns']),
                    f"{session['duration_minutes']}m"
                )
            
            console.print(Panel(
                sessions_table, 
                title=f"📚 All Sessions (Page {page + 1}/{total_pages})",
                border_style="green"
            ))
            
            # Ask to continue to next page
            if page < total_pages - 1:
                if not Confirm.ask("Continue to next page?", default=True):
                    break
    
    async def _show_bookmarked_sessions(self) -> None:
        """Show bookmarked sessions."""
        bookmarked = self.session_manager.get_bookmarked_sessions()
        
        if not bookmarked:
            console.print(Panel(
                Text("No bookmarked sessions found.", style="yellow"),
                title="📌 Bookmarked Sessions",
                border_style="yellow"
            ))
            return
        
        sessions_table = Table(show_header=True, header_style="bold magenta")
        sessions_table.add_column("ID", style="cyan", width=10)
        sessions_table.add_column("Title", style="white", width=35)
        sessions_table.add_column("Created", style="dim", width=12)
        sessions_table.add_column("Turns", style="green", width=8)
        sessions_table.add_column("Tags", style="yellow")
        
        for session in bookmarked:
            created = datetime.fromisoformat(session['created_at'])
            created_str = created.strftime("%m/%d %H:%M")
            title = session['title'][:32] + "..." if len(session['title']) > 35 else session['title']
            tags_str = ", ".join(session.get('tags', []))
            
            sessions_table.add_row(
                session['session_id'][:8],
                f"📌 {title}",
                created_str,
                str(session['total_turns']),
                tags_str
            )
        
        console.print(Panel(sessions_table, title="📌 Bookmarked Sessions", border_style="green"))
    
    async def _show_session_statistics(self) -> None:
        """Show session statistics."""
        stats = self.session_manager.get_session_statistics()
        
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Metric", style="cyan", width=20)
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("📊 Total Sessions", str(stats['total_sessions']))
        stats_table.add_row("💬 Total Conversations", str(stats['total_conversations']))
        stats_table.add_row("📝 Total Messages", str(stats['total_messages']))
        stats_table.add_row("📈 Avg Session Length", f"{stats['avg_session_length']} turns")
        stats_table.add_row("📌 Bookmarked Sessions", str(stats['bookmarked_sessions']))
        stats_table.add_row("🏷️  Sessions with Tags", str(stats['sessions_with_tags']))
        
        # Current session info
        if self.session_manager.current_session:
            current = self.session_manager.current_session
            stats_table.add_row("", "")  # Separator
            stats_table.add_row("🔄 Current Session", current.session_id[:8])
            stats_table.add_row("📋 Current Title", current.title or "Untitled")
            stats_table.add_row("💭 Current Turns", str(len(current.conversation_turns)))
        
        console.print(Panel(stats_table, title="📊 Session Statistics", border_style="magenta"))
    
    async def _load_session_interactive(self) -> None:
        """Interactive session loading."""
        # Show recent sessions first
        recent_sessions = self.session_manager.get_recent_sessions(limit=10)
        
        if not recent_sessions:
            console.print(Panel(
                Text("No sessions available to load.", style="yellow"),
                title="📂 Load Session",
                border_style="yellow"
            ))
            return
        
        console.print("\n📂 [bold blue]Recent Sessions:[/bold blue]")
        
        # Display sessions with numbers
        for i, session in enumerate(recent_sessions, 1):
            title = session['title'] or session['session_id'][:8]
            last_access = datetime.fromisoformat(session['last_accessed'])
            time_ago = self._format_time_ago(last_access)
            bookmark = "📌 " if session.get('bookmarked', False) else ""
            
            console.print(f"{i:2}. {bookmark}{title} ({session['total_turns']} turns, {time_ago})")
        
        console.print(f"{len(recent_sessions) + 1:2}. Enter session ID manually")
        console.print(f"{len(recent_sessions) + 2:2}. Cancel")
        
        choice = Prompt.ask(
            "Select session to load",
            choices=[str(i) for i in range(1, len(recent_sessions) + 3)],
            default=str(len(recent_sessions) + 2)
        )
        
        choice_int = int(choice)
        
        if choice_int == len(recent_sessions) + 2:  # Cancel
            return
        elif choice_int == len(recent_sessions) + 1:  # Manual ID entry
            session_id = Prompt.ask("Enter session ID")
            if not session_id:
                return
        else:  # Select from list
            session_id = recent_sessions[choice_int - 1]['session_id']
        
        # Confirm loading if we have a current session
        if self.session_manager.current_session and len(self.session_manager.current_session.conversation_turns) > 0:
            if not Confirm.ask("This will save and replace your current session. Continue?", default=False):
                return
            
            # Save current session
            await self.session_manager.save_session(force=True)
        
        # Load the selected session
        success = await self.session_manager.restore_session(session_id)
        if success:
            # Restore conversation history to CLI
            self._restore_session_to_cli()
            console.print(f"[green]✅ Session loaded successfully![/green]")
        else:
            console.print(f"[red]❌ Failed to load session {session_id[:8]}[/red]")
    
    async def _delete_session_interactive(self) -> None:
        """Interactive session deletion."""
        # Show recent sessions
        recent_sessions = self.session_manager.get_recent_sessions(limit=15)
        
        if not recent_sessions:
            console.print(Panel(
                Text("No sessions available to delete.", style="yellow"),
                title="🗑️  Delete Session",
                border_style="yellow"
            ))
            return
        
        console.print("\n🗑️  [bold red]Delete Session:[/bold red]")
        console.print("[yellow]⚠️  This action cannot be undone![/yellow]")
        
        # Display sessions
        for i, session in enumerate(recent_sessions, 1):
            title = session['title'] or session['session_id'][:8]
            bookmark = "📌 " if session.get('bookmarked', False) else ""
            console.print(f"{i:2}. {bookmark}{title} ({session['total_turns']} turns)")
        
        console.print(f"{len(recent_sessions) + 1:2}. Cancel")
        
        choice = Prompt.ask(
            "Select session to delete",
            choices=[str(i) for i in range(1, len(recent_sessions) + 2)],
            default=str(len(recent_sessions) + 1)
        )
        
        choice_int = int(choice)
        
        if choice_int == len(recent_sessions) + 1:  # Cancel
            return
        
        session_to_delete = recent_sessions[choice_int - 1]
        
        # Double confirmation for bookmarked sessions
        if session_to_delete.get('bookmarked', False):
            console.print("[yellow]⚠️  This is a bookmarked session![/yellow]")
            if not Confirm.ask("Are you sure you want to delete this bookmarked session?", default=False):
                return
        
        if Confirm.ask(f"Delete session '{session_to_delete['title']}'?", default=False):
            success = await self.session_manager.delete_session(session_to_delete['session_id'])
            if success:
                console.print("[green]✅ Session deleted successfully[/green]")
            else:
                console.print("[red]❌ Failed to delete session[/red]")
    
    async def _manage_current_session(self) -> None:
        """Manage the current session."""
        if not self.session_manager.current_session:
            console.print(Panel(
                Text("No active session to manage.", style="yellow"),
                title="⚙️  Manage Current Session",
                border_style="yellow"
            ))
            return
        
        current = self.session_manager.current_session
        
        while True:
            # Display current session info
            info_table = Table(show_header=False, box=None, padding=(0, 2))
            info_table.add_column("Property", style="cyan", width=15)
            info_table.add_column("Value", style="white")
            
            info_table.add_row("Session ID", current.session_id[:8])
            info_table.add_row("Title", current.title or "Untitled")
            info_table.add_row("Description", current.description or "No description")
            info_table.add_row("Turns", str(len(current.conversation_turns)))
            info_table.add_row("Tags", ", ".join(current.tags) if current.tags else "No tags")
            info_table.add_row("Bookmarked", "Yes" if current.bookmarked else "No")
            info_table.add_row("Auto-save", "Yes" if current.auto_save else "No")
            
            console.print(Panel(info_table, title="⚙️  Current Session", border_style="blue"))
            
            # Management options
            console.print("\n[bold cyan]Management Options:[/bold cyan]")
            console.print("1. Edit title")
            console.print("2. Edit description")
            console.print("3. Add tag")
            console.print("4. Remove tag")
            console.print("5. Toggle bookmark")
            console.print("6. Toggle auto-save")
            console.print("7. Save session now")
            console.print("8. Back to session browser")
            
            choice = Prompt.ask(
                "Select option",
                choices=["1", "2", "3", "4", "5", "6", "7", "8"],
                default="8"
            )
            
            if choice == "1":
                new_title = Prompt.ask("Enter new title", default=current.title)
                current.title = new_title
                console.print("[green]✅ Title updated[/green]")
                
            elif choice == "2":
                new_description = Prompt.ask("Enter description", default=current.description)
                current.description = new_description
                console.print("[green]✅ Description updated[/green]")
                
            elif choice == "3":
                new_tag = Prompt.ask("Enter tag to add")
                if new_tag:
                    current.add_tag(new_tag)
                    console.print(f"[green]✅ Tag '{new_tag}' added[/green]")
                    
            elif choice == "4":
                if current.tags:
                    console.print("Current tags:", ", ".join(current.tags))
                    tag_to_remove = Prompt.ask("Enter tag to remove")
                    if tag_to_remove in current.tags:
                        current.remove_tag(tag_to_remove)
                        console.print(f"[green]✅ Tag '{tag_to_remove}' removed[/green]")
                    else:
                        console.print("[yellow]Tag not found[/yellow]")
                else:
                    console.print("[yellow]No tags to remove[/yellow]")
                    
            elif choice == "5":
                current.bookmarked = not current.bookmarked
                status = "bookmarked" if current.bookmarked else "unbookmarked"
                console.print(f"[green]✅ Session {status}[/green]")
                
            elif choice == "6":
                current.auto_save = not current.auto_save
                status = "enabled" if current.auto_save else "disabled"
                console.print(f"[green]✅ Auto-save {status}[/green]")
                
            elif choice == "7":
                success = await self.session_manager.save_session(current, force=True)
                if success:
                    console.print("[green]✅ Session saved[/green]")
                else:
                    console.print("[red]❌ Failed to save session[/red]")
                    
            elif choice == "8":
                break
    
    async def _save_current_session(self) -> None:
        """Save the current session."""
        if not self.session_manager or not self.session_manager.current_session:
            console.print("[yellow]⚠️  No active session to save[/yellow]")
            return
        
        current = self.session_manager.current_session
        
        # Update session duration
        current.update_duration(self.session_start_time)
        
        success = await self.session_manager.save_session(current, force=True)
        if success:
            console.print(f"[green]✅ Session saved: {current.title or current.session_id[:8]}[/green]")
        else:
            console.print("[red]❌ Failed to save session[/red]")
    
    async def _create_new_session(self) -> None:
        """Create a new session."""
        if not self.session_manager:
            console.print("[red]❌ Session manager not available[/red]")
            return
        
        # Save current session if it has content
        if (self.session_manager.current_session and 
            len(self.session_manager.current_session.conversation_turns) > 0):
            
            if Confirm.ask("Save current session before creating new one?", default=True):
                await self.session_manager.save_session(force=True)
        
        # Get optional title
        title = Prompt.ask("Enter session title (optional)", default="")
        
        # Create new session
        new_session = self.session_manager.create_new_session(title if title else None)
        
        # Clear current conversation history
        self.conversation_history.current_session_turns.clear()
        self.total_messages = 0
        self.session_start_time = datetime.now()
        
        console.print(f"[green]🆕 New session created: {new_session.title or new_session.session_id[:8]}[/green]")
    
    def _restore_session_to_cli(self) -> None:
        """Restore a session's conversation history to the CLI."""
        if not self.session_manager or not self.session_manager.current_session:
            return
        
        current = self.session_manager.current_session
        
        # Clear current history
        self.conversation_history.current_session_turns.clear()
        
        # Restore from session
        for turn in current.conversation_turns:
            self.conversation_history.add_turn(
                turn['user_message'],
                turn['assistant_response'],
                turn['timestamp']
            )
        
        # Update stats
        self.total_messages = current.total_messages
        self.session_start_time = current.created_at
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as 'time ago' string."""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}m ago"
        else:
            return "just now"
    
    async def _show_configuration_menu(self) -> None:
        """Show the main configuration menu."""
        if not self.cli_config:
            console.print(Panel(
                Text("❌ CLI configuration not available", style="red"),
                title="⚙️  Configuration",
                border_style="red"
            ))
            return
        
        while True:
            # Get current config summary
            summary = self.cli_config.get_config_summary()
            
            # Display current settings
            current_table = Table(show_header=False, box=None, padding=(0, 2))
            current_table.add_column("Setting", style="cyan", width=20)
            current_table.add_column("Value", style="white")
            
            current_table.add_row("🎨 Theme", summary['theme'])
            current_table.add_row("💬 Response Style", summary['response_style'])
            current_table.add_row("📺 Display Mode", summary['display_mode'])
            current_table.add_row("🌊 Streaming", "Enabled" if summary['streaming_enabled'] else "Disabled")
            current_table.add_row("💾 Auto-save", f"{summary['auto_save_frequency']}s")
            current_table.add_row("⚡ Aliases", str(summary['aliases_count']))
            current_table.add_row("🚀 Quick Commands", str(summary['quick_commands_count']))
            
            console.print(Panel(current_table, title="⚙️  Current Configuration", border_style="blue"))
            
            # Configuration menu
            console.print("\n[bold cyan]Configuration Options:[/bold cyan]")
            console.print("1. Change theme")
            console.print("2. Set response style")
            console.print("3. Set display mode")
            console.print("4. Configure display settings")
            console.print("5. Configure behavior settings")
            console.print("6. Manage aliases")
            console.print("7. Manage quick commands")
            console.print("8. Export configuration")
            console.print("9. Import configuration")
            console.print("10. Reset to defaults")
            console.print("11. Back to main chat")
            
            choice = Prompt.ask(
                "Select option",
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
                default="11"
            )
            
            if choice == "1":
                await self._show_theme_selector()
            elif choice == "2":
                await self._configure_response_style()
            elif choice == "3":
                await self._configure_display_mode()
            elif choice == "4":
                await self._configure_display_settings()
            elif choice == "5":
                await self._configure_behavior_settings()
            elif choice == "6":
                await self._show_aliases_menu()
            elif choice == "7":
                await self._show_quick_commands()
            elif choice == "8":
                await self._export_configuration()
            elif choice == "9":
                await self._import_configuration()
            elif choice == "10":
                if Confirm.ask("⚠️  Reset all settings to defaults?", default=False):
                    await self._reset_configuration()
            elif choice == "11":
                break
    
    async def _show_theme_selector(self) -> None:
        """Show theme selection interface."""
        if not self.cli_config:
            console.print("[red]❌ CLI configuration not available[/red]")
            return
        
        themes = self.cli_config.get_available_themes()
        current_theme = self.cli_config.config.ui_theme.name
        
        console.print("\n🎨 [bold blue]Available Themes:[/bold blue]")
        
        for i, theme_name in enumerate(themes, 1):
            indicator = "→ " if theme_name == current_theme else "  "
            console.print(f"{indicator}{i}. {theme_name.title()}")
        
        console.print(f"  {len(themes) + 1}. Cancel")
        
        choice = Prompt.ask(
            "Select theme",
            choices=[str(i) for i in range(1, len(themes) + 2)],
            default=str(len(themes) + 1)
        )
        
        choice_int = int(choice)
        
        if choice_int <= len(themes):
            theme_name = themes[choice_int - 1]
            success = await self.cli_config.set_theme(theme_name)
            if success:
                console.print(f"[green]✅ Theme changed to '{theme_name}'[/green]")
                # Apply theme to current console (would need Rich theme integration)
                console.print("[dim]💡 Theme will take full effect on next restart[/dim]")
            else:
                console.print(f"[red]❌ Failed to set theme '{theme_name}'[/red]")
    
    async def _configure_response_style(self) -> None:
        """Configure response style."""
        if not self.cli_config:
            return
        
        styles = list(ResponseStyle)
        current_style = self.cli_config.config.behavior.response_style
        
        console.print("\n💬 [bold blue]Response Styles:[/bold blue]")
        
        for i, style in enumerate(styles, 1):
            indicator = "→ " if style == current_style else "  "
            console.print(f"{indicator}{i}. {style.value.title()}")
        
        console.print(f"  {len(styles) + 1}. Cancel")
        
        choice = Prompt.ask(
            "Select response style",
            choices=[str(i) for i in range(1, len(styles) + 2)],
            default=str(len(styles) + 1)
        )
        
        choice_int = int(choice)
        
        if choice_int <= len(styles):
            selected_style = styles[choice_int - 1]
            await self.cli_config.set_response_style(selected_style)
            console.print(f"[green]✅ Response style set to '{selected_style.value}'[/green]")
    
    async def _configure_display_mode(self) -> None:
        """Configure display mode."""
        if not self.cli_config:
            return
        
        modes = list(DisplayMode)
        current_mode = self.cli_config.config.display.mode
        
        console.print("\n📺 [bold blue]Display Modes:[/bold blue]")
        
        for i, mode in enumerate(modes, 1):
            indicator = "→ " if mode == current_mode else "  "
            console.print(f"{indicator}{i}. {mode.value.title()}")
        
        console.print(f"  {len(modes) + 1}. Cancel")
        
        choice = Prompt.ask(
            "Select display mode",
            choices=[str(i) for i in range(1, len(modes) + 2)],
            default=str(len(modes) + 1)
        )
        
        choice_int = int(choice)
        
        if choice_int <= len(modes):
            selected_mode = modes[choice_int - 1]
            await self.cli_config.set_display_mode(selected_mode)
            console.print(f"[green]✅ Display mode set to '{selected_mode.value}'[/green]")
    
    async def _configure_display_settings(self) -> None:
        """Configure detailed display settings."""
        if not self.cli_config:
            return
        
        config = self.cli_config.config.display
        
        while True:
            settings_table = Table(show_header=False, box=None, padding=(0, 2))
            settings_table.add_column("Setting", style="cyan", width=25)
            settings_table.add_column("Value", style="white")
            
            settings_table.add_row("Max History Display", str(config.max_history_display))
            settings_table.add_row("Max Search Results", str(config.max_search_results))
            settings_table.add_row("Show Timestamps", "Yes" if config.show_timestamps else "No")
            settings_table.add_row("Show Session Stats", "Yes" if config.show_session_stats else "No")
            settings_table.add_row("Enable Streaming", "Yes" if config.enable_streaming else "No")
            settings_table.add_row("Show Progress Bars", "Yes" if config.show_progress_bars else "No")
            settings_table.add_row("Syntax Highlighting", "Yes" if config.syntax_highlighting else "No")
            settings_table.add_row("Markdown Rendering", "Yes" if config.markdown_rendering else "No")
            
            console.print(Panel(settings_table, title="📺 Display Settings", border_style="blue"))
            
            console.print("\n[bold cyan]Settings to modify:[/bold cyan]")
            console.print("1. Max history display")
            console.print("2. Max search results")
            console.print("3. Toggle timestamps")
            console.print("4. Toggle session stats")
            console.print("5. Toggle streaming")
            console.print("6. Toggle progress bars")
            console.print("7. Toggle syntax highlighting")
            console.print("8. Toggle markdown rendering")
            console.print("9. Back")
            
            choice = Prompt.ask("Select setting", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"], default="9")
            
            if choice == "1":
                new_value = Prompt.ask("Max history display", default=str(config.max_history_display))
                try:
                    config.max_history_display = int(new_value)
                    await self.cli_config.save_config()
                    console.print("[green]✅ Setting updated[/green]")
                except ValueError:
                    console.print("[red]❌ Invalid number[/red]")
            elif choice == "2":
                new_value = Prompt.ask("Max search results", default=str(config.max_search_results))
                try:
                    config.max_search_results = int(new_value)
                    await self.cli_config.save_config()
                    console.print("[green]✅ Setting updated[/green]")
                except ValueError:
                    console.print("[red]❌ Invalid number[/red]")
            elif choice == "3":
                config.show_timestamps = not config.show_timestamps
                await self.cli_config.save_config()
                console.print("[green]✅ Timestamps toggled[/green]")
            elif choice == "4":
                config.show_session_stats = not config.show_session_stats
                await self.cli_config.save_config()
                console.print("[green]✅ Session stats toggled[/green]")
            elif choice == "5":
                config.enable_streaming = not config.enable_streaming
                await self.cli_config.save_config()
                console.print("[green]✅ Streaming toggled[/green]")
            elif choice == "6":
                config.show_progress_bars = not config.show_progress_bars
                await self.cli_config.save_config()
                console.print("[green]✅ Progress bars toggled[/green]")
            elif choice == "7":
                config.syntax_highlighting = not config.syntax_highlighting
                await self.cli_config.save_config()
                console.print("[green]✅ Syntax highlighting toggled[/green]")
            elif choice == "8":
                config.markdown_rendering = not config.markdown_rendering
                await self.cli_config.save_config()
                console.print("[green]✅ Markdown rendering toggled[/green]")
            elif choice == "9":
                break
    
    async def _configure_behavior_settings(self) -> None:
        """Configure behavior settings."""
        if not self.cli_config:
            return
        
        config = self.cli_config.config.behavior
        
        while True:
            settings_table = Table(show_header=False, box=None, padding=(0, 2))
            settings_table.add_column("Setting", style="cyan", width=25)
            settings_table.add_column("Value", style="white")
            
            settings_table.add_row("Auto-save Frequency", f"{config.auto_save_frequency}s")
            settings_table.add_row("Confirmation Prompts", "Yes" if config.confirmation_prompts else "No")
            settings_table.add_row("Auto Greetings", "Yes" if config.auto_greetings else "No")
            settings_table.add_row("Memory Verbosity", config.memory_verbosity)
            settings_table.add_row("Show Memory Reasoning", "Yes" if config.show_memory_reasoning else "No")
            settings_table.add_row("Auto-tag Sessions", "Yes" if config.auto_tag_sessions else "No")
            settings_table.add_row("Debug Mode Default", "Yes" if config.debug_mode_default else "No")
            
            console.print(Panel(settings_table, title="⚙️  Behavior Settings", border_style="blue"))
            
            console.print("\n[bold cyan]Settings to modify:[/bold cyan]")
            console.print("1. Auto-save frequency")
            console.print("2. Toggle confirmation prompts")
            console.print("3. Toggle auto greetings")
            console.print("4. Set memory verbosity")
            console.print("5. Toggle memory reasoning display")
            console.print("6. Toggle auto-tag sessions")
            console.print("7. Toggle debug mode default")
            console.print("8. Back")
            
            choice = Prompt.ask("Select setting", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="8")
            
            if choice == "1":
                new_value = Prompt.ask("Auto-save frequency (seconds)", default=str(config.auto_save_frequency))
                try:
                    config.auto_save_frequency = int(new_value)
                    await self.cli_config.save_config()
                    console.print("[green]✅ Auto-save frequency updated[/green]")
                except ValueError:
                    console.print("[red]❌ Invalid number[/red]")
            elif choice == "2":
                config.confirmation_prompts = not config.confirmation_prompts
                await self.cli_config.save_config()
                console.print("[green]✅ Confirmation prompts toggled[/green]")
            elif choice == "3":
                config.auto_greetings = not config.auto_greetings
                await self.cli_config.save_config()
                console.print("[green]✅ Auto greetings toggled[/green]")
            elif choice == "4":
                console.print("Memory verbosity options: minimal, normal, detailed")
                new_verbosity = Prompt.ask("Memory verbosity", choices=["minimal", "normal", "detailed"], default=config.memory_verbosity)
                config.memory_verbosity = new_verbosity
                await self.cli_config.save_config()
                console.print("[green]✅ Memory verbosity updated[/green]")
            elif choice == "5":
                config.show_memory_reasoning = not config.show_memory_reasoning
                await self.cli_config.save_config()
                console.print("[green]✅ Memory reasoning display toggled[/green]")
            elif choice == "6":
                config.auto_tag_sessions = not config.auto_tag_sessions
                await self.cli_config.save_config()
                console.print("[green]✅ Auto-tag sessions toggled[/green]")
            elif choice == "7":
                config.debug_mode_default = not config.debug_mode_default
                await self.cli_config.save_config()
                console.print("[green]✅ Debug mode default toggled[/green]")
            elif choice == "8":
                break
    
    async def _show_preferences_menu(self) -> None:
        """Show user preferences management."""
        if not self.cli_config:
            console.print("[red]❌ CLI configuration not available[/red]")
            return
        
        while True:
            prefs = self.cli_config.config.user_preferences
            
            if prefs:
                prefs_table = Table(show_header=True, header_style="bold magenta")
                prefs_table.add_column("Key", style="cyan")
                prefs_table.add_column("Value", style="white")
                
                for key, value in prefs.items():
                    prefs_table.add_row(key, str(value))
                
                console.print(Panel(prefs_table, title="👤 User Preferences", border_style="blue"))
            else:
                console.print(Panel(
                    Text("No user preferences set yet.", style="dim"),
                    title="👤 User Preferences",
                    border_style="blue"
                ))
            
            console.print("\n[bold cyan]Preferences Options:[/bold cyan]")
            console.print("1. Add preference")
            console.print("2. Update preference")
            console.print("3. Remove preference")
            console.print("4. Clear all preferences")
            console.print("5. Back")
            
            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"], default="5")
            
            if choice == "1":
                key = Prompt.ask("Preference key")
                value = Prompt.ask("Preference value")
                if key and value:
                    await self.cli_config.update_preference(key, value)
                    console.print(f"[green]✅ Preference '{key}' added[/green]")
            elif choice == "2":
                if prefs:
                    key = Prompt.ask("Preference key to update", choices=list(prefs.keys()))
                    value = Prompt.ask("New value", default=str(prefs[key]))
                    await self.cli_config.update_preference(key, value)
                    console.print(f"[green]✅ Preference '{key}' updated[/green]")
                else:
                    console.print("[yellow]No preferences to update[/yellow]")
            elif choice == "3":
                if prefs:
                    key = Prompt.ask("Preference key to remove", choices=list(prefs.keys()))
                    if key in prefs:
                        del self.cli_config.config.user_preferences[key]
                        await self.cli_config.save_config()
                        console.print(f"[green]✅ Preference '{key}' removed[/green]")
                else:
                    console.print("[yellow]No preferences to remove[/yellow]")
            elif choice == "4":
                if prefs and Confirm.ask("Clear all preferences?", default=False):
                    self.cli_config.config.user_preferences.clear()
                    await self.cli_config.save_config()
                    console.print("[green]✅ All preferences cleared[/green]")
            elif choice == "5":
                break
    
    async def _show_aliases_menu(self) -> None:
        """Show aliases management."""
        if not self.cli_config:
            console.print("[red]❌ CLI configuration not available[/red]")
            return
        
        while True:
            aliases = self.cli_config.config.shortcuts.aliases
            
            if aliases:
                aliases_table = Table(show_header=True, header_style="bold magenta")
                aliases_table.add_column("Alias", style="cyan", width=10)
                aliases_table.add_column("Command", style="white")
                
                for alias, command in aliases.items():
                    aliases_table.add_row(alias, command)
                
                console.print(Panel(aliases_table, title="⚡ Command Aliases", border_style="blue"))
            else:
                console.print(Panel(
                    Text("No aliases configured.", style="dim"),
                    title="⚡ Command Aliases",
                    border_style="blue"
                ))
            
            console.print("\n[bold cyan]Aliases Options:[/bold cyan]")
            console.print("1. Add alias")
            console.print("2. Remove alias")
            console.print("3. Test alias")
            console.print("4. Reset to defaults")
            console.print("5. Back")
            
            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"], default="5")
            
            if choice == "1":
                alias = Prompt.ask("Alias (short form)")
                command = Prompt.ask("Full command")
                if alias and command:
                    await self.cli_config.add_alias(alias, command)
                    console.print(f"[green]✅ Alias '{alias}' → '{command}' added[/green]")
            elif choice == "2":
                if aliases:
                    alias = Prompt.ask("Alias to remove", choices=list(aliases.keys()))
                    success = await self.cli_config.remove_alias(alias)
                    if success:
                        console.print(f"[green]✅ Alias '{alias}' removed[/green]")
                else:
                    console.print("[yellow]No aliases to remove[/yellow]")
            elif choice == "3":
                if aliases:
                    test_input = Prompt.ask("Test input")
                    resolved = self.cli_config.resolve_alias(test_input)
                    console.print(f"[cyan]Input:[/cyan] {test_input}")
                    console.print(f"[cyan]Resolved:[/cyan] {resolved}")
                else:
                    console.print("[yellow]No aliases to test[/yellow]")
            elif choice == "4":
                if Confirm.ask("Reset aliases to defaults?", default=False):
                    # Reset to default aliases
                    from .cli_config import ShortcutsConfig
                    default_shortcuts = ShortcutsConfig()
                    self.cli_config.config.shortcuts.aliases = default_shortcuts.aliases
                    await self.cli_config.save_config()
                    console.print("[green]✅ Aliases reset to defaults[/green]")
            elif choice == "5":
                break
    
    async def _show_quick_commands(self) -> None:
        """Show quick commands management."""
        if not self.cli_config:
            console.print("[red]❌ CLI configuration not available[/red]")
            return
        
        while True:
            quick_commands = self.cli_config.config.shortcuts.quick_commands
            
            if quick_commands:
                commands_table = Table(show_header=True, header_style="bold magenta")
                commands_table.add_column("Quick Command", style="cyan", width=15)
                commands_table.add_column("Full Message", style="white")
                
                for name, message in quick_commands.items():
                    # Truncate long messages for display
                    display_message = message[:50] + "..." if len(message) > 50 else message
                    commands_table.add_row(name, display_message)
                
                console.print(Panel(commands_table, title="🚀 Quick Commands", border_style="blue"))
            else:
                console.print(Panel(
                    Text("No quick commands configured.", style="dim"),
                    title="🚀 Quick Commands",
                    border_style="blue"
                ))
            
            console.print("\n[bold cyan]Quick Commands Options:[/bold cyan]")
            console.print("1. Add quick command")
            console.print("2. Remove quick command")
            console.print("3. Test quick command")
            console.print("4. Reset to defaults")
            console.print("5. Back")
            
            choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"], default="5")
            
            if choice == "1":
                name = Prompt.ask("Quick command name")
                message = Prompt.ask("Full message")
                if name and message:
                    await self.cli_config.add_quick_command(name, message)
                    console.print(f"[green]✅ Quick command '{name}' added[/green]")
            elif choice == "2":
                if quick_commands:
                    name = Prompt.ask("Quick command to remove", choices=list(quick_commands.keys()))
                    success = await self.cli_config.remove_quick_command(name)
                    if success:
                        console.print(f"[green]✅ Quick command '{name}' removed[/green]")
                else:
                    console.print("[yellow]No quick commands to remove[/yellow]")
            elif choice == "3":
                if quick_commands:
                    test_name = Prompt.ask("Test quick command", choices=list(quick_commands.keys()))
                    resolved = self.cli_config.resolve_quick_command(test_name)
                    console.print(f"[cyan]Quick command:[/cyan] {test_name}")
                    console.print(f"[cyan]Full message:[/cyan] {resolved}")
                else:
                    console.print("[yellow]No quick commands to test[/yellow]")
            elif choice == "4":
                if Confirm.ask("Reset quick commands to defaults?", default=False):
                    # Reset to default quick commands
                    from .cli_config import ShortcutsConfig
                    default_shortcuts = ShortcutsConfig()
                    self.cli_config.config.shortcuts.quick_commands = default_shortcuts.quick_commands
                    await self.cli_config.save_config()
                    console.print("[green]✅ Quick commands reset to defaults[/green]")
            elif choice == "5":
                break
    
    async def _export_configuration(self) -> None:
        """Export configuration to file."""
        if not self.cli_config:
            console.print("[red]❌ CLI configuration not available[/red]")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"config_export_{timestamp}.json"
        
        export_path = Prompt.ask("Export file path", default=default_path)
        
        try:
            await self.cli_config.export_config(Path(export_path))
            console.print(f"[green]✅ Configuration exported to {export_path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Export failed: {e}[/red]")
    
    async def _import_configuration(self) -> None:
        """Import configuration from file."""
        if not self.cli_config:
            console.print("[red]❌ CLI configuration not available[/red]")
            return
        
        import_path = Prompt.ask("Import file path")
        
        if not Path(import_path).exists():
            console.print(f"[red]❌ File not found: {import_path}[/red]")
            return
        
        if Confirm.ask("⚠️  This will overwrite current configuration. Continue?", default=False):
            success = await self.cli_config.import_config(Path(import_path))
            if success:
                console.print("[green]✅ Configuration imported successfully[/green]")
                console.print("[dim]💡 Some changes may require restart to take effect[/dim]")
            else:
                console.print("[red]❌ Import failed[/red]")
    
    async def _reset_configuration(self) -> None:
        """Reset configuration to defaults."""
        if not self.cli_config:
            console.print("[red]❌ CLI configuration not available[/red]")
            return
        
        await self.cli_config.reset_to_defaults()
        console.print("[green]✅ Configuration reset to defaults[/green]")
        console.print("[dim]💡 Restart recommended for full effect[/dim]")
    
    async def _cleanup(self) -> None:
        """Enhanced cleanup with goodbye message."""
        console.print()
        console.print("🧹 [dim]Cleaning up resources...[/dim]")
        
        # Save current session if it exists and has content
        if (self.session_manager and 
            self.session_manager.current_session and 
            len(self.session_manager.current_session.conversation_turns) > 0):
            
            console.print("💾 [dim]Saving current session...[/dim]")
            self.session_manager.current_session.update_duration(self.session_start_time)
            await self.session_manager.save_session(force=True)
        
        if self.agent:
            await self.agent.shutdown()
        
        # Show session summary
        session_duration = datetime.now() - self.session_start_time
        minutes = int(session_duration.total_seconds() / 60)
        
        summary_text = Text()
        summary_text.append("📊 Session Summary:\n", style="bold")
        summary_text.append(f"• Duration: {minutes} minutes\n")
        summary_text.append(f"• Messages: {self.total_messages}\n")
        summary_text.append(f"• Conversations: {len(self.conversation_history.current_session_turns)} turns\n")
        
        # Add session persistence info
        if self.session_manager and self.session_manager.current_session:
            current = self.session_manager.current_session
            summary_text.append(f"• Session: {current.title or current.session_id[:8]} (saved)\n")
        
        summary_text.append("\nThank you for using Personal AI Assistant! 🙏")
        
        console.print(Panel(summary_text, title="👋 Goodbye", border_style="blue"))


@click.group()
def cli():
    """Personal AI Assistant CLI"""
    pass


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--model', '-m', help='Model ID to use')
@click.option('--temperature', '-t', type=float, help='Model temperature')
def chat(config, model, temperature):
    """Start an interactive chat session with the assistant."""
    
    async def run_chat():
        # Create CLI instance
        config_path = Path(config) if config else None
        agent_cli = AgentCLI(config_path)
        
        # Override config with CLI options
        if model or temperature:
            # This would require modifying the config after loading
            console.print("[yellow]CLI parameter overrides not yet implemented[/yellow]")
        
        # Initialize and run
        if await agent_cli.initialize_agent():
            await agent_cli.run_interactive_session()
        else:
            sys.exit(1)
    
    # Run the async chat session
    asyncio.run(run_chat())


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
def status(config):
    """Show agent status without starting interactive session."""
    
    async def show_status():
        config_path = Path(config) if config else None
        agent_cli = AgentCLI(config_path)
        
        if await agent_cli.initialize_agent():
            await agent_cli._show_status()
            await agent_cli._cleanup()
        else:
            sys.exit(1)
    
    asyncio.run(show_status())


@cli.command()
@click.argument('message')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--stream/--no-stream', default=True, help='Enable/disable streaming response')
def ask(message, config, stream):
    """Ask the assistant a single question."""
    
    async def process_question():
        config_path = Path(config) if config else None
        agent_cli = AgentCLI(config_path)
        
        if await agent_cli.initialize_agent():
            console.print(f"[bold cyan]Question:[/bold cyan] {message}")
            console.print("[bold green]Assistant:[/bold green] ", end="")
            
            if stream:
                async for chunk in agent_cli.agent.stream_response(message):
                    console.print(chunk, end="", highlight=False)
                console.print()
            else:
                response = await agent_cli.agent.process_message(message)
                console.print(response)
            
            await agent_cli._cleanup()
        else:
            sys.exit(1)
    
    asyncio.run(process_question())


if __name__ == '__main__':
    cli() 