"""
Command Line Interface for Personal AI Assistant

Provides an interactive CLI for communicating with the agent,
including support for streaming responses and agent management.
"""

import asyncio
import sys
from typing import Optional
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from agent.core_agent import PersonalAssistantAgent
from agent.agent_config import AgentConfig


console = Console()


class AgentCLI:
    """Command-line interface for the Personal AI Assistant."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the CLI.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.agent: Optional[PersonalAssistantAgent] = None
        self.config_path = config_path
        self.is_running = False
        
    async def initialize_agent(self) -> bool:
        """Initialize the agent with configuration."""
        try:
            # Load configuration
            if self.config_path and self.config_path.exists():
                config = AgentConfig.from_file(self.config_path)
                console.print(f"[green]Loaded configuration from {self.config_path}[/green]")
            else:
                config = AgentConfig.from_env()
                console.print("[yellow]Using default configuration with environment overrides[/yellow]")
            
            # Create and initialize agent
            self.agent = PersonalAssistantAgent(config)
            
            with console.status("[bold blue]Initializing Personal AI Assistant..."):
                success = await self.agent.initialize()
            
            if success:
                console.print("[green]✓ Personal AI Assistant initialized successfully![/green]")
                return True
            else:
                console.print("[red]✗ Failed to initialize Personal AI Assistant[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error initializing agent: {e}[/red]")
            return False
    
    async def run_interactive_session(self) -> None:
        """Run an interactive chat session with the agent."""
        if not self.agent:
            console.print("[red]Agent not initialized[/red]")
            return
        
        self.is_running = True
        
        # Display welcome message
        self._display_welcome()
        
        try:
            while self.is_running:
                # Get user input
                try:
                    user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
                except (KeyboardInterrupt, EOFError):
                    break
                
                if not user_input.strip():
                    continue
                
                # Handle special commands
                if await self._handle_special_commands(user_input.strip()):
                    continue
                
                # Process message with streaming
                await self._process_message_with_streaming(user_input)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted by user[/yellow]")
        finally:
            await self._cleanup()
    
    def _display_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_text = Text()
        welcome_text.append("Personal AI Assistant", style="bold blue")
        welcome_text.append("\n\nI'm your AI assistant with persistent memory capabilities.")
        welcome_text.append("\nI can help you with various tasks and remember our conversations.")
        welcome_text.append("\n\nSpecial commands:")
        welcome_text.append("\n  /status  - Show agent status")
        welcome_text.append("\n  /history - Show conversation history")
        welcome_text.append("\n  /clear   - Clear conversation history")
        welcome_text.append("\n  /help    - Show this help message")
        welcome_text.append("\n  /quit    - Exit the assistant")
        welcome_text.append("\n\nType your message and press Enter to start!")
        
        panel = Panel(welcome_text, title="Welcome", border_style="blue")
        console.print(panel)
    
    async def _handle_special_commands(self, command: str) -> bool:
        """
        Handle special CLI commands.
        
        Args:
            command: User input command
            
        Returns:
            True if command was handled, False otherwise
        """
        if command.lower() in ['/quit', '/exit', '/q']:
            self.is_running = False
            console.print("[yellow]Goodbye![/yellow]")
            return True
        
        elif command.lower() == '/status':
            await self._show_status()
            return True
        
        elif command.lower() == '/history':
            self._show_history()
            return True
        
        elif command.lower() == '/clear':
            self._clear_history()
            return True
        
        elif command.lower() == '/help':
            self._display_welcome()
            return True
        
        return False
    
    async def _process_message_with_streaming(self, message: str) -> None:
        """Process a message with streaming response display."""
        console.print("[bold green]Assistant:[/bold green] ", end="")
        
        try:
            response_text = ""
            async for chunk in self.agent.stream_response(message):
                response_text += chunk
                console.print(chunk, end="", highlight=False)
            
            console.print()  # New line after response
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    async def _show_status(self) -> None:
        """Display agent status information."""
        try:
            status = await self.agent.get_agent_status()
            
            status_text = Text()
            status_text.append(f"Agent: {status['agent_name']} v{status['agent_version']}\n", style="bold")
            status_text.append(f"Initialized: {'✓' if status['initialized'] else '✗'}\n")
            status_text.append(f"Memory Enabled: {'✓' if status['memory_enabled'] else '✗'}\n")
            status_text.append(f"MCP Enabled: {'✓' if status['mcp_enabled'] else '✗'}\n")
            status_text.append(f"Conversation Turns: {status['conversation_turns']}\n")
            
            # Memory stats
            if 'memory_stats' in status:
                mem_stats = status['memory_stats']
                status_text.append(f"\nMemory Statistics:\n", style="bold")
                status_text.append(f"  Total Memories: {mem_stats.get('total_memories', 'N/A')}\n")
                status_text.append(f"  Total Interactions: {mem_stats.get('total_interactions', 'N/A')}\n")
            
            # MCP status
            if 'mcp_status' in status:
                mcp_status = status['mcp_status']
                status_text.append(f"\nMCP Status:\n", style="bold")
                status_text.append(f"  Overall: {mcp_status.get('overall_status', 'unknown')}\n")
                status_text.append(f"  Connected Servers: {mcp_status.get('connected_servers', 0)}/{mcp_status.get('total_servers', 0)}\n")
                status_text.append(f"  Available Tools: {mcp_status.get('available_tools', 0)}\n")
            
            panel = Panel(status_text, title="Agent Status", border_style="green")
            console.print(panel)
            
        except Exception as e:
            console.print(f"[red]Error getting status: {e}[/red]")
    
    def _show_history(self) -> None:
        """Display conversation history."""
        if not self.agent:
            console.print("[red]Agent not available[/red]")
            return
        
        history = self.agent.get_conversation_history()
        
        if not history:
            console.print("[yellow]No conversation history available[/yellow]")
            return
        
        history_text = Text()
        for i, turn in enumerate(history[-10:], 1):  # Show last 10 turns
            timestamp = turn.get('timestamp', 'Unknown')
            user_msg = turn.get('user_message', '')
            assistant_msg = turn.get('assistant_response', '')
            
            history_text.append(f"{i}. [{timestamp}]\n", style="dim")
            history_text.append(f"You: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}\n", style="cyan")
            history_text.append(f"Assistant: {assistant_msg[:100]}{'...' if len(assistant_msg) > 100 else ''}\n\n", style="green")
        
        panel = Panel(history_text, title="Recent Conversation History", border_style="blue")
        console.print(panel)
    
    def _clear_history(self) -> None:
        """Clear conversation history."""
        if self.agent:
            self.agent.conversation_history.clear()
            console.print("[green]Conversation history cleared[/green]")
        else:
            console.print("[red]Agent not available[/red]")
    
    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.agent:
            await self.agent.shutdown()


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