#!/usr/bin/env python3
"""
Personal AI Assistant - Main Entry Point

This is the primary entry point for the Personal AI Assistant.
Simply run: python src/main.py

Features:
- Automatic environment detection and setup
- System health checks and validation
- Graceful startup with helpful error messages
- Direct launch into interactive chat experience
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
import time

console = Console()


class StartupValidator:
    """Validates system requirements and configuration for startup."""
    
    def __init__(self):
        self.console = console
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        
    def validate_python_version(self) -> bool:
        """Check Python version compatibility."""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        self.validation_results['python'] = {
            'status': current_version >= min_version,
            'message': f"Python {'.'.join(map(str, current_version))}",
            'requirement': f"Python {'.'.join(map(str, min_version))}+",
            'fix': "Please upgrade to Python 3.8 or higher"
        }
        
        return current_version >= min_version
    
    def validate_environment_file(self) -> bool:
        """Check for .env file and required environment variables with smart API key detection."""
        env_path = Path('.env')
        env_example_path = Path('.env-example')
        
        if not env_path.exists():
            if env_example_path.exists():
                # Offer to create .env from .env-example
                self.validation_results['env_file'] = {
                    'status': False,
                    'message': ".env file missing",
                    'requirement': ".env file with API key",
                    'fix': "Copy .env-example to .env and add your API key",
                    'auto_fix': True
                }
                return False
            else:
                self.validation_results['env_file'] = {
                    'status': False,
                    'message': ".env and .env-example missing",
                    'requirement': ".env file with API key",
                    'fix': "Create .env file with ANTHROPIC_API_KEY=your_key_here"
                }
                return False
        
        # Check if .env has API key and auto-configure provider
        try:
            with open(env_path, 'r') as f:
                content = f.read()
            
            # Smart API key detection and auto-configuration
            detected_provider = None
            api_key_found = False
            
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    
                    if 'API_KEY' in key and value and value != 'your_api_key_here' and value != 'your_anthropic_api_key_here':
                        api_key_found = True
                        
                        # Auto-detect provider based on API key format
                        if key == 'ANTHROPIC_API_KEY' or value.startswith('sk-ant-'):
                            detected_provider = 'anthropic'
                        elif key == 'OPENAI_API_KEY' or (value.startswith('sk-') and not value.startswith('sk-ant-')):
                            detected_provider = 'openai'
                        elif key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']:
                            detected_provider = 'bedrock'
            
            if api_key_found:
                # Auto-configure optimal settings based on detected provider
                auto_config_applied = self._apply_auto_configuration(env_path, content, detected_provider)
                
                provider_msg = f" (auto-detected: {detected_provider})" if detected_provider else ""
                config_msg = " with optimized settings" if auto_config_applied else ""
                
                self.validation_results['env_file'] = {
                    'status': True,
                    'message': f".env file ready{provider_msg}{config_msg}",
                    'requirement': ".env file with API key",
                    'detected_provider': detected_provider,
                    'auto_configured': auto_config_applied
                }
                return True
            else:
                self.validation_results['env_file'] = {
                    'status': False,
                    'message': ".env file exists but no valid API key found",
                    'requirement': "Valid API key in .env file",
                    'fix': "Add a valid API key (ANTHROPIC_API_KEY=sk-ant-... or OPENAI_API_KEY=sk-...)"
                }
                return False
                
        except Exception as e:
            self.validation_results['env_file'] = {
                'status': False,
                'message': f".env file error: {e}",
                'requirement': "Valid .env file",
                'fix': "Fix .env file format or recreate from .env-example"
            }
            return False
    
    def _apply_auto_configuration(self, env_path: Path, content: str, detected_provider: Optional[str]) -> bool:
        """Apply automatic configuration optimizations based on detected provider."""
        if not detected_provider:
            return False
            
        try:
            lines = content.split('\n')
            modified = False
            
            # Add provider-specific optimizations
            if detected_provider == 'anthropic':
                # Set optimal Anthropic model if not already set
                if not any('AI_MODEL_ID=' in line and not line.strip().startswith('#') for line in lines):
                    lines.append('')
                    lines.append('# Auto-configured for optimal Anthropic experience (Claude 3.7)')
                    lines.append('AI_MODEL_ID=claude-3-7-sonnet-latest')
                    lines.append('AI_MODEL_TEMPERATURE=0.7')
                    modified = True
                    
            elif detected_provider == 'openai':
                # Set optimal OpenAI model if not already set
                if not any('AI_MODEL_ID=' in line and not line.strip().startswith('#') for line in lines):
                    lines.append('')
                    lines.append('# Auto-configured for optimal OpenAI experience')
                    lines.append('AI_MODEL_ID=gpt-4-turbo-preview')
                    lines.append('AI_MODEL_TEMPERATURE=0.7')
                    modified = True
            
            # Add memory optimizations if not set
            if not any('MEMORY_BASE_PATH=' in line and not line.strip().startswith('#') for line in lines):
                lines.append('')
                lines.append('# Memory location (default: ~/.assistant_memory)')
                lines.append('MEMORY_BASE_PATH=~/.assistant_memory')
                modified = True
            
            # Write back optimized configuration
            if modified:
                with open(env_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                console.print(f"\n✨ [cyan]Auto-configured optimal settings for {detected_provider}![/cyan]")
                console.print("   [dim]Added model and memory optimizations to .env file[/dim]")
                return True
                
        except Exception as e:
            console.print(f"[yellow]Note: Could not auto-configure settings: {e}[/yellow]")
        
        return False
    
    def validate_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        required_modules = [
            ('rich', 'Rich terminal library'),
            ('anthropic', 'Anthropic API client'),
            ('click', 'CLI framework'),
            ('dotenv', 'Environment variable loading')
        ]
        
        missing_modules = []
        for module_name, description in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                missing_modules.append((module_name, description))
        
        if missing_modules:
            self.validation_results['dependencies'] = {
                'status': False,
                'message': f"Missing {len(missing_modules)} required modules",
                'requirement': "All dependencies installed",
                'fix': "Run: pip install -r requirements.txt",
                'missing': missing_modules
            }
            return False
        
        self.validation_results['dependencies'] = {
            'status': True,
            'message': "All required dependencies found",
            'requirement': "All dependencies installed"
        }
        return True
    
    def validate_memory_directory(self) -> bool:
        """Check if memory directory structure exists and initialize if needed."""
        from dotenv import load_dotenv
        load_dotenv()
        
        memory_base = os.getenv('MEMORY_BASE_PATH', '~/.assistant_memory')
        memory_path = Path(memory_base).expanduser()
        
        # Check for migration from old location
        self._check_memory_migration(memory_path)
        
        try:
            # Import MemoryInitializer - handle both package and direct execution
            try:
                from src.memory.memory_initializer import MemoryInitializer
            except ImportError:
                # For direct execution, add parent to path
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from src.memory.memory_initializer import MemoryInitializer
            
            initializer = MemoryInitializer(str(memory_path))
            
            # Check if already initialized
            if initializer.is_initialized():
                # Verify directory is writable
                if not os.access(memory_path, os.W_OK):
                    self.validation_results['memory'] = {
                        'status': False,
                        'message': f"Memory directory not writable: {memory_path}",
                        'requirement': "Writable memory directory",
                        'fix': f"Check permissions for {memory_path}"
                    }
                    return False
                
                # Get memory info for status message
                memory_info = initializer.get_memory_info()
                core_count = len(memory_info.get('core_files', []))
                month_count = len(memory_info.get('interaction_months', []))
                
                self.validation_results['memory'] = {
                    'status': True,
                    'message': f"Memory system ready: {memory_path} ({core_count} core files, {month_count} months)",
                    'requirement': "Memory system initialized",
                    'path': str(memory_path),
                    'already_exists': True
                }
                return True
            
            # Need to initialize - check if parent directory can be created
            try:
                memory_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.validation_results['memory'] = {
                    'status': False,
                    'message': f"Cannot create memory directory: {e}",
                    'requirement': "Memory directory access",
                    'fix': f"Check permissions for {memory_path.parent} or set MEMORY_BASE_PATH"
                }
                return False
            
            # Initialize the memory structure
            console.print(f"\n🧠 [cyan]Initializing memory system at {memory_path}...[/cyan]")
            
            success = initializer.initialize_memory_structure()
            
            if success:
                # Show what was created
                console.print("✅ [green]Memory system initialized successfully![/green]")
                console.print("\n📁 [dim]Created memory structure:[/dim]")
                console.print("   [cyan]├── core/[/cyan]          [dim]Essential memory files (profile, context, preferences)[/dim]")
                console.print("   [cyan]├── interactions/[/cyan]  [dim]Conversation logs organized by month[/dim]")
                console.print("   [cyan]├── condensed/[/cyan]     [dim]Time-organized memory summaries[/dim]")
                console.print("   [cyan]└── system/[/cyan]       [dim]Configuration and analytics[/dim]")
                
                # Show first-time user message
                console.print("\n💡 [yellow]First time setup complete![/yellow]")
                console.print("   [dim]Your AI assistant will remember conversations across sessions.[/dim]")
                console.print("   [dim]You can explore your memory files anytime in:[/dim] [cyan]" + str(memory_path) + "[/cyan]")
                
                memory_info = initializer.get_memory_info()
                core_count = len(memory_info.get('core_files', []))
                
                self.validation_results['memory'] = {
                    'status': True,
                    'message': f"Memory system initialized: {memory_path} ({core_count} core files created)",
                    'requirement': "Memory system initialized",
                    'path': str(memory_path),
                    'newly_created': True,
                    'core_files': core_count
                }
                return True
            else:
                self.validation_results['memory'] = {
                    'status': False,
                    'message': f"Failed to initialize memory system at {memory_path}",
                    'requirement': "Memory system initialized",
                    'fix': f"Check permissions and disk space for {memory_path}"
                }
                return False
                
        except ImportError as e:
            self.validation_results['memory'] = {
                'status': False,
                'message': f"Memory system components not found: {e}",
                'requirement': "Memory system functional",
                'fix': "Ensure src/memory/ modules are properly installed"
            }
            return False
            
        except Exception as e:
            self.validation_results['memory'] = {
                'status': False,
                'message': f"Memory system error: {e}",
                'requirement': "Memory system functional",
                'fix': f"Check {memory_base} permissions or set MEMORY_BASE_PATH environment variable"
            }
            return False
    
    def validate_cli_components(self) -> bool:
        """Check if CLI components can be imported."""
        try:
            # Check if the CLI files exist in the expected location
            cli_path = Path(__file__).parent / "agent" / "cli.py"
            config_path = Path(__file__).parent / "agent" / "agent_config.py"
            
            if not cli_path.exists():
                self.validation_results['cli'] = {
                    'status': False,
                    'message': f"CLI file not found: {cli_path}",
                    'requirement': "CLI system functional",
                    'fix': "Ensure src/agent/cli.py exists"
                }
                return False
                
            if not config_path.exists():
                self.validation_results['cli'] = {
                    'status': False,
                    'message': f"Config file not found: {config_path}",
                    'requirement': "CLI system functional", 
                    'fix': "Ensure src/agent/agent_config.py exists"
                }
                return False
            
            self.validation_results['cli'] = {
                'status': True,
                'message': "CLI components found",
                'requirement': "CLI system functional"
            }
            return True
            
        except Exception as e:
            self.validation_results['cli'] = {
                'status': False,
                'message': f"CLI validation error: {e}",
                'requirement': "CLI system functional",
                'fix': "Check src/agent/ directory structure"
            }
            return False
    
    async def auto_fix_env_file(self) -> bool:
        """Automatically create .env file from .env-example with enhanced user guidance."""
        env_path = Path('.env')
        env_example_path = Path('.env-example')
        
        try:
            if env_example_path.exists():
                shutil.copy2(env_example_path, env_path)
                
                console.print("✅ [green]Created .env file from .env-example[/green]")
                console.print("📝 [yellow]Please edit .env and add your API key![/yellow]")
                
                return True
            else:
                # Create a basic .env file if .env-example doesn't exist
                basic_env_content = """# Personal AI Assistant Configuration
# =============================================================================

# 🔑 REQUIRED: Add your AI provider API key
# Get Anthropic key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your_api_key_here

# OR get OpenAI key from: https://platform.openai.com/api-keys  
# OPENAI_API_KEY=your_api_key_here

# =============================================================================
# ✨ That's it! The assistant will auto-configure everything else.
# =============================================================================
"""
                with open(env_path, 'w') as f:
                    f.write(basic_env_content)
                
                console.print()
                console.print("✅ [green]Created .env configuration file[/green]")
                console.print("🔑 [yellow]Please add your API key to the .env file to continue[/yellow]")
                console.print()
                return True
                
        except Exception as e:
            console.print(f"❌ [red]Failed to create .env file: {e}[/red]")
            
        return False
    
    def show_validation_results(self, show_all: bool = False) -> bool:
        """Display validation results in a nice table."""
        table = Table(title="🔍 System Validation Results")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")
        
        all_passed = True
        
        for component, result in self.validation_results.items():
            status = result['status']
            all_passed = all_passed and status
            
            if status:
                status_icon = "✅"
                status_style = "green"
            else:
                status_icon = "❌"
                status_style = "red"
            
            if show_all or not status:
                table.add_row(
                    component.replace('_', ' ').title(),
                    f"[{status_style}]{status_icon}[/{status_style}]",
                    result['message']
                )
        
        console.print(table)
        
        # Show fixes for failed validations
        if not all_passed:
            console.print()
            console.print("🔧 [yellow]Required fixes:[/yellow]")
            for component, result in self.validation_results.items():
                if not result['status'] and 'fix' in result:
                    console.print(f"   • {result['fix']}")
        
        return all_passed
    
    def _check_memory_migration(self, new_memory_path: Path) -> None:
        """Check for migration from old memory location."""
        old_memory_path = Path("~/assistant_memory").expanduser()
        
        # If new location already exists, no migration needed
        if new_memory_path.exists():
            return
            
        # If old location exists, offer migration
        if old_memory_path.exists():
            console.print("\n🔄 [yellow]Memory Directory Migration Available[/yellow]")
            console.print(f"   Found existing memory at: [cyan]{old_memory_path}[/cyan]")
            console.print(f"   New location will be:     [cyan]{new_memory_path}[/cyan]")
            console.print("   [dim](Hidden directory following standard conventions)[/dim]")
            
            try:
                from rich.prompt import Confirm
                should_migrate = Confirm.ask("\n   Migrate your memory to the new location?", default=True)
                
                if should_migrate:
                    console.print(f"\n📦 [cyan]Migrating memory from {old_memory_path} to {new_memory_path}...[/cyan]")
                    
                    # Create parent directory if needed
                    new_memory_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move the directory
                    import shutil
                    shutil.move(str(old_memory_path), str(new_memory_path))
                    
                    console.print("✅ [green]Memory migration completed successfully![/green]")
                    console.print(f"   Your memories are now at: [cyan]{new_memory_path}[/cyan]")
                    
                    # Update .env file if it uses the old path
                    self._update_env_memory_path(new_memory_path)
                    
                else:
                    console.print("\n⚠️  [yellow]Migration skipped.[/yellow]")
                    console.print(f"   Your memories remain at: [cyan]{old_memory_path}[/cyan]")
                    console.print("   To use the old location, set in your .env:")
                    console.print(f"   [dim]MEMORY_BASE_PATH={old_memory_path}[/dim]")
                    
            except Exception as e:
                console.print(f"\n❌ [red]Migration failed: {e}[/red]")
                console.print("   You can manually move the directory later.")
                
    def _update_env_memory_path(self, new_path: Path) -> None:
        """Update .env file with new memory path if it was using old default."""
        env_path = Path('.env')
        if not env_path.exists():
            return
            
        try:
            with open(env_path, 'r') as f:
                lines = f.readlines()
            
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith('MEMORY_BASE_PATH=~/assistant_memory'):
                    lines[i] = f'MEMORY_BASE_PATH={new_path}\n'
                    updated = True
                    break
            
            if updated:
                with open(env_path, 'w') as f:
                    f.writelines(lines)
                console.print(f"   📝 Updated .env file with new path")
                
        except Exception as e:
            console.print(f"   ⚠️  Could not update .env file: {e}")


class StartupManager:
    """Manages the startup process and user experience."""
    
    def __init__(self):
        self.console = console
        self.validator = StartupValidator()
        
    def show_welcome_banner(self):
        """Display welcome banner and introduction."""
        banner_text = Text()
        banner_text.append("🧠 Personal AI Assistant\n", style="bold cyan")
        banner_text.append("AI that remembers you across conversations\n\n", style="dim")
        banner_text.append("Starting up...", style="green")
        
        panel = Panel(
            banner_text,
            title="Welcome",
            border_style="blue",
            padding=(1, 2)
        )
        
        console.print()
        console.print(panel)
        console.print()
    
    async def run_startup_validation(self) -> bool:
        """Run all startup validations with enhanced diagnostics and progress indicator."""
        # First run our enhanced validation for comprehensive checks
        from src.utils.startup_validator import StartupValidator
        enhanced_validator = StartupValidator()
        
        console.print("\n🔍 [bold blue]Running comprehensive startup diagnostics...[/bold blue]\n")
        
        # Run enhanced validation first for complete health check
        enhanced_success = enhanced_validator.validate_all()
        
        if not enhanced_success:
            # If enhanced validation fails, show helpful error guidance
            console.print("\n📋 [yellow]For detailed setup instructions, visit:[/yellow]")
            console.print("   [cyan]https://github.com/yourusername/personal-ai-assistant#setup[/cyan]\n")
            return False
            
        # If enhanced validation passes, run the main app validations quickly
        console.print("✅ [green]Core diagnostics passed! Finalizing setup...[/green]\n")
        
        validations = [
            ("Finalizing environment configuration", self.validator.validate_environment_file),
            ("Finalizing memory system", self.validator.validate_memory_directory),
            ("Finalizing CLI components", self.validator.validate_cli_components),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            for description, validation_func in validations:
                task = progress.add_task(description, total=None)
                
                # Add small delay for better UX
                await asyncio.sleep(0.2)
                
                if asyncio.iscoroutinefunction(validation_func):
                    result = await validation_func()
                else:
                    result = validation_func()
                
                progress.update(task, completed=True)
                
                # If env file validation failed but auto-fix is available
                if (not result and 
                    description == "Checking environment file" and 
                    self.validator.validation_results.get('env_file', {}).get('auto_fix')):
                    
                    progress.update(task, description="Creating .env file")
                    await self.validator.auto_fix_env_file()
                    
                    # Re-validate
                    progress.update(task, description="Re-checking environment file")
                    result = self.validator.validate_environment_file()
        
        # Show results
        all_passed = self.validator.show_validation_results()
        
        # Check for different scenarios and show appropriate messages
        memory_result = self.validator.validation_results.get('memory', {})
        env_result = self.validator.validation_results.get('env_file', {})
        
        if all_passed:
            # Determine which celebration to show
            if memory_result.get('newly_created') and env_result.get('auto_configured'):
                # Perfect 30-second first-time setup
                console.print()
                self.show_30_second_success()
                self.show_first_time_demo()
            elif memory_result.get('newly_created'):
                # First-time memory setup (already shows demo)
                self.show_first_time_memory_setup()
            elif env_result.get('auto_configured'):
                # Just auto-configured environment
                console.print()
                self.show_30_second_success()
            # Otherwise show regular tips in launch_interactive_cli
        elif memory_result.get('newly_created'):
            # Memory was created but other issues exist
            self.show_first_time_memory_setup()
        
        return all_passed
    
    def show_first_time_memory_setup(self):
        """Show special welcome message for first-time memory system setup."""
        memory_result = self.validator.validation_results.get('memory', {})
        memory_path = memory_result.get('path', '~/.assistant_memory')
        core_files = memory_result.get('core_files', 0)
        
        welcome_text = Text()
        welcome_text.append("🎉 Welcome to your Personal AI Assistant!\n\n", style="bold green")
        welcome_text.append("Your memory system has been set up with:\n", style="yellow")
        welcome_text.append(f"• {core_files} core memory files ready to learn about you\n", style="dim")
        welcome_text.append("• Automatic conversation logging and organization\n", style="dim")
        welcome_text.append("• Smart memory condensation over time\n", style="dim")
        welcome_text.append("• Complete conversation continuity across sessions\n\n", style="dim")
        
        welcome_text.append("🗂️  Your memories are stored at:\n", style="cyan")
        welcome_text.append(f"   {memory_path}\n\n", style="bright_white")
        
        welcome_text.append("💡 Pro tip: ", style="yellow")
        welcome_text.append("The assistant will start learning about you immediately. ", style="dim")
        welcome_text.append("Feel free to tell it about your work, interests, and preferences!", style="dim")
        
        panel = Panel(
            welcome_text,
            title="🧠 Memory System Ready",
            border_style="green",
            padding=(1, 2)
        )
        
        console.print()
        console.print(panel)
        console.print()
        
        # Show first-time demo experience
        self.show_first_time_demo()
    
    def show_first_time_demo(self):
        """Show interactive demo experience for first-time users."""
        from rich.prompt import Confirm
        
        # Ask if user wants to see demos
        console.print("🎪 [bold cyan]Would you like to see what your assistant can do?[/bold cyan]")
        console.print("   [dim]I can show you example workflows and memory features![/dim]")
        
        show_demos = Confirm.ask(
            "\n[yellow]See demo workflows?[/yellow]",
            default=True
        )
        
        if show_demos:
            try:
                from src.utils.demo_workflows import show_demo_workflows
                
                # Launch the comprehensive demo system
                continue_to_chat = show_demo_workflows()
                
                if continue_to_chat:
                    console.print("\n🎉 [bold green]Great! Let's start chatting![/bold green]")
                    return
                    
            except ImportError:
                console.print("\n⚠️  [yellow]Demo system not available, showing quick examples instead.[/yellow]")
                self._show_fallback_demo()
        else:
            self._show_quick_start_tips()
    
    def _show_fallback_demo(self):
        """Show fallback demo if demo workflows system isn't available."""
        demo_text = Text()
        demo_text.append("🚀 Here's what your assistant can do:\n\n", style="bold cyan")
        demo_text.append("Great conversation starters:\n\n", style="yellow")
        
        starters = [
            ("👋 Personal Introduction", "Hi, I'm [name] and I work as a [job title] at [company]"),
            ("💼 Current Project", "I'm working on [project] and need help with [specific area]"),
            ("🎯 Goals & Learning", "I'm learning [skill/technology] and want to [goal]"),
            ("🛠️ Tech Stack", "I work with [technologies] and prefer [tools/frameworks]"),
            ("📚 Interests", "I'm interested in [hobbies/topics] and enjoy [activities]")
        ]
        
        for title, example in starters:
            demo_text.append(f"  {title}\n", style="bright_blue")
            demo_text.append(f"  \"{example}\"\n\n", style="dim italic")
        
        demo_text.append("💫 Memory Features:\n", style="yellow")
        demo_text.append("• Remembers everything you share\n", style="dim")
        demo_text.append("• References past conversations naturally\n", style="dim")
        demo_text.append("• Your preferences shape future interactions\n", style="dim")
        demo_text.append("• Context builds up over time for better assistance\n\n", style="dim")
        
        demo_text.append("🎮 Ready to start! Pick a conversation starter above!", style="green bold")
        
        panel = Panel(
            demo_text,
            title="✨ Quick Start Guide",
            border_style="cyan",
            padding=(1, 2)
        )
        
        console.print()
        console.print(panel)
        console.print()
    
    def _show_quick_start_tips(self):
        """Show quick start tips for users who skip demos."""
        tips_text = Text()
        tips_text.append("💡 Quick Start Tips:\n\n", style="bold yellow")
        tips_text.append("Your assistant learns about you through conversation:\n", style="white")
        tips_text.append("• Tell it about your work, interests, and projects\n", style="dim")
        tips_text.append("• Ask questions about your areas of interest\n", style="dim") 
        tips_text.append("• Share your goals and challenges\n", style="dim")
        tips_text.append("• Mention your preferred tools and technologies\n\n", style="dim")
        
        tips_text.append("Special commands:\n", style="white")
        tips_text.append("• Type ", style="dim")
        tips_text.append("/help", style="cyan")
        tips_text.append(" to see all available commands\n", style="dim")
        tips_text.append("• Type ", style="dim")
        tips_text.append("/memory", style="cyan")
        tips_text.append(" to explore your memory files\n", style="dim")
        tips_text.append("• Type ", style="dim")
        tips_text.append("/configure", style="cyan")
        tips_text.append(" to customize settings\n\n", style="dim")
        
        tips_text.append("🚀 Ready to chat! Just start with \"Hello\" or introduce yourself!", style="green bold")
        
        panel = Panel(
            tips_text,
            title="🎯 Getting Started",
            border_style="green",
            padding=(1, 2)
        )
        
        console.print()
        console.print(panel)
        console.print()
    
    def show_30_second_success(self):
        """Show celebration for successful 30-second startup."""
        env_result = self.validator.validation_results.get('env_file', {})
        provider = env_result.get('detected_provider', 'your AI provider')
        auto_configured = env_result.get('auto_configured', False)
        
        success_text = Text()
        success_text.append("🎉 30-Second Setup Complete!\n\n", style="bold green")
        
        if auto_configured:
            success_text.append(f"✨ Auto-configured for {provider}\n", style="cyan")
            success_text.append("🧠 Memory system initialized\n", style="cyan")
            success_text.append("⚙️  Optimal settings applied\n\n", style="cyan")
        else:
            success_text.append(f"✅ Connected to {provider}\n", style="cyan")
            success_text.append("🧠 Memory system ready\n\n", style="cyan")
        
        success_text.append("Your AI assistant is now:\n", style="yellow")
        success_text.append("• Ready to learn about you and remember everything\n", style="dim")
        success_text.append("• Configured with optimal settings for great performance\n", style="dim")
        success_text.append("• Set up to maintain conversation continuity\n", style="dim")
        success_text.append("• Prepared to grow smarter with each interaction\n\n", style="dim")
        
        success_text.append("🚀 Let's start chatting!", style="green bold")
        
        panel = Panel(
            success_text,
            title="🎯 Ready to Go!",
            border_style="green",
            padding=(1, 2)
        )
        
        console.print()
        console.print(panel)
        console.print()
    
    def show_startup_tips(self, skip_basics: bool = False):
        """Show helpful tips for first-time users."""
        if skip_basics:
            # Show advanced tips when user has seen the demo
            tips_text = Text()
            tips_text.append("⚡ Pro Tips:\n", style="bold yellow")
            tips_text.append("• Type ", style="dim")
            tips_text.append("/memory", style="cyan")
            tips_text.append(" to explore your memory files\n", style="dim")
            tips_text.append("• Type ", style="dim")
            tips_text.append("/configure", style="cyan")
            tips_text.append(" to customize the assistant\n", style="dim")
            tips_text.append("• Type ", style="dim")
            tips_text.append("/help", style="cyan")
            tips_text.append(" to see all available commands\n", style="dim")
            tips_text.append("• Your conversations are automatically saved and organized!", style="dim")
            
            console.print(Panel(tips_text, title="Advanced Features", border_style="yellow"))
        else:
            # Show basic tips for regular users
            tips_text = Text()
            tips_text.append("💡 Quick Tips:\n", style="bold yellow")
            tips_text.append("• Type ", style="dim")
            tips_text.append("/help", style="cyan")
            tips_text.append(" to see all available commands\n", style="dim")
            tips_text.append("• Type ", style="dim")
            tips_text.append("/memory", style="cyan")
            tips_text.append(" to explore your memory files\n", style="dim")
            tips_text.append("• Type ", style="dim")
            tips_text.append("/configure", style="cyan")
            tips_text.append(" to customize the assistant\n", style="dim")
            tips_text.append("• Your conversations are automatically saved!", style="dim")
            
            console.print(Panel(tips_text, title="Getting Started", border_style="green"))
        console.print()
    
    def show_startup_error_help(self):
        """Show help for resolving startup errors."""
        help_text = Text()
        help_text.append("🆘 Need help getting started?\n\n", style="bold red")
        help_text.append("Common solutions:\n", style="yellow")
        help_text.append("1. Install dependencies: ", style="dim")
        help_text.append("pip install -r requirements.txt\n", style="cyan")
        help_text.append("2. Add API key to .env: ", style="dim")
        help_text.append("ANTHROPIC_API_KEY=your_key\n", style="cyan")
        help_text.append("3. Check Python version: ", style="dim")
        help_text.append("python --version\n", style="cyan")
        help_text.append("\nFor more help, check the README.md file", style="dim")
        
        console.print(Panel(help_text, title="Troubleshooting", border_style="red"))
        console.print()
    
    async def launch_interactive_cli(self):
        """Launch the interactive CLI interface."""
        try:
            import subprocess
            
            console.print("🚀 [green]Launching Personal AI Assistant...[/green]")
            console.print()
            
            # Check if we've already shown the demo experience
            memory_result = self.validator.validation_results.get('memory', {})
            env_result = self.validator.validation_results.get('env_file', {})
            
            shown_demo = (memory_result.get('newly_created') or 
                         env_result.get('auto_configured'))
            
            if shown_demo:
                # Show advanced tips since user saw the demo
                self.show_startup_tips(skip_basics=True)
            else:
                # Show regular tips for returning users
                self.show_startup_tips(skip_basics=False)
            
            # Launch the CLI using the click command directly  
            console.print("🎯 [cyan]Starting interactive chat session...[/cyan]")
            console.print("💡 [dim]Type /help for available commands, Ctrl+C to exit[/dim]")
            console.print()
            
            # Change to project directory and run the CLI
            project_root = Path(__file__).parent.parent
            
            # Execute the CLI module directly using the clean __main__.py approach
            result = subprocess.run([
                sys.executable, "-m", "src.agent", "chat"
            ], cwd=project_root)
            
            # Handle clean exit
            if result.returncode == 0:
                console.print("\n✨ [green]Thanks for using Personal AI Assistant![/green]")
            elif result.returncode == 130:  # Ctrl+C
                console.print("\n👋 [yellow]Session ended by user[/yellow]")
            else:
                console.print(f"\n⚠️  [yellow]CLI exited with code {result.returncode}[/yellow]")
                
        except KeyboardInterrupt:
            console.print("\n👋 [yellow]Goodbye![/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"❌ [red]Failed to launch CLI: {e}[/red]")
            console.print()
            self.show_startup_error_help()
            sys.exit(1)


async def main():
    """Main entry point for the Personal AI Assistant."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Personal AI Assistant with Persistent Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                    # Start interactive session
  python src/main.py --validate-only    # Check system without starting
  python src/main.py --setup            # Run setup wizard
        """
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Run system validation checks only, do not start CLI'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run interactive setup wizard'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose startup information'
    )
    
    args = parser.parse_args()
    
    # Create startup manager
    startup_manager = StartupManager()
    
    # Show welcome banner
    if not args.validate_only:
        startup_manager.show_welcome_banner()
    
    # Run startup validation
    try:
        validation_passed = await startup_manager.run_startup_validation()
        
        if args.validate_only:
            if validation_passed:
                console.print("\n✅ [green]All validation checks passed![/green]")
                sys.exit(0)
            else:
                console.print("\n❌ [red]Some validation checks failed[/red]")
                sys.exit(1)
        
        if not validation_passed:
            console.print()
            # Check if it's just the API key missing (most common issue)
            env_failed = startup_manager.validator.validation_results.get('env_file', {}).get('status', True) == False
            if env_failed and len([r for r in startup_manager.validator.validation_results.values() if not r['status']]) == 1:
                console.print("🔑 [yellow]Only missing API key - this is normal for first setup![/yellow]")
                if Confirm.ask("🤔 Continue and set up your API key in the CLI?", default=True):
                    console.print("✅ [green]Great! You can configure your API key when the CLI starts[/green]")
                else:
                    startup_manager.show_startup_error_help()
                    sys.exit(1)
            elif Confirm.ask("🤔 Some checks failed. Continue anyway?", default=False):
                console.print("⚠️  [yellow]Continuing with potential issues...[/yellow]")
            else:
                startup_manager.show_startup_error_help()
                sys.exit(1)
        
        # Setup wizard
        if args.setup:
            console.print("🔧 [blue]Setup wizard not yet implemented[/blue]")
            console.print("For now, please configure .env file manually")
            return
        
        # Launch interactive CLI
        await startup_manager.launch_interactive_cli()
        
    except KeyboardInterrupt:
        console.print("\n\n👋 [yellow]Goodbye![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n❌ [red]Unexpected error: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        startup_manager.show_startup_error_help()
        sys.exit(1)


if __name__ == '__main__':
    # Ensure we're running with the correct Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again")
        sys.exit(1)
    
    # Run the main function
    asyncio.run(main()) 