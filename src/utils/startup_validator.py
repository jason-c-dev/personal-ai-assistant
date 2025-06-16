#!/usr/bin/env python3
"""
Enhanced startup validation system with comprehensive checks and helpful error messages.
Provides clear diagnostics and actionable fix suggestions for common issues.
"""

import os
import sys
import json
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


class StartupValidator:
    """
    Comprehensive startup validation with helpful error messages and fix suggestions.
    """
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success_messages: List[str] = []
        self.project_root = Path(__file__).parent.parent.parent
        self.validation_results: Dict[str, Dict] = {}
    
    def validate_all(self) -> bool:
        """
        Run all startup validations with comprehensive health check.
        Returns True if no critical errors found.
        """
        self.errors.clear()
        self.warnings.clear()
        self.success_messages.clear()
        self.validation_results.clear()
        
        console.print("\n🔍 [bold blue]Running startup health check...[/bold blue]\n")
        
        # Critical validations (must pass)
        checks = [
            ("Python Version", self._validate_python_version),
            ("Environment File", self._validate_environment_file),
            ("API Key", self._validate_api_key),
            ("Dependencies", self._validate_dependencies),
            ("Config Files", self._validate_config_files),
            ("Memory System", self._validate_memory_system),
            ("Network Access", self._validate_network_connectivity),
            ("File Permissions", self._validate_file_permissions),
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                self.validation_results[check_name] = result
                if result['status']:
                    self.success_messages.append(f"✅ {check_name}: {result['message']}")
                else:
                    if result.get('critical', True):
                        self.errors.append(f"❌ {check_name}: {result['message']}")
                    else:
                        self.warnings.append(f"⚠️  {check_name}: {result['message']}")
            except Exception as e:
                self.errors.append(f"❌ {check_name}: Unexpected error - {e}")
        
        return len(self.errors) == 0
    
    def _validate_python_version(self) -> Dict:
        """Check Python version compatibility."""
        version = sys.version_info
        min_version = (3, 8)
        
        if version[:2] >= min_version:
            return {
                'status': True,
                'message': f"Python {version.major}.{version.minor}.{version.micro} ✓",
                'critical': True
            }
        else:
            return {
                'status': False,
                'message': f"Python {version.major}.{version.minor} not supported. Requires Python {min_version[0]}.{min_version[1]}+",
                'fix': f"Install Python {min_version[0]}.{min_version[1]}+ from https://python.org/downloads/",
                'critical': True
            }
    
    def _validate_environment_file(self) -> Dict:
        """Check for .env file existence and provide guidance."""
        env_file = self.project_root / ".env"
        example_file = self.project_root / ".env-example"
        
        if env_file.exists():
            return {
                'status': True,
                'message': ".env file found ✓",
                'critical': True
            }
        else:
            if example_file.exists():
                return {
                    'status': False,
                    'message': "Missing .env file",
                    'fix': f"Copy .env-example to .env:\n   cp {example_file} {env_file}\n   Then edit .env and add your ANTHROPIC_API_KEY",
                    'auto_fix': True,
                    'critical': True
                }
            else:
                return {
                    'status': False,
                    'message': "Missing .env file and template",
                    'fix': "Create .env file:\n   echo 'ANTHROPIC_API_KEY=your_key_here' > .env",
                    'critical': True
                }
    
    def _validate_api_key(self) -> Dict:
        """Check for valid API key configuration with provider detection."""
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for different API providers
        api_providers = {
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'openai': os.getenv('OPENAI_API_KEY'),
            'aws': os.getenv('AWS_ACCESS_KEY_ID')
        }
        
        valid_keys = []
        for provider, key in api_providers.items():
            if key and key != 'your_anthropic_api_key_here' and key != 'your_key_here':
                # Basic format validation
                if provider == 'anthropic' and key.startswith('sk-ant-'):
                    valid_keys.append(f"Anthropic (sk-ant-...)")
                elif provider == 'openai' and key.startswith('sk-') and not key.startswith('sk-ant-'):
                    valid_keys.append(f"OpenAI (sk-...)")
                elif provider == 'aws' and len(key) > 10:
                    valid_keys.append(f"AWS ({key[:8]}...)")
                else:
                    valid_keys.append(f"{provider.title()} (format unknown)")
        
        if valid_keys:
            return {
                'status': True,
                'message': f"API key configured: {', '.join(valid_keys)} ✓",
                'critical': True
            }
        else:
            return {
                'status': False,
                'message': "No valid API key found",
                'fix': "Add API key to .env file:\n   ANTHROPIC_API_KEY=sk-ant-your_key_here\n   Get your key from: https://console.anthropic.com/",
                'critical': True
            }
    
    def _validate_dependencies(self) -> Dict:
        """Check for critical dependencies with helpful installation guidance."""
        critical_packages = [
            ('rich', 'Rich terminal library'),
            ('anthropic', 'Anthropic API client'),
            ('click', 'CLI framework'),
            ('python-dotenv', 'Environment variable loading'),
            ('pyyaml', 'YAML file processing'),
            ('pydantic', 'Data validation')
        ]
        
        missing = []
        installed = []
        
        for package, description in critical_packages:
            try:
                # Handle package name variations
                import_name = package
                if package == 'python-dotenv':
                    import_name = 'dotenv'
                elif package == 'pyyaml':
                    import_name = 'yaml'
                
                __import__(import_name)
                installed.append(package)
            except ImportError:
                missing.append((package, description))
        
        if missing:
            missing_list = [f"{pkg} ({desc})" for pkg, desc in missing]
            return {
                'status': False,
                'message': f"Missing {len(missing)} required packages",
                'fix': f"Install missing packages:\n   pip install {' '.join([pkg for pkg, _ in missing])}\n   Or: pip install -r requirements.txt",
                'missing': missing_list,
                'critical': True
            }
        else:
            return {
                'status': True,
                'message': f"All {len(installed)} critical dependencies installed ✓",
                'critical': True
            }
    
    def _validate_config_files(self) -> Dict:
        """Check for required configuration files."""
        config_files = [
            ("config/system_prompts.json", "System prompts configuration"),
            ("config/model_config.json", "Model configuration")
        ]
        
        missing_files = []
        invalid_files = []
        valid_files = []
        
        for file_path, description in config_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append((file_path, description))
            else:
                # Try to parse JSON
                try:
                    with open(full_path, 'r') as f:
                        json.load(f)
                    valid_files.append(file_path)
                except json.JSONDecodeError as e:
                    invalid_files.append((file_path, str(e)))
        
        if missing_files or invalid_files:
            issues = []
            if missing_files:
                issues.extend([f"Missing: {path}" for path, _ in missing_files])
            if invalid_files:
                issues.extend([f"Invalid JSON: {path}" for path, _ in invalid_files])
            
            return {
                'status': False,
                'message': f"Configuration file issues: {', '.join(issues)}",
                'fix': "Check config/ directory and fix JSON syntax errors",
                'critical': False  # Non-critical - app can work with defaults
            }
        else:
            return {
                'status': True,
                'message': f"Configuration files valid ✓",
                'critical': False
            }
    
    def _validate_memory_system(self) -> Dict:
        """Check memory system setup and permissions."""
        from dotenv import load_dotenv
        load_dotenv()
        
        memory_path = os.getenv('MEMORY_BASE_PATH', '~/.assistant_memory')
        expanded_path = Path(memory_path).expanduser()
        
        try:
            # Check if path exists and is writable
            expanded_path.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = expanded_path / '.health_check'
            test_file.write_text('Health check test')
            test_file.unlink()
            
            return {
                'status': True,
                'message': f"Memory system ready at {expanded_path} ✓",
                'critical': True
            }
            
        except PermissionError:
            return {
                'status': False,
                'message': f"Cannot write to memory directory: {expanded_path}",
                'fix': f"Fix permissions:\n   chmod 755 {expanded_path.parent}\n   Or change MEMORY_BASE_PATH in .env",
                'critical': True
            }
        except Exception as e:
            return {
                'status': False,
                'message': f"Memory system error: {e}",
                'fix': f"Check path and permissions for: {expanded_path}",
                'critical': True
            }
    
    def _validate_network_connectivity(self) -> Dict:
        """Test network connectivity to AI provider APIs."""
        test_urls = [
            ("Anthropic API", "https://api.anthropic.com"),
            ("OpenAI API", "https://api.openai.com"),
            ("DNS Resolution", "https://google.com")
        ]
        
        connectivity_results = []
        
        for name, url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code in [200, 401, 403]:  # 401/403 means API is reachable
                    connectivity_results.append(f"{name} ✓")
                else:
                    connectivity_results.append(f"{name} ⚠️")
            except requests.exceptions.RequestException:
                connectivity_results.append(f"{name} ❌")
        
        working_connections = [r for r in connectivity_results if "✓" in r]
        
        if len(working_connections) >= 2:  # At least API + DNS working
            return {
                'status': True,
                'message': f"Network connectivity good ({len(working_connections)}/3 services reachable) ✓",
                'critical': False
            }
        else:
            return {
                'status': False,
                'message': f"Network connectivity issues ({len(working_connections)}/3 services reachable)",
                'fix': "Check internet connection and firewall settings",
                'critical': False  # Can work offline for setup
            }
    
    def _validate_file_permissions(self) -> Dict:
        """Check file system permissions for critical operations."""
        test_locations = [
            (self.project_root, "Project directory"),
            (Path.home(), "Home directory"),
            (Path.cwd(), "Current directory")
        ]
        
        permission_issues = []
        working_locations = []
        
        for location, name in test_locations:
            try:
                test_file = location / '.permission_test'
                test_file.write_text('test')
                test_file.unlink()
                working_locations.append(name)
            except Exception as e:
                permission_issues.append(f"{name}: {e}")
        
        if len(working_locations) >= 2:
            return {
                'status': True,
                'message': f"File permissions OK ({len(working_locations)}/3 locations writable) ✓",
                'critical': False
            }
        else:
            return {
                'status': False,
                'message': f"File permission issues in {len(permission_issues)} locations",
                'fix': "Check file permissions and disk space",
                'critical': True
            }
    
    def print_health_dashboard(self):
        """Print a comprehensive health dashboard with fix suggestions."""
        if not self.validation_results:
            console.print("❌ No validation results available. Run validate_all() first.")
            return
        
        # Create status table
        table = Table(title="🏥 Startup Health Dashboard")
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")
        
        for check_name, result in self.validation_results.items():
            status_icon = "✅" if result['status'] else ("❌" if result.get('critical', True) else "⚠️")
            details = result['message']
            
            table.add_row(check_name, status_icon, details)
        
        console.print(table)
        
        # Show fix suggestions for any issues
        issues_with_fixes = [(name, result) for name, result in self.validation_results.items() 
                           if not result['status'] and 'fix' in result]
        
        if issues_with_fixes:
            console.print("\n🔧 [bold yellow]Fix Suggestions:[/bold yellow]")
            for i, (name, result) in enumerate(issues_with_fixes, 1):
                console.print(f"\n[bold red]{i}. {name}:[/bold red]")
                console.print(f"   [red]Issue:[/red] {result['message']}")
                console.print(f"   [green]Fix:[/green] {result['fix']}")
        
        # Show overall status
        critical_errors = len([r for r in self.validation_results.values() 
                              if not r['status'] and r.get('critical', True)])
        warnings = len([r for r in self.validation_results.values() 
                       if not r['status'] and not r.get('critical', True)])
        
        if critical_errors == 0:
            if warnings == 0:
                console.print(Panel("🎉 [bold green]All systems ready! You can start using the assistant.[/bold green]", 
                                   style="green"))
            else:
                console.print(Panel(f"✅ [bold green]Ready to start![/bold green] ({warnings} non-critical warnings)", 
                                   style="yellow"))
        else:
            console.print(Panel(f"❌ [bold red]Please fix {critical_errors} critical issue(s) before starting.[/bold red]", 
                               style="red"))
    
    def get_auto_fix_suggestions(self) -> List[str]:
        """Get actionable commands for auto-fixing common issues."""
        suggestions = []
        
        for name, result in self.validation_results.items():
            if not result['status'] and result.get('auto_fix'):
                if name == "Environment File":
                    suggestions.append("cp .env-example .env")
                    suggestions.append("# Edit .env and add your ANTHROPIC_API_KEY")
        
        return suggestions


def validate_startup() -> bool:
    """
    Main validation function for startup with rich output.
    Returns True if validation passes, False otherwise.
    """
    validator = StartupValidator()
    success = validator.validate_all()
    validator.print_health_dashboard()
    
    if not success:
        auto_fixes = validator.get_auto_fix_suggestions()
        if auto_fixes:
            console.print("\n🚀 [bold cyan]Quick Auto-Fix Commands:[/bold cyan]")
            for cmd in auto_fixes:
                console.print(f"   {cmd}")
    
    return success


if __name__ == "__main__":
    success = validate_startup()
    sys.exit(0 if success else 1) 