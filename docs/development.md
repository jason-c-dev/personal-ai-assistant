# Development Guide

Complete guide for developers contributing to the Personal AI Assistant project.

## 🚀 Quick Start for Developers

### Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/yourusername/personal-ai-assistant.git
cd personal-ai-assistant

# 2. Create development environment  
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# 3. Setup development environment
cp .env.example .env
# Add your API keys for testing

# 4. Verify setup
python src/main.py --validate-only
pytest
```

### First Contribution

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make your changes
# ... code, tests, docs ...

# 3. Run quality checks
black . && flake8 . && mypy src/ && pytest

# 4. Commit and push
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name

# 5. Open Pull Request on GitHub
```

## 🏗️ Project Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Rich CLI      │────│  Strands Agent   │────│  AI Provider    │
│   Interface     │    │ (Native MCP)     │    │ (Claude/GPT)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        
         ▼                        ▼                        
┌─────────────────┐    ┌──────────────────┐               
│ Session Manager │    │   Memory System  │               
│ (Conversation   │    │ (File-based MD)  │               
│  Persistence)   │    │                  │               
└─────────────────┘    └──────────────────┘               
                                │                          
                                ▼                          
                       ┌──────────────────┐               
                       │   MCP Servers    │               
                       │ ┌──────────────┐ │ ←── Native Strands MCP
                       │ │ Memory       │ │ ←── Multiple server support
                       │ │ Filesystem   │ │ ←── Error isolation
                       │ │ External API │ │ ←── HTTP/SSE transports
                       │ └──────────────┘ │               
                       └──────────────────┘               
```

### Code Organization

```
src/
├── main.py                      # 🚀 Entry point & auto-configuration
├── agent/                       # 🤖 Core agent components
│   ├── core_agent.py           #     Main assistant with native Strands MCP
│   ├── agent_config.py         #     Configuration with MCP server management
│   ├── cli.py                  #     Rich CLI interface
│   └── session_manager.py      #     Session persistence
├── memory/                      # 🧠 Memory management system
│   ├── memory_manager.py       #     Core CRUD operations
│   ├── file_operations.py      #     File I/O abstraction
│   ├── memory_condensation.py  #     AI-powered summarization
│   ├── memory_prioritization.py#     Importance scoring
│   ├── memory_reasoning.py     #     Chain-of-thought decisions
│   ├── memory_cleanup.py       #     Maintenance & optimization
│   ├── memory_analytics.py     #     Usage statistics
│   └── memory_backup.py        #     Backup & recovery
├── mcp_servers/                 # 🔧 MCP server implementations
│   ├── filesystem_server.py    #     File system operations
│   └── memory_server.py        #     Memory search capabilities
└── utils/                       # 🛠️ Shared utilities
    └── config.py               #     Configuration helpers
```

## 🧩 Core Components

### 1. Main Entry Point (`src/main.py`)

**Purpose**: User-facing entry point with auto-configuration

**Key Features**:
- API key auto-detection
- Environment validation  
- Memory system initialization
- Provider auto-configuration
- First-time user experience

**Key Classes**:
- `StartupValidator`: System health checks
- `StartupManager`: Orchestrates initialization

### 2. Core Agent (`src/agent/core_agent.py`)

**Purpose**: Main AI assistant logic using Strands framework

**Key Features**:
- Conversation management
- Memory integration
- AI provider abstraction
- Tool integration

**Key Classes**:
- `PersonalAssistantAgent`: Main agent class
- `SessionContext`: Conversation state management

### 3. Memory System (`src/memory/`)

**Purpose**: Persistent memory with intelligent organization

**Key Components**:
- **MemoryManager**: Core CRUD operations for memory files
- **MemoryCondensation**: AI-powered summarization of old memories
- **MemoryPrioritization**: Intelligent importance scoring
- **MemoryReasoning**: Chain-of-thought for memory decisions
- **MemoryCleanup**: Automated maintenance and optimization

### 4. CLI Interface (`src/agent/cli.py`)

**Purpose**: Rich terminal interface using Rich library

**Key Features**:
- Interactive conversation
- Memory browsing commands
- Real-time status display
- Command system

### 5. Configuration (`src/agent/agent_config.py`)

**Purpose**: Centralized configuration management using Pydantic

**Key Features**:
- Type-safe configuration
- Environment variable binding
- Validation and defaults
- Multiple provider support

## 🧪 Testing Strategy

### Test Organization

```
tests/
├── integration/              # End-to-end tests
├── unit/                     # Component tests  
├── fixtures/                 # Test data
└── conftest.py              # Pytest configuration
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/            # Unit tests only
pytest tests/integration/     # Integration tests only

# Run specific test file
pytest tests/unit/test_memory_manager.py -v

# Run tests matching pattern
pytest -k "memory" -v
```

### Writing Tests

#### Unit Test Example

```python
# tests/unit/test_memory_manager.py
import pytest
from unittest.mock import Mock, patch
from src.memory.memory_manager import MemoryManager

class TestMemoryManager:
    @pytest.fixture
    def memory_manager(self):
        return MemoryManager(base_path="/tmp/test_memory")
    
    def test_store_memory(self, memory_manager):
        entry = {
            "content": "Test memory",
            "importance": 7.5,
            "timestamp": "2024-01-01T00:00:00"
        }
        
        result = memory_manager.store_memory(entry)
        
        assert result["success"] == True
        assert "memory_id" in result
```

#### Integration Test Example

```python
# tests/integration/test_full_conversation.py
import pytest
from src.agent.core_agent import PersonalAssistantAgent
from src.agent.agent_config import AgentConfig

class TestFullConversation:
    @pytest.fixture
    def agent(self):
        config = AgentConfig()
        agent = PersonalAssistantAgent(config)
        yield agent
        agent.cleanup()
    
    async def test_conversation_with_memory(self, agent):
        # First message
        response1 = await agent.process_message(
            "Hi, I'm Sarah and I work as a data scientist"
        )
        assert "Sarah" in response1
        
        # Follow-up message should remember
        response2 = await agent.process_message(
            "What did I tell you about my job?"
        )
        assert "data scientist" in response2.lower()
```

### Test Data Management

Create realistic test fixtures:

```python
# tests/fixtures/memory_data.py
@pytest.fixture
def sample_memory_entry():
    return {
        "content": "User mentioned working on machine learning project",
        "importance": 8.0,
        "timestamp": "2024-01-01T10:00:00Z",
        "tags": ["work", "ml", "project"],
        "memory_type": "interaction"
    }

@pytest.fixture
def sample_conversation_history():
    return [
        {"role": "user", "content": "Hi, I'm working on a ML project"},
        {"role": "assistant", "content": "Tell me more about your ML project"},
        {"role": "user", "content": "It's about image classification"}
    ]
```

## 🏃 Development Workflow

### Code Style and Quality

We use several tools to maintain code quality:

```bash
# Code formatting
black src/ tests/

# Import sorting
isort src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Run all quality checks
black . && isort . && flake8 . && mypy src/ && pytest
```

### Configuration Files

#### `.flake8`
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,venv,build,dist
```

#### `pyproject.toml`
```toml
[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.9"
disallow_untyped_defs = true
ignore_missing_imports = true
```

### Git Workflow

#### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

#### Commit Messages
Follow conventional commits:
```
feat: add memory analytics dashboard
fix: resolve API key validation issue
docs: update installation guide
refactor: simplify memory condensation logic
test: add integration tests for CLI
```

#### Pull Request Process
1. Create feature branch
2. Implement changes with tests
3. Run quality checks
4. Update documentation
5. Submit PR with clear description
6. Address review feedback
7. Merge after approval

## 🔧 Adding New Features

### Adding a New Memory Component

1. **Create the module**:
```python
# src/memory/new_feature.py
from typing import Dict, Any, List
from .memory_manager import MemoryManager

class NewMemoryFeature:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    def process_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Implementation here
        pass
```

2. **Add tests**:
```python
# tests/unit/test_new_feature.py
import pytest
from src.memory.new_feature import NewMemoryFeature

class TestNewMemoryFeature:
    def test_process_memories(self):
        # Test implementation
        pass
```

3. **Update configuration**:
```python
# src/agent/agent_config.py
class MemoryConfig(BaseModel):
    # Add new configuration options
    new_feature_enabled: bool = Field(default=True)
```

4. **Integrate with core agent**:
```python
# src/agent/core_agent.py
from ..memory.new_feature import NewMemoryFeature

class PersonalAssistantAgent:
    def __init__(self, config: AgentConfig):
        # Initialize new feature
        self.new_feature = NewMemoryFeature(self.memory_manager)
```

### Adding a New CLI Command

1. **Add command to CLI**:
```python
# src/agent/cli.py
def handle_command(self, command: str) -> str:
    if command.startswith('/newcommand'):
        return self._handle_new_command(command)
    # ... existing commands

def _handle_new_command(self, command: str) -> str:
    # Implementation here
    return "Command executed"
```

2. **Add tests**:
```python
# tests/unit/test_cli.py
def test_new_command_handling(self, cli):
    result = cli.handle_command("/newcommand test")
    assert "executed" in result
```

3. **Update help system**:
```python
# Update help text to include new command
HELP_TEXT = """
Available commands:
...
/newcommand - Description of new command
"""
```

## 🐛 Debugging

### Debug Configuration

```bash
# Enable debug mode
DEBUG=true python src/main.py --verbose

# Show memory operations
VERBOSE_MEMORY_OPERATIONS=true python src/main.py

# Enable development mode
DEV_MODE=true python src/main.py
```

### Logging

```python
import logging

# Use structured logging
logger = logging.getLogger(__name__)

# Different log levels
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning about potential issues")
logger.error("Error occurred")
```

### Memory System Debugging

```bash
# Inspect memory files directly
cat ~/.assistant_memory/core/user_profile.md

# Check memory system health
python -c "
from src.memory.memory_manager import MemoryManager
mm = MemoryManager()
print(mm.get_system_status())
"

# Validate memory file format
python -c "
import yaml
with open('~/.assistant_memory/core/user_profile.md') as f:
    content = f.read()
    # Parse YAML frontmatter
    print('Valid format')
"
```

### Performance Profiling

```bash
# Profile memory usage
pip install memory-profiler
python -m memory_profiler src/main.py

# Profile execution time
pip install line-profiler
kernprof -l -v src/main.py
```

## 📚 Documentation Standards

### Code Documentation

Use comprehensive docstrings:

```python
class MemoryManager:
    """
    Manages persistent memory storage and retrieval for the AI assistant.
    
    This class handles CRUD operations for memory files, automatic organization
    by importance and time, and integration with the AI agent for context.
    
    Args:
        base_path: Directory where memory files are stored
        config: Memory configuration settings
        
    Attributes:
        base_path: Path to memory directory
        file_ops: File operations handler
        
    Example:
        >>> manager = MemoryManager(base_path="~/.assistant_memory")
        >>> entry = {"content": "User likes coffee", "importance": 7.0}
        >>> result = manager.store_memory(entry)
        >>> print(result["memory_id"])
    """
    
    def store_memory(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a new memory entry in the appropriate location.
        
        Args:
            entry: Memory data with content, importance, and metadata
            
        Returns:
            Dict containing success status and memory_id
            
        Raises:
            MemoryStorageError: If storage operation fails
            ValidationError: If entry format is invalid
        """
        pass
```

### API Documentation

Document all public APIs:

```python
from typing import Dict, Any, List, Optional

def process_message(
    self, 
    message: str, 
    user_id: str = "default",
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Process a user message and return AI response with memory integration.
    
    Args:
        message: User's input message
        user_id: Unique identifier for the user session
        context: Additional context for processing
        
    Returns:
        AI assistant's response incorporating relevant memories
        
    Example:
        >>> agent = PersonalAssistantAgent()
        >>> response = await agent.process_message("How's my project going?")
        >>> print(response)
        "Based on our previous discussions about your ML project..."
    """
```

## 🔄 Release Process

### Version Management

We use semantic versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Checklist

1. **Update version numbers**:
   - `src/agent/agent_config.py`
   - `pyproject.toml` 
   - `README.md`

2. **Update documentation**:
   - CHANGELOG.md
   - Migration guides (if needed)
   - API documentation

3. **Quality assurance**:
   - All tests pass
   - Code coverage > 90%
   - Performance benchmarks
   - Security scan

4. **Create release**:
   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```

## 🤝 Contributing Guidelines

### What We Need

- 🐛 **Bug Reports**: Clear reproduction steps
- ✨ **Feature Requests**: Use cases and requirements  
- 📚 **Documentation**: Improvements and clarifications
- 🧪 **Tests**: Better coverage and edge cases
- 🎨 **UI/UX**: CLI experience improvements
- 🔧 **Integrations**: New AI providers or tools

### Code Review Process

1. **Automated checks** must pass
2. **At least one approving review** from maintainer
3. **All conversations resolved**
4. **Documentation updated** if needed
5. **Tests added** for new functionality

### Getting Help

- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/personal-ai-assistant/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/personal-ai-assistant/issues)
- 📧 **Email**: For security issues only
- 📖 **Docs**: Check existing documentation first

---

*Happy coding! Building meaningful AI relationships requires thoughtful development.* 🤖✨ 

## 🔧 MCP Server Development

### Native Strands MCP Integration

The Personal AI Assistant uses native Strands MCP integration (completed in Task 7.0), providing:

- **96% Code Reduction**: From 771 lines of custom implementation to ~50 lines
- **Multiple Server Support**: Connect to stdio, HTTP, and SSE MCP servers simultaneously
- **Error Isolation**: Individual server failures don't break the system
- **Tool Namespacing**: Automatic conflict resolution between server tools
- **Health Monitoring**: Built-in server status tracking and recovery

### Creating Custom MCP Servers

#### Basic Server Structure

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.models import InitializationOptions
import asyncio

# Create server instance
mcp = FastMCP("custom-server")

@mcp.tool()
def custom_operation(param: str) -> str:
    """
    Custom tool description that will appear to the AI agent.
    
    Args:
        param: Description of the parameter
        
    Returns:
        Description of what this tool returns
    """
    # Implement your logic here
    result = f"Processed: {param}"
    return result

@mcp.tool()
async def async_operation(data: dict) -> dict:
    """
    Async tools for I/O operations or external API calls.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Processed result dictionary
    """
    # Async operations like file I/O or API calls
    await asyncio.sleep(0.1)  # Simulate async work
    return {"status": "complete", "result": data}

if __name__ == "__main__":
    mcp.run()
```

#### Server Configuration

Add your server to the MCP configuration:

```json
{
  "mcp": {
    "enabled": true,
    "servers": [
      {
        "name": "custom_server",
        "transport": "stdio",
        "command": "python",
        "args": ["path/to/your/custom_server.py"],
        "timeout": 30,
        "enabled": true
      }
    ]
  }
}
```

### Testing MCP Servers

#### Unit Testing Individual Servers

```python
import pytest
import asyncio
from your_custom_server import mcp

@pytest.mark.asyncio
async def test_custom_operation():
    """Test individual server tools"""
    result = mcp.custom_operation("test_input")
    assert "Processed: test_input" in result

@pytest.mark.asyncio 
async def test_async_operation():
    """Test async server operations"""
    result = await mcp.async_operation({"key": "value"})
    assert result["status"] == "complete"
```

#### Integration Testing with Agent

```python
import pytest
from src.agent.core_agent import PersonalAssistantAgent
from src.agent.agent_config import AgentConfig

@pytest.mark.asyncio
async def test_mcp_server_integration():
    """Test that custom MCP server integrates with agent"""
    config = AgentConfig()
    agent = PersonalAssistantAgent(config)
    
    success = await agent.initialize()
    assert success, "Agent should initialize with MCP servers"
    
    # Check that custom server tools are available
    status = await agent.get_agent_status()
    server_names = [s["name"] for s in status["mcp_system"]["active_servers"]]
    assert "custom_server" in server_names
```

### Server Development Best Practices

#### Error Handling

```python
from mcp.server.fastmcp import FastMCP
from mcp.types import McpError, ErrorCode

mcp = FastMCP("robust-server")

@mcp.tool()
def robust_operation(param: str) -> str:
    """Tool with proper error handling"""
    try:
        if not param:
            raise McpError(
                ErrorCode.INVALID_PARAMS,
                "Parameter cannot be empty"
            )
        
        # Your operation logic
        result = process_param(param)
        return result
        
    except Exception as e:
        raise McpError(
            ErrorCode.INTERNAL_ERROR,
            f"Operation failed: {str(e)}"
        )
```

#### Resource Management

```python
@mcp.tool()
async def file_operation(filepath: str) -> str:
    """Proper resource management for file operations"""
    try:
        async with aiofiles.open(filepath, 'r') as f:
            content = await f.read()
        return content
    except FileNotFoundError:
        raise McpError(
            ErrorCode.INVALID_PARAMS,
            f"File not found: {filepath}"
        )
```

#### Performance Optimization

```python
import functools
import asyncio
from typing import Any

# Caching for expensive operations
@functools.lru_cache(maxsize=128)
def expensive_computation(param: str) -> str:
    """Cache results of expensive operations"""
    # Expensive computation here
    return result

# Rate limiting for external APIs
class RateLimiter:
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.last_call = 0
    
    async def acquire(self):
        now = asyncio.get_event_loop().time()
        time_since_last = now - self.last_call
        if time_since_last < 1.0 / self.calls_per_second:
            await asyncio.sleep(1.0 / self.calls_per_second - time_since_last)
        self.last_call = asyncio.get_event_loop().time()

rate_limiter = RateLimiter(10.0)  # 10 calls per second

@mcp.tool()
async def api_call(endpoint: str) -> dict:
    """Rate-limited API calls"""
    await rate_limiter.acquire()
    # Make API call
    return result
```

### Multiple Server Configurations

#### Development Setup

```json
{
  "mcp": {
    "enabled": true,
    "global_timeout": 30,
    "retry_attempts": 3,
    "servers": [
      {
        "name": "memory",
        "transport": "stdio",
        "command": "python",
        "args": ["src/mcp_servers/memory_server.py", "dev_memory"],
        "env": {"DEBUG": "1", "LOG_LEVEL": "DEBUG"},
        "enabled": true
      },
      {
        "name": "test_database",
        "transport": "stdio",
        "command": "python", 
        "args": ["dev/test_db_server.py"],
        "enabled": true
      },
      {
        "name": "mock_api",
        "transport": "http",
        "url": "http://localhost:8080/mcp",
        "enabled": false
      }
    ]
  }
}
```

#### Production Setup

```json
{
  "mcp": {
    "enabled": true,
    "servers": [
      {
        "name": "memory",
        "transport": "stdio",
        "command": "python",
        "args": ["src/mcp_servers/memory_server.py", "memory"],
        "enabled": true
      },
      {
        "name": "filesystem",
        "transport": "stdio",
        "command": "python",
        "args": ["src/mcp_servers/filesystem_server.py", "memory"],
        "enabled": true
      },
      {
        "name": "analytics_api",
        "transport": "http",
        "url": "https://analytics.company.com/mcp",
        "timeout": 45,
        "enabled": true
      },
      {
        "name": "realtime_updates",
        "transport": "sse",
        "url": "https://updates.company.com/mcp/stream",
        "timeout": 60,
        "enabled": true
      }
    ]
  }
}
```

### Monitoring and Debugging

#### Server Health Monitoring

```python
# Check server status programmatically
from src.agent.core_agent import PersonalAssistantAgent

async def monitor_mcp_health():
    agent = PersonalAssistantAgent()
    await agent.initialize()
    
    status = await agent.get_agent_status()
    mcp_status = status["mcp_system"]
    
    print(f"Health: {mcp_status['health_summary']['health_percentage']:.1f}%")
    print(f"Servers: {mcp_status['health_summary']['operational_servers']}/{mcp_status['health_summary']['total_servers']}")
    
    for server in mcp_status["active_servers"]:
        status_icon = "✅" if server["status"] == "operational" else "❌"
        print(f"  {status_icon} {server['name']}: {server['tools_count']} tools")
```

#### Debug Mode

```bash
# Enable comprehensive MCP debugging
export DEBUG=true
export MCP_DEBUG=true
export LOG_LEVEL=DEBUG

python -m src.main
```

#### Server Logs

```python
import logging

# Configure logging in your server
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("custom-server")

@mcp.tool()
def logged_operation(param: str) -> str:
    """Tool with comprehensive logging"""
    logger.debug(f"Starting operation with param: {param}")
    
    try:
        result = process_param(param)
        logger.info(f"Operation completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise
```

### Migration from Custom Implementation

If you were using the old custom MCP implementation (removed in Task 7.0):

#### What Changed
- Removed `MCPClient`, `StrandsMCPTools`, `MCPIntegration` classes
- Simplified configuration format to support multiple servers
- Native Strands context manager handling
- Automatic tool discovery and registration

#### Migration Steps
1. **Update imports**: Remove custom MCP class imports
2. **Update configuration**: Use new multiple server format
3. **Update server implementations**: Ensure compatibility with FastMCP
4. **Test integration**: Verify tools work with native Strands integration

#### Old vs New Patterns

**Old (Custom Implementation):**
```python
# Don't use - removed in Task 7.0
from src.agent.mcp_client import MCPClient
from src.agent.strands_mcp_tools import StrandsMCPTools

mcp_client = MCPClient(memory_path) 
await mcp_client.initialize()
tools = StrandsMCPTools(mcp_client).get_tools()
```

**New (Native Strands):**
```python
# Use this pattern instead
from strands.tools.mcp import MCPClient
from mcp.client.stdio import stdio_client, StdioServerParameters

server_params = StdioServerParameters(
    command="python",
    args=["src/mcp_servers/memory_server.py"]
)

client = MCPClient(lambda: stdio_client(server_params))
with client:
    tools = client.list_tools_sync()
    agent = Agent(tools=tools)
```

For complete MCP integration documentation, see [MCP Integration Guide](mcp-integration.md). 