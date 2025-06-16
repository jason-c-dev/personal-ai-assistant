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
│   Interface     │    │  (Core Logic)    │    │ (Claude/GPT)    │
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
                       │ ┌──────────────┐ │               
                       │ │ Filesystem   │ │               
                       │ │ Memory Search│ │               
                       │ └──────────────┘ │               
                       └──────────────────┘               
```

### Code Organization

```
src/
├── main.py                      # 🚀 Entry point & auto-configuration
├── agent/                       # 🤖 Core agent components
│   ├── core_agent.py           #     Main assistant logic
│   ├── agent_config.py         #     Configuration management
│   ├── cli.py                  #     Rich CLI interface
│   ├── session_manager.py      #     Session persistence
│   ├── mcp_client.py           #     MCP server communication
│   └── strands_mcp_tools.py    #     Strands-MCP integration
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