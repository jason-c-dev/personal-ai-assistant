# Personal AI Assistant with Persistent Memory

> 🧠 An AI assistant that remembers who you are, building genuine relationships through persistent memory across conversations.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 What Makes This Different

Unlike traditional AI assistants that forget everything between conversations, this assistant:

- **Remembers You**: Maintains persistent memory of your preferences, ongoing projects, and conversation history
- **Learns Over Time**: Builds deeper understanding through every interaction
- **Transparent Memory**: You can see, edit, and control exactly what it remembers about you
- **Privacy-First**: All memory stored locally on your machine
- **Relationship Building**: Conversations feel natural and continuous, like talking to someone who knows you

## ✨ Key Features

### Core Capabilities
- 🧠 **Persistent Memory System** - Never repeat yourself again
- 🔄 **Contextual Conversations** - Picks up where you left off
- 📁 **Transparent File Storage** - Memory stored in readable Markdown files
- ⚡ **Intelligent Memory Management** - Automatically organizes and condenses memories
- 🛡️ **Privacy-Focused** - Local-first design with full user control

### Technical Features
- Built on [Strands Agents](https://strandsagents.com/) framework
- Model Context Protocol (MCP) integration
- Configurable AI providers (Anthropic Claude, OpenAI, AWS Bedrock)
- File-based memory with automatic time-windowed organization
- Chain-of-thought reasoning for memory decisions
- Rich CLI interface with conversation history

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- API key for your preferred AI provider (Anthropic Claude recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/personal-ai-assistant.git
   cd personal-ai-assistant
   ```

2. **Set up Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and preferences
   ```

5. **Run the assistant:**
   ```bash
   python src/main.py
   ```

## ⚙️ Configuration

### Environment Setup

Copy `.env.example` to `.env` and configure:

```bash
# Required: Your AI provider API key
ANTHROPIC_API_KEY=your_api_key_here

# Optional: Customize memory settings
MEMORY_BASE_PATH=~/assistant_memory
MEMORY_RECENT_DAYS=30
MEMORY_MEDIUM_DAYS=180

# Optional: Assistant personality
SYSTEM_PROMPT_FILE=config/system_prompts.json
```

### AI Provider Configuration

The assistant supports multiple AI providers:

#### Anthropic Claude (Default)
```bash
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

#### OpenAI
```bash
AI_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4-turbo-preview
```

#### AWS Bedrock
```bash
AI_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
```

## 💬 Usage Examples

### First Conversation
```
You: Hi, I'm Sarah and I'm a software engineer working on mobile apps.
Assistant: Nice to meet you, Sarah! I'll remember that you're a software engineer focusing on mobile apps. What kind of mobile development are you working on?

You: I'm building a React Native app for fitness tracking.
Assistant: That sounds interesting! I'll keep track of your React Native fitness app project. How's the development going?
```

### Later Conversation (Days/Weeks Later)
```
Assistant: Good morning, Sarah! How's your React Native fitness app coming along? Last time we talked, you mentioned you were working on the tracking features.

You: Actually, I decided to switch to Flutter instead.
Assistant: That's a significant change! I'll update my notes about your project. What made you decide to switch from React Native to Flutter?
```

## 🧠 Memory System

### Memory Organization

Your memories are stored in `~/assistant_memory/` with this structure:

```
~/assistant_memory/
├── core/
│   ├── user_profile.md           # Your basic info and preferences
│   ├── active_context.md         # Current conversation topics
│   ├── relationship_evolution.md # How our relationship has developed
│   ├── preferences_patterns.md   # Your communication style & preferences
│   └── life_context.md          # Work, interests, ongoing situations
├── interactions/
│   └── 2024-06/                 # Monthly conversation logs
├── condensed/
│   ├── recent/     (0-30 days)  # Full detail memories
│   ├── medium/     (30-180 days) # Summarized key points
│   └── archive/    (180+ days)  # Essential facts only
└── system/
    ├── config.json              # Memory management settings
    └── embeddings/              # Search indexes (if enabled)
```

### Memory Features

- **Automatic Organization**: Memories are automatically sorted by importance and time
- **Smart Condensation**: Older memories are summarized while preserving key facts
- **Chain-of-Thought**: You can see the assistant's reasoning for memory decisions
- **Full Control**: Edit or delete any memory file directly
- **Importance Scoring**: Memories rated 1-10 for retention priority

## 🛠️ Development

### Project Structure

```
personal-ai-assistant/
├── src/
│   ├── agent/              # Strands agent implementation
│   ├── memory/             # Memory management system
│   ├── mcp_servers/        # MCP server integrations
│   └── main.py            # CLI entry point
├── config/
│   ├── system_prompts.json # Assistant personality configuration
│   └── model_config.json   # AI model settings
├── tests/
│   ├── integration/        # Integration tests
│   └── unit/              # Unit tests
├── requirements.txt        # Python dependencies
├── .env.example           # Environment configuration template
└── README.md             # This file
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/unit/test_memory_manager.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 🔧 Advanced Configuration

### Custom System Prompts

Edit `config/system_prompts.json` to customize the assistant's personality:

```json
{
  "base_prompt": "You are a helpful AI assistant with persistent memory...",
  "memory_review_prompt": "Before responding, review relevant memories...",
  "conversation_style": "friendly_professional",
  "memory_importance_guidelines": "Rate memories 1-10 based on..."
}
```

### Memory Management Settings

Customize memory behavior in your `.env` file:

```bash
# Memory time windows (days)
MEMORY_RECENT_DAYS=30      # Full detail retention
MEMORY_MEDIUM_DAYS=180     # Summarized memories
MEMORY_ARCHIVE_DAYS=365    # Essential facts only

# Importance thresholds
MEMORY_HIGH_IMPORTANCE_THRESHOLD=7
MEMORY_MEDIUM_IMPORTANCE_THRESHOLD=4

# File size limits
MEMORY_MAX_FILE_SIZE_MB=5
```

## 🚨 Troubleshooting

### Common Issues

**Assistant doesn't remember previous conversations:**
- Check that `MEMORY_BASE_PATH` exists and is writable
- Verify memory files are being created in `~/assistant_memory/`
- Check logs for memory operation errors

**API key errors:**
- Ensure your `.env` file has the correct API key
- Verify the API key has sufficient permissions
- Check that the API provider is correctly set

**Slow response times:**
- Check your internet connection
- Verify API provider status
- Consider reducing `MAX_CONTEXT_TOKENS` if memory files are large

**Memory files corrupted:**
- Memory files are in Markdown format and can be manually edited
- Backup files are created automatically (if enabled)
- Delete corrupted files to reset that memory component

### Getting Help

1. Check the [Issues](https://github.com/yourusername/personal-ai-assistant/issues) page
2. Review memory files in `~/assistant_memory/` for debugging
3. Enable debug logging with `DEBUG=true` in `.env`
4. Check the assistant's reasoning in memory files' YAML frontmatter

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests and ensure code quality: `pytest && black . && flake8`
5. Submit a pull request

## 📊 Architecture

### High-Level Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │────│  Strands Agent   │────│  AI Provider    │
│                 │    │                  │    │ (Claude/GPT/etc)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   MCP Servers    │
                       │ ┌──────────────┐ │
                       │ │ Filesystem   │ │
                       │ │ Memory Search│ │
                       │ └──────────────┘ │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Memory Files     │
                       │ (Markdown + YAML)│
                       └──────────────────┘
```

### Key Components

- **Strands Agent**: Orchestrates conversation flow and memory operations
- **MCP Servers**: Handle file operations and memory search
- **Memory Manager**: Organizes, condenses, and maintains memory files
- **CLI Interface**: Rich terminal interface for user interaction

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Strands Agents](https://strandsagents.com/) for the agent framework
- [Model Context Protocol](https://modelcontextprotocol.io/) for standardized tool integration
- [Anthropic](https://www.anthropic.com/) for Claude AI capabilities
- The open-source community for inspiration and tools

## 🔄 Version History

- **v0.1.0** - Initial release with core memory functionality
- **v0.2.0** - Added intelligent memory condensation
- **v0.3.0** - Multi-provider support and improved CLI

---

**Built with ❤️ for meaningful AI relationships**

*Questions? Feedback? [Open an issue](https://github.com/yourusername/personal-ai-assistant/issues) or start a discussion!* 