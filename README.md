# Personal AI Assistant with Persistent Memory

> 🧠 **30-Second Setup** • An AI assistant that remembers who you are, builds relationships through persistent memory, and gets smarter with every conversation.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 What Makes This Different

Unlike traditional AI assistants that forget everything between conversations, this assistant:

- **🧠 Remembers Everything**: Maintains persistent memory of your preferences, projects, and conversation history
- **📈 Learns Over Time**: Builds deeper understanding through every interaction
- **👁️ Transparent Memory**: You can view, edit, and control exactly what it remembers
- **🏠 Privacy-First**: All memory stored locally on your machine - nothing sent to external servers
- **💬 Natural Conversations**: Feels like talking to someone who knows you and your context
- **⚡ 30-Second Setup**: Just add your API key and start chatting

## 🚀 30-Second Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/jason-c-dev/personal-ai-assistant.git
cd personal-ai-assistant
python -m venv venv && source venv/bin/activate

# 2. Install (choose your speed)
pip install -r requirements-minimal.txt      # Fast: ~15MB, essentials only
# OR pip install -r requirements.txt         # Standard: ~50MB, includes dev tools

# 3. Add your API key to .env
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# 4. Start chatting!
python -m src.mainy
```

**That's it!** The assistant will auto-configure everything else and create your memory system on first run.

## 📦 Installation Options

Choose the installation that fits your needs:

### 🚀 **Minimal (Recommended for Users)**
```bash
pip install -r requirements-minimal.txt
```
- **Size**: ~15MB download
- **Contains**: Core runtime dependencies only
- **Perfect for**: Just want to chat with the assistant

### 📦 **Standard (Recommended for Contributors)** 
```bash
pip install -r requirements.txt
```
- **Size**: ~50MB download  
- **Contains**: Core + development/testing tools
- **Perfect for**: Contributing code or running tests

### 🔧 **Development (Full Features)**
```bash
pip install -r requirements-dev.txt
```
- **Size**: ~200MB download
- **Contains**: Everything + ML features (vector search, embeddings)
- **Perfect for**: Advanced features and research

## ⚙️ Configuration

### 🎯 Essential Setup (Required)

The assistant auto-detects your AI provider from your API key format:

```bash
# Anthropic Claude (Recommended)
ANTHROPIC_API_KEY=sk-ant-your_key_here

# OpenAI GPT
OPENAI_API_KEY=sk-your_key_here  

# AWS Bedrock (auto-detected from AWS credentials)
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
```

### 🎨 Optional Customization

**Auto-created `.env` file** includes sensible defaults. Customize if needed:

```bash
# Memory location (default: ~/.assistant_memory)
MEMORY_BASE_PATH=~/my_ai_memory

# Assistant personality (default: friendly and helpful)
AGENT_NAME="My AI Assistant"

# Advanced: AI model selection (auto-configured by provider)
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022-v2:0
OPENAI_MODEL=gpt-4-turbo-preview
```

**See [Configuration Guide](docs/configuration.md) for all options.**

## 💬 Real Usage Examples

### First Meeting
```
You: Hi! I'm Alex, a product manager at a startup working on a mobile fitness app.
Assistant: Nice to meet you, Alex! I'll remember that you're a product manager at a startup working on mobile fitness apps. What challenges are you facing with the app development?

You: We're struggling with user retention. People download it but stop using it after a week.
Assistant: I'll make a note about your user retention challenge. Have you identified what's causing users to drop off? Is it the onboarding, the features, or something else?
```

### Returning User (Weeks Later)
```
Assistant: Welcome back, Alex! How's the user retention issue going with your fitness app? Last time we discussed strategies for improving the onboarding experience.

You: We implemented your suggestions and retention improved by 40%!
Assistant: That's fantastic news! I'm updating my notes with this success. What specific changes made the biggest impact?
```

### Memory in Action
```
You: I'm thinking about changing careers from product management to UX design.
Assistant: That's a significant shift, Alex! I remember you've been working as a PM at your startup with the fitness app. What's driving your interest in UX design? Have you noticed design challenges in your current role?

You: Yes, exactly! Working on the app made me realize I love the design problem-solving aspect.
Assistant: That makes perfect sense - your experience with user retention issues probably gave you insights into user experience. I'll keep track of this career exploration. Are you looking into formal UX training?
```

## 🧠 Memory System

### Memory Organization

Your memories are stored in `~/.assistant_memory/` with this structure:

```
~/.assistant_memory/
├── core/                          # 🎯 Essential facts about you
│   ├── user_profile.md           #     Basic info, role, interests
│   ├── active_context.md         #     Current projects & topics
│   ├── relationship_evolution.md #     How you work with the assistant
│   ├── preferences_patterns.md   #     Communication style & habits
│   └── life_context.md          #     Work, goals, important situations
├── interactions/                  # 📚 Full conversation logs
│   ├── 2024-12/
│   │   ├── 2024-12-15-conversation-001.md
│   │   └── 2024-12-15-conversation-002.md
│   └── 2024-11/
├── condensed/                     # 🗂️ Time-organized summaries
│   ├── recent/     (0-30 days)   #     Full detail, current relevance
│   ├── medium/     (30-180 days) #     Key points, important context
│   └── archive/    (180+ days)   #     Essential facts only
└── system/                        # ⚙️ Assistant operations
    ├── config.json               #     Memory settings
    ├── analytics.db              #     Usage patterns (optional)
    └── backups/                  #     Automatic memory backups
```

### Memory Features

- **Automatic Organization**: Memories are automatically sorted by importance and time
- **Smart Condensation**: Older memories are summarized while preserving key facts
- **Chain-of-Thought**: You can see the assistant's reasoning for memory decisions
- **Full Control**: Edit or delete any memory file directly
- **Importance Scoring**: Memories rated 1-10 for retention priority

## 🛠️ Development

### Architecture

**🎯 Native Strands MCP Integration**: Uses Strands agent SDK's native MCP (Model Context Protocol) integration for robust, production-ready tool access:

- **Memory Server**: Provides semantic search and memory management through MCP
- **Filesystem Server**: Handles file operations through standardized MCP protocol  
- **Multiple Server Support**: Can connect to additional MCP servers simultaneously
- **Error Isolation**: Individual server failures don't break the entire system
- **Tool Namespacing**: Prevents conflicts between servers with similar tools

### Project Structure

```
personal-ai-assistant/
├── src/
│   ├── agent/              # Strands agent with native MCP integration
│   │   ├── core_agent.py   # Main agent using Strands framework
│   │   ├── agent_config.py # Configuration with MCP server management
│   │   └── cli.py          # Command-line interface
│   ├── memory/             # Memory management system
│   │   ├── memory_manager.py     # Core memory operations
│   │   ├── memory_condensation.py  # AI-powered summarization
│   │   ├── memory_prioritization.py # Intelligent prioritization
│   │   └── memory_*.py     # Specialized memory components
│   ├── mcp_servers/        # MCP server implementations
│   │   ├── memory_server.py      # Memory search and management
│   │   └── filesystem_server.py  # File operations
│   └── main.py             # CLI entry point
├── config/
│   ├── system_prompts.json       # Assistant personality
│   ├── model_config.json         # AI model settings
│   └── mcp_multi_server_example.json  # Example multi-server config
├── docs/                          # Complete documentation
├── tests/                         # Comprehensive test suite
├── requirements*.txt              # Python dependencies
└── .env-example                   # Environment configuration
```

### MCP Server Configuration

The assistant automatically configures essential MCP servers, but you can add more:

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
        "name": "external_api",
        "transport": "http",
        "url": "https://api.example.com/mcp",
        "enabled": false
      }
    ]
  }
}
```

**Transport Types Supported:**
- **stdio**: Local Python servers (default for memory/filesystem)
- **http**: Remote HTTP APIs 
- **sse**: Server-Sent Events for real-time services

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
- Verify memory files are being created in `~/.assistant_memory/`
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

**📚 [Complete Documentation](docs/README.md)** - Full guides and advanced configuration

**Quick Help:**
1. **Memory Issues**: Check files in `~/.assistant_memory/` directly
2. **Configuration**: See [Configuration Guide](docs/configuration.md) 
3. **Memory System**: Read [Memory System Guide](docs/memory-system.md)
4. **Development**: Check [Development Guide](docs/development.md)

**Debug Logging** - Add to your `.env` file:
```bash
LOG_LEVEL=DEBUG
MEMORY_DEBUG=true
MCP_DEBUG=true
```

**Community Support:**
- [GitHub Issues](https://github.com/jason-c-dev/personal-ai-assistant/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/jason-c-dev/personal-ai-assistant/discussions) - Questions and community help

---

**Ready to have AI conversations that actually remember you? Get started with the 30-second setup above! 🚀**