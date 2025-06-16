# Personal AI Assistant Documentation

Welcome to the Personal AI Assistant documentation! This AI assistant provides **persistent memory** across conversations, learning about you and your projects over time to become increasingly helpful and personalized.

## 📚 Documentation Overview

### Getting Started
- **[Main README](../README.md)** - Quick setup and basic usage
- **[Configuration Guide](configuration.md)** - Complete setup and customization options
- **[Memory System Deep Dive](memory-system.md)** - Understanding how memories work

### Advanced Topics  
- **[Development Guide](development.md)** - Contributing, architecture, and code standards
- **[MCP Integration Guide](mcp-integration.md)** - Model Context Protocol servers and custom tools

## 🚀 Quick Start

1. **30-Second Setup**: Follow the [Quick Start](../README.md#-quick-start) in the main README
2. **First Conversation**: Try the interactive demos when you first run the assistant
3. **Customize**: Use the [Configuration Guide](configuration.md) to personalize your setup

## 🧠 Key Features

### Persistent Memory
- **Learns About You**: Remembers your role, projects, preferences, and context
- **Cross-Session Continuity**: Picks up conversations from weeks or months ago
- **Smart Organization**: Automatically organizes memories by importance and time
- **Full Control**: Edit or delete any memory file in plain Markdown

### Intelligent Management
- **Auto-Condensation**: Older memories are summarized while preserving key facts
- **Importance Scoring**: Memories rated 1-10 for retention priority
- **Chain-of-Thought**: See the assistant's reasoning for memory decisions
- **Backup & Recovery**: Automatic backups with easy restoration

### Professional CLI
- **Rich Interface**: Beautiful terminal experience with colors and formatting
- **Memory Browser**: View and manage your memory files directly from the CLI
- **Session Management**: Save and restore conversation states
- **Error Recovery**: Graceful handling of issues with helpful suggestions

## 🎯 Use Cases

### Personal Productivity
- **Project Tracking**: Remembers your ongoing projects and their status
- **Learning Companion**: Tracks your learning journey and adapts teaching style
- **Decision Support**: Provides context-aware advice based on your history

### Professional Work
- **Team Collaboration**: Maintains context about team members and project details
- **Technical Support**: Learns your tech stack and common issues
- **Meeting Prep**: Recalls previous discussions and action items

### Creative Projects
- **Writing Assistant**: Remembers your style, ongoing projects, and feedback
- **Idea Development**: Tracks concept evolution over time
- **Research Helper**: Organizes findings and connects related information

## 📖 Documentation Structure

```
docs/
├── README.md              # This overview (you are here)
├── configuration.md       # Complete setup guide
├── memory-system.md       # Memory organization & control
├── mcp-integration.md     # MCP servers and custom tools
└── development.md         # Architecture & contributing
```

## 🔗 External Resources

- **[Strands Agent Framework](https://strandsagents.com/latest/)** - The AI agent framework powering the assistant
- **[Model Context Protocol](https://modelcontextprotocol.io/)** - How the assistant interfaces with your files
- **[GitHub Repository](https://github.com/jason-c-dev/personal-ai-assistant)** - Source code and issue tracking

## 💡 Getting Help

### Self-Help
1. **Memory Issues**: Check `~/.assistant_memory/` files directly
2. **Configuration Problems**: Review [Configuration Guide](configuration.md)
3. **Technical Issues**: See [Development Guide](development.md) troubleshooting section

### Community Support
1. **[GitHub Issues](https://github.com/jason-c-dev/personal-ai-assistant/issues)** - Bug reports and feature requests
2. **[Discussions](https://github.com/jason-c-dev/personal-ai-assistant/discussions)** - Questions and community help

### Debug Information
Enable debug logging in your `.env` file:
```bash
LOG_LEVEL=DEBUG
MEMORY_DEBUG=true
MCP_DEBUG=true
```

## 🎉 What Makes This Special

Unlike other AI assistants that forget everything between conversations, this assistant:

- **Remembers Your Context**: Knows your role, projects, and preferences
- **Evolves With You**: Adapts communication style based on interaction history  
- **Maintains Relationships**: Tracks how you work together over time
- **Stays Organized**: Automatically manages memory without overwhelming you
- **Gives You Control**: All memories stored in readable Markdown files you can edit

Ready to get started? Head back to the [main README](../README.md) for setup instructions! 