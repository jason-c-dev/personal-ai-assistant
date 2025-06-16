# Configuration Guide

Complete guide to configuring your Personal AI Assistant for optimal performance and personalization.

## 🎯 Quick Configuration

### Essential Setup (Required)

The assistant needs only an API key to get started:

```bash
# Create .env file with your API key
echo "ANTHROPIC_API_KEY=sk-ant-your_key_here" > .env

# That's it! Everything else is auto-configured
python src/main.py
```

### Auto-Detection Features

The assistant automatically detects and configures:

- **AI Provider**: From your API key format (Anthropic/OpenAI/AWS)
- **Optimal Model**: Best model for your provider
- **Memory Location**: Creates `~/.assistant_memory/` directory
- **Configuration Files**: Generates optimal `.env` settings

## 📋 Complete Configuration Reference

### AI Provider Settings

#### Anthropic Claude (Recommended)
```bash
# Required
ANTHROPIC_API_KEY=sk-ant-your_key_here

# Optional - auto-configured to optimal values
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022-v2:0
ANTHROPIC_MAX_TOKENS=4000
ANTHROPIC_TEMPERATURE=0.7
```

#### OpenAI GPT
```bash
# Required  
OPENAI_API_KEY=sk-your_key_here

# Optional - auto-configured to optimal values
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.7
```

#### AWS Bedrock
```bash
# Required - AWS credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Optional
AWS_REGION=us-west-2
BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

### Memory System Configuration

#### Basic Memory Settings
```bash
# Memory location (default: ~/.assistant_memory)
MEMORY_BASE_PATH=~/my_ai_memory

# Time-based organization (days)
MEMORY_RECENT_DAYS=30      # Full detail retention
MEMORY_MEDIUM_DAYS=180     # Summarized retention  
MEMORY_ARCHIVE_DAYS=365    # Essential facts only
```

#### Memory Management
```bash
# Importance scoring thresholds (1-10 scale)
MEMORY_HIGH_IMPORTANCE_THRESHOLD=7.0
MEMORY_MEDIUM_IMPORTANCE_THRESHOLD=4.0
MEMORY_LOW_IMPORTANCE_THRESHOLD=2.0

# File size limits
MEMORY_MAX_FILE_SIZE_MB=5
MEMORY_MAX_INTERACTION_LENGTH=10000

# Automatic cleanup settings
MEMORY_AUTO_CLEANUP_ENABLED=true
MEMORY_CLEANUP_INTERVAL_DAYS=30
```

### Assistant Personality

#### Basic Personality Settings
```bash
# Assistant identity
AGENT_NAME="My AI Assistant"
AGENT_VERSION="1.0.0"

# Communication style
CONVERSATION_STYLE=friendly_professional
# Options: friendly_professional, casual, formal, technical

# Response preferences
RESPONSE_LENGTH=medium
# Options: brief, medium, detailed

MEMORY_SHARING_STYLE=contextual
# Options: explicit, contextual, minimal
```

#### Advanced Personality Configuration

Edit `config/system_prompts.json` for detailed personality customization:

```json
{
  "base_prompt": "You are a helpful AI assistant with persistent memory...",
  "memory_review_prompt": "Before responding, review relevant memories...",
  "conversation_style": "friendly_professional",
  "memory_importance_guidelines": "Rate memories 1-10 based on...",
  "relationship_building": {
    "remember_preferences": true,
    "track_goals": true,
    "note_communication_style": true
  }
}
```

### CLI Interface Settings

#### Display & Interaction
```bash
# CLI appearance
CLI_THEME=default
# Options: default, dark, light, colorful

CLI_PROMPT_STYLE=friendly
# Options: friendly, professional, minimal

# Terminal behavior
CLI_AUTO_SCROLL=true
CLI_HISTORY_SIZE=1000
CLI_ENABLE_RICH_DISPLAY=true
```

#### Command Behavior
```bash
# Startup behavior
CLI_SHOW_WELCOME=true
CLI_SHOW_MEMORY_STATS=true
CLI_AUTO_LOAD_CONTEXT=true

# Session management
SESSION_AUTO_SAVE=true
SESSION_TIMEOUT_MINUTES=60
```

### Performance & Technical Settings

#### Model Performance
```bash
# Context and generation limits
MAX_CONTEXT_TOKENS=8000
MAX_GENERATION_TOKENS=4000

# Response timing
REQUEST_TIMEOUT_SECONDS=30
RETRY_ATTEMPTS=3
RETRY_DELAY_SECONDS=1
```

#### MCP Server Configuration
```bash
# Enable/disable MCP features
MCP_ENABLED=true
MCP_FILESYSTEM_ENABLED=true
MCP_MEMORY_SEARCH_ENABLED=true

# MCP server settings
MCP_SERVER_PORT=8000
MCP_LOG_LEVEL=INFO
```

### Logging & Debugging

#### Log Configuration
```bash
# Logging levels
LOG_LEVEL=INFO
# Options: DEBUG, INFO, WARNING, ERROR

# Log destinations
LOG_TO_FILE=false
LOG_FILE_PATH=./logs/assistant.log

# Debug features
DEBUG=false
VERBOSE_MEMORY_OPERATIONS=false
SHOW_REASONING_CHAIN=false
```

#### Development Settings
```bash
# Development features
DEV_MODE=false
ENABLE_MEMORY_ANALYTICS=true
ENABLE_PERFORMANCE_TRACKING=false

# Testing
TEST_MODE=false
MOCK_AI_RESPONSES=false
```

## 🔧 Advanced Configuration

### Custom Memory Templates

Create custom memory file templates in `config/memory_templates/`:

```markdown
# user_profile_template.md
---
memory_type: core
importance: 10
created: {timestamp}
last_updated: {timestamp}
---

# User Profile

## Basic Information
- **Name**: {user_name}
- **Role**: {user_role}
- **Interests**: {user_interests}

## Communication Preferences
- **Style**: {communication_style}
- **Detail Level**: {detail_preference}

## Current Context
{current_context}
```

### Environment-Specific Configurations

#### Development Environment
```bash
# .env.development
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-dev-key
DEBUG=true
LOG_LEVEL=DEBUG
MEMORY_BASE_PATH=./dev_memory
```

#### Production Environment  
```bash
# .env.production
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-prod-key
DEBUG=false
LOG_LEVEL=INFO
MEMORY_BASE_PATH=~/.assistant_memory
```

### Integration Settings

#### API Rate Limiting
```bash
# Rate limiting (requests per minute)
ANTHROPIC_RATE_LIMIT=60
OPENAI_RATE_LIMIT=60

# Batch processing
ENABLE_BATCH_PROCESSING=false
BATCH_SIZE=10
```

#### External Integrations
```bash
# Calendar integration (future feature)
CALENDAR_ENABLED=false
CALENDAR_PROVIDER=google

# Task management (future feature)  
TASK_MANAGEMENT_ENABLED=false
TASK_PROVIDER=notion
```

## 🛠️ Configuration Management

### Configuration Validation

Check your configuration:
```bash
# Validate current settings
python src/main.py --validate-only

# Show effective configuration
python src/main.py --show-config

# Test API connectivity
python src/main.py --test-api
```

### Configuration Backup & Restore

```bash
# Backup current configuration
cp .env .env.backup
cp -r config/ config_backup/

# Restore configuration
cp .env.backup .env
cp -r config_backup/ config/
```

### Environment-Based Configuration

```bash
# Use different configs for different environments
export AI_ENV=development
python src/main.py  # Uses .env.development

export AI_ENV=production  
python src/main.py  # Uses .env.production
```

## 🚨 Troubleshooting Configuration

### Common Configuration Issues

#### API Key Problems
```bash
# Test your API key
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
     https://api.anthropic.com/v1/messages \
     -d '{"model":"claude-3-sonnet-20240229","max_tokens":10,"messages":[{"role":"user","content":"test"}]}'
```

#### Memory Directory Issues
```bash
# Check permissions
ls -la ~/.assistant_memory/

# Fix permissions
chmod -R 755 ~/.assistant_memory/

# Recreate if corrupted
rm -rf ~/.assistant_memory/
python src/main.py  # Auto-recreates
```

#### Configuration File Issues
```bash
# Validate JSON configuration files
python -m json.tool config/system_prompts.json
python -m json.tool config/model_config.json

# Reset to defaults
cp config/system_prompts.json.example config/system_prompts.json
cp config/model_config.json.example config/model_config.json
```

### Getting Help

1. **Validate**: `python src/main.py --validate-only`
2. **Check logs**: `DEBUG=true python src/main.py --verbose`
3. **Reset config**: Delete `.env` and restart for auto-configuration
4. **Ask for help**: [Open an issue](https://github.com/yourusername/personal-ai-assistant/issues) with your configuration details

## 📚 Configuration Examples

### Minimal Setup (Just Works)
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

### Power User Setup
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022-v2:0
MEMORY_BASE_PATH=~/ai_memories
AGENT_NAME="Claude Memory Assistant"
CONVERSATION_STYLE=technical
DEBUG=false
CLI_THEME=dark
MEMORY_HIGH_IMPORTANCE_THRESHOLD=8.0
```

### Multi-Provider Setup
```bash
# .env  
# Primary provider
ANTHROPIC_API_KEY=sk-ant-your_key_here

# Backup providers
OPENAI_API_KEY=sk-your_openai_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret

# Let assistant choose best provider per request
AUTO_PROVIDER_SELECTION=true
```

---

*This configuration guide covers all available options. Start with the basics and customize as needed!* 