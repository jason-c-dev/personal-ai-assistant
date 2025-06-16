# MCP Integration Guide

Complete guide to the Model Context Protocol (MCP) integration in the Personal AI Assistant.

## Overview

The Personal AI Assistant uses native Strands MCP integration to provide robust, extensible tool access. This system replaced our previous custom implementation with a 96% code reduction while improving reliability and capabilities.

## Architecture

### Native Strands Integration

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
                       │ │ Memory       │ │ ←── stdio transport
                       │ │ Filesystem   │ │ ←── stdio transport  
                       │ │ External API │ │ ←── HTTP transport
                       │ │ Real-time    │ │ ←── SSE transport
                       │ └──────────────┘ │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Memory Files     │
                       │ (Markdown + YAML)│
                       └──────────────────┘
```

## Default Servers

### Memory Server
- **Purpose**: Semantic search and memory management
- **Transport**: stdio (local Python process)
- **Tools**: `memory_search_memories`, `memory_store_memory`, `memory_get_memory_stats`
- **Auto-configured**: Yes

### Filesystem Server  
- **Purpose**: File operations within memory directory
- **Transport**: stdio (local Python process)
- **Tools**: `filesystem_read_file`, `filesystem_write_file`, `filesystem_list_files`, `filesystem_search_files`
- **Auto-configured**: Yes

## Multiple Server Support

### Configuration Format

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
        "args": ["src/mcp_servers/memory_server.py", "memory"],
        "timeout": 30,
        "enabled": true
      },
      {
        "name": "filesystem",
        "transport": "stdio",
        "command": "python",
        "args": ["src/mcp_servers/filesystem_server.py", "memory"],
        "timeout": 30,
        "enabled": true
      }
    ]
  }
}
```

### Error Isolation

Each server operates independently:
- Memory server failure doesn't affect filesystem operations
- External API downtime doesn't break core functionality  
- Individual server timeouts don't crash the system

### Tool Namespacing

Tools are automatically namespaced to prevent conflicts:
- `memory_search_memories` vs `filesystem_search_files`
- `[memory] Search through stored memories` vs `[filesystem] Search for files`

## Transport Types

### stdio Transport
**Best for**: Local Python servers, core functionality
```json
{
  "name": "memory",
  "transport": "stdio", 
  "command": "python",
  "args": ["src/mcp_servers/memory_server.py", "memory"],
  "env": {"DEBUG": "1"}
}
```

### HTTP Transport  
**Best for**: Remote APIs, external services
```json
{
  "name": "external_api",
  "transport": "http",
  "url": "https://api.example.com/mcp",
  "timeout": 45
}
```

### SSE Transport
**Best for**: Real-time data, live updates
```json
{
  "name": "realtime_feed", 
  "transport": "sse",
  "url": "https://stream.example.com/mcp/sse",
  "timeout": 60
}
```

## Development Guide

### Creating Custom MCP Servers

1. **Use FastMCP Framework:**
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def my_custom_tool(param: str) -> str:
    """
    Custom tool description.
    
    Args:
        param: Parameter description
        
    Returns:
        Result description
    """
    return f"Processed: {param}"

if __name__ == "__main__":
    mcp.run()
```

2. **Add Server Configuration:**
```json
{
  "name": "my_server",
  "transport": "stdio", 
  "command": "python",
  "args": ["path/to/my_server.py"],
  "enabled": true
}
```

### Testing MCP Servers

```bash
# Test individual server
python src/mcp_servers/memory_server.py

# Test agent with servers
python -c "
import asyncio
from src.agent.core_agent import PersonalAssistantAgent

async def test():
    agent = PersonalAssistantAgent()
    await agent.initialize()
    status = await agent.get_agent_status()
    print(status['mcp_system'])

asyncio.run(test())
"
```

## Health Monitoring

### Server Status
```python
# Get comprehensive MCP status
status = await agent.get_agent_status()
mcp_status = status['mcp_system']

print(f"Active servers: {mcp_status['health_summary']['operational_servers']}")
print(f"Total tools: {mcp_status['health_summary']['total_tools']}")

# Check individual servers
for server in mcp_status['active_servers']:
    print(f"{server['name']}: {server['status']} ({server['tools_count']} tools)")
```

### Configuration Management
```python
# Enable/disable servers dynamically
agent.config.mcp.enable_server("memory")
agent.config.mcp.disable_server("external_api")

# Get health overview  
health = agent.config.mcp.get_server_health_status()
print(f"Total: {health['total_servers']}, Enabled: {health['enabled_servers']}")
```

## Advanced Examples

### Multi-Environment Setup

**Development Configuration:**
```json
{
  "mcp": {
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
        "name": "test_api",
        "transport": "http", 
        "url": "http://localhost:8080/mcp",
        "enabled": false
      }
    ]
  }
}
```

**Production Configuration:**
```json
{
  "mcp": {
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
        "enabled": true
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

**Server Not Starting:**
- Check command path and arguments
- Verify Python environment has required dependencies
- Check server logs for startup errors
- Ensure memory directory exists and is writable

**Tool Not Found:**
- Verify server is enabled and operational
- Check tool naming (may have namespace prefix like `memory_search_memories`)
- Confirm server implements expected tools
- Check server health status

**Connection Timeouts:**
- Increase server timeout setting in configuration
- Check network connectivity for HTTP/SSE transports
- Verify server is responsive and not overloaded
- Check firewall settings for remote servers

**Memory/Performance Issues:**
- Monitor server resource usage
- Check for memory leaks in custom servers
- Adjust timeout and retry settings
- Consider disabling unused servers

### Debug Mode
```bash
# Enable detailed MCP logging
export DEBUG=true
export MCP_DEBUG=true
export LOG_LEVEL=DEBUG

python -m src.main
```

### Health Check Commands
```bash
# Quick server health check
python -c "
import asyncio
from src.agent.agent_config import AgentConfig

async def health_check():
    config = AgentConfig()
    health = config.mcp.get_server_health_status()
    print(f'Servers: {health[\"total_servers\"]} total, {health[\"enabled_servers\"]} enabled')
    for server in health['server_list']:
        print(f'  - {server[\"name\"]}: {\"✅\" if server[\"enabled\"] else \"❌\"} ({server[\"transport\"]})')

asyncio.run(health_check())
"

# Detailed status with tool counts
python -c "
import asyncio
from src.agent.core_agent import PersonalAssistantAgent

async def detailed_status():
    agent = PersonalAssistantAgent()
    success = await agent.initialize()
    if success:
        status = await agent.get_agent_status()
        mcp = status['mcp_system']
        if 'health_summary' in mcp:
            print(f'MCP Health: {mcp[\"health_summary\"][\"health_percentage\"]:.1f}%')
            print(f'Operational: {mcp[\"health_summary\"][\"operational_servers\"]}/{mcp[\"health_summary\"][\"total_servers\"]} servers')
            print(f'Total tools: {mcp[\"health_summary\"][\"total_tools\"]}')

asyncio.run(detailed_status())
"
```

## Migration Notes

This system replaced our previous custom MCP implementation with significant improvements:

### What Changed
- **Code Reduction**: 96% fewer lines (771 → ~50 lines)
- **Reliability**: Using battle-tested Strands implementation
- **Features**: Native multiple server support, error isolation
- **Compatibility**: All existing memory/filesystem tools work unchanged

### Migration Benefits
- **Maintainability**: Much simpler codebase to understand and modify
- **Performance**: Native implementation optimized for production use
- **Future-proofing**: Automatic benefits from Strands framework improvements
- **Community**: Access to broader MCP ecosystem and tools

### For Developers
If you were working with the old custom implementation:
- Remove imports of `MCPClient`, `StrandsMCPTools`, `MCPIntegration`
- Update to use native Strands `Agent` with MCP tools
- Configuration format has changed to support multiple servers
- All tool operations now happen within MCP context managers

For detailed migration information, see `tasks/task-7-1-strands-mcp-research-documentation.md`.

## Getting Help

- **Configuration Issues**: See [Configuration Guide](configuration.md)
- **Development Questions**: Check [Development Guide](development.md)
- **Memory System**: Read [Memory System Guide](memory-system.md)
- **General Help**: Visit [Documentation Index](README.md)

## Example Projects

For complete examples of custom MCP servers and configurations, see:
- `config/mcp_multi_server_example.json` - Multi-server configuration example
- `src/mcp_servers/memory_server.py` - Reference memory server implementation
- `src/mcp_servers/filesystem_server.py` - Reference filesystem server implementation 