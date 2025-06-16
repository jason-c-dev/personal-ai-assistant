"""
Agent Configuration Module

Manages configuration settings for the Personal AI Assistant agent,
including model providers, memory settings, and MCP server connections.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import os
import sys


class ModelConfig(BaseModel):
    """Configuration for AI model providers"""
    model_config = {"protected_namespaces": ()}  # Allow model_ fields
    
    provider: str = Field(default="bedrock", description="Model provider (bedrock, anthropic, openai, etc.)")
    model_id: str = Field(default="anthropic.claude-3-5-sonnet-20241022-v2:0", description="Model identifier")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=4000, description="Maximum tokens for response")
    streaming: bool = Field(default=True, description="Enable streaming responses")
    region_name: Optional[str] = Field(default="us-west-2", description="AWS region for Bedrock")


class MemoryConfig(BaseModel):
    """Configuration for memory system integration"""
    enabled: bool = Field(default=True, description="Enable memory system")
    memory_base_path: Path = Field(default_factory=lambda: Path("memory"), description="Base path for memory files")
    max_context_memories: int = Field(default=10, description="Maximum memories to include in context")
    importance_threshold: float = Field(default=0.3, description="Minimum importance score for memory inclusion")
    enable_semantic_search: bool = Field(default=True, description="Enable semantic memory search")


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server"""
    name: str = Field(description="Server name identifier")
    transport: str = Field(default="stdio", description="Transport type (stdio, http, sse)")
    command: Optional[str] = Field(default=None, description="Command to run for stdio transport")
    args: Optional[List[str]] = Field(default_factory=list, description="Arguments for stdio command")
    url: Optional[str] = Field(default=None, description="URL for HTTP/SSE transport")
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables for stdio transport")
    timeout: int = Field(default=30, description="Connection timeout in seconds")
    enabled: bool = Field(default=True, description="Enable this server")
    
    @field_validator('transport')
    @classmethod
    def validate_transport(cls, v):
        """Validate transport type"""
        valid_transports = ['stdio', 'http', 'sse']
        if v not in valid_transports:
            raise ValueError(f"Transport must be one of {valid_transports}")
        return v
    
    @field_validator('command')
    @classmethod
    def validate_stdio_command(cls, v, info):
        """Validate command is provided for stdio transport"""
        if info.data.get('transport') == 'stdio' and not v:
            raise ValueError("Command is required for stdio transport")
        return v
    
    @field_validator('url')
    @classmethod
    def validate_url_for_transport(cls, v, info):
        """Validate URL is provided for HTTP/SSE transport"""
        transport = info.data.get('transport')
        if transport in ['http', 'sse'] and not v:
            raise ValueError(f"URL is required for {transport} transport")
        return v


class MCPConfig(BaseModel):
    """Configuration for MCP server connections"""
    enabled: bool = Field(default=True, description="Enable MCP integration")
    servers: List[MCPServerConfig] = Field(default_factory=list, description="MCP server configurations")
    global_timeout: int = Field(default=30, description="Global connection timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed connections")
    
    # Legacy fields for backward compatibility
    filesystem_server_url: Optional[str] = Field(default=None, description="Legacy filesystem server URL")
    memory_server_url: Optional[str] = Field(default=None, description="Legacy memory server URL")
    connection_timeout: Optional[int] = Field(default=None, description="Legacy connection timeout")
    
    def __init__(self, **data):
        """Initialize MCP config with backward compatibility and defaults"""
        # Handle legacy configuration migration
        if 'servers' not in data or not data['servers']:
            data['servers'] = self._create_default_servers(data)
        
        # Handle legacy timeout field
        if 'connection_timeout' in data and data['connection_timeout']:
            data['global_timeout'] = data['connection_timeout']
            
        super().__init__(**data)
    
    def _create_default_servers(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create default server configurations"""
        servers = []
        
        # Get memory base path for server arguments
        memory_base_path = data.get('memory_base_path', 'memory')
        if hasattr(memory_base_path, '__str__'):
            memory_base_path = str(memory_base_path)
        
        # Default memory server (stdio)
        memory_server = {
            "name": "memory",
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                str(Path(__file__).parent.parent / "mcp_servers" / "memory_server.py"),
                memory_base_path
            ],
            "timeout": data.get('connection_timeout', 30),
            "enabled": True
        }
        servers.append(memory_server)
        
        # Default filesystem server (stdio)
        filesystem_server = {
            "name": "filesystem", 
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                str(Path(__file__).parent.parent / "mcp_servers" / "filesystem_server.py"),
                memory_base_path
            ],
            "timeout": data.get('connection_timeout', 30),
            "enabled": True
        }
        servers.append(filesystem_server)
        
        # Handle legacy URL configurations (convert to HTTP if provided)
        if data.get('memory_server_url') and 'localhost' not in data['memory_server_url']:
            servers.append({
                "name": "legacy_memory",
                "transport": "http",
                "url": data['memory_server_url'],
                "timeout": data.get('connection_timeout', 30),
                "enabled": True
            })
            
        if data.get('filesystem_server_url') and 'localhost' not in data['filesystem_server_url']:
            servers.append({
                "name": "legacy_filesystem",
                "transport": "http", 
                "url": data['filesystem_server_url'],
                "timeout": data.get('connection_timeout', 30),
                "enabled": True
            })
        
        return servers
    
    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get list of enabled server configurations"""
        return [server for server in self.servers if server.enabled]
    
    def get_servers_by_transport(self, transport: str) -> List[MCPServerConfig]:
        """Get servers filtered by transport type"""
        return [server for server in self.servers if server.transport == transport and server.enabled]
    
    def add_server(self, server_config: Union[MCPServerConfig, Dict[str, Any]]) -> None:
        """Add a new server configuration"""
        if isinstance(server_config, dict):
            server_config = MCPServerConfig(**server_config)
        self.servers.append(server_config)
    
    def remove_server(self, server_name: str) -> bool:
        """Remove a server configuration by name"""
        original_count = len(self.servers)
        self.servers = [s for s in self.servers if s.name != server_name]
        return len(self.servers) < original_count
    
    def get_server(self, server_name: str) -> Optional[MCPServerConfig]:
        """Get server configuration by name"""
        for server in self.servers:
            if server.name == server_name:
                return server
        return None
    
    def update_memory_base_path(self, memory_base_path: str) -> None:
        """Update memory base path for all stdio servers"""
        for server in self.servers:
            if server.transport == "stdio" and server.args:
                # Update the memory base path argument (usually the last argument)
                if len(server.args) >= 2:
                    server.args[-1] = memory_base_path
    
    def enable_server(self, server_name: str) -> bool:
        """Enable a specific server by name"""
        server = self.get_server(server_name)
        if server:
            server.enabled = True
            return True
        return False
    
    def disable_server(self, server_name: str) -> bool:
        """Disable a specific server by name"""
        server = self.get_server(server_name)
        if server:
            server.enabled = False
            return True
        return False
    
    def get_server_health_status(self) -> Dict[str, Any]:
        """Get health status information for all servers"""
        return {
            "total_servers": len(self.servers),
            "enabled_servers": len(self.get_enabled_servers()),
            "servers_by_transport": {
                "stdio": len(self.get_servers_by_transport("stdio")),
                "http": len(self.get_servers_by_transport("http")),
                "sse": len(self.get_servers_by_transport("sse"))
            },
            "server_list": [
                {
                    "name": server.name,
                    "transport": server.transport,
                    "enabled": server.enabled,
                    "timeout": server.timeout
                }
                for server in self.servers
            ]
        }
    
    def validate_server_configurations(self) -> List[str]:
        """Validate all server configurations and return list of errors"""
        errors = []
        
        for server in self.servers:
            try:
                # Validate the server configuration
                MCPServerConfig.model_validate(server.model_dump())
            except Exception as e:
                errors.append(f"Server '{server.name}': {str(e)}")
        
        return errors
    
    def update_server_config(self, server_name: str, updates: Dict[str, Any]) -> bool:
        """Update specific server configuration"""
        server = self.get_server(server_name)
        if not server:
            return False
        
        try:
            # Apply updates to server
            for key, value in updates.items():
                if hasattr(server, key):
                    setattr(server, key, value)
            return True
        except Exception:
            return False


class AgentConfig(BaseModel):
    """Main configuration class for the Personal AI Assistant agent"""
    
    # Agent Identity
    agent_name: str = Field(default="Personal AI Assistant", description="Agent display name")
    agent_version: str = Field(default="1.0.0", description="Agent version")
    
    # Model Configuration
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model provider configuration")
    
    # Memory Configuration
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="Memory system configuration")
    
    # MCP Configuration
    mcp: MCPConfig = Field(default_factory=MCPConfig, description="MCP integration configuration")
    
    # Agent Behavior
    system_prompt: str = Field(
        default="""You are a Personal AI Assistant with persistent memory capabilities. 
        
Your core abilities include:
- Remembering conversations and user preferences across sessions
- Learning from interactions to provide increasingly personalized assistance
- Accessing and managing files through your filesystem tools
- Searching and retrieving relevant information from your memory system
- Maintaining context awareness across multiple conversation threads

Guidelines:
- Always be helpful, accurate, and respectful
- Use your memory system to provide personalized responses based on past interactions
- When uncertain, acknowledge limitations and ask clarifying questions
- Prioritize user privacy and data security in all operations
- Continuously learn and adapt to user preferences and communication style

Your memory system allows you to:
- Store important information from conversations
- Retrieve relevant context for current discussions
- Track user preferences and interests over time
- Maintain continuity across multiple sessions""",
        description="System prompt defining agent behavior and capabilities"
    )
    
    enable_conversation_logging: bool = Field(default=True, description="Enable conversation logging")
    max_conversation_history: int = Field(default=50, description="Maximum conversation turns to maintain")
    
    # Tool Configuration
    enable_builtin_tools: bool = Field(default=True, description="Enable built-in Strands tools")
    custom_tools: List[str] = Field(default_factory=list, description="List of custom tool modules to load")
    
    def __init__(self, **data):
        """Initialize config with MCP memory path coordination"""
        super().__init__(**data)
        
        # Ensure MCP servers use the same memory base path as memory config
        memory_path = str(self.memory.memory_base_path)
        self.mcp.update_memory_base_path(memory_path)
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables"""
        config_data = {}
        
        # Model configuration from environment
        if os.getenv("AI_MODEL_PROVIDER"):
            config_data.setdefault("model", {})["provider"] = os.getenv("AI_MODEL_PROVIDER")
        if os.getenv("AI_MODEL_ID"):
            config_data.setdefault("model", {})["model_id"] = os.getenv("AI_MODEL_ID")
        if os.getenv("AI_MODEL_TEMPERATURE"):
            config_data.setdefault("model", {})["temperature"] = float(os.getenv("AI_MODEL_TEMPERATURE"))
        if os.getenv("AWS_REGION"):
            config_data.setdefault("model", {})["region_name"] = os.getenv("AWS_REGION")
            
        # Memory configuration from environment
        if os.getenv("MEMORY_BASE_PATH"):
            config_data.setdefault("memory", {})["memory_base_path"] = Path(os.getenv("MEMORY_BASE_PATH"))
        if os.getenv("MEMORY_MAX_CONTEXT"):
            config_data.setdefault("memory", {})["max_context_memories"] = int(os.getenv("MEMORY_MAX_CONTEXT"))
            
        # MCP configuration from environment (legacy support)
        mcp_data = {}
        if os.getenv("MCP_FILESYSTEM_URL"):
            mcp_data["filesystem_server_url"] = os.getenv("MCP_FILESYSTEM_URL")
        if os.getenv("MCP_MEMORY_URL"):
            mcp_data["memory_server_url"] = os.getenv("MCP_MEMORY_URL")
        if mcp_data:
            config_data["mcp"] = mcp_data
            
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.model_dump()
    
    @classmethod
    def from_file(cls, config_path: Path) -> "AgentConfig":
        """Load configuration from YAML file"""
        import yaml
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        return cls(**config_data)
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file"""
        import yaml
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2) 