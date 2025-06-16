"""
Agent Configuration Module

Manages configuration settings for the Personal AI Assistant agent,
including model providers, memory settings, and MCP server connections.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import os


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


class MCPConfig(BaseModel):
    """Configuration for MCP server connections"""
    enabled: bool = Field(default=True, description="Enable MCP integration")
    filesystem_server_url: str = Field(default="http://localhost:8001/mcp", description="Filesystem MCP server URL")
    memory_server_url: str = Field(default="http://localhost:8002/mcp", description="Memory MCP server URL")
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed connections")


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
            
        # MCP configuration from environment
        if os.getenv("MCP_FILESYSTEM_URL"):
            config_data.setdefault("mcp", {})["filesystem_server_url"] = os.getenv("MCP_FILESYSTEM_URL")
        if os.getenv("MCP_MEMORY_URL"):
            config_data.setdefault("mcp", {})["memory_server_url"] = os.getenv("MCP_MEMORY_URL")
            
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