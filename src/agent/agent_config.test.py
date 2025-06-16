"""
Unit tests for agent_config module

Tests for configuration management including model settings,
memory configuration, MCP settings, and environment loading.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from agent_config import AgentConfig, ModelConfig, MemoryConfig, MCPConfig


class TestModelConfig:
    """Test cases for ModelConfig class"""
    
    def test_default_configuration(self):
        """Test default model configuration values"""
        config = ModelConfig()
        
        assert config.provider == "bedrock"
        assert config.model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert config.temperature == 0.7
        assert config.max_tokens == 4000
        assert config.streaming is True
        assert config.region_name == "us-west-2"
    
    def test_custom_configuration(self):
        """Test custom model configuration"""
        config = ModelConfig(
            provider="openai",
            model_id="gpt-4",
            temperature=0.5,
            max_tokens=2000,
            streaming=False,
            region_name="us-east-1"
        )
        
        assert config.provider == "openai"
        assert config.model_id == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.streaming is False
        assert config.region_name == "us-east-1"
    
    def test_temperature_validation(self):
        """Test temperature validation constraints"""
        # Valid temperatures
        ModelConfig(temperature=0.0)
        ModelConfig(temperature=1.0)
        ModelConfig(temperature=2.0)
        
        # Invalid temperatures should raise validation error
        with pytest.raises(ValueError):
            ModelConfig(temperature=-0.1)
        
        with pytest.raises(ValueError):
            ModelConfig(temperature=2.1)


class TestMemoryConfig:
    """Test cases for MemoryConfig class"""
    
    def test_default_configuration(self):
        """Test default memory configuration values"""
        config = MemoryConfig()
        
        assert config.enabled is True
        assert config.memory_base_path == Path("memory")
        assert config.max_context_memories == 10
        assert config.importance_threshold == 0.3
        assert config.enable_semantic_search is True
    
    def test_custom_configuration(self):
        """Test custom memory configuration"""
        custom_path = Path("/custom/memory")
        config = MemoryConfig(
            enabled=False,
            memory_base_path=custom_path,
            max_context_memories=5,
            importance_threshold=0.5,
            enable_semantic_search=False
        )
        
        assert config.enabled is False
        assert config.memory_base_path == custom_path
        assert config.max_context_memories == 5
        assert config.importance_threshold == 0.5
        assert config.enable_semantic_search is False


class TestMCPConfig:
    """Test cases for MCPConfig class"""
    
    def test_default_configuration(self):
        """Test default MCP configuration values"""
        config = MCPConfig()
        
        assert config.enabled is True
        assert config.filesystem_server_url == "http://localhost:8001/mcp"
        assert config.memory_server_url == "http://localhost:8002/mcp"
        assert config.connection_timeout == 30
        assert config.retry_attempts == 3
    
    def test_custom_configuration(self):
        """Test custom MCP configuration"""
        config = MCPConfig(
            enabled=False,
            filesystem_server_url="http://custom:9001/mcp",
            memory_server_url="http://custom:9002/mcp",
            connection_timeout=60,
            retry_attempts=5
        )
        
        assert config.enabled is False
        assert config.filesystem_server_url == "http://custom:9001/mcp"
        assert config.memory_server_url == "http://custom:9002/mcp"
        assert config.connection_timeout == 60
        assert config.retry_attempts == 5


class TestAgentConfig:
    """Test cases for AgentConfig class"""
    
    def test_default_configuration(self):
        """Test default agent configuration"""
        config = AgentConfig()
        
        assert config.agent_name == "Personal AI Assistant"
        assert config.agent_version == "1.0.0"
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.mcp, MCPConfig)
        assert config.enable_conversation_logging is True
        assert config.max_conversation_history == 50
        assert config.enable_builtin_tools is True
        assert config.custom_tools == []
        assert "Personal AI Assistant" in config.system_prompt
    
    def test_custom_configuration(self):
        """Test custom agent configuration"""
        custom_model = ModelConfig(provider="openai")
        custom_memory = MemoryConfig(enabled=False)
        custom_mcp = MCPConfig(enabled=False)
        
        config = AgentConfig(
            agent_name="Test Agent",
            agent_version="2.0.0",
            model=custom_model,
            memory=custom_memory,
            mcp=custom_mcp,
            system_prompt="Custom prompt",
            enable_conversation_logging=False,
            max_conversation_history=100,
            enable_builtin_tools=False,
            custom_tools=["tool1", "tool2"]
        )
        
        assert config.agent_name == "Test Agent"
        assert config.agent_version == "2.0.0"
        assert config.model.provider == "openai"
        assert config.memory.enabled is False
        assert config.mcp.enabled is False
        assert config.system_prompt == "Custom prompt"
        assert config.enable_conversation_logging is False
        assert config.max_conversation_history == 100
        assert config.enable_builtin_tools is False
        assert config.custom_tools == ["tool1", "tool2"]
    
    @patch.dict(os.environ, {
        'AI_MODEL_PROVIDER': 'anthropic',
        'AI_MODEL_ID': 'claude-3-haiku',
        'AI_MODEL_TEMPERATURE': '0.9',
        'AWS_REGION': 'eu-west-1',
        'MEMORY_BASE_PATH': '/custom/memory',
        'MEMORY_MAX_CONTEXT': '15',
        'MCP_FILESYSTEM_URL': 'http://fs:8001/mcp',
        'MCP_MEMORY_URL': 'http://mem:8002/mcp'
    })
    def test_from_env(self):
        """Test configuration creation from environment variables"""
        config = AgentConfig.from_env()
        
        assert config.model.provider == "anthropic"
        assert config.model.model_id == "claude-3-haiku"
        assert config.model.temperature == 0.9
        assert config.model.region_name == "eu-west-1"
        assert config.memory.memory_base_path == Path("/custom/memory")
        assert config.memory.max_context_memories == 15
        assert config.mcp.filesystem_server_url == "http://fs:8001/mcp"
        assert config.mcp.memory_server_url == "http://mem:8002/mcp"
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_no_variables(self):
        """Test configuration creation with no environment variables"""
        config = AgentConfig.from_env()
        
        # Should use defaults when no environment variables are set
        assert config.model.provider == "bedrock"
        assert config.model.model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert config.memory.memory_base_path == Path("memory")
    
    def test_to_dict(self):
        """Test configuration conversion to dictionary"""
        config = AgentConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["agent_name"] == "Personal AI Assistant"
        assert config_dict["agent_version"] == "1.0.0"
        assert "model" in config_dict
        assert "memory" in config_dict
        assert "mcp" in config_dict
        assert isinstance(config_dict["model"], dict)
        assert isinstance(config_dict["memory"], dict)
        assert isinstance(config_dict["mcp"], dict)
    
    def test_from_file_nonexistent(self):
        """Test loading from non-existent file"""
        with pytest.raises(FileNotFoundError):
            AgentConfig.from_file(Path("nonexistent.yaml"))
    
    def test_from_file_and_save(self):
        """Test saving to and loading from YAML file"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Create and save configuration
            original_config = AgentConfig(
                agent_name="Test Agent",
                agent_version="2.0.0"
            )
            original_config.save_to_file(temp_path)
            
            # Load configuration from file
            loaded_config = AgentConfig.from_file(temp_path)
            
            assert loaded_config.agent_name == "Test Agent"
            assert loaded_config.agent_version == "2.0.0"
            assert loaded_config.model.provider == "bedrock"  # Default value
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_save_to_file_creates_directory(self):
        """Test that save_to_file creates parent directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "subdir" / "config.yaml"
            
            config = AgentConfig()
            config.save_to_file(config_path)
            
            assert config_path.exists()
            assert config_path.parent.exists()


class TestConfigIntegration:
    """Integration tests for configuration classes"""
    
    def test_nested_configuration_validation(self):
        """Test that nested configuration objects validate properly"""
        # Should not raise any exceptions
        config = AgentConfig(
            model=ModelConfig(
                provider="bedrock",
                temperature=1.5
            ),
            memory=MemoryConfig(
                max_context_memories=20,
                importance_threshold=0.7
            ),
            mcp=MCPConfig(
                connection_timeout=45,
                retry_attempts=2
            )
        )
        
        assert config.model.temperature == 1.5
        assert config.memory.max_context_memories == 20
        assert config.mcp.connection_timeout == 45
    
    def test_configuration_serialization_roundtrip(self):
        """Test configuration serialization and deserialization"""
        original_config = AgentConfig(
            agent_name="Roundtrip Test",
            model=ModelConfig(provider="openai", temperature=0.8),
            memory=MemoryConfig(enabled=False),
            mcp=MCPConfig(enabled=True, connection_timeout=120)
        )
        
        # Convert to dict and back
        config_dict = original_config.to_dict()
        reconstructed_config = AgentConfig(**config_dict)
        
        assert reconstructed_config.agent_name == original_config.agent_name
        assert reconstructed_config.model.provider == original_config.model.provider
        assert reconstructed_config.model.temperature == original_config.model.temperature
        assert reconstructed_config.memory.enabled == original_config.memory.enabled
        assert reconstructed_config.mcp.connection_timeout == original_config.mcp.connection_timeout


if __name__ == "__main__":
    pytest.main([__file__]) 