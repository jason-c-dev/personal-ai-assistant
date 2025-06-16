"""
Unit tests for core_agent module

Tests for the PersonalAssistantAgent class including initialization,
message processing, memory integration, and tool functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path
import tempfile

from core_agent import PersonalAssistantAgent
from agent_config import AgentConfig, ModelConfig, MemoryConfig, MCPConfig
from ..memory.memory_manager import MemoryEntry


@pytest.fixture
def temp_memory_path(tmp_path):
    """Create a temporary memory directory for tests"""
    memory_dir = tmp_path / "test_memory"
    memory_dir.mkdir()
    return str(memory_dir)


@pytest.fixture
def test_config(temp_memory_path):
    """Create a test configuration"""
    return AgentConfig(
        agent_name="Test Agent",
        memory=MemoryConfig(
            enabled=True,
            memory_base_path=Path(temp_memory_path)
        ),
        mcp=MCPConfig(enabled=False),  # Disable MCP for unit tests
        enable_conversation_logging=True
    )


@pytest.fixture
def agent(test_config):
    """Create a PersonalAssistantAgent instance for testing"""
    return PersonalAssistantAgent(test_config)


class TestAgentInitialization:
    """Test cases for agent initialization"""
    
    def test_agent_creation_default_config(self):
        """Test agent creation with default configuration"""
        agent = PersonalAssistantAgent()
        
        assert agent.config is not None
        assert isinstance(agent.config, AgentConfig)
        assert agent.memory_manager is None
        assert agent.file_ops is None
        assert agent.mcp_client is None
        assert agent.agent is None
        assert agent.conversation_history == []
        assert agent.is_initialized is False
    
    def test_agent_creation_custom_config(self, test_config):
        """Test agent creation with custom configuration"""
        agent = PersonalAssistantAgent(test_config)
        
        assert agent.config == test_config
        assert agent.config.agent_name == "Test Agent"
        assert agent.is_initialized is False
    
    @patch('core_agent.PersonalAssistantAgent._initialize_memory_system')
    @patch('core_agent.PersonalAssistantAgent._initialize_mcp_client')
    @patch('core_agent.PersonalAssistantAgent._initialize_strands_agent')
    async def test_initialize_success(self, mock_strands, mock_mcp, mock_memory, agent):
        """Test successful agent initialization"""
        mock_memory.return_value = None
        mock_mcp.return_value = None
        mock_strands.return_value = None
        
        result = await agent.initialize()
        
        assert result is True
        assert agent.is_initialized is True
        mock_memory.assert_called_once()
        mock_strands.assert_called_once()
        # MCP should not be called since it's disabled in test config
        mock_mcp.assert_not_called()
    
    @patch('core_agent.PersonalAssistantAgent._initialize_memory_system')
    async def test_initialize_memory_failure(self, mock_memory, agent):
        """Test initialization failure in memory system"""
        mock_memory.side_effect = Exception("Memory initialization failed")
        
        result = await agent.initialize()
        
        assert result is False
        assert agent.is_initialized is False
    
    @patch('core_agent.PersonalAssistantAgent._initialize_memory_system')
    @patch('core_agent.PersonalAssistantAgent._initialize_strands_agent')
    async def test_initialize_strands_failure(self, mock_strands, mock_memory, agent):
        """Test initialization failure in Strands agent"""
        mock_memory.return_value = None
        mock_strands.side_effect = Exception("Strands initialization failed")
        
        result = await agent.initialize()
        
        assert result is False
        assert agent.is_initialized is False


class TestMemorySystemInitialization:
    """Test cases for memory system initialization"""
    
    @patch('core_agent.MemoryManager')
    @patch('core_agent.MemoryFileOperations')
    async def test_initialize_memory_system(self, mock_file_ops, mock_memory_manager, agent):
        """Test memory system initialization"""
        mock_manager_instance = MagicMock()
        mock_memory_manager.return_value = mock_manager_instance
        
        await agent._initialize_memory_system()
        
        assert agent.file_ops == mock_file_ops
        assert agent.memory_manager == mock_manager_instance
        
        # Verify memory directory creation
        assert agent.config.memory.memory_base_path.exists()
        
        # Verify MemoryManager was called with correct path
        mock_memory_manager.assert_called_once_with(
            base_path=str(agent.config.memory.memory_base_path)
        )


class TestMCPClientInitialization:
    """Test cases for MCP client initialization"""
    
    @patch('core_agent.MCPClient')
    @patch('core_agent.StrandsMCPTools')
    async def test_initialize_mcp_client_success(self, mock_strands_tools, mock_mcp_client, agent):
        """Test successful MCP client initialization"""
        # Enable MCP for this test
        agent.config.mcp.enabled = True
        
        mock_client_instance = AsyncMock()
        mock_client_instance.initialize.return_value = True
        mock_mcp_client.return_value = mock_client_instance
        
        mock_tools_instance = MagicMock()
        mock_strands_tools.return_value = mock_tools_instance
        
        await agent._initialize_mcp_client()
        
        assert agent.mcp_client == mock_client_instance
        assert agent.strands_mcp_tools == mock_tools_instance
        mock_client_instance.initialize.assert_called_once()
    
    @patch('core_agent.MCPClient')
    async def test_initialize_mcp_client_failure(self, mock_mcp_client, agent):
        """Test MCP client initialization failure"""
        # Enable MCP for this test
        agent.config.mcp.enabled = True
        
        mock_client_instance = AsyncMock()
        mock_client_instance.initialize.return_value = False
        mock_mcp_client.return_value = mock_client_instance
        
        await agent._initialize_mcp_client()
        
        assert agent.mcp_client == mock_client_instance
        assert agent.strands_mcp_tools is None


class TestStrandsAgentInitialization:
    """Test cases for Strands agent initialization"""
    
    @patch('core_agent.Agent')
    @patch('core_agent.PersonalAssistantAgent._create_model')
    @patch('core_agent.PersonalAssistantAgent._collect_tools')
    async def test_initialize_strands_agent(self, mock_collect_tools, mock_create_model, mock_agent_class, agent):
        """Test Strands agent initialization"""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_tools = [MagicMock(), MagicMock()]
        mock_collect_tools.return_value = mock_tools
        
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance
        
        await agent._initialize_strands_agent()
        
        assert agent.agent == mock_agent_instance
        mock_agent_class.assert_called_once_with(
            model=mock_model,
            tools=mock_tools,
            system_prompt=agent.config.system_prompt
        )
    
    @patch('core_agent.BedrockModel')
    def test_create_model(self, mock_bedrock_model, agent):
        """Test model creation"""
        mock_model_instance = MagicMock()
        mock_bedrock_model.return_value = mock_model_instance
        
        model = agent._create_model()
        
        assert model == mock_model_instance
        mock_bedrock_model.assert_called_once_with(
            model_id=agent.config.model.model_id,
            region_name=agent.config.model.region_name,
            temperature=agent.config.model.temperature,
            max_tokens=agent.config.model.max_tokens,
            streaming=agent.config.model.streaming
        )


class TestToolCollection:
    """Test cases for tool collection"""
    
    @patch('core_agent.calculator')
    @patch('core_agent.current_time')
    async def test_collect_tools_builtin_only(self, mock_current_time, mock_calculator, agent):
        """Test tool collection with only built-in tools"""
        agent.config.enable_builtin_tools = True
        agent.memory_manager = None
        agent.strands_mcp_tools = None
        
        tools = await agent._collect_tools()
        
        assert mock_calculator in tools
        assert mock_current_time in tools
        assert len(tools) == 2
    
    @patch('core_agent.calculator')
    @patch('core_agent.current_time')
    @patch('core_agent.PersonalAssistantAgent._create_memory_tools')
    async def test_collect_tools_with_memory(self, mock_create_memory_tools, mock_current_time, mock_calculator, agent):
        """Test tool collection with memory tools"""
        agent.config.enable_builtin_tools = True
        agent.memory_manager = MagicMock()
        agent.strands_mcp_tools = None
        
        mock_memory_tools = [MagicMock(), MagicMock()]
        mock_create_memory_tools.return_value = mock_memory_tools
        
        tools = await agent._collect_tools()
        
        assert mock_calculator in tools
        assert mock_current_time in tools
        for tool in mock_memory_tools:
            assert tool in tools
        assert len(tools) == 4
    
    async def test_collect_tools_with_mcp(self, agent):
        """Test tool collection with MCP tools"""
        agent.config.enable_builtin_tools = False
        agent.memory_manager = None
        
        mock_strands_mcp_tools = MagicMock()
        mock_mcp_tools = [MagicMock(), MagicMock(), MagicMock()]
        mock_strands_mcp_tools.get_tools.return_value = mock_mcp_tools
        agent.strands_mcp_tools = mock_strands_mcp_tools
        
        tools = await agent._collect_tools()
        
        for tool in mock_mcp_tools:
            assert tool in tools
        assert len(tools) == 3


class TestMemoryTools:
    """Test cases for memory tool creation"""
    
    def test_create_memory_tools(self, agent):
        """Test memory tool creation"""
        mock_memory_manager = MagicMock()
        agent.memory_manager = mock_memory_manager
        
        tools = agent._create_memory_tools()
        
        assert len(tools) == 3
        assert all(callable(tool) for tool in tools)
    
    def test_search_memories_tool(self, agent):
        """Test search memories tool functionality"""
        mock_memory_manager = MagicMock()
        mock_results = [
            MagicMock(relevance_score=0.8, content_snippet="Test memory 1"),
            MagicMock(relevance_score=0.6, content_snippet="Test memory 2")
        ]
        mock_memory_manager.search_memories.return_value = mock_results
        agent.memory_manager = mock_memory_manager
        
        tools = agent._create_memory_tools()
        search_tool = tools[0]  # First tool should be search_memories
        
        result = search_tool("test query", 5)
        
        assert "Test memory 1" in result
        assert "Test memory 2" in result
        assert "Score: 0.80" in result
        mock_memory_manager.search_memories.assert_called_once_with("test query", limit=5)
    
    def test_store_memory_tool(self, agent):
        """Test store memory tool functionality"""
        mock_memory_manager = MagicMock()
        mock_memory_manager.create_interaction_memory.return_value = "mem_123"
        agent.memory_manager = mock_memory_manager
        
        tools = agent._create_memory_tools()
        store_tool = tools[1]  # Second tool should be store_memory
        
        result = store_tool("Test content", 0.7)
        
        assert "Memory stored successfully with ID: mem_123" in result
        mock_memory_manager.create_interaction_memory.assert_called_once()
        
        # Verify the MemoryEntry was created correctly
        call_args = mock_memory_manager.create_interaction_memory.call_args[0][0]
        assert call_args.content == "Test content"
        assert call_args.importance_score == 7  # 0.7 * 10
        assert call_args.category == "manual_storage"


class TestMessageProcessing:
    """Test cases for message processing"""
    
    @patch('core_agent.PersonalAssistantAgent._enhance_message_with_context')
    @patch('core_agent.PersonalAssistantAgent._store_interaction')
    async def test_process_message_success(self, mock_store, mock_enhance, agent):
        """Test successful message processing"""
        agent.is_initialized = True
        
        # Mock the Strands agent
        mock_strands_agent = MagicMock()
        mock_strands_agent.return_value = "Test response"
        agent.agent = mock_strands_agent
        
        mock_enhance.return_value = "Enhanced message"
        mock_store.return_value = None
        
        result = await agent.process_message("Hello", "user123")
        
        assert result == "Test response"
        mock_enhance.assert_called_once_with("Hello", "user123")
        mock_strands_agent.assert_called_once_with("Enhanced message")
        mock_store.assert_called_once_with("Hello", "Test response", "user123")
    
    async def test_process_message_not_initialized(self, agent):
        """Test message processing when agent is not initialized"""
        agent.is_initialized = False
        
        with pytest.raises(RuntimeError, match="Agent not initialized"):
            await agent.process_message("Hello")
    
    @patch('core_agent.PersonalAssistantAgent._enhance_message_with_context')
    async def test_process_message_exception_handling(self, mock_enhance, agent):
        """Test exception handling in message processing"""
        agent.is_initialized = True
        mock_enhance.side_effect = Exception("Context enhancement failed")
        
        result = await agent.process_message("Hello")
        
        assert "I apologize, but I encountered an error" in result
        assert "Context enhancement failed" in result
    
    def test_response_format_handling(self, agent):
        """Test handling of different response formats from Strands"""
        agent.is_initialized = True
        
        # Test with AgentResult-like object (has __class__ attribute)
        mock_agent_result = MagicMock()
        mock_agent_result.__class__.__name__ = "AgentResult"
        str(mock_agent_result)  # Should return string representation
        
        # Test with list format
        list_response = [{"text": "List response"}]
        
        # Test with dict format
        dict_response = {"content": [{"text": "Dict response"}]}
        
        # Test with simple string
        string_response = "String response"
        
        # These would be tested in integration tests with actual Strands responses
        pass


class TestContextEnhancement:
    """Test cases for message context enhancement"""
    
    async def test_enhance_message_memory_disabled(self, agent):
        """Test context enhancement when memory is disabled"""
        agent.config.memory.enabled = False
        
        result = await agent._enhance_message_with_context("Hello", "user123")
        
        assert result == "Hello"
    
    @patch('core_agent.PersonalAssistantAgent._store_interaction')
    async def test_enhance_message_with_mcp(self, mock_store, agent):
        """Test context enhancement with MCP client"""
        agent.config.memory.enabled = True
        
        mock_mcp_client = MagicMock()
        mock_tools = {"search_memories_mcp": AsyncMock(return_value="Relevant context")}
        mock_mcp_client.get_available_tools.return_value = mock_tools
        agent.mcp_client = mock_mcp_client
        
        result = await agent._enhance_message_with_context("Hello", "user123")
        
        assert "Previous relevant context:" in result
        assert "Relevant context" in result
        assert "Current message: Hello" in result
    
    async def test_enhance_message_with_memory_manager(self, agent):
        """Test context enhancement with direct memory manager"""
        agent.config.memory.enabled = True
        agent.config.memory.max_context_memories = 3
        agent.config.memory.importance_threshold = 0.5
        
        mock_memory_manager = MagicMock()
        mock_memories = [
            MagicMock(relevance_score=0.8, content_snippet="Important memory"),
            MagicMock(relevance_score=0.4, content_snippet="Less important memory"),
            MagicMock(relevance_score=0.7, content_snippet="Another important memory")
        ]
        mock_memory_manager.search_memories.return_value = mock_memories
        agent.memory_manager = mock_memory_manager
        
        result = await agent._enhance_message_with_context("Hello", "user123")
        
        assert "Previous relevant context:" in result
        assert "Important memory" in result
        assert "Another important memory" in result
        assert "Less important memory" not in result  # Below threshold
        assert "Current message: Hello" in result


class TestConversationHistory:
    """Test cases for conversation history management"""
    
    def test_update_conversation_history(self, agent):
        """Test conversation history update"""
        agent._update_conversation_history("Hello", "Hi there!")
        
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["user_message"] == "Hello"
        assert agent.conversation_history[0]["assistant_response"] == "Hi there!"
        assert "timestamp" in agent.conversation_history[0]
    
    def test_conversation_history_trimming(self, agent):
        """Test conversation history trimming when max length exceeded"""
        agent.config.max_conversation_history = 2
        
        # Add 3 conversations
        agent._update_conversation_history("Message 1", "Response 1")
        agent._update_conversation_history("Message 2", "Response 2")
        agent._update_conversation_history("Message 3", "Response 3")
        
        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0]["user_message"] == "Message 2"
        assert agent.conversation_history[1]["user_message"] == "Message 3"
    
    def test_get_conversation_history(self, agent):
        """Test getting conversation history"""
        agent._update_conversation_history("Hello", "Hi!")
        
        history = agent.get_conversation_history()
        
        assert len(history) == 1
        assert history[0]["user_message"] == "Hello"
        assert history is not agent.conversation_history  # Should be a copy


class TestAgentStatus:
    """Test cases for agent status reporting"""
    
    async def test_get_agent_status_basic(self, agent):
        """Test basic agent status"""
        agent.is_initialized = True
        
        status = await agent.get_agent_status()
        
        assert status["agent_name"] == agent.config.agent_name
        assert status["initialized"] is True
        assert status["memory_enabled"] == agent.config.memory.enabled
        assert status["mcp_enabled"] == agent.config.mcp.enabled
        assert status["conversation_turns"] == 0
    
    async def test_get_agent_status_with_memory(self, agent):
        """Test agent status with memory statistics"""
        agent.is_initialized = True
        
        mock_memory_manager = MagicMock()
        mock_stats = {"total_memories": 10, "categories": ["conversation", "facts"]}
        mock_memory_manager.get_memory_statistics.return_value = mock_stats
        agent.memory_manager = mock_memory_manager
        
        status = await agent.get_agent_status()
        
        assert status["memory_stats"] == mock_stats
    
    async def test_get_agent_status_with_mcp(self, agent):
        """Test agent status with MCP health check"""
        agent.is_initialized = True
        
        mock_mcp_client = AsyncMock()
        mock_health = {"status": "healthy", "servers": ["memory", "filesystem"]}
        mock_mcp_client.health_check.return_value = mock_health
        agent.mcp_client = mock_mcp_client
        
        status = await agent.get_agent_status()
        
        assert status["mcp_status"] == mock_health


class TestAgentShutdown:
    """Test cases for agent shutdown"""
    
    async def test_shutdown_success(self, agent):
        """Test successful agent shutdown"""
        agent.is_initialized = True
        agent.conversation_history = [{"test": "data"}]
        
        mock_mcp_client = AsyncMock()
        agent.mcp_client = mock_mcp_client
        
        await agent.shutdown()
        
        assert agent.is_initialized is False
        assert len(agent.conversation_history) == 0
        mock_mcp_client.shutdown.assert_called_once()
    
    async def test_shutdown_with_mcp_error(self, agent):
        """Test shutdown with MCP client error"""
        agent.is_initialized = True
        
        mock_mcp_client = AsyncMock()
        mock_mcp_client.shutdown.side_effect = Exception("Shutdown failed")
        agent.mcp_client = mock_mcp_client
        
        # Should not raise exception
        await agent.shutdown()
        
        assert agent.is_initialized is False


if __name__ == "__main__":
    pytest.main([__file__]) 