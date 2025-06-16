"""
Integration tests for the Personal AI Assistant agent system

Tests the complete integration between agent, memory system, MCP client,
and Strands framework components.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json

import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from agent.core_agent import PersonalAssistantAgent
from agent.agent_config import AgentConfig, ModelConfig, MemoryConfig, MCPConfig
from memory.memory_manager import MemoryManager, MemoryEntry


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create a temporary memory directory for integration tests"""
    memory_dir = tmp_path / "integration_memory"
    memory_dir.mkdir()
    return memory_dir


@pytest.fixture
def integration_config(temp_memory_dir):
    """Create an integration test configuration"""
    return AgentConfig(
        agent_name="Integration Test Agent",
        model=ModelConfig(
            provider="bedrock",
            temperature=0.5,
            max_tokens=1000
        ),
        memory=MemoryConfig(
            enabled=True,
            memory_base_path=temp_memory_dir,
            max_context_memories=5,
            importance_threshold=0.3
        ),
        mcp=MCPConfig(
            enabled=True,
            connection_timeout=10
        ),
        enable_conversation_logging=True,
        max_conversation_history=10
    )


@pytest.fixture
async def agent_with_mocks(integration_config):
    """Create an agent with mocked external dependencies"""
    agent = PersonalAssistantAgent(integration_config)
    
    # Mock the Strands Agent
    with patch('agent.core_agent.Agent') as mock_agent_class:
        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = "Mocked response"
        mock_agent_class.return_value = mock_agent_instance
        
        # Mock BedrockModel
        with patch('agent.core_agent.BedrockModel') as mock_model:
            mock_model.return_value = MagicMock()
            
            # Mock MCP Client
            with patch('agent.core_agent.MCPClient') as mock_mcp_client_class:
                mock_mcp_client = AsyncMock()
                mock_mcp_client.initialize.return_value = True
                mock_mcp_client.get_available_tools.return_value = {}
                mock_mcp_client_class.return_value = mock_mcp_client
                
                await agent.initialize()
                agent.mock_strands_agent = mock_agent_instance
                agent.mock_mcp_client = mock_mcp_client
                yield agent


class TestAgentInitializationIntegration:
    """Integration tests for agent initialization"""
    
    async def test_full_initialization_sequence(self, integration_config):
        """Test complete agent initialization with all components"""
        with patch('agent.core_agent.Agent'), \
             patch('agent.core_agent.BedrockModel'), \
             patch('agent.core_agent.MCPClient') as mock_mcp_class:
            
            mock_mcp_client = AsyncMock()
            mock_mcp_client.initialize.return_value = True
            mock_mcp_class.return_value = mock_mcp_client
            
            agent = PersonalAssistantAgent(integration_config)
            result = await agent.initialize()
            
            assert result is True
            assert agent.is_initialized is True
            assert agent.memory_manager is not None
            assert agent.mcp_client is not None
            assert agent.agent is not None
            
            # Verify memory directory was created
            assert integration_config.memory.memory_base_path.exists()
    
    async def test_initialization_with_memory_disabled(self, integration_config):
        """Test initialization with memory system disabled"""
        integration_config.memory.enabled = False
        
        with patch('agent.core_agent.Agent'), \
             patch('agent.core_agent.BedrockModel'):
            
            agent = PersonalAssistantAgent(integration_config)
            result = await agent.initialize()
            
            assert result is True
            assert agent.memory_manager is None
    
    async def test_initialization_with_mcp_disabled(self, integration_config):
        """Test initialization with MCP disabled"""
        integration_config.mcp.enabled = False
        
        with patch('agent.core_agent.Agent'), \
             patch('agent.core_agent.BedrockModel'):
            
            agent = PersonalAssistantAgent(integration_config)
            result = await agent.initialize()
            
            assert result is True
            assert agent.mcp_client is None


class TestMemoryIntegration:
    """Integration tests for memory system"""
    
    async def test_memory_system_workflow(self, integration_config):
        """Test complete memory system workflow"""
        with patch('agent.core_agent.Agent'), \
             patch('agent.core_agent.BedrockModel'):
            
            agent = PersonalAssistantAgent(integration_config)
            await agent.initialize()
            
            # Test memory creation
            test_memory = MemoryEntry(
                content="Test conversation about AI",
                importance_score=8,
                category="conversation"
            )
            memory_id = agent.memory_manager.create_interaction_memory(test_memory)
            assert memory_id is not None
            
            # Test memory retrieval
            results = agent.memory_manager.search_memories("AI", limit=5)
            assert len(results) > 0
            assert any("AI" in result.content_snippet for result in results)


class TestMCPIntegration:
    """Integration tests for MCP client integration"""
    
    async def test_mcp_tool_integration(self, agent_with_mocks):
        """Test MCP tool integration with agent"""
        agent = agent_with_mocks
        
        # Mock MCP tools
        mock_search_tool = AsyncMock(return_value="MCP search results")
        mock_tools = {"search_memories_mcp": mock_search_tool}
        agent.mock_mcp_client.get_available_tools.return_value = mock_tools
        
        # Test context enhancement with MCP
        result = await agent._enhance_message_with_context("test query", "user123")
        
        assert "Previous relevant context:" in result
        assert "MCP search results" in result
    
    async def test_mcp_fallback_to_memory_manager(self, agent_with_mocks):
        """Test fallback to memory manager when MCP is unavailable"""
        agent = agent_with_mocks
        
        # Disable MCP client
        agent.mcp_client = None
        
        # Mock memory manager search
        mock_results = [
            MagicMock(relevance_score=0.8, content_snippet="Direct memory result")
        ]
        agent.memory_manager.search_memories = MagicMock(return_value=mock_results)
        
        result = await agent._enhance_message_with_context("test query", "user123")
        
        assert "Previous relevant context:" in result
        assert "Direct memory result" in result
    
    async def test_mcp_error_handling(self, agent_with_mocks):
        """Test MCP error handling and graceful degradation"""
        agent = agent_with_mocks
        
        # Mock MCP tool to raise exception
        mock_search_tool = AsyncMock(side_effect=Exception("MCP connection failed"))
        mock_tools = {"search_memories_mcp": mock_search_tool}
        agent.mock_mcp_client.get_available_tools.return_value = mock_tools
        
        # Should not raise exception, should fallback
        result = await agent._enhance_message_with_context("test query", "user123")
        
        # Should fallback to original message since MCP failed and no memory manager fallback
        assert result == "test query"


class TestToolIntegration:
    """Integration tests for tool collection and usage"""
    
    async def test_builtin_tools_integration(self, agent_with_mocks):
        """Test built-in Strands tools integration"""
        agent = agent_with_mocks
        
        tools = await agent._collect_tools()
        
        # Should have built-in tools plus memory tools
        assert len(tools) >= 2  # At least calculator and current_time
        
        # Memory tools should be included if memory is enabled
        if agent.memory_manager:
            assert len(tools) >= 5  # Built-in + memory tools
    
    async def test_memory_tools_functionality(self, agent_with_mocks):
        """Test memory tools functionality"""
        agent = agent_with_mocks
        
        memory_tools = agent._create_memory_tools()
        assert len(memory_tools) == 3
        
        # Test search_memories tool
        search_tool = memory_tools[0]
        agent.memory_manager.search_memories = MagicMock(return_value=[
            MagicMock(relevance_score=0.9, content_snippet="Test memory content")
        ])
        
        result = search_tool("test query", 3)
        assert "Test memory content" in result
        assert "Score: 0.90" in result
        
        # Test store_memory tool
        store_tool = memory_tools[1]
        agent.memory_manager.create_interaction_memory = MagicMock(return_value="mem_456")
        
        result = store_tool("New memory content", 0.8)
        assert "Memory stored successfully with ID: mem_456" in result
    
    async def test_strands_mcp_tools_integration(self, agent_with_mocks):
        """Test Strands MCP tools integration"""
        agent = agent_with_mocks
        
        # Mock StrandsMCPTools
        with patch('agent.core_agent.StrandsMCPTools') as mock_strands_mcp_class:
            mock_strands_mcp = MagicMock()
            mock_tools = [MagicMock(), MagicMock()]
            mock_strands_mcp.get_tools.return_value = mock_tools
            mock_strands_mcp_class.return_value = mock_strands_mcp
            
            agent.strands_mcp_tools = mock_strands_mcp
            
            tools = await agent._collect_tools()
            
            # Should include MCP tools
            for tool in mock_tools:
                assert tool in tools


class TestResponseProcessingIntegration:
    """Integration tests for response processing and formatting"""
    
    async def test_response_format_handling_integration(self, agent_with_mocks):
        """Test handling of different response formats from Strands"""
        agent = agent_with_mocks
        
        # Test with different response formats
        test_cases = [
            # Dictionary with content list
            {"content": [{"text": "Dict format response"}]},
            # List format
            [{"text": "List format response"}],
            # Simple string
            "Simple string response",
            # Dictionary with text key
            {"text": "Direct text response"}
        ]
        
        for mock_response in test_cases:
            agent.mock_strands_agent.return_value = mock_response
            
            with patch.object(agent, '_enhance_message_with_context', return_value="test"):
                result = await agent.process_message("test message")
                
                # Should extract text properly regardless of format
                assert isinstance(result, str)
                assert len(result) > 0
                assert "response" in result.lower() or result == str(mock_response)


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components"""
    
    async def test_memory_system_error_recovery(self, agent_with_mocks):
        """Test error recovery when memory system fails"""
        agent = agent_with_mocks
        
        # Mock memory manager to raise exception
        agent.memory_manager.search_memories = MagicMock(side_effect=Exception("Memory error"))
        
        # Should not raise exception, should continue with original message
        result = await agent._enhance_message_with_context("test message", "user123")
        assert result == "test message"
    
    async def test_strands_agent_error_handling(self, agent_with_mocks):
        """Test error handling when Strands agent fails"""
        agent = agent_with_mocks
        
        # Mock Strands agent to raise exception
        agent.mock_strands_agent.side_effect = Exception("Strands processing error")
        
        with patch.object(agent, '_enhance_message_with_context', return_value="test"):
            result = await agent.process_message("test message")
            
            assert "I apologize, but I encountered an error" in result
            assert "Strands processing error" in result
    
    async def test_complete_system_error_resilience(self, agent_with_mocks):
        """Test system resilience with multiple component failures"""
        agent = agent_with_mocks
        
        # Simulate multiple failures
        agent.memory_manager = None  # Memory system down
        agent.mcp_client = None      # MCP system down
        agent.mock_strands_agent.side_effect = Exception("Complete failure")
        
        result = await agent.process_message("test message")
        
        # Should still return an error message rather than crashing
        assert isinstance(result, str)
        assert "error" in result.lower()


class TestAgentStatusIntegration:
    """Integration tests for agent status and health monitoring"""
    
    async def test_comprehensive_agent_status(self, agent_with_mocks):
        """Test comprehensive agent status reporting"""
        agent = agent_with_mocks
        
        # Add some conversation history
        agent._update_conversation_history("test", "response")
        
        # Mock memory statistics
        agent.memory_manager.get_memory_statistics = MagicMock(return_value={
            "total_memories": 5,
            "categories": ["conversation", "facts"],
            "last_updated": "2024-01-01T00:00:00Z"
        })
        
        # Mock MCP health check
        agent.mock_mcp_client.health_check = AsyncMock(return_value={
            "status": "healthy",
            "servers": ["memory", "filesystem"],
            "uptime": "5m"
        })
        
        status = await agent.get_agent_status()
        
        assert status["agent_name"] == agent.config.agent_name
        assert status["initialized"] is True
        assert status["conversation_turns"] == 1
        assert "memory_stats" in status
        assert status["memory_stats"]["total_memories"] == 5
        assert "mcp_status" in status
        assert status["mcp_status"]["status"] == "healthy"
    
    async def test_agent_shutdown_integration(self, agent_with_mocks):
        """Test complete agent shutdown process"""
        agent = agent_with_mocks
        
        # Add some state
        agent._update_conversation_history("test", "response")
        assert len(agent.conversation_history) == 1
        assert agent.is_initialized is True
        
        # Shutdown
        await agent.shutdown()
        
        # Verify cleanup
        assert agent.is_initialized is False
        assert len(agent.conversation_history) == 0
        agent.mock_mcp_client.shutdown.assert_called_once()


class TestEndToEndFlow:
    """End-to-end workflow tests"""
    
    async def test_message_processing_workflow(self, integration_config):
        """Test complete message processing workflow"""
        with patch('agent.core_agent.Agent') as mock_agent_class, \
             patch('agent.core_agent.BedrockModel'):
            
            # Mock Strands agent
            mock_agent_instance = MagicMock()
            mock_agent_instance.return_value = "Test response"
            mock_agent_class.return_value = mock_agent_instance
            
            agent = PersonalAssistantAgent(integration_config)
            await agent.initialize()
            
            # Process a message
            result = await agent.process_message("Hello, test message", "test_user")
            
            assert result == "Test response"
            assert len(agent.conversation_history) == 1
            assert agent.conversation_history[0]["user_message"] == "Hello, test message"


if __name__ == "__main__":
    pytest.main([__file__]) 