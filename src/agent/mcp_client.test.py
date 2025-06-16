"""
Unit tests for mcp_client module

Tests for MCP client functionality including server connections,
tool wrappers, and protocol communication.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path
import json

from mcp_client import MCPClient


@pytest.fixture
def temp_memory_path(tmp_path):
    """Create a temporary memory directory for tests"""
    memory_dir = tmp_path / "test_memory"
    memory_dir.mkdir()
    return str(memory_dir)


@pytest.fixture
def mcp_client(temp_memory_path):
    """Create an MCP client instance for testing"""
    return MCPClient(temp_memory_path)


class TestMCPClientInitialization:
    """Test cases for MCP client initialization"""
    
    def test_client_creation(self, temp_memory_path):
        """Test basic client creation"""
        client = MCPClient(temp_memory_path)
        
        assert client.memory_base_path == Path(temp_memory_path)
        assert isinstance(client.server_processes, dict)
        assert isinstance(client.available_tools, dict)
        assert "memory" in client.server_configs
        assert "filesystem" in client.server_configs
    
    def test_server_configs_structure(self, mcp_client):
        """Test server configuration structure"""
        assert "memory" in mcp_client.server_configs
        assert "filesystem" in mcp_client.server_configs
        
        for server_name, config in mcp_client.server_configs.items():
            assert "command" in config
            assert "args" in config
            assert "env" in config
            assert isinstance(config["args"], list)
    
    @patch('mcp_client.MCPClient._create_tool_wrappers')
    async def test_initialize_success(self, mock_create_tools, mcp_client):
        """Test successful initialization"""
        mock_create_tools.return_value = None
        
        result = await mcp_client.initialize()
        
        assert result is True
        mock_create_tools.assert_called_once()
    
    @patch('mcp_client.MCPClient._create_tool_wrappers')
    async def test_initialize_failure(self, mock_create_tools, mcp_client):
        """Test initialization failure"""
        mock_create_tools.side_effect = Exception("Initialization failed")
        
        result = await mcp_client.initialize()
        
        assert result is False


class TestToolWrapperCreation:
    """Test cases for tool wrapper creation"""
    
    async def test_create_tool_wrappers(self, mcp_client):
        """Test tool wrapper creation"""
        await mcp_client._create_tool_wrappers()
        
        # Check that all expected tools are created
        expected_tools = [
            "search_memories_mcp",
            "get_user_profile_mcp", 
            "update_memory_mcp",
            "get_recent_context_mcp",
            "read_memory_file_mcp",
            "write_memory_file_mcp",
            "list_memory_files_mcp"
        ]
        
        for tool_name in expected_tools:
            assert tool_name in mcp_client.available_tools
            assert callable(mcp_client.available_tools[tool_name])
    
    def test_get_available_tools(self, mcp_client):
        """Test get_available_tools method"""
        # Add some mock tools
        mcp_client.available_tools = {"tool1": lambda: None, "tool2": lambda: None}
        
        tools = mcp_client.get_available_tools()
        
        assert "tool1" in tools
        assert "tool2" in tools
        assert callable(tools["tool1"])
        assert callable(tools["tool2"])


class TestMemoryServerTools:
    """Test cases for memory server tool wrappers"""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock MCP session"""
        session = AsyncMock()
        session.initialize = AsyncMock()
        return session
    
    @pytest.fixture
    def mock_server_session_context(self, mock_session):
        """Mock the server session context manager"""
        context_manager = AsyncMock()
        context_manager.__aenter__ = AsyncMock(return_value=mock_session)
        context_manager.__aexit__ = AsyncMock(return_value=None)
        return context_manager
    
    async def test_search_memories_tool_success(self, mcp_client, mock_session, mock_server_session_context):
        """Test successful memory search"""
        # Setup mock response
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock()]
        mock_result.content[0].text = json.dumps({
            "success": True,
            "results": [
                {"content": "Test memory content", "relevance_score": 0.8},
                {"content": "Another memory", "relevance_score": 0.6}
            ]
        })
        mock_session.call_tool.return_value = mock_result
        
        with patch.object(mcp_client, '_get_server_session', return_value=mock_server_session_context):
            search_tool = mcp_client._create_memory_search_tool()
            result = await search_tool("test query", 5)
        
        assert "Found 2 memories" in result
        assert "Test memory content" in result
        assert "Score: 0.80" in result
        mock_session.call_tool.assert_called_once_with("search_memories", {
            "query": "test query",
            "limit": 5
        })
    
    async def test_search_memories_tool_no_results(self, mcp_client, mock_session, mock_server_session_context):
        """Test memory search with no results"""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock()]
        mock_result.content[0].text = json.dumps({
            "success": True,
            "results": []
        })
        mock_session.call_tool.return_value = mock_result
        
        with patch.object(mcp_client, '_get_server_session', return_value=mock_server_session_context):
            search_tool = mcp_client._create_memory_search_tool()
            result = await search_tool("test query")
        
        assert "No memories found for query: test query" in result
    
    async def test_search_memories_tool_error(self, mcp_client, mock_session, mock_server_session_context):
        """Test memory search with server error"""
        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = "Server error occurred"
        mock_session.call_tool.return_value = mock_result
        
        with patch.object(mcp_client, '_get_server_session', return_value=mock_server_session_context):
            search_tool = mcp_client._create_memory_search_tool()
            result = await search_tool("test query")
        
        assert "Error searching memories: Server error occurred" in result
    
    async def test_get_user_profile_tool_success(self, mcp_client, mock_session, mock_server_session_context):
        """Test successful user profile retrieval"""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock()]
        mock_result.content[0].text = json.dumps({
            "success": True,
            "profile": {"content": "User profile information"}
        })
        mock_session.call_tool.return_value = mock_result
        
        with patch.object(mcp_client, '_get_server_session', return_value=mock_server_session_context):
            profile_tool = mcp_client._create_user_profile_tool()
            result = await profile_tool()
        
        assert "User Profile:" in result
        assert "User profile information" in result
        mock_session.call_tool.assert_called_once_with("get_user_profile", {})
    
    async def test_update_memory_tool_success(self, mcp_client, mock_session, mock_server_session_context):
        """Test successful memory update"""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock()]
        mock_result.content[0].text = json.dumps({
            "success": True,
            "memory_id": "mem_123"
        })
        mock_session.call_tool.return_value = mock_result
        
        with patch.object(mcp_client, '_get_server_session', return_value=mock_server_session_context):
            update_tool = mcp_client._create_update_memory_tool()
            result = await update_tool("Test content", "conversation")
        
        assert "Memory stored successfully with ID: mem_123" in result
        mock_session.call_tool.assert_called_once_with("update_memory", {
            "content": "Test content",
            "memory_type": "conversation"
        })


class TestFilesystemServerTools:
    """Test cases for filesystem server tool wrappers"""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock MCP session"""
        session = AsyncMock()
        session.initialize = AsyncMock()
        return session
    
    @pytest.fixture
    def mock_server_session_context(self, mock_session):
        """Mock the server session context manager"""
        context_manager = AsyncMock()
        context_manager.__aenter__ = AsyncMock(return_value=mock_session)
        context_manager.__aexit__ = AsyncMock(return_value=None)
        return context_manager
    
    async def test_read_file_tool_success(self, mcp_client, mock_session, mock_server_session_context):
        """Test successful file read"""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock()]
        mock_result.content[0].text = json.dumps({
            "success": True,
            "content": "File content here"
        })
        mock_session.call_tool.return_value = mock_result
        
        with patch.object(mcp_client, '_get_server_session', return_value=mock_server_session_context):
            read_tool = mcp_client._create_read_file_tool()
            result = await read_tool("test_file.txt")
        
        assert "File content here" in result
        mock_session.call_tool.assert_called_once_with("read_file", {
            "file_path": "test_file.txt"
        })
    
    async def test_write_file_tool_success(self, mcp_client, mock_session, mock_server_session_context):
        """Test successful file write"""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock()]
        mock_result.content[0].text = json.dumps({
            "success": True
        })
        mock_session.call_tool.return_value = mock_result
        
        with patch.object(mcp_client, '_get_server_session', return_value=mock_server_session_context):
            write_tool = mcp_client._create_write_file_tool()
            result = await write_tool("test_file.txt", "Test content")
        
        assert "File written successfully" in result
        mock_session.call_tool.assert_called_once_with("write_file", {
            "file_path": "test_file.txt",
            "content": "Test content"
        })
    
    async def test_list_files_tool_success(self, mcp_client, mock_session, mock_server_session_context):
        """Test successful file listing"""
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock()]
        mock_result.content[0].text = json.dumps({
            "success": True,
            "files": ["file1.txt", "file2.md", "subdir/file3.yaml"]
        })
        mock_session.call_tool.return_value = mock_result
        
        with patch.object(mcp_client, '_get_server_session', return_value=mock_server_session_context):
            list_tool = mcp_client._create_list_files_tool()
            result = await list_tool("memory_dir")
        
        assert "Memory files in" in result
        assert "file1.txt" in result
        assert "file2.md" in result
        assert "subdir/file3.yaml" in result
        mock_session.call_tool.assert_called_once_with("list_files", {
            "directory": "memory_dir"
        })


class TestServerSessionManagement:
    """Test cases for server session management"""
    
    def test_unknown_server_error(self, mcp_client):
        """Test error handling for unknown server"""
        async def test_unknown():
            async with mcp_client._get_server_session("unknown_server"):
                pass
        
        with pytest.raises(ValueError, match="Unknown server: unknown_server"):
            asyncio.run(test_unknown())
    
    @patch('mcp_client.stdio_client')
    async def test_server_session_creation(self, mock_stdio_client, mcp_client):
        """Test server session creation"""
        mock_context = AsyncMock()
        mock_read, mock_write = MagicMock(), MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_context
        
        mock_session = AsyncMock()
        with patch('mcp_client.ClientSession', return_value=mock_session):
            async with mcp_client._get_server_session("memory") as session:
                assert session == mock_session
                mock_session.initialize.assert_called_once()


class TestErrorHandling:
    """Test cases for error handling in MCP client"""
    
    async def test_tool_execution_exception(self, mcp_client):
        """Test exception handling in tool execution"""
        with patch.object(mcp_client, '_get_server_session', side_effect=Exception("Connection failed")):
            search_tool = mcp_client._create_memory_search_tool()
            result = await search_tool("test query")
        
        assert "Error searching memories: Connection failed" in result
    
    async def test_malformed_json_response(self, mcp_client):
        """Test handling of malformed JSON responses"""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock()]
        mock_result.content[0].text = "invalid json"
        mock_session.call_tool.return_value = mock_result
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(mcp_client, '_get_server_session', return_value=mock_context):
            search_tool = mcp_client._create_memory_search_tool()
            result = await search_tool("test query")
        
        assert "Error searching memories:" in result


class TestShutdown:
    """Test cases for client shutdown"""
    
    async def test_shutdown_success(self, mcp_client):
        """Test successful shutdown"""
        # Add mock processes to shutdown
        mock_process = MagicMock()
        mcp_client.server_processes["test_server"] = mock_process
        
        await mcp_client.shutdown()
        
        mock_process.terminate.assert_called_once()
        assert len(mcp_client.server_processes) == 0
    
    async def test_shutdown_with_exception(self, mcp_client):
        """Test shutdown with exception during process termination"""
        mock_process = MagicMock()
        mock_process.terminate.side_effect = Exception("Termination failed")
        mcp_client.server_processes["test_server"] = mock_process
        
        # Should not raise exception
        await mcp_client.shutdown()
        
        assert len(mcp_client.server_processes) == 0


if __name__ == "__main__":
    pytest.main([__file__]) 