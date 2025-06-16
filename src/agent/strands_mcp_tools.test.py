"""
Unit tests for strands_mcp_tools module

Tests for Strands-compatible MCP tool wrappers including
tool creation, integration, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from strands_mcp_tools import StrandsMCPTools


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client for testing"""
    client = MagicMock()
    client.get_available_tools.return_value = {}
    return client


@pytest.fixture
def strands_tools(mock_mcp_client):
    """Create a StrandsMCPTools instance for testing"""
    return StrandsMCPTools(mock_mcp_client)


class TestStrandsMCPToolsInitialization:
    """Test cases for StrandsMCPTools initialization"""
    
    def test_initialization_with_client(self, mock_mcp_client):
        """Test initialization with MCP client"""
        tools = StrandsMCPTools(mock_mcp_client)
        
        assert tools.mcp_client == mock_mcp_client
        assert isinstance(tools._tools, list)
        assert len(tools._tools) == 7  # All 7 tools should be created
    
    def test_initialization_without_client(self):
        """Test initialization without MCP client"""
        tools = StrandsMCPTools(None)
        
        assert tools.mcp_client is None
        assert isinstance(tools._tools, list)
        assert len(tools._tools) == 7  # Tools should still be created
    
    def test_get_tools(self, strands_tools):
        """Test getting the tools list"""
        tools = strands_tools.get_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 7
        assert all(callable(tool) for tool in tools)


class TestMemoryTools:
    """Test cases for memory-related tools"""
    
    async def test_search_memories_success(self, strands_tools):
        """Test successful memory search"""
        mock_search_tool = AsyncMock(return_value="Search results")
        strands_tools.mcp_client.get_available_tools.return_value = {
            "search_memories_mcp": mock_search_tool
        }
        
        tools = strands_tools.get_tools()
        search_memories = tools[0]  # First tool should be search_memories
        
        result = await search_memories("test query", 5)
        
        assert result == "Search results"
        mock_search_tool.assert_called_once_with("test query", 5)
    
    async def test_search_memories_no_client(self):
        """Test memory search with no MCP client"""
        tools = StrandsMCPTools(None)
        search_memories = tools.get_tools()[0]
        
        result = await search_memories("test query")
        
        assert result == "MCP client not available"
    
    async def test_search_memories_tool_not_available(self, strands_tools):
        """Test memory search when tool is not available"""
        strands_tools.mcp_client.get_available_tools.return_value = {}
        
        search_memories = strands_tools.get_tools()[0]
        result = await search_memories("test query")
        
        assert result == "Memory search tool not available"
    
    async def test_search_memories_exception(self, strands_tools):
        """Test memory search with exception"""
        mock_search_tool = AsyncMock(side_effect=Exception("Search failed"))
        strands_tools.mcp_client.get_available_tools.return_value = {
            "search_memories_mcp": mock_search_tool
        }
        
        search_memories = strands_tools.get_tools()[0]
        result = await search_memories("test query")
        
        assert "Error searching memories: Search failed" in result
    
    async def test_get_user_profile_success(self, strands_tools):
        """Test successful user profile retrieval"""
        mock_profile_tool = AsyncMock(return_value="User profile data")
        strands_tools.mcp_client.get_available_tools.return_value = {
            "get_user_profile_mcp": mock_profile_tool
        }
        
        get_user_profile = strands_tools.get_tools()[1]  # Second tool
        result = await get_user_profile()
        
        assert result == "User profile data"
        mock_profile_tool.assert_called_once_with()
    
    async def test_get_user_profile_no_client(self):
        """Test user profile retrieval with no MCP client"""
        tools = StrandsMCPTools(None)
        get_user_profile = tools.get_tools()[1]
        
        result = await get_user_profile()
        
        assert result == "MCP client not available"
    
    async def test_get_user_profile_tool_not_available(self, strands_tools):
        """Test user profile retrieval when tool is not available"""
        strands_tools.mcp_client.get_available_tools.return_value = {}
        
        get_user_profile = strands_tools.get_tools()[1]
        result = await get_user_profile()
        
        assert result == "User profile tool not available"
    
    async def test_store_memory_success(self, strands_tools):
        """Test successful memory storage"""
        mock_store_tool = AsyncMock(return_value="Memory stored with ID: mem_123")
        strands_tools.mcp_client.get_available_tools.return_value = {
            "update_memory_mcp": mock_store_tool
        }
        
        store_memory = strands_tools.get_tools()[2]  # Third tool
        result = await store_memory("Test content", "conversation")
        
        assert result == "Memory stored with ID: mem_123"
        mock_store_tool.assert_called_once_with("Test content", "conversation")
    
    async def test_store_memory_default_type(self, strands_tools):
        """Test memory storage with default type"""
        mock_store_tool = AsyncMock(return_value="Memory stored")
        strands_tools.mcp_client.get_available_tools.return_value = {
            "update_memory_mcp": mock_store_tool
        }
        
        store_memory = strands_tools.get_tools()[2]
        result = await store_memory("Test content")
        
        assert result == "Memory stored"
        mock_store_tool.assert_called_once_with("Test content", "conversation")
    
    async def test_get_recent_context_success(self, strands_tools):
        """Test successful recent context retrieval"""
        mock_context_tool = AsyncMock(return_value="Recent context data")
        strands_tools.mcp_client.get_available_tools.return_value = {
            "get_recent_context_mcp": mock_context_tool
        }
        
        get_recent_context = strands_tools.get_tools()[3]  # Fourth tool
        result = await get_recent_context(12, 5)
        
        assert result == "Recent context data"
        mock_context_tool.assert_called_once_with(12, 5)
    
    async def test_get_recent_context_defaults(self, strands_tools):
        """Test recent context retrieval with default parameters"""
        mock_context_tool = AsyncMock(return_value="Recent context")
        strands_tools.mcp_client.get_available_tools.return_value = {
            "get_recent_context_mcp": mock_context_tool
        }
        
        get_recent_context = strands_tools.get_tools()[3]
        result = await get_recent_context()
        
        assert result == "Recent context"
        mock_context_tool.assert_called_once_with(24, 10)


class TestFileSystemTools:
    """Test cases for filesystem-related tools"""
    
    async def test_read_memory_file_success(self, strands_tools):
        """Test successful file reading"""
        mock_read_tool = AsyncMock(return_value="File content")
        strands_tools.mcp_client.get_available_tools.return_value = {
            "read_memory_file_mcp": mock_read_tool
        }
        
        read_memory_file = strands_tools.get_tools()[4]  # Fifth tool
        result = await read_memory_file("test_file.txt")
        
        assert result == "File content"
        mock_read_tool.assert_called_once_with("test_file.txt")
    
    async def test_read_memory_file_no_client(self):
        """Test file reading with no MCP client"""
        tools = StrandsMCPTools(None)
        read_memory_file = tools.get_tools()[4]
        
        result = await read_memory_file("test_file.txt")
        
        assert result == "MCP client not available"
    
    async def test_read_memory_file_tool_not_available(self, strands_tools):
        """Test file reading when tool is not available"""
        strands_tools.mcp_client.get_available_tools.return_value = {}
        
        read_memory_file = strands_tools.get_tools()[4]
        result = await read_memory_file("test_file.txt")
        
        assert result == "File read tool not available"
    
    async def test_read_memory_file_exception(self, strands_tools):
        """Test file reading with exception"""
        mock_read_tool = AsyncMock(side_effect=Exception("Read failed"))
        strands_tools.mcp_client.get_available_tools.return_value = {
            "read_memory_file_mcp": mock_read_tool
        }
        
        read_memory_file = strands_tools.get_tools()[4]
        result = await read_memory_file("test_file.txt")
        
        assert "Error reading file: Read failed" in result
    
    async def test_write_memory_file_success(self, strands_tools):
        """Test successful file writing"""
        mock_write_tool = AsyncMock(return_value="File written successfully")
        strands_tools.mcp_client.get_available_tools.return_value = {
            "write_memory_file_mcp": mock_write_tool
        }
        
        write_memory_file = strands_tools.get_tools()[5]  # Sixth tool
        result = await write_memory_file("test_file.txt", "Test content")
        
        assert result == "File written successfully"
        mock_write_tool.assert_called_once_with("test_file.txt", "Test content")
    
    async def test_write_memory_file_no_client(self):
        """Test file writing with no MCP client"""
        tools = StrandsMCPTools(None)
        write_memory_file = tools.get_tools()[5]
        
        result = await write_memory_file("test_file.txt", "content")
        
        assert result == "MCP client not available"
    
    async def test_write_memory_file_tool_not_available(self, strands_tools):
        """Test file writing when tool is not available"""
        strands_tools.mcp_client.get_available_tools.return_value = {}
        
        write_memory_file = strands_tools.get_tools()[5]
        result = await write_memory_file("test_file.txt", "content")
        
        assert result == "File write tool not available"
    
    async def test_list_memory_files_success(self, strands_tools):
        """Test successful file listing"""
        mock_list_tool = AsyncMock(return_value='["file1.txt", "file2.md"]')
        strands_tools.mcp_client.get_available_tools.return_value = {
            "list_memory_files_mcp": mock_list_tool
        }
        
        list_memory_files = strands_tools.get_tools()[6]  # Seventh tool
        result = await list_memory_files("memories")
        
        assert result == '["file1.txt", "file2.md"]'
        mock_list_tool.assert_called_once_with("memories")
    
    async def test_list_memory_files_default_directory(self, strands_tools):
        """Test file listing with default directory"""
        mock_list_tool = AsyncMock(return_value='["root_file.txt"]')
        strands_tools.mcp_client.get_available_tools.return_value = {
            "list_memory_files_mcp": mock_list_tool
        }
        
        list_memory_files = strands_tools.get_tools()[6]
        result = await list_memory_files()
        
        assert result == '["root_file.txt"]'
        mock_list_tool.assert_called_once_with("")
    
    async def test_list_memory_files_no_client(self):
        """Test file listing with no MCP client"""
        tools = StrandsMCPTools(None)
        list_memory_files = tools.get_tools()[6]
        
        result = await list_memory_files()
        
        assert result == "MCP client not available"
    
    async def test_list_memory_files_tool_not_available(self, strands_tools):
        """Test file listing when tool is not available"""
        strands_tools.mcp_client.get_available_tools.return_value = {}
        
        list_memory_files = strands_tools.get_tools()[6]
        result = await list_memory_files()
        
        assert result == "File list tool not available"


class TestToolsIntegration:
    """Integration tests for StrandsMCPTools"""
    
    async def test_all_tools_with_full_mcp_client(self, strands_tools):
        """Test all tools with fully configured MCP client"""
        # Setup all tools
        mock_tools = {
            "search_memories_mcp": AsyncMock(return_value="Search results"),
            "get_user_profile_mcp": AsyncMock(return_value="Profile data"),
            "update_memory_mcp": AsyncMock(return_value="Memory stored"),
            "get_recent_context_mcp": AsyncMock(return_value="Context data"),
            "read_memory_file_mcp": AsyncMock(return_value="File content"),
            "write_memory_file_mcp": AsyncMock(return_value="File written"),
            "list_memory_files_mcp": AsyncMock(return_value='["file.txt"]')
        }
        strands_tools.mcp_client.get_available_tools.return_value = mock_tools
        
        tools = strands_tools.get_tools()
        
        # Test each tool
        result1 = await tools[0]("query")  # search_memories
        result2 = await tools[1]()  # get_user_profile
        result3 = await tools[2]("content")  # store_memory
        result4 = await tools[3]()  # get_recent_context
        result5 = await tools[4]("file.txt")  # read_memory_file
        result6 = await tools[5]("file.txt", "content")  # write_memory_file
        result7 = await tools[6]()  # list_memory_files
        
        assert result1 == "Search results"
        assert result2 == "Profile data"
        assert result3 == "Memory stored"
        assert result4 == "Context data"
        assert result5 == "File content"
        assert result6 == "File written"
        assert result7 == '["file.txt"]'
        
        # Verify all tools were called correctly
        mock_tools["search_memories_mcp"].assert_called_once_with("query", 10)
        mock_tools["get_user_profile_mcp"].assert_called_once_with()
        mock_tools["update_memory_mcp"].assert_called_once_with("content", "conversation")
        mock_tools["get_recent_context_mcp"].assert_called_once_with(24, 10)
        mock_tools["read_memory_file_mcp"].assert_called_once_with("file.txt")
        mock_tools["write_memory_file_mcp"].assert_called_once_with("file.txt", "content")
        mock_tools["list_memory_files_mcp"].assert_called_once_with("")
    
    def test_tool_names_and_docstrings(self, strands_tools):
        """Test that tools have proper names and documentation"""
        tools = strands_tools.get_tools()
        
        # Check that all tools are callable and have docstrings
        for tool in tools:
            assert callable(tool)
            assert hasattr(tool, '__doc__')
            assert tool.__doc__ is not None
            assert len(tool.__doc__.strip()) > 0
            
        # Check specific tool documentation
        search_tool = tools[0]
        assert "search" in search_tool.__doc__.lower()
        assert "query" in search_tool.__doc__.lower()
        
        profile_tool = tools[1]
        assert "profile" in profile_tool.__doc__.lower()
        
        store_tool = tools[2]
        assert "store" in store_tool.__doc__.lower()
        
        context_tool = tools[3]
        assert "context" in context_tool.__doc__.lower()
        
        read_tool = tools[4]
        assert "read" in read_tool.__doc__.lower()
        
        write_tool = tools[5]
        assert "write" in write_tool.__doc__.lower()
        
        list_tool = tools[6]
        assert "list" in list_tool.__doc__.lower()


class TestErrorHandling:
    """Test cases for error handling across all tools"""
    
    async def test_all_tools_no_mcp_client(self):
        """Test all tools when MCP client is None"""
        tools = StrandsMCPTools(None)
        tool_list = tools.get_tools()
        
        # Test all tools return appropriate error message
        for tool in tool_list:
            # Get function signature to call with appropriate arguments
            func_name = tool.__name__
            if func_name in ["get_user_profile", "list_memory_files", "get_recent_context"]:
                result = await tool()
            elif func_name == "search_memories":
                result = await tool("query")
            elif func_name == "store_memory":
                result = await tool("content")
            elif func_name == "read_memory_file":
                result = await tool("file.txt")
            elif func_name == "write_memory_file":
                result = await tool("file.txt", "content")
            else:
                # Fallback for any other tools
                result = await tool()
                
            assert result == "MCP client not available"
    
    async def test_all_tools_missing_mcp_tools(self, strands_tools):
        """Test all tools when MCP tools are not available"""
        strands_tools.mcp_client.get_available_tools.return_value = {}
        tool_list = strands_tools.get_tools()
        
        expected_messages = [
            "Memory search tool not available",
            "User profile tool not available",
            "Memory storage tool not available",
            "Recent context tool not available",
            "File read tool not available",
            "File write tool not available",
            "File list tool not available"
        ]
        
        # Test each tool
        results = []
        results.append(await tool_list[0]("query"))  # search_memories
        results.append(await tool_list[1]())  # get_user_profile
        results.append(await tool_list[2]("content"))  # store_memory
        results.append(await tool_list[3]())  # get_recent_context
        results.append(await tool_list[4]("file.txt"))  # read_memory_file
        results.append(await tool_list[5]("file.txt", "content"))  # write_memory_file
        results.append(await tool_list[6]())  # list_memory_files
        
        for result, expected in zip(results, expected_messages):
            assert result == expected


if __name__ == "__main__":
    pytest.main([__file__]) 