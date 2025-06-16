"""
Strands MCP Tools Integration

Proper Strands framework tool wrappers for MCP functionality.
This module creates tools that are properly recognized by the Strands framework.
"""

from typing import Any, Dict, Optional
import logging
import asyncio

from strands import tool

logger = logging.getLogger(__name__)


class StrandsMCPTools:
    """
    Strands-compatible tool wrappers for MCP functionality.
    
    This class creates tools that properly integrate with the Strands framework
    and avoid tool registration warnings.
    """
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self._tools = []
        self._create_tools()
    
    def _create_tools(self):
        """Create all Strands-compatible MCP tools."""
        
        # Memory tools
        @tool
        async def search_memories(query: str, limit: int = 10) -> str:
            """
            Search through stored memories using semantic search.
            
            Args:
                query: Search query string
                limit: Maximum number of results to return
                
            Returns:
                Formatted search results
            """
            if not self.mcp_client:
                return "MCP client not available"
            
            tools = self.mcp_client.get_available_tools()
            if "search_memories_mcp" in tools:
                try:
                    return await tools["search_memories_mcp"](query, limit)
                except Exception as e:
                    return f"Error searching memories: {str(e)}"
            else:
                return "Memory search tool not available"
        
        @tool
        async def get_user_profile() -> str:
            """
            Get the current user profile and preferences.
            
            Returns:
                User profile information
            """
            if not self.mcp_client:
                return "MCP client not available"
            
            tools = self.mcp_client.get_available_tools()
            if "get_user_profile_mcp" in tools:
                try:
                    return await tools["get_user_profile_mcp"]()
                except Exception as e:
                    return f"Error getting user profile: {str(e)}"
            else:
                return "User profile tool not available"
        
        @tool
        async def store_memory(content: str, memory_type: str = "conversation") -> str:
            """
            Store a new memory or conversation.
            
            Args:
                content: The content to store
                memory_type: Type of memory (default: conversation)
                
            Returns:
                Success message with memory ID
            """
            if not self.mcp_client:
                return "MCP client not available"
            
            tools = self.mcp_client.get_available_tools()
            if "update_memory_mcp" in tools:
                try:
                    return await tools["update_memory_mcp"](content, memory_type)
                except Exception as e:
                    return f"Error storing memory: {str(e)}"
            else:
                return "Memory storage tool not available"
        
        @tool
        async def get_recent_context(hours: int = 24, limit: int = 10) -> str:
            """
            Get recent conversation context.
            
            Args:
                hours: Number of hours back to search
                limit: Maximum number of memories to return
                
            Returns:
                Recent conversation context
            """
            if not self.mcp_client:
                return "MCP client not available"
            
            tools = self.mcp_client.get_available_tools()
            if "get_recent_context_mcp" in tools:
                try:
                    return await tools["get_recent_context_mcp"](hours, limit)
                except Exception as e:
                    return f"Error getting recent context: {str(e)}"
            else:
                return "Recent context tool not available"
        
        # File system tools
        @tool
        async def read_memory_file(file_path: str) -> str:
            """
            Read a memory file from the filesystem.
            
            Args:
                file_path: Relative path to the memory file
                
            Returns:
                File contents
            """
            if not self.mcp_client:
                return "MCP client not available"
            
            tools = self.mcp_client.get_available_tools()
            if "read_memory_file_mcp" in tools:
                try:
                    return await tools["read_memory_file_mcp"](file_path)
                except Exception as e:
                    return f"Error reading file: {str(e)}"
            else:
                return "File read tool not available"
        
        @tool
        async def write_memory_file(file_path: str, content: str) -> str:
            """
            Write content to a memory file.
            
            Args:
                file_path: Relative path to the memory file
                content: Content to write
                
            Returns:
                Success or error message
            """
            if not self.mcp_client:
                return "MCP client not available"
            
            tools = self.mcp_client.get_available_tools()
            if "write_memory_file_mcp" in tools:
                try:
                    return await tools["write_memory_file_mcp"](file_path, content)
                except Exception as e:
                    return f"Error writing file: {str(e)}"
            else:
                return "File write tool not available"
        
        @tool
        async def list_memory_files(directory: str = "") -> str:
            """
            List memory files in a directory.
            
            Args:
                directory: Relative directory path (empty for root)
                
            Returns:
                JSON string of file list
            """
            if not self.mcp_client:
                return "MCP client not available"
            
            tools = self.mcp_client.get_available_tools()
            if "list_memory_files_mcp" in tools:
                try:
                    return await tools["list_memory_files_mcp"](directory)
                except Exception as e:
                    return f"Error listing files: {str(e)}"
            else:
                return "File list tool not available"
        
        # Store tools for access
        self._tools = [
            search_memories,
            get_user_profile,
            store_memory,
            get_recent_context,
            read_memory_file,
            write_memory_file,
            list_memory_files
        ]
    
    def get_tools(self):
        """Get all Strands-compatible tools."""
        return self._tools 