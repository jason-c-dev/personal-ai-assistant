"""
MCP Client Implementation

Proper Model Context Protocol client implementation using stdio transport
for connecting to MCP servers and integrating with Strands agents.
"""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """
    MCP client for connecting to stdio-based MCP servers.
    
    This client follows the official MCP protocol standards and provides
    proper integration with Strands agents through tool wrappers.
    """
    
    def __init__(self, memory_base_path: str):
        self.memory_base_path = Path(memory_base_path)
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self.available_tools: Dict[str, Callable] = {}
        self.server_configs = {
            "memory": {
                "command": sys.executable,
                "args": [
                    str(Path(__file__).parent.parent / "mcp_servers" / "memory_server.py"),
                    str(self.memory_base_path)
                ],
                "env": None
            },
            "filesystem": {
                "command": sys.executable,
                "args": [
                    str(Path(__file__).parent.parent / "mcp_servers" / "filesystem_server.py"),
                    str(self.memory_base_path)
                ],
                "env": None
            }
        }
        
    async def initialize(self) -> bool:
        """Initialize MCP client and create tool wrappers."""
        logger.info("Initializing MCP client with stdio transport...")
        
        try:
            # Create tool wrappers for each server
            await self._create_tool_wrappers()
            
            logger.info(f"MCP client initialized with {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            return False
    
    async def _create_tool_wrappers(self) -> None:
        """Create Strands-compatible tool wrappers for MCP tools."""
        
        # Memory server tools
        self.available_tools.update({
            "search_memories_mcp": self._create_memory_search_tool(),
            "get_user_profile_mcp": self._create_user_profile_tool(),
            "update_memory_mcp": self._create_update_memory_tool(),
            "get_recent_context_mcp": self._create_recent_context_tool(),
        })
        
        # Filesystem server tools
        self.available_tools.update({
            "read_memory_file_mcp": self._create_read_file_tool(),
            "write_memory_file_mcp": self._create_write_file_tool(),
            "list_memory_files_mcp": self._create_list_files_tool(),
        })
    
    @asynccontextmanager
    async def _get_server_session(self, server_name: str):
        """Get a session for the specified MCP server."""
        if server_name not in self.server_configs:
            raise ValueError(f"Unknown server: {server_name}")
        
        config = self.server_configs[server_name]
        server_params = StdioServerParameters(
            command=config["command"],
            args=config["args"],
            env=config.get("env")
        )
        
        async with stdio_client(server_params) as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            yield session
    
    def _create_memory_search_tool(self) -> Callable:
        """Create memory search tool wrapper."""
        async def search_memories_mcp(query: str, limit: int = 10) -> str:
            """Search through stored memories using the MCP memory server."""
            try:
                async with self._get_server_session("memory") as session:
                    result = await session.call_tool("search_memories", {
                        "query": query,
                        "limit": limit
                    })
                    
                    if result.isError:
                        return f"Error searching memories: {result.content}"
                    
                    # Parse the JSON response
                    if result.content and len(result.content) > 0:
                        response_data = json.loads(result.content[0].text)
                        
                        if not response_data.get("success", False):
                            return f"Search failed: {response_data.get('error', 'Unknown error')}"
                        
                        results = response_data.get("results", [])
                        if not results:
                            return f"No memories found for query: {query}"
                        
                        # Format results for display
                        formatted = [f"Found {len(results)} memories for '{query}':"]
                        for i, result in enumerate(results, 1):
                            content = result.get('content', 'No content')[:100]
                            score = result.get('relevance_score', 0)
                            formatted.append(f"{i}. {content}... (Score: {score:.2f})")
                        
                        return "\n".join(formatted)
                    else:
                        return "No response from memory server"
                        
            except Exception as e:
                logger.error(f"Error in search_memories_mcp: {e}")
                return f"Error searching memories: {str(e)}"
        
        return search_memories_mcp
    
    def _create_user_profile_tool(self) -> Callable:
        """Create user profile tool wrapper."""
        async def get_user_profile_mcp() -> str:
            """Get user profile information using the MCP memory server."""
            try:
                async with self._get_server_session("memory") as session:
                    result = await session.call_tool("get_user_profile", {})
                    
                    if result.isError:
                        return f"Error getting user profile: {result.content}"
                    
                    if result.content and len(result.content) > 0:
                        response_data = json.loads(result.content[0].text)
                        
                        if not response_data.get("success", False):
                            return f"Failed to get user profile: {response_data.get('error', 'Unknown error')}"
                        
                        profile = response_data.get("profile", {})
                        content = profile.get("content", "No profile content available")
                        
                        return f"User Profile:\n{content}"
                    else:
                        return "No response from memory server"
                        
            except Exception as e:
                logger.error(f"Error in get_user_profile_mcp: {e}")
                return f"Error getting user profile: {str(e)}"
        
        return get_user_profile_mcp
    
    def _create_update_memory_tool(self) -> Callable:
        """Create update memory tool wrapper."""
        async def update_memory_mcp(content: str, memory_type: str = "conversation") -> str:
            """Store a new memory using the MCP memory server."""
            try:
                async with self._get_server_session("memory") as session:
                    result = await session.call_tool("update_memory", {
                        "content": content,
                        "memory_type": memory_type
                    })
                    
                    if result.isError:
                        return f"Error storing memory: {result.content}"
                    
                    if result.content and len(result.content) > 0:
                        response_data = json.loads(result.content[0].text)
                        
                        if not response_data.get("success", False):
                            return f"Failed to store memory: {response_data.get('error', 'Unknown error')}"
                        
                        memory_id = response_data.get("memory_id", "unknown")
                        return f"Memory stored successfully with ID: {memory_id}"
                    else:
                        return "No response from memory server"
                        
            except Exception as e:
                logger.error(f"Error in update_memory_mcp: {e}")
                return f"Error storing memory: {str(e)}"
        
        return update_memory_mcp
    
    def _create_recent_context_tool(self) -> Callable:
        """Create recent context tool wrapper."""
        async def get_recent_context_mcp(hours: int = 24, limit: int = 10) -> str:
            """Get recent context using the MCP memory server."""
            try:
                async with self._get_server_session("memory") as session:
                    result = await session.call_tool("get_recent_context", {
                        "hours": hours,
                        "limit": limit
                    })
                    
                    if result.isError:
                        return f"Error getting recent context: {result.content}"
                    
                    if result.content and len(result.content) > 0:
                        response_data = json.loads(result.content[0].text)
                        
                        if not response_data.get("success", False):
                            return f"Failed to get recent context: {response_data.get('error', 'Unknown error')}"
                        
                        memories = response_data.get("memories", [])
                        if not memories:
                            return f"No recent memories found in the last {hours} hours"
                        
                        # Format memories for display
                        formatted = [f"Recent context ({len(memories)} memories):"]
                        for i, memory in enumerate(memories, 1):
                            content = memory.get('content', 'No content')[:80]
                            timestamp = memory.get('timestamp', 'Unknown time')
                            formatted.append(f"{i}. [{timestamp}] {content}...")
                        
                        return "\n".join(formatted)
                    else:
                        return "No response from memory server"
                        
            except Exception as e:
                logger.error(f"Error in get_recent_context_mcp: {e}")
                return f"Error getting recent context: {str(e)}"
        
        return get_recent_context_mcp
    
    def _create_read_file_tool(self) -> Callable:
        """Create read file tool wrapper."""
        async def read_memory_file_mcp(file_path: str) -> str:
            """Read a memory file using the MCP filesystem server."""
            try:
                async with self._get_server_session("filesystem") as session:
                    result = await session.call_tool("read_memory_file", {
                        "file_path": file_path
                    })
                    
                    if result.isError:
                        return f"Error reading file: {result.content}"
                    
                    if result.content and len(result.content) > 0:
                        return result.content[0].text
                    else:
                        return "No response from filesystem server"
                        
            except Exception as e:
                logger.error(f"Error in read_memory_file_mcp: {e}")
                return f"Error reading file: {str(e)}"
        
        return read_memory_file_mcp
    
    def _create_write_file_tool(self) -> Callable:
        """Create write file tool wrapper."""
        async def write_memory_file_mcp(file_path: str, content: str) -> str:
            """Write to a memory file using the MCP filesystem server."""
            try:
                async with self._get_server_session("filesystem") as session:
                    result = await session.call_tool("write_memory_file", {
                        "file_path": file_path,
                        "content": content
                    })
                    
                    if result.isError:
                        return f"Error writing file: {result.content}"
                    
                    if result.content and len(result.content) > 0:
                        return result.content[0].text
                    else:
                        return "No response from filesystem server"
                        
            except Exception as e:
                logger.error(f"Error in write_memory_file_mcp: {e}")
                return f"Error writing file: {str(e)}"
        
        return write_memory_file_mcp
    
    def _create_list_files_tool(self) -> Callable:
        """Create list files tool wrapper."""
        async def list_memory_files_mcp(directory: str = "") -> str:
            """List memory files using the MCP filesystem server."""
            try:
                async with self._get_server_session("filesystem") as session:
                    result = await session.call_tool("list_memory_files", {
                        "directory": directory
                    })
                    
                    if result.isError:
                        return f"Error listing files: {result.content}"
                    
                    if result.content and len(result.content) > 0:
                        return result.content[0].text
                    else:
                        return "No response from filesystem server"
                        
            except Exception as e:
                logger.error(f"Error in list_memory_files_mcp: {e}")
                return f"Error listing files: {str(e)}"
        
        return list_memory_files_mcp
    
    def get_available_tools(self) -> Dict[str, Callable]:
        """Get all available MCP tools."""
        return self.available_tools.copy()
    
    async def shutdown(self) -> None:
        """Shutdown the MCP client."""
        logger.info("Shutting down MCP client...")
        
        # Stop any server processes
        for name, process in self.server_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping {name} process: {e}")
        
        self.server_processes.clear()
        self.available_tools.clear()
        logger.info("MCP client shutdown complete") 