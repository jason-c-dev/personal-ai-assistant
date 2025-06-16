"""
MCP Integration Module

Handles integration with Model Context Protocol (MCP) servers,
providing seamless access to filesystem and memory tools.
"""

from typing import List, Optional, Dict, Any
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import timedelta

from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client

from agent.agent_config import MCPConfig


logger = logging.getLogger(__name__)


class MCPIntegration:
    """
    Manages MCP server connections and tool discovery for the Personal AI Assistant.
    
    This class handles:
    - Connection management to multiple MCP servers
    - Tool discovery and registration
    - Error handling and retry logic
    - Health monitoring of MCP connections
    """
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.clients: Dict[str, MCPClient] = {}
        self.connected_servers: Dict[str, bool] = {}
        self.available_tools: List[Any] = []
        
    async def initialize(self) -> bool:
        """
        Initialize MCP connections and discover available tools.
        
        Returns:
            bool: True if at least one MCP server connected successfully
        """
        if not self.config.enabled:
            logger.info("MCP integration disabled in configuration")
            return True
            
        logger.info("Initializing MCP integration...")
        
        # Define MCP servers to connect to
        servers = {
            "filesystem": self.config.filesystem_server_url,
            "memory": self.config.memory_server_url
        }
        
        success_count = 0
        
        for server_name, server_url in servers.items():
            try:
                success = await self._connect_to_server(server_name, server_url)
                if success:
                    success_count += 1
                    logger.info(f"Successfully connected to {server_name} MCP server")
                else:
                    logger.warning(f"Failed to connect to {server_name} MCP server")
                    
            except Exception as e:
                logger.error(f"Error connecting to {server_name} MCP server: {e}")
                self.connected_servers[server_name] = False
        
        # Discover tools from connected servers
        await self._discover_tools()
        
        logger.info(f"MCP integration initialized. Connected to {success_count}/{len(servers)} servers")
        logger.info(f"Discovered {len(self.available_tools)} MCP tools")
        
        return success_count > 0
    
    async def _connect_to_server(self, server_name: str, server_url: str) -> bool:
        """
        Connect to a specific MCP server with retry logic.
        
        Args:
            server_name: Name identifier for the server
            server_url: URL of the MCP server
            
        Returns:
            bool: True if connection successful
        """
        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(f"Attempting to connect to {server_name} (attempt {attempt + 1})")
                
                # Create MCP client with timeout
                client = MCPClient(
                    lambda: streamablehttp_client(
                        server_url,
                        timeout=timedelta(seconds=self.config.connection_timeout)
                    )
                )
                
                # Test connection by listing tools
                async with client:
                    tools = await client.list_tools()
                    if tools:
                        self.clients[server_name] = client
                        self.connected_servers[server_name] = True
                        logger.debug(f"Found {len(tools)} tools on {server_name} server")
                        return True
                        
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed for {server_name}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        self.connected_servers[server_name] = False
        return False
    
    async def _discover_tools(self) -> None:
        """Discover and collect tools from all connected MCP servers."""
        self.available_tools = []
        
        for server_name, client in self.clients.items():
            if not self.connected_servers.get(server_name, False):
                continue
                
            try:
                async with client:
                    tools = await client.list_tools()
                    self.available_tools.extend(tools)
                    logger.debug(f"Added {len(tools)} tools from {server_name} server")
                    
            except Exception as e:
                logger.error(f"Error discovering tools from {server_name}: {e}")
                self.connected_servers[server_name] = False
    
    def get_available_tools(self) -> List[Any]:
        """
        Get list of all available MCP tools.
        
        Returns:
            List of MCP tools ready for use with Strands agents
        """
        return self.available_tools.copy()
    
    def get_connected_servers(self) -> Dict[str, bool]:
        """
        Get status of MCP server connections.
        
        Returns:
            Dictionary mapping server names to connection status
        """
        return self.connected_servers.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all MCP connections.
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            "overall_status": "healthy",
            "connected_servers": 0,
            "total_servers": len(self.clients),
            "available_tools": len(self.available_tools),
            "server_details": {}
        }
        
        for server_name, client in self.clients.items():
            server_status = {
                "connected": False,
                "tools_count": 0,
                "last_error": None
            }
            
            try:
                async with client:
                    tools = await client.list_tools()
                    server_status["connected"] = True
                    server_status["tools_count"] = len(tools)
                    health_status["connected_servers"] += 1
                    
            except Exception as e:
                server_status["last_error"] = str(e)
                logger.warning(f"Health check failed for {server_name}: {e}")
                
            health_status["server_details"][server_name] = server_status
        
        # Update overall status
        if health_status["connected_servers"] == 0:
            health_status["overall_status"] = "unhealthy"
        elif health_status["connected_servers"] < health_status["total_servers"]:
            health_status["overall_status"] = "degraded"
            
        return health_status
    
    async def reconnect_failed_servers(self) -> int:
        """
        Attempt to reconnect to failed MCP servers.
        
        Returns:
            Number of servers successfully reconnected
        """
        reconnected = 0
        
        servers = {
            "filesystem": self.config.filesystem_server_url,
            "memory": self.config.memory_server_url
        }
        
        for server_name, server_url in servers.items():
            if not self.connected_servers.get(server_name, False):
                logger.info(f"Attempting to reconnect to {server_name} server...")
                
                success = await self._connect_to_server(server_name, server_url)
                if success:
                    reconnected += 1
                    logger.info(f"Successfully reconnected to {server_name} server")
        
        if reconnected > 0:
            await self._discover_tools()
            logger.info(f"Reconnected to {reconnected} servers, rediscovered tools")
            
        return reconnected
    
    @asynccontextmanager
    async def get_client_context(self, server_name: str):
        """
        Get an async context manager for a specific MCP client.
        
        Args:
            server_name: Name of the MCP server
            
        Yields:
            MCPClient instance ready for use
        """
        if server_name not in self.clients:
            raise ValueError(f"MCP server '{server_name}' not found")
            
        if not self.connected_servers.get(server_name, False):
            raise ConnectionError(f"MCP server '{server_name}' is not connected")
            
        client = self.clients[server_name]
        async with client:
            yield client
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all MCP connections."""
        logger.info("Shutting down MCP integration...")
        
        for server_name in list(self.clients.keys()):
            try:
                # MCP clients are context managers, they'll clean up automatically
                self.connected_servers[server_name] = False
                logger.debug(f"Disconnected from {server_name} server")
                
            except Exception as e:
                logger.warning(f"Error during shutdown of {server_name}: {e}")
        
        self.clients.clear()
        self.available_tools.clear()
        logger.info("MCP integration shutdown complete") 