"""
MCP server for memory operations and retrieval.
Provides tools for searching memories, retrieving context, and managing user profiles.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, TextResourceContents

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_manager import MemoryManager
from memory.file_operations import MemoryFileOperations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global MCP server instance
mcp = FastMCP("memory-server")

# Global instances
memory_manager: Optional[MemoryManager] = None
file_ops: Optional[MemoryFileOperations] = None

def initialize_memory_server(memory_base_path: str):
    """Initialize the memory server with the given base path."""
    global memory_manager, file_ops
    
    base_path = Path(memory_base_path)
    file_ops = MemoryFileOperations()
    memory_manager = MemoryManager(str(memory_base_path))
    
    logger.info(f"Memory server initialized with base path: {base_path}")

@mcp.tool()
def search_memories(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search through stored memories using semantic search.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        Dictionary containing search results and metadata
    """
    if not memory_manager:
        raise RuntimeError("Memory server not initialized")
    
    try:
        results = memory_manager.search_memories(query, limit=limit)
        
        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "content": result.content_snippet,
                    "timestamp": result.created,
                    "memory_type": result.category,
                    "tags": [],  # Tags not available in SearchResult
                    "metadata": {"file_path": result.file_path},
                    "relevance_score": result.relevance_score
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results_count": 0,
            "results": []
        }

@mcp.tool()
def get_recent_context(hours: int = 24, limit: int = 20) -> Dict[str, Any]:
    """
    Retrieve recent memories for context.
    
    Args:
        hours: Number of hours back to search (default: 24)
        limit: Maximum number of memories to return (default: 20)
    
    Returns:
        Dictionary containing recent memories and metadata
    """
    if not memory_manager:
        raise RuntimeError("Memory server not initialized")
    
    try:
        # Convert hours to days for the method
        days = max(1, hours // 24)
        memories = memory_manager.get_recent_interactions(days=days, limit=limit)
        
        return {
            "success": True,
            "hours_back": hours,
            "memories_count": len(memories),
            "memories": [
                {
                    "content": memory.get("content", ""),
                    "timestamp": memory.get("created", ""),
                    "memory_type": memory.get("category", "interaction"),
                    "tags": memory.get("tags", []),
                    "metadata": memory.get("metadata", {})
                }
                for memory in memories
            ]
        }
    except Exception as e:
        logger.error(f"Error getting recent context: {e}")
        return {
            "success": False,
            "error": str(e),
            "hours_back": hours,
            "memories_count": 0,
            "memories": []
        }

@mcp.tool()
def get_user_profile() -> Dict[str, Any]:
    """
    Retrieve the current user profile and preferences.
    
    Returns:
        Dictionary containing user profile information
    """
    if not memory_manager:
        raise RuntimeError("Memory server not initialized")
    
    try:
        # Get user profile from core memory
        frontmatter, content = memory_manager.get_core_memory("user_profile")
        
        return {
            "success": True,
            "profile": {
                "content": content,
                "metadata": frontmatter
            }
        }
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return {
            "success": False,
            "error": str(e),
            "profile": {}
        }

@mcp.tool()
def update_memory(content: str, memory_type: str = "conversation", 
                 tags: Optional[List[str]] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Store a new memory or update existing memory.
    
    Args:
        content: The content to store
        memory_type: Type of memory (default: "conversation")
        tags: Optional list of tags
        metadata: Optional metadata dictionary
    
    Returns:
        Dictionary containing operation result
    """
    if not memory_manager:
        raise RuntimeError("Memory server not initialized")
    
    try:
        from memory.memory_manager import MemoryEntry
        
        # Create a memory entry
        entry = MemoryEntry(
            content=content,
            importance_score=5,  # Default importance
            category=memory_type,
            file_type="interaction",
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Create the memory using the correct method
        file_path = memory_manager.create_interaction_memory(entry)
        
        return {
            "success": True,
            "memory_id": Path(file_path).stem,  # Use filename as ID
            "timestamp": datetime.now().isoformat(),
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def get_memory_stats() -> Dict[str, Any]:
    """
    Get statistics about the memory system.
    
    Returns:
        Dictionary containing memory statistics
    """
    if not memory_manager:
        raise RuntimeError("Memory server not initialized")
    
    try:
        stats = memory_manager.get_memory_statistics()
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {
            "success": False,
            "error": str(e),
            "stats": {}
        }

@mcp.tool()
def find_related_memories(memory_id: str, limit: int = 5) -> Dict[str, Any]:
    """
    Find memories related to a specific memory.
    
    Args:
        memory_id: ID of the memory to find relations for
        limit: Maximum number of related memories to return (default: 5)
    
    Returns:
        Dictionary containing related memories
    """
    if not memory_manager:
        raise RuntimeError("Memory server not initialized")
    
    try:
        # Use search to find related memories based on the memory_id as a query
        # This is a simplified approach - in a real implementation you'd load the memory content first
        related = memory_manager.search_memories(
            query=memory_id,  # Use memory_id as search term
            limit=limit
        )
        
        return {
            "success": True,
            "source_memory_id": memory_id,
            "related_count": len(related),
            "related_memories": [
                {
                    "memory_id": Path(result.file_path).stem,
                    "content": result.content_snippet,
                    "timestamp": result.created,
                    "memory_type": result.category,
                    "tags": [],  # Tags not available in SearchResult
                    "similarity_score": result.relevance_score
                }
                for result in related
            ]
        }
    except Exception as e:
        logger.error(f"Error finding related memories: {e}")
        return {
            "success": False,
            "error": str(e),
            "source_memory_id": memory_id,
            "related_count": 0,
            "related_memories": []
        }

@mcp.resource("memory://search")
def memory_search_resource() -> Resource:
    """Resource for memory search capabilities."""
    return Resource(
        uri="memory://search",
        name="Memory Search",
        description="Search through stored memories using semantic search",
        mimeType="application/json"
    )

@mcp.resource("memory://profile")
def user_profile_resource() -> Resource:
    """Resource for user profile access."""
    return Resource(
        uri="memory://profile", 
        name="User Profile",
        description="Access to user profile and preferences",
        mimeType="application/json"
    )

@mcp.resource("memory://recent")
def recent_context_resource() -> Resource:
    """Resource for recent context access."""
    return Resource(
        uri="memory://recent",
        name="Recent Context", 
        description="Access to recent memories for context",
        mimeType="application/json"
    )

def main():
    """Main entry point for the memory server."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python memory_server.py <memory_base_path>", file=sys.stderr)
        sys.exit(1)
    
    memory_base_path = sys.argv[1]
    initialize_memory_server(memory_base_path)
    
    # Let mcp framework handle the event loop
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()