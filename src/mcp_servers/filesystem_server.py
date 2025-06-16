"""
MCP Filesystem Server for Personal AI Assistant

This server provides filesystem operations through the Model Context Protocol (MCP)
for the personal AI assistant's memory system.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import time
import shutil

from mcp.server.fastmcp import FastMCP


# Create the MCP server instance
mcp = FastMCP("filesystem-server")

# Global memory base path - will be set during initialization
MEMORY_BASE_PATH = None


def initialize_server(memory_base_path: str):
    """Initialize the server with the memory base path."""
    global MEMORY_BASE_PATH
    MEMORY_BASE_PATH = Path(memory_base_path)


def _is_safe_path(path: Path) -> bool:
    """
    Check if a path is safe (within memory base path).
    
    Args:
        path: Path to check
        
    Returns:
        True if path is safe
    """
    try:
        # Resolve to absolute path and check if it's within memory base
        resolved_path = path.resolve()
        resolved_base = MEMORY_BASE_PATH.resolve()
        
        return str(resolved_path).startswith(str(resolved_base))
    except Exception:
        return False


@mcp.tool()
def read_memory_file(file_path: str) -> str:
    """
    Read a memory file from the filesystem.
    
    Args:
        file_path: Relative path to the memory file
        
    Returns:
        File contents as string
    """
    try:
        full_path = MEMORY_BASE_PATH / file_path
        if not full_path.exists():
            return f"Error: File {file_path} does not exist"
        
        if not _is_safe_path(full_path):
            return f"Error: Access denied to {file_path}"
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
        
    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool()
def write_memory_file(file_path: str, content: str) -> str:
    """
    Write content to a memory file.
    
    Args:
        file_path: Relative path to the memory file
        content: Content to write
        
    Returns:
        Success or error message
    """
    try:
        full_path = MEMORY_BASE_PATH / file_path
        
        if not _is_safe_path(full_path):
            return f"Error: Access denied to {file_path}"
        
        # Ensure directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote to {file_path}"
        
    except Exception as e:
        return f"Error writing file: {str(e)}"


@mcp.tool()
def append_memory_file(file_path: str, content: str) -> str:
    """
    Append content to a memory file.
    
    Args:
        file_path: Relative path to the memory file
        content: Content to append
        
    Returns:
        Success or error message
    """
    try:
        full_path = MEMORY_BASE_PATH / file_path
        
        if not _is_safe_path(full_path):
            return f"Error: Access denied to {file_path}"
        
        # Ensure directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'a', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully appended to {file_path}"
        
    except Exception as e:
        return f"Error appending to file: {str(e)}"


@mcp.tool()
def list_memory_files(directory: str = "") -> str:
    """
    List memory files in a directory.
    
    Args:
        directory: Relative directory path (empty for root)
        
    Returns:
        JSON string of file list
    """
    try:
        full_path = MEMORY_BASE_PATH / directory
        
        if not _is_safe_path(full_path):
            return f"Error: Access denied to {directory}"
        
        if not full_path.exists():
            return f"Error: Directory {directory} does not exist"
        
        files = []
        for item in full_path.iterdir():
            if item.is_file():
                files.append({
                    "name": item.name,
                    "path": str(item.relative_to(MEMORY_BASE_PATH)),
                    "size": item.stat().st_size,
                    "modified": item.stat().st_mtime
                })
            elif item.is_dir():
                files.append({
                    "name": item.name,
                    "path": str(item.relative_to(MEMORY_BASE_PATH)),
                    "type": "directory"
                })
        
        return json.dumps(files, indent=2)
        
    except Exception as e:
        return f"Error listing files: {str(e)}"


@mcp.tool()
def file_exists(file_path: str) -> str:
    """
    Check if a memory file exists.
    
    Args:
        file_path: Relative path to the memory file
        
    Returns:
        "true" or "false"
    """
    try:
        full_path = MEMORY_BASE_PATH / file_path
        
        if not _is_safe_path(full_path):
            return "false"
        
        exists = full_path.exists()
        return "true" if exists else "false"
        
    except Exception as e:
        return "false"


@mcp.tool()
def create_backup(file_path: str) -> str:
    """
    Create a backup of a memory file.
    
    Args:
        file_path: Relative path to the memory file
        
    Returns:
        Success message with backup path
    """
    try:
        full_path = MEMORY_BASE_PATH / file_path
        
        if not _is_safe_path(full_path):
            return f"Error: Access denied to {file_path}"
        
        if not full_path.exists():
            return f"Error: File {file_path} does not exist"
        
        # Create backup with timestamp
        timestamp = int(time.time())
        backup_path = full_path.with_suffix(f".backup.{timestamp}{full_path.suffix}")
        
        shutil.copy2(full_path, backup_path)
        
        backup_relative = backup_path.relative_to(MEMORY_BASE_PATH)
        return f"Backup created: {backup_relative}"
        
    except Exception as e:
        return f"Error creating backup: {str(e)}"


def main():
    """Main entry point for the filesystem server."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python filesystem_server.py <memory_base_path>", file=sys.stderr)
        sys.exit(1)
    
    memory_base_path = sys.argv[1]
    initialize_server(memory_base_path)
    
    # Let mcp framework handle the event loop
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()