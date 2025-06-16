"""
Memory system for the Personal AI Assistant.

This package handles persistent memory storage, retrieval, and management
for building long-term relationships with users.
"""

from .memory_initializer import MemoryInitializer, initialize_memory_system

__all__ = ['MemoryInitializer', 'initialize_memory_system'] 