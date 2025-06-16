#!/usr/bin/env python3
"""
Agent module main entry point.

This allows running the CLI via: python -m src.agent
"""

if __name__ == "__main__":
    from .cli import cli
    cli() 