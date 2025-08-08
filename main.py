#!/usr/bin/env python3
"""
Enterprise Doc Bot - Main entry point

This is a RAG system that ingests GitHub repositories and Confluence docs
to answer questions about codebases and documentation using local LLMs.

Usage:
    python main.py ingest              # Ingest documents
    python main.py serve               # Start web interface
    python main.py chat --interactive  # Interactive CLI chat
    python main.py status              # Show system status
"""

from src.cli import main

if __name__ == '__main__':
    main()
