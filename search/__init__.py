"""
Search module for Obsidian vault.

Provides multiple search strategies:
- fuzzy: Fuzzy filename matching using rapidfuzz
- semantic: Semantic/vector similarity search (coming soon)
"""

from . import fuzzy
from . import semantic

__all__ = ['fuzzy', 'semantic']
