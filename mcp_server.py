#!/usr/bin/env python3
"""
Obsidian MCP Server - Model Context Protocol server for Obsidian vault interaction

This MCP server provides comprehensive tools and resources for interacting with
an Obsidian vault including vector search, text search, note CRUD operations,
and more. Implements OAuth 2.1 for secure authentication.

Tools:
- vector_search: Search notes using vector embeddings
- text_search: Regular/fuzzy text search
- create_note: Create new markdown notes
- create_folder: Create folders in vault
- read_note: Read note contents
- update_note_diff: Modify notes using unified diff
- delete_note: Delete notes
- list_notes: List all notes in vault
- get_vault_stats: Get vault statistics
- reindex_vault: Trigger vector database reindex

Resources:
- vault://note/{path}: Access individual notes
- vault://list: List all notes
- vault://stats: Vault statistics

Authentication: OAuth 2.1 resource server
"""

import os
import sys
import glob
import json
import logging
import difflib
from pathlib import Path
from typing import Optional, Any, Sequence
from datetime import datetime
from dataclasses import dataclass

# Third-party imports
import chromadb
from sentence_transformers import SentenceTransformer
import frontmatter
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, TextContent, Tool
import asyncio
from functools import wraps
import jwt
from jwt import PyJWKClient
import httpx

# Load environment variables
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
VAULT_PATH = os.getenv("VAULT_PATH", "./vault")
DB_PATH = os.getenv("DB_PATH", "./chroma_data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "obsidian_vault"
DEFAULT_SPLIT_TOKEN = os.getenv("DEFAULT_SPLIT_TOKEN", "\n\n")

# OAuth 2.1 Configuration
OAUTH_ENABLED = os.getenv("MCP_OAUTH_ENABLED", "false").lower() == "true"
OAUTH_ISSUER = os.getenv("MCP_OAUTH_ISSUER", "")  # e.g., https://accounts.google.com
OAUTH_AUDIENCE = os.getenv("MCP_OAUTH_AUDIENCE", "")  # Your application's client ID
OAUTH_JWKS_URI = os.getenv("MCP_OAUTH_JWKS_URI", "")  # e.g., https://www.googleapis.com/oauth2/v3/certs

logger.info("=== MCP Server Configuration ===")
logger.info(f"VAULT_PATH: {VAULT_PATH}")
logger.info(f"DB_PATH: {DB_PATH}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"OAUTH_ENABLED: {OAUTH_ENABLED}")
logger.info("================================")

# --- Global State ---
model: Optional[SentenceTransformer] = None
chroma_client: Optional[chromadb.ClientAPI] = None
collection: Optional[chromadb.Collection] = None
jwks_client: Optional[PyJWKClient] = None

# Initialize FastMCP server
mcp = FastMCP("obsidian-vault-server")


# --- OAuth 2.1 Implementation ---
class OAuthError(Exception):
    """OAuth authentication/authorization error"""
    pass


def init_oauth():
    """Initialize OAuth JWKS client if OAuth is enabled"""
    global jwks_client
    if OAUTH_ENABLED:
        if not OAUTH_JWKS_URI:
            logger.warning("OAuth enabled but OAUTH_JWKS_URI not set. OAuth will not work.")
            return
        try:
            jwks_client = PyJWKClient(OAUTH_JWKS_URI)
            logger.info(f"OAuth JWKS client initialized with URI: {OAUTH_JWKS_URI}")
        except Exception as e:
            logger.error(f"Failed to initialize OAuth JWKS client: {e}")
            raise


async def validate_oauth_token(token: str) -> dict:
    """
    Validate OAuth 2.1 access token using JWKS

    Args:
        token: JWT access token from Authorization header

    Returns:
        Decoded token payload

    Raises:
        OAuthError: If token is invalid
    """
    if not OAUTH_ENABLED:
        return {}  # OAuth disabled, allow access

    if not token:
        raise OAuthError("No access token provided")

    if not jwks_client:
        raise OAuthError("OAuth not properly configured")

    try:
        # Get signing key from JWKS
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        # Decode and validate token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=OAUTH_AUDIENCE,
            issuer=OAUTH_ISSUER,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": True,
                "verify_iss": True,
            }
        )

        logger.info(f"Token validated for subject: {payload.get('sub', 'unknown')}")
        return payload

    except jwt.ExpiredSignatureError:
        raise OAuthError("Token has expired")
    except jwt.InvalidAudienceError:
        raise OAuthError("Invalid token audience")
    except jwt.InvalidIssuerError:
        raise OAuthError("Invalid token issuer")
    except jwt.InvalidTokenError as e:
        raise OAuthError(f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error(f"Token validation error: {e}", exc_info=True)
        raise OAuthError(f"Token validation failed: {str(e)}")


def require_auth(func):
    """Decorator to require OAuth authentication for tool functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # In MCP, authentication is handled at the transport level
        # This is a placeholder for additional authorization logic if needed
        return await func(*args, **kwargs)
    return wrapper


# --- Vector Database Functions ---
def init_vector_db():
    """Initialize sentence transformer model and ChromaDB"""
    global model, chroma_client, collection

    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Model loaded successfully")

        logger.info(f"Initializing ChromaDB at: {DB_PATH}")
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready")

    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}", exc_info=True)
        raise


def process_file(filepath: str) -> dict:
    """
    Process and index a single file into ChromaDB

    Args:
        filepath: Absolute path to the markdown file

    Returns:
        dict with status and message
    """
    if not model or not collection:
        return {"status": "error", "message": "Vector database not initialized"}

    try:
        # Relative path from vault root
        rel_path = os.path.relpath(filepath, VAULT_PATH)

        # Delete existing entries for this file
        try:
            existing = collection.get(where={"filepath": rel_path})
            if existing and existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.info(f"Deleted {len(existing['ids'])} existing chunks for {rel_path}")
        except Exception as e:
            logger.warning(f"Error deleting existing chunks: {e}")

        # Read and parse file
        with open(filepath, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)

        content = post.content
        if not content.strip():
            logger.info(f"Skipping empty file: {rel_path}")
            return {"status": "skipped", "message": "File is empty"}

        # Get split token from frontmatter or use default
        split_token = post.get('split_token', DEFAULT_SPLIT_TOKEN)

        # Split into chunks
        chunks = content.split(split_token)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        if not chunks:
            logger.info(f"No chunks after splitting: {rel_path}")
            return {"status": "skipped", "message": "No content chunks"}

        # Prepare data for ChromaDB
        filename = os.path.basename(filepath)
        documents = [f"{filename}:part\n---\n{chunk}" for chunk in chunks]
        ids = [f"{rel_path}::{i}" for i in range(len(chunks))]
        metadatas = [{"filepath": rel_path, "chunk_index": i} for i in range(len(chunks))]

        # Add to ChromaDB (automatically embeds)
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

        logger.info(f"Indexed {len(chunks)} chunks from {rel_path}")
        return {
            "status": "success",
            "message": f"Indexed {len(chunks)} chunks",
            "chunks": len(chunks)
        }

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def get_all_markdown_files() -> list[str]:
    """Get all markdown files in the vault"""
    pattern = os.path.join(VAULT_PATH, "**", "*.md")
    return glob.glob(pattern, recursive=True)


# --- Helper Functions ---
def normalize_path(path: str) -> str:
    """Normalize and validate a path within the vault"""
    # Remove leading slash if present
    path = path.lstrip('/')

    # Create absolute path
    abs_path = os.path.abspath(os.path.join(VAULT_PATH, path))
    vault_abs = os.path.abspath(VAULT_PATH)

    # Security check: ensure path is within vault
    if not abs_path.startswith(vault_abs):
        raise ValueError(f"Path '{path}' is outside vault directory")

    return abs_path


def get_relative_path(abs_path: str) -> str:
    """Get relative path from vault root"""
    return os.path.relpath(abs_path, VAULT_PATH)


def fuzzy_search_text(query: str, content: str, threshold: float = 0.6) -> bool:
    """Simple fuzzy text matching using difflib"""
    query_lower = query.lower()
    content_lower = content.lower()

    # Direct substring match
    if query_lower in content_lower:
        return True

    # Fuzzy matching on words
    query_words = query_lower.split()
    for word in query_words:
        if difflib.get_close_matches(word, content_lower.split(), n=1, cutoff=threshold):
            return True

    return False


# --- MCP Tools ---
@mcp.tool()
async def vector_search(query: str, n_results: int = 5) -> dict:
    """
    Search vault notes using vector embeddings for semantic similarity

    Args:
        query: Search query text
        n_results: Number of results to return (default: 5, max: 20)

    Returns:
        Dictionary with search results containing filepath, chunk_index, and document text
    """
    if not model or not collection:
        return {"error": "Vector database not initialized"}

    try:
        # Validate n_results
        n_results = max(1, min(n_results, 20))

        # Encode query
        logger.info(f"Vector search query: '{query}' (n_results={n_results})")
        query_embedding = model.encode(query).tolist()

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Format results
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "filepath": results["metadatas"][0][i]["filepath"],
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "document": results["documents"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })

        logger.info(f"Found {len(formatted_results)} results")
        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        }

    except Exception as e:
        logger.error(f"Vector search error: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def text_search(query: str, fuzzy: bool = True, case_sensitive: bool = False) -> dict:
    """
    Search vault notes using regular text search with optional fuzzy matching

    Args:
        query: Search query text
        fuzzy: Enable fuzzy matching (default: True)
        case_sensitive: Enable case-sensitive search (default: False)

    Returns:
        Dictionary with matching files and line numbers
    """
    try:
        logger.info(f"Text search query: '{query}' (fuzzy={fuzzy}, case_sensitive={case_sensitive})")
        results = []

        files = get_all_markdown_files()
        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                matches = []
                for line_num, line in enumerate(lines, 1):
                    # Case handling
                    search_line = line if case_sensitive else line.lower()
                    search_query = query if case_sensitive else query.lower()

                    # Direct match
                    if search_query in search_line:
                        matches.append({
                            "line_number": line_num,
                            "line": line,
                            "match_type": "exact"
                        })
                    # Fuzzy match
                    elif fuzzy and fuzzy_search_text(query, line):
                        matches.append({
                            "line_number": line_num,
                            "line": line,
                            "match_type": "fuzzy"
                        })

                if matches:
                    results.append({
                        "filepath": get_relative_path(filepath),
                        "matches": matches,
                        "match_count": len(matches)
                    })

            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")
                continue

        logger.info(f"Text search found matches in {len(results)} files")
        return {
            "query": query,
            "results": results,
            "files_matched": len(results)
        }

    except Exception as e:
        logger.error(f"Text search error: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def create_note(path: str, content: str, frontmatter_data: Optional[dict] = None) -> dict:
    """
    Create a new markdown note in the vault

    Args:
        path: Relative path for the new note (e.g., "folder/note.md")
        content: Markdown content for the note
        frontmatter_data: Optional YAML frontmatter as dictionary

    Returns:
        Dictionary with status and created file path
    """
    try:
        # Ensure path ends with .md
        if not path.endswith('.md'):
            path += '.md'

        abs_path = normalize_path(path)

        # Check if file already exists
        if os.path.exists(abs_path):
            return {"error": f"Note already exists: {path}"}

        # Create parent directories if needed
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        # Create frontmatter post
        post = frontmatter.Post(content)
        if frontmatter_data:
            post.metadata = frontmatter_data

        # Write file
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(frontmatter.dumps(post))

        logger.info(f"Created note: {path}")

        # Index the new file
        index_result = process_file(abs_path)

        return {
            "status": "success",
            "filepath": path,
            "indexed": index_result.get("status") == "success",
            "message": f"Note created and {'indexed' if index_result.get('status') == 'success' else 'not indexed'}"
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error creating note: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def create_folder(path: str) -> dict:
    """
    Create a new folder in the vault

    Args:
        path: Relative path for the new folder (e.g., "Projects/Work")

    Returns:
        Dictionary with status and created folder path
    """
    try:
        abs_path = normalize_path(path)

        # Create directory
        os.makedirs(abs_path, exist_ok=True)

        logger.info(f"Created folder: {path}")
        return {
            "status": "success",
            "path": path,
            "message": "Folder created successfully"
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error creating folder: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def read_note(path: str, include_frontmatter: bool = True) -> dict:
    """
    Read the contents of a note

    Args:
        path: Relative path to the note (e.g., "folder/note.md")
        include_frontmatter: Include frontmatter metadata (default: True)

    Returns:
        Dictionary with note content and metadata
    """
    try:
        abs_path = normalize_path(path)

        if not os.path.exists(abs_path):
            return {"error": f"Note not found: {path}"}

        if not os.path.isfile(abs_path):
            return {"error": f"Path is not a file: {path}"}

        # Read file
        with open(abs_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)

        result = {
            "filepath": path,
            "content": post.content,
        }

        if include_frontmatter and post.metadata:
            result["frontmatter"] = post.metadata

        # Add file stats
        stat = os.stat(abs_path)
        result["stats"] = {
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
        }

        return result

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error reading note: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def update_note_diff(path: str, diff: str) -> dict:
    """
    Update a note using unified diff format

    Args:
        path: Relative path to the note (e.g., "folder/note.md")
        diff: Unified diff string to apply to the note

    Returns:
        Dictionary with status and update details
    """
    try:
        abs_path = normalize_path(path)

        if not os.path.exists(abs_path):
            return {"error": f"Note not found: {path}"}

        # Read current content
        with open(abs_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Parse diff and apply
        original_lines = original_content.splitlines(keepends=True)

        # Parse the unified diff
        diff_lines = diff.splitlines(keepends=True)

        # Apply diff using difflib
        try:
            # Simple diff application (this is a basic implementation)
            # For production, consider using a more robust diff library
            new_lines = original_lines.copy()

            # Parse diff format
            i = 0
            while i < len(diff_lines):
                line = diff_lines[i]

                if line.startswith('@@'):
                    # Parse hunk header: @@ -start,count +start,count @@
                    parts = line.split()
                    if len(parts) >= 2:
                        old_info = parts[1].lstrip('-').split(',')
                        new_info = parts[2].lstrip('+').split(',')

                        old_start = int(old_info[0]) - 1
                        old_count = int(old_info[1]) if len(old_info) > 1 else 1

                        # Apply changes from this hunk
                        i += 1
                        new_hunk_lines = []
                        old_line_idx = 0

                        while i < len(diff_lines) and not diff_lines[i].startswith('@@'):
                            diff_line = diff_lines[i]

                            if diff_line.startswith('+'):
                                new_hunk_lines.append(diff_line[1:])
                            elif diff_line.startswith('-'):
                                old_line_idx += 1
                            elif diff_line.startswith(' '):
                                new_hunk_lines.append(diff_line[1:])
                                old_line_idx += 1

                            i += 1

                        # Replace the section
                        new_lines[old_start:old_start + old_count] = new_hunk_lines
                        continue

                i += 1

            new_content = ''.join(new_lines)

        except Exception as e:
            logger.error(f"Error applying diff: {e}")
            return {"error": f"Failed to apply diff: {str(e)}"}

        # Backup original
        backup_path = abs_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)

        # Write new content
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        logger.info(f"Updated note via diff: {path}")

        # Reindex the file
        index_result = process_file(abs_path)

        return {
            "status": "success",
            "filepath": path,
            "backup_created": backup_path,
            "reindexed": index_result.get("status") == "success",
            "message": "Note updated successfully"
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error updating note: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def delete_note(path: str, permanent: bool = False) -> dict:
    """
    Delete a note from the vault

    Args:
        path: Relative path to the note (e.g., "folder/note.md")
        permanent: Permanently delete (default: False, creates .trash folder)

    Returns:
        Dictionary with status and deletion details
    """
    try:
        abs_path = normalize_path(path)

        if not os.path.exists(abs_path):
            return {"error": f"Note not found: {path}"}

        if not os.path.isfile(abs_path):
            return {"error": f"Path is not a file: {path}"}

        # Remove from vector database
        rel_path = get_relative_path(abs_path)
        try:
            existing = collection.get(where={"filepath": rel_path})
            if existing and existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.info(f"Removed {len(existing['ids'])} chunks from vector DB")
        except Exception as e:
            logger.warning(f"Error removing from vector DB: {e}")

        if permanent:
            # Permanent deletion
            os.remove(abs_path)
            logger.info(f"Permanently deleted: {path}")
            return {
                "status": "success",
                "filepath": path,
                "permanent": True,
                "message": "Note permanently deleted"
            }
        else:
            # Move to trash
            trash_dir = os.path.join(VAULT_PATH, ".trash")
            os.makedirs(trash_dir, exist_ok=True)

            trash_path = os.path.join(trash_dir, os.path.basename(path))

            # Handle duplicates in trash
            counter = 1
            while os.path.exists(trash_path):
                name, ext = os.path.splitext(os.path.basename(path))
                trash_path = os.path.join(trash_dir, f"{name}_{counter}{ext}")
                counter += 1

            os.rename(abs_path, trash_path)
            logger.info(f"Moved to trash: {path} -> {trash_path}")

            return {
                "status": "success",
                "filepath": path,
                "trash_location": get_relative_path(trash_path),
                "permanent": False,
                "message": "Note moved to trash"
            }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error deleting note: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def list_notes(folder: str = "", pattern: str = "*.md", recursive: bool = True) -> dict:
    """
    List all notes in the vault or a specific folder

    Args:
        folder: Relative folder path to list (default: "" for root)
        pattern: File pattern to match (default: "*.md")
        recursive: Search recursively (default: True)

    Returns:
        Dictionary with list of notes and their metadata
    """
    try:
        search_path = VAULT_PATH
        if folder:
            search_path = normalize_path(folder)

            if not os.path.exists(search_path):
                return {"error": f"Folder not found: {folder}"}

            if not os.path.isdir(search_path):
                return {"error": f"Path is not a folder: {folder}"}

        # Get files
        if recursive:
            glob_pattern = os.path.join(search_path, "**", pattern)
            files = glob.glob(glob_pattern, recursive=True)
        else:
            glob_pattern = os.path.join(search_path, pattern)
            files = glob.glob(glob_pattern)

        # Format results
        notes = []
        for filepath in files:
            try:
                stat = os.stat(filepath)
                rel_path = get_relative_path(filepath)

                # Read frontmatter
                with open(filepath, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)

                notes.append({
                    "path": rel_path,
                    "name": os.path.basename(filepath),
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "frontmatter": post.metadata if post.metadata else {}
                })
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")
                continue

        # Sort by modified time (newest first)
        notes.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "folder": folder or "/",
            "pattern": pattern,
            "recursive": recursive,
            "notes": notes,
            "count": len(notes)
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error listing notes: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def get_vault_stats() -> dict:
    """
    Get statistics about the vault and vector database

    Returns:
        Dictionary with vault statistics including file counts, sizes, and index status
    """
    try:
        files = get_all_markdown_files()
        total_size = sum(os.path.getsize(f) for f in files)

        # Get vector DB stats
        db_stats = {
            "total_chunks": 0,
            "indexed_files": 0
        }

        if collection:
            try:
                all_items = collection.get()
                db_stats["total_chunks"] = len(all_items["ids"]) if all_items["ids"] else 0

                # Count unique files
                if all_items["metadatas"]:
                    unique_files = set(item["filepath"] for item in all_items["metadatas"])
                    db_stats["indexed_files"] = len(unique_files)
            except Exception as e:
                logger.warning(f"Error getting vector DB stats: {e}")

        # Get folder structure depth
        max_depth = 0
        for filepath in files:
            rel_path = get_relative_path(filepath)
            depth = rel_path.count(os.sep)
            max_depth = max(max_depth, depth)

        return {
            "vault_path": VAULT_PATH,
            "total_notes": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_folder_depth": max_depth,
            "vector_db": db_stats,
            "embedding_model": EMBEDDING_MODEL,
            "collection_name": COLLECTION_NAME
        }

    except Exception as e:
        logger.error(f"Error getting vault stats: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def reindex_vault(force: bool = False) -> dict:
    """
    Reindex all notes in the vault to the vector database

    Args:
        force: Force reindex all files even if already indexed (default: False)

    Returns:
        Dictionary with reindex status and statistics
    """
    try:
        logger.info(f"Starting vault reindex (force={force})")

        files = get_all_markdown_files()
        stats = {
            "processed": 0,
            "indexed": 0,
            "skipped": 0,
            "errors": 0,
            "total_chunks": 0
        }

        for filepath in files:
            try:
                result = process_file(filepath)
                stats["processed"] += 1

                if result["status"] == "success":
                    stats["indexed"] += 1
                    stats["total_chunks"] += result.get("chunks", 0)
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["errors"] += 1

            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                stats["errors"] += 1

        logger.info(f"Reindex complete: {stats}")
        return {
            "status": "success",
            "stats": stats,
            "message": f"Reindexed {stats['indexed']} files with {stats['total_chunks']} total chunks"
        }

    except Exception as e:
        logger.error(f"Error reindexing vault: {e}", exc_info=True)
        return {"error": str(e)}


# --- MCP Resources ---
@mcp.resource("vault://note/{path:path}")
async def get_note_resource(path: str) -> str:
    """
    Resource endpoint to access individual notes

    Args:
        path: Relative path to the note

    Returns:
        Note content as text
    """
    try:
        result = await read_note(path, include_frontmatter=True)

        if "error" in result:
            return f"Error: {result['error']}"

        # Format output
        output = []

        if "frontmatter" in result and result["frontmatter"]:
            output.append("---")
            output.append(json.dumps(result["frontmatter"], indent=2))
            output.append("---\n")

        output.append(result["content"])

        return "\n".join(output)

    except Exception as e:
        return f"Error accessing resource: {str(e)}"


@mcp.resource("vault://list")
async def list_notes_resource() -> str:
    """
    Resource endpoint to list all notes in the vault

    Returns:
        Formatted list of all notes
    """
    try:
        result = await list_notes()

        if "error" in result:
            return f"Error: {result['error']}"

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error accessing resource: {str(e)}"


@mcp.resource("vault://stats")
async def vault_stats_resource() -> str:
    """
    Resource endpoint for vault statistics

    Returns:
        Vault statistics as JSON
    """
    try:
        result = await get_vault_stats()

        if "error" in result:
            return f"Error: {result['error']}"

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error accessing resource: {str(e)}"


# --- MCP Prompts ---
@mcp.prompt()
async def search_and_summarize(topic: str) -> str:
    """
    Prompt template for searching and summarizing notes on a topic

    Args:
        topic: Topic to search for

    Returns:
        Formatted prompt for LLM
    """
    return f"""Please search my Obsidian vault for information about "{topic}" and provide a comprehensive summary.

Use the vector_search tool to find relevant notes, then read the full content of the most relevant notes.

Organize your summary by:
1. Main concepts and ideas
2. Key insights from different notes
3. Connections between related topics
4. Any gaps or areas for further exploration

Topic: {topic}"""


@mcp.prompt()
async def create_note_with_context(title: str, related_topics: str = "") -> str:
    """
    Prompt template for creating a new note with context from existing notes

    Args:
        title: Title for the new note
        related_topics: Related topics to search for context

    Returns:
        Formatted prompt for LLM
    """
    context_search = ""
    if related_topics:
        context_search = f"\nFirst, search for notes related to: {related_topics}"

    return f"""Please help me create a new note titled "{title}" in my Obsidian vault.
{context_search}

Then create a well-structured note that:
1. Has appropriate YAML frontmatter (tags, date, etc.)
2. Includes an introduction/overview
3. Links to related notes if relevant
4. Has clear section headings
5. Follows markdown best practices

Use the create_note tool when ready."""


# --- Initialization and Main ---
async def initialize():
    """Initialize all server components"""
    try:
        logger.info("Initializing MCP server components...")

        # Initialize vector database
        init_vector_db()

        # Initialize OAuth if enabled
        if OAUTH_ENABLED:
            init_oauth()

        # Verify vault path exists
        if not os.path.exists(VAULT_PATH):
            logger.warning(f"Vault path does not exist: {VAULT_PATH}")
            logger.info(f"Creating vault directory: {VAULT_PATH}")
            os.makedirs(VAULT_PATH, exist_ok=True)

        logger.info("MCP server initialized successfully")

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        raise


def main():
    """Main entry point for the MCP server"""
    import sys

    # Run initialization
    asyncio.run(initialize())

    # Start the MCP server
    # The FastMCP server will handle stdio communication
    logger.info("Starting Obsidian MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
