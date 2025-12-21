"""
MCP Tool Handlers for Obsidian Vault
Separated tool definitions and execution logic for better maintainability.
"""

import io
import logging
from pathlib import Path
from typing import Any
from contextlib import redirect_stdout

import mcp.types as types
from search import fuzzy, semantic
import base_engine


logger = logging.getLogger("mcp-obsidian.tools")


# Helper Functions
def make_text(text: str) -> list[types.TextContent]:
    """Create a text content response."""
    return [types.TextContent(type="text", text=text)]


def make_error(message: str) -> list[types.TextContent]:
    """Create an error message response."""
    return make_text(f"Error: {message}")


# Tool Definitions
def get_tool_definitions() -> list[types.Tool]:
    """Return list of all available MCP tools."""
    return [
        types.Tool(
            name="read_note",
            description="Read a markdown note with numbered lines. MUST be called before edit_note to see current content and line numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the note (e.g., 'Projects/Alpha.md')",
                    }
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="edit_note",
            description="Edit a markdown note using line-based operations. MUST call read_note first to see current line numbers. Operations: 'replace' replaces lines start-end (inclusive) with new content, 'insert' adds content after the specified line (use 0 to insert at beginning), 'delete' removes lines start-end (inclusive).",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the note",
                    },
                    "edits": {
                        "type": "array",
                        "description": "List of edit operations, applied in order. Line numbers refer to the ORIGINAL file, not intermediate states.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {
                                    "type": "string",
                                    "enum": ["replace", "insert", "delete"],
                                    "description": "Operation type",
                                },
                                "start": {
                                    "type": "integer",
                                    "description": "Starting line number (1-indexed). For 'insert', content is added AFTER this line (use 0 for beginning).",
                                },
                                "end": {
                                    "type": "integer",
                                    "description": "Ending line number (inclusive). Required for 'replace' and 'delete'. Not used for 'insert'.",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "New content (for 'replace' and 'insert'). Can be multiple lines separated by newlines.",
                                },
                            },
                            "required": ["op", "start"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        ),
        types.Tool(
            name="search_vault",
            description="Search for notes using fuzzy matching on filenames, content, or both",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "search_mode": {
                        "type": "string",
                        "enum": ["filename", "content", "both"],
                        "description": "Where to search",
                        "default": "filename",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_notes",
            description="List all markdown notes in the vault",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="create_note",
            description="Create a new note. Always check get_templates first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path for the note"},
                    "content": {"type": "string", "description": "Note content"},
                },
                "required": ["path", "content"],
            },
        ),
        types.Tool(
            name="semantic_search",
            description="Search notes using semantic/vector similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="show_folder_structure",
            description="Display vault folder structure (directories only)",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="reindex_vault",
            description="Refresh embeddings for semantic search",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="list_bases",
            description="List all Obsidian Base files (.base) in the vault",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="read_base",
            description="Read an Obsidian Base file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to .base file",
                    }
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="get_templates",
            description="Get templates for new notes. ALWAYS call before create_note.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# Individual Tool Handlers
def handle_read_note(vault_path: Path, arguments: dict) -> list[types.TextContent]:
    """Read a note with line numbers."""
    path = vault_path / arguments["path"]

    if not path.exists():
        return make_error(f"Note not found: {arguments['path']}")

    if not path.resolve().is_relative_to(vault_path.resolve()):
        return make_error("Invalid path")

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))

    return make_text(
        f"File: {arguments['path']} ({len(lines)} lines)\n"
        f"{'─' * 60}\n{numbered}"
    )


def handle_edit_note(vault_path: Path, arguments: dict) -> list[types.TextContent]:
    """Edit a note using line-based operations."""
    path = vault_path / arguments["path"]

    if not path.exists():
        return make_error(
            f"Note not found: {arguments['path']}. Use create_note for new files."
        )

    if not path.resolve().is_relative_to(vault_path.resolve()):
        return make_error("Invalid path")

    edits = arguments.get("edits", [])
    if not edits:
        return make_error("No edits provided")

    content = path.read_text(encoding="utf-8")
    original_lines = content.splitlines()
    result_lines = original_lines.copy()

    offset = 0
    changes = []
    sorted_edits = sorted(enumerate(edits), key=lambda x: x[1].get("start", 0))

    for orig_idx, edit in sorted_edits:
        op = edit.get("op")
        start = edit.get("start", 1)
        end = edit.get("end", start)
        new_content = edit.get("content", "")

        adj_start = start + offset - 1
        adj_end = end + offset

        if op == "replace":
            if start < 1 or end > len(original_lines) or start > end:
                return make_error(
                    f"Edit {orig_idx + 1}: Invalid line range {start}-{end} "
                    f"(file has {len(original_lines)} lines)"
                )
            new_lines = new_content.splitlines() if new_content else []
            old_section = result_lines[adj_start:adj_end]
            result_lines[adj_start:adj_end] = new_lines
            offset += len(new_lines) - (end - start + 1)
            changes.append(
                f"Replaced lines {start}-{end}:\n  - "
                + "\n  - ".join(old_section)
                + "\n  + "
                + "\n  + ".join(new_lines)
            )

        elif op == "insert":
            if start < 0 or start > len(original_lines):
                return make_error(
                    f"Edit {orig_idx + 1}: Invalid insert position {start} "
                    f"(file has {len(original_lines)} lines)"
                )
            new_lines = new_content.splitlines() if new_content else []
            insert_pos = start + offset
            result_lines[insert_pos:insert_pos] = new_lines
            offset += len(new_lines)
            changes.append(
                f"Inserted after line {start}:\n  + " + "\n  + ".join(new_lines)
            )

        elif op == "delete":
            if start < 1 or end > len(original_lines) or start > end:
                return make_error(
                    f"Edit {orig_idx + 1}: Invalid line range {start}-{end} "
                    f"(file has {len(original_lines)} lines)"
                )
            old_section = result_lines[adj_start:adj_end]
            del result_lines[adj_start:adj_end]
            offset -= end - start + 1
            changes.append(
                f"Deleted lines {start}-{end}:\n  - " + "\n  - ".join(old_section)
            )

        else:
            return make_error(f"Edit {orig_idx + 1}: Unknown operation '{op}'")

    path.write_text("\n".join(result_lines), encoding="utf-8")

    summary = f"Edited {arguments['path']}\n{'─' * 60}\n"
    summary += "\n\n".join(changes)
    summary += f"\n{'─' * 60}\nResult: {len(original_lines)} → {len(result_lines)} lines"

    return make_text(summary)


def handle_search_vault(vault_path: Path, arguments: dict) -> list[types.TextContent]:
    """Search vault using fuzzy matching."""
    mode = arguments.get("search_mode", "filename")
    result = fuzzy.search_vault(arguments["query"], vault_path, search_mode=mode)
    return make_text(result)


def handle_list_notes(vault_path: Path, arguments: dict) -> list[types.TextContent]:
    """List all markdown notes in the vault."""
    files = [str(p.relative_to(vault_path)) for p in vault_path.rglob("*.md")]
    result = f"Found {len(files)} notes:\n"
    result += "\n".join(f"- {f}" for f in sorted(files)[:100])
    if len(files) > 100:
        result += f"\n... and {len(files) - 100} more"
    return make_text(result)


def handle_create_note(vault_path: Path, arguments: dict) -> list[types.TextContent]:
    """Create a new note."""
    path = vault_path / arguments["path"]

    if not path.resolve().is_relative_to(vault_path.resolve()):
        return make_error("Invalid path")

    if path.exists():
        return make_error("File already exists")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(arguments["content"], encoding="utf-8")

    return make_text(f"Created note: {arguments['path']}")


def handle_semantic_search(
    vault_path: Path, arguments: dict
) -> list[types.TextContent]:
    """Search vault using semantic/vector similarity."""
    limit = arguments.get("limit", 10)
    result = semantic.search_vault(arguments["query"], vault_path, limit=limit)
    return make_text(result)


def handle_show_folder_structure(
    vault_path: Path, arguments: dict, generate_fn
) -> list[types.TextContent]:
    """Display vault folder structure."""
    result = f"### Vault Folder Structure:\n\n{vault_path.name}/\n"
    result += generate_fn(vault_path)
    return make_text(result)


def handle_reindex_vault(
    vault_path: Path, arguments: dict, file_watcher
) -> list[types.TextContent]:
    """Reindex vault for semantic search."""
    search_engine = file_watcher.get_search_engine()
    search_engine.reindex()

    count = 0
    if search_engine.collection:
        count = search_engine.collection.count()

    return make_text(
        f"Vault reindexed successfully.\nVectors in database: {count}"
    )


def handle_list_bases(vault_path: Path, arguments: dict) -> list[types.TextContent]:
    """List all Obsidian Base files."""
    base_files = base_engine.list_bases(str(vault_path))

    if not base_files:
        return make_text("No .base files found")

    relative = [str(Path(f).relative_to(vault_path)) for f in base_files]
    return make_text(
        f"Found {len(base_files)} Base files:\n"
        + "\n".join(f"- {f}" for f in sorted(relative))
    )


def handle_read_base(vault_path: Path, arguments: dict) -> list[types.TextContent]:
    """Read an Obsidian Base file."""
    path = vault_path / arguments["path"]

    if not path.exists():
        return make_error(f"Base not found: {arguments['path']}")

    if not path.resolve().is_relative_to(vault_path.resolve()):
        return make_error("Invalid path")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        base_engine.view_base(str(vault_path), str(path))

    return make_text(buffer.getvalue())


def handle_get_templates(vault_path: Path, arguments: dict) -> list[types.TextContent]:
    """Get templates for new notes."""
    templates_path = Path("templates.md")

    if not templates_path.exists():
        return make_error("templates.md not found")

    return make_text(templates_path.read_text(encoding="utf-8"))


# Tool Dispatcher
class ToolHandler:
    """Manages tool execution with proper dependency injection."""

    def __init__(self, vault_path: Path, file_watcher, folder_structure_generator):
        self.vault_path = vault_path
        self.file_watcher = file_watcher
        self.folder_structure_generator = folder_structure_generator

    async def execute_tool(
        self, name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Execute a tool by name with given arguments."""
        try:
            handlers = {
                "read_note": lambda: handle_read_note(self.vault_path, arguments),
                "edit_note": lambda: handle_edit_note(self.vault_path, arguments),
                "search_vault": lambda: handle_search_vault(self.vault_path, arguments),
                "list_notes": lambda: handle_list_notes(self.vault_path, arguments),
                "create_note": lambda: handle_create_note(self.vault_path, arguments),
                "semantic_search": lambda: handle_semantic_search(self.vault_path, arguments),
                "show_folder_structure": lambda: handle_show_folder_structure(
                    self.vault_path, arguments, self.folder_structure_generator
                ),
                "reindex_vault": lambda: handle_reindex_vault(
                    self.vault_path, arguments, self.file_watcher
                ),
                "list_bases": lambda: handle_list_bases(self.vault_path, arguments),
                "read_base": lambda: handle_read_base(self.vault_path, arguments),
                "get_templates": lambda: handle_get_templates(self.vault_path, arguments),
            }

            handler = handlers.get(name)
            if handler is None:
                return make_error(f"Unknown tool: {name}")

            return handler()

        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            return make_error(str(e))
