"""Vaultkeeper MCP Server.

Main entrypoint that wires up all tools, auth, and the internal agent.
Uses FastMCP (standalone) with Streamable HTTP transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_request

from vaultkeeper.agent.orchestrator import Orchestrator
from vaultkeeper.agent.router import QueryRouter
from vaultkeeper.bases.parser import BasesParser
from vaultkeeper.config import Config
from vaultkeeper.index.indexer import VaultIndexer
from vaultkeeper.llm.client import EmbeddingClient, LLMClient
from vaultkeeper.tools.undo import UndoManager
from vaultkeeper.vault.reader import VaultReader
from vaultkeeper.vault.watcher import VaultWatcher
from vaultkeeper.vault.writer import PatchOperation, VaultWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("vaultkeeper")


# ---------------------------------------------------------------------------
# Application context (passed through lifespan)
# ---------------------------------------------------------------------------
@dataclass
class AppContext:
    vault_reader: VaultReader
    vault_writer: VaultWriter
    undo_manager: UndoManager
    bases_parser: BasesParser
    indexer: VaultIndexer
    orchestrator: Orchestrator
    llm_client: LLMClient
    embedding_client: EmbeddingClient
    watcher: VaultWatcher
    config: Config


# ---------------------------------------------------------------------------
# Lifespan: startup and shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize all services on startup, clean up on shutdown."""
    config_path = os.environ.get("VAULTKEEPER_CONFIG", "/etc/vaultkeeper/config.yml")
    config = Config.from_yaml(config_path)

    logger.info(f"Vault path: {config.vault.path}")
    logger.info(f"LanceDB path: {config.index.db_path}")

    # Core services
    vault_reader = VaultReader(config.vault.path)
    vault_writer = VaultWriter(config.vault.path)
    undo_manager = UndoManager(vault_writer, max_operations=config.undo.max_operations)
    bases_parser = BasesParser(vault_reader)

    # LLM clients
    llm_client = LLMClient(config.models)
    embedding_client = EmbeddingClient(config.embeddings, api_key=config.models.api_key)

    # Indexer
    indexer = VaultIndexer(
        config=config,
        vault_reader=vault_reader,
        embedding_client=embedding_client,
        llm_client=llm_client,
    )

    # Agent
    base_names = [Path(b).stem for b in vault_reader.list_all_bases()]
    router = QueryRouter(llm_client, base_names=base_names)
    orchestrator = Orchestrator(
        config=config,
        vault_reader=vault_reader,
        indexer=indexer,
        embedding_client=embedding_client,
        llm_client=llm_client,
        bases_parser=bases_parser,
        router=router,
    )

    # File watcher
    async def on_file_changes(changes: list[tuple[str, str]]) -> None:
        md_changes = [path for path, _ in changes if path.endswith(".md")]
        if md_changes:
            logger.info(f"Re-indexing {len(md_changes)} changed notes")
            await indexer.incremental_index(md_changes)

    watcher = VaultWatcher(
        vault_path=config.vault.path,
        debounce_seconds=config.vault.debounce_seconds,
        on_changes=on_file_changes,
    )

    # Start background tasks
    index_task = None
    if config.vault.index_on_startup:
        logger.info("Starting background vault indexing...")
        index_task = asyncio.create_task(_background_index(indexer))

    if config.vault.watch:
        await watcher.start()

    logger.info("Vaultkeeper ready")

    try:
        yield AppContext(
            vault_reader=vault_reader,
            vault_writer=vault_writer,
            undo_manager=undo_manager,
            bases_parser=bases_parser,
            indexer=indexer,
            orchestrator=orchestrator,
            llm_client=llm_client,
            embedding_client=embedding_client,
            watcher=watcher,
            config=config,
        )
    finally:
        logger.info("Vaultkeeper shutting down")
        await watcher.stop()
        if index_task and not index_task.done():
            index_task.cancel()
        await llm_client.close()
        await embedding_client.close()


async def _background_index(indexer: VaultIndexer) -> None:
    try:
        stats = await indexer.full_index()
        logger.info(
            f"Indexing complete: {stats.indexed} indexed, {stats.skipped} skipped, "
            f"{stats.chunks_created} chunks, {stats.facts_extracted} facts, "
            f"{stats.errors} errors in {stats.duration_seconds:.1f}s"
        )
    except Exception as e:
        logger.error(f"Background indexing failed: {e}")


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------
class BearerTokenMiddleware(Middleware):
    """Validates a static Bearer token on every tool call."""

    def __init__(self, expected_token: str):
        self.expected_token = expected_token

    async def _check(self) -> None:
        if not self.expected_token:
            return

        try:
            request = get_http_request()
        except RuntimeError:
            return  # Not in HTTP context (stdio transport)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            raise ToolError("Missing or invalid Authorization header")

        token = auth_header.removeprefix("Bearer ").strip()
        if token != self.expected_token:
            raise ToolError("Invalid authentication token")

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        await self._check()
        return await call_next(context)

    async def on_list_tools(self, context: MiddlewareContext, call_next):
        await self._check()
        return await call_next(context)


# ---------------------------------------------------------------------------
# Helper to get AppContext from lifespan
# ---------------------------------------------------------------------------
def _ctx(ctx: Context) -> AppContext:
    """Extract AppContext from the FastMCP lifespan context."""
    return ctx.lifespan_context


# ---------------------------------------------------------------------------
# FastMCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Vaultkeeper",
    description=(
        "Intelligent Obsidian vault access with semantic search, "
        "Bases support, and note mutation."
    ),
    lifespan=app_lifespan,
)

# Auth token from env (read at import time for middleware registration)
_auth_token = os.environ.get("VAULTKEEPER_TOKEN", "")
if _auth_token:
    mcp.add_middleware(BearerTokenMiddleware(_auth_token))
    logger.info("Bearer token authentication enabled")
else:
    logger.warning("No VAULTKEEPER_TOKEN set — running without authentication")


# ===================================================================
# NAVIGATION TOOLS
# ===================================================================

@mcp.tool()
def vault_ls(ctx: Context, path: str = "") -> str:
    """List contents of a directory in the vault.

    Args:
        path: Relative path within the vault. Empty string for vault root.
    """
    app = _ctx(ctx)
    entries = app.vault_reader.list_dir(path)
    lines = []
    for e in entries:
        if e.is_dir:
            lines.append(f"📁 {e.name}/")
        else:
            size_kb = e.size / 1024
            lines.append(f"  {e.name} ({size_kb:.1f} KB)")
    return "\n".join(lines) if lines else "(empty directory)"


@mcp.tool()
def vault_read(ctx: Context, path: str) -> str:
    """Read a note's full content with parsed metadata.

    Args:
        path: Relative path to the .md file.
    """
    app = _ctx(ctx)
    note = app.vault_reader.read_note(path)

    sections = [
        f"# {note.title}",
        f"Path: {note.path}",
        f"Lines: {note.line_count}",
    ]

    if note.frontmatter:
        fm_lines = [f"  {k}: {v}" for k, v in note.frontmatter.items()]
        sections.append("Frontmatter:\n" + "\n".join(fm_lines))

    if note.tags:
        sections.append(f"Tags: {', '.join(note.tags)}")

    if note.wikilinks:
        sections.append(f"Links: {', '.join(note.wikilinks)}")

    if note.headers:
        header_lines = [
            f"  {'#' * h.level} {h.text} (line {h.line_number})" for h in note.headers
        ]
        sections.append("Headers:\n" + "\n".join(header_lines))

    sections.append("\n--- Content ---\n")
    sections.append(note.content)

    return "\n\n".join(sections)


@mcp.tool()
def vault_tree(ctx: Context, path: str = "", depth: int = 3) -> str:
    """Show the directory tree of the vault.

    Args:
        path: Starting path. Empty for vault root.
        depth: How many levels deep to show (default 3).
    """
    app = _ctx(ctx)
    tree = app.vault_reader.tree(path, depth)
    return _format_tree(tree, indent=0)


def _format_tree(node: dict, indent: int) -> str:
    prefix = "  " * indent
    lines = []
    if node["type"] == "directory":
        lines.append(f"{prefix}📁 {node['name']}/")
        for child in node.get("children", []):
            lines.append(_format_tree(child, indent + 1))
    else:
        lines.append(f"{prefix}{node['name']}")
    return "\n".join(lines)


@mcp.tool()
def vault_tags(ctx: Context, tag: str = "") -> str:
    """List all tags in the vault, or find notes with a specific tag.

    Args:
        tag: If provided, return notes with this tag. Otherwise list all tags.
    """
    app = _ctx(ctx)
    all_notes = app.vault_reader.list_all_notes()
    tag_counts: dict[str, list[str]] = {}

    for note_path in all_notes:
        try:
            note = app.vault_reader.read_note(note_path)
            for t in note.tags:
                tag_counts.setdefault(t, []).append(note_path)
        except Exception:
            continue

    if tag:
        tag_clean = tag.lstrip("#")
        matching = tag_counts.get(tag_clean, [])
        if matching:
            return f"Notes with #{tag_clean} ({len(matching)}):\n" + "\n".join(
                f"  {p}" for p in matching
            )
        return f"No notes found with tag #{tag_clean}"

    sorted_tags = sorted(tag_counts.items(), key=lambda x: len(x[1]), reverse=True)
    lines = [f"#{t} ({len(notes)})" for t, notes in sorted_tags]
    return "\n".join(lines) if lines else "No tags found in vault."


# ===================================================================
# BASES TOOLS
# ===================================================================

@mcp.tool()
def base_list(ctx: Context) -> str:
    """List all Obsidian Bases in the vault with their schemas."""
    app = _ctx(ctx)
    bases = app.bases_parser.list_bases()
    if not bases:
        return "No Bases found in vault."

    lines = []
    for b in bases:
        lines.append(f"📊 {b['name']} ({b['path']})")
        lines.append(f"   Filters: {', '.join(b['filters'])}")
        for v in b["views"]:
            lines.append(f"   View '{v['name']}': {', '.join(v['columns'])}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def base_read(ctx: Context, path: str) -> str:
    """Read the raw schema of an Obsidian Base file.

    Args:
        path: Relative path to the .base file.
    """
    app = _ctx(ctx)
    base = app.bases_parser.parse_base(path)
    return json.dumps({
        "name": base.name,
        "path": base.path,
        "filter_logic": base.filter_logic,
        "filters": base.filters,
        "views": [
            {"name": v.name, "type": v.view_type, "columns": v.columns, "sort": v.sort}
            for v in base.views
        ],
    }, indent=2)


@mcp.tool()
async def base_query(ctx: Context, base_path: str, query: str = "") -> str:
    """Query an Obsidian Base, optionally filtering with natural language.

    Args:
        base_path: Path to the .base file.
        query: Optional natural language filter (e.g., "restaurants I haven't visited").
    """
    app = _ctx(ctx)
    base = app.bases_parser.parse_base(base_path)
    rows = app.bases_parser.resolve_base(base)

    if query and rows:
        rows_json = json.dumps(rows[:50], indent=2, default=str)
        filter_prompt = (
            f'Given these rows from an Obsidian Base named "{base.name}":\n\n'
            f"{rows_json}\n\n"
            f'The user wants: "{query}"\n\n'
            "Return a JSON array containing ONLY the matching rows. "
            "Return the rows exactly as they are, just filtered. Return ONLY the JSON array."
        )
        try:
            filtered = await app.llm_client.complete_json(
                "router", [{"role": "user", "content": filter_prompt}]
            )
            if isinstance(filtered, list):
                rows = filtered
        except Exception as e:
            logger.warning(f"NL base filtering failed, returning all rows: {e}")

    if not rows:
        return f"No results found in Base '{base.name}'."

    lines = [f"Base: {base.name} ({len(rows)} results)\n"]
    for row in rows:
        parts = [f"{k}: {v}" for k, v in row.items() if k != "_path" and v]
        lines.append(f"• {' | '.join(parts)}")
        path = row.get("_path", "")
        if path:
            lines.append(f"  → {path}")
    return "\n".join(lines)


# ===================================================================
# SEARCH TOOLS (agent-backed)
# ===================================================================

@mcp.tool()
async def vault_query(ctx: Context, query: str) -> str:
    """Ask a natural language question about your vault.

    This is the primary intelligence tool. It classifies your query, runs
    multiple retrieval strategies in parallel, and synthesizes an answer.

    Args:
        query: Natural language question (e.g., "What's John Smith's garage code?")
    """
    app = _ctx(ctx)
    response = await app.orchestrator.query(query)

    parts = [response.synthesized_answer]
    if response.sources:
        parts.append("\nSources:")
        for s in response.sources:
            parts.append(f"  • {s}")
    return "\n".join(parts)


@mcp.tool()
async def vault_search(ctx: Context, query: str, strategy: str = "") -> str:
    """Search the vault with optional strategy override.

    Args:
        query: Search query.
        strategy: Optional: semantic, text, facts, frontmatter, base, graph.
                  If empty, the router picks automatically.
    """
    app = _ctx(ctx)
    results = await app.orchestrator.search(query, strategy=strategy or None)

    if not results:
        return "No results found."

    lines = []
    for r in results:
        score_pct = int(r.score * 100)
        lines.append(f"[{score_pct}%] {r.title} ({r.path})")
        lines.append(f"  Strategy: {r.source}")
        lines.append(f"  {r.content[:200]}")
        lines.append("")
    return "\n".join(lines)


# ===================================================================
# MUTATION TOOLS
# ===================================================================

def _new_message_id() -> str:
    return str(uuid.uuid4())[:8]


@mcp.tool()
def note_create(ctx: Context, path: str, content: str = "", frontmatter_json: str = "{}") -> str:
    """Create a new note in the vault.

    Args:
        path: Relative path for the new note (must end with .md).
        content: Note body content.
        frontmatter_json: JSON string of frontmatter properties.
    """
    app = _ctx(ctx)
    msg_id = _new_message_id()
    app.undo_manager.set_message_id(msg_id)

    fm = json.loads(frontmatter_json) if frontmatter_json and frontmatter_json != "{}" else None
    result = app.vault_writer.create_note(path, content, note_frontmatter=fm)
    app.undo_manager.record(result)
    return f"Created: {path}"


@mcp.tool()
def note_delete(ctx: Context, path: str) -> str:
    """Delete a note from the vault.

    Args:
        path: Relative path to the note to delete.
    """
    app = _ctx(ctx)
    msg_id = _new_message_id()
    app.undo_manager.set_message_id(msg_id)

    result = app.vault_writer.delete_note(path)
    app.undo_manager.record(result)
    lines = result.before_content.count("\n") + 1 if result.before_content else 0
    return f"Deleted: {path} ({lines} lines)"


@mcp.tool()
def note_patch(ctx: Context, path: str, operations_json: str) -> str:
    """Apply line-level edits to a note.

    Args:
        path: Relative path to the note.
        operations_json: JSON array of operations, each with:
            - "action": "insert" | "delete" | "replace"
            - "line": line number (1-indexed)
            - "content": new content (for insert/replace)
    """
    app = _ctx(ctx)
    msg_id = _new_message_id()
    app.undo_manager.set_message_id(msg_id)

    ops_data = json.loads(operations_json)
    operations = [
        PatchOperation(action=op["action"], line=op["line"], content=op.get("content", ""))
        for op in ops_data
    ]

    result = app.vault_writer.patch_note(path, operations)
    app.undo_manager.record(result)
    return f"Patched {path}: {len(operations)} operation(s) applied."


@mcp.tool()
def note_append(ctx: Context, path: str, content: str) -> str:
    """Append content to the end of a note.

    Args:
        path: Relative path to the note.
        content: Content to append.
    """
    app = _ctx(ctx)
    msg_id = _new_message_id()
    app.undo_manager.set_message_id(msg_id)

    result = app.vault_writer.append_note(path, content)
    app.undo_manager.record(result)
    return f"Appended to {path}."


@mcp.tool()
def note_frontmatter_update(ctx: Context, path: str, updates_json: str) -> str:
    """Update specific frontmatter fields on a note.

    Args:
        path: Relative path to the note.
        updates_json: JSON object of field updates. Set a value to null to remove it.
    """
    app = _ctx(ctx)
    msg_id = _new_message_id()
    app.undo_manager.set_message_id(msg_id)

    updates = json.loads(updates_json)
    result = app.vault_writer.update_frontmatter(path, updates)
    app.undo_manager.record(result)
    return f"Updated frontmatter on {path}: {', '.join(updates.keys())}"


@mcp.tool()
def vault_undo(ctx: Context) -> str:
    """Undo all changes from the most recent operation.

    Reverts creates, deletes, patches, appends, and frontmatter updates.
    """
    app = _ctx(ctx)
    results = app.undo_manager.undo_last_message()
    return "\n".join(results)


# ===================================================================
# Entrypoint
# ===================================================================

def main():
    """Run the server."""
    port = int(os.environ.get("VAULTKEEPER_PORT", "8000"))
    host = os.environ.get("VAULTKEEPER_HOST", "0.0.0.0")
    mcp.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    main()
