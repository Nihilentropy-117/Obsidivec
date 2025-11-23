import os
import logging
import yaml
import frontmatter
import secrets
import uuid
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

# MCP Imports
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
from search import fuzzy, semantic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-obsidian")

VAULT_PATH = Path(os.getenv("VAULT_PATH", "/vault"))
SKIP_AUTH_DEBUG = os.getenv("SKIP_AUTH_DEBUG", "false").lower() == "true"
SERVER_URL = os.getenv("SERVER_URL", "https://localhost:8000")


def generate_folder_structure(root_path: Path, prefix: str = "", is_last: bool = True) -> str:
    """
    Generate a tree-like folder structure (directories only).

    Args:
        root_path: Root directory to scan
        prefix: Prefix for tree formatting
        is_last: Whether this is the last item in current level

    Returns:
        Formatted string representing the folder structure
    """
    output = ""

    # Get all subdirectories, sorted
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith('.')], key=lambda x: x.name)

    for idx, subdir in enumerate(subdirs):
        is_last_item = (idx == len(subdirs) - 1)

        # Add the current directory with tree characters
        connector = "└── " if is_last_item else "├── "
        output += prefix + connector + subdir.name + "/\n"

        # Recursively process subdirectories
        extension = "    " if is_last_item else "│   "
        output += generate_folder_structure(subdir, prefix + extension, is_last_item)

    return output

# OAuth storage (in-memory for simplicity)
oauth_clients: Dict[str, dict] = {}
oauth_codes: Dict[str, dict] = {}
oauth_tokens: Dict[str, dict] = {}

# 1. Create Server
server = Server("Obsidian Vault")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="read_note",
            description="Read the full content of a markdown note from the vault",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the note (e.g., 'Projects/Alpha.md')"
                    }
                },
                "required": ["path"]
            }
        ),
        types.Tool(
            name="search_vault",
            description="Search for notes in the vault using fuzzy matching on filenames, content, or both",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "search_mode": {
                        "type": "string",
                        "enum": ["filename", "content", "both"],
                        "description": "Where to search: 'filename' for note names, 'content' for note contents, 'both' for both",
                        "default": "filename"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="list_notes",
            description="List all markdown notes in the vault",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="create_note",
            description="Create a new note in the vault",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path where the note should be created"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content of the note"
                    }
                },
                "required": ["path", "content"]
            }
        ),
        types.Tool(
            name="semantic_search",
            description="Search for notes using semantic/vector similarity. Chunks content by double newlines and returns full file contents when a chunk matches.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="show_folder_structure",
            description="Display the complete folder structure of the vault (directories only, no files)",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="reindex_vault",
            description="Manually refresh and regenerate embeddings for semantic search (use if vault contents changed)",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="list_bases",
            description="List all Obsidian Base files (.base) in the vault. Bases are database-like views of notes with filters and sorting.",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="read_base",
            description="Read and parse an Obsidian Base file to show its structure: filters, formulas, display mappings, and view configurations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the .base file (e.g., 'Databases/Books.base')"
                    }
                },
                "required": ["path"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    try:
        if name == "read_note":
            p = VAULT_PATH / arguments["path"]
            if p.exists():
                return [types.TextContent(type="text", text=p.read_text(encoding='utf-8'))]
            else:
                return [types.TextContent(type="text", text=f"Error: Note not found at {arguments['path']}")]

        elif name == "search_vault":
            search_mode = arguments.get("search_mode", "filename")
            result = fuzzy.search_vault(arguments["query"], VAULT_PATH, search_mode=search_mode)
            return [types.TextContent(type="text", text=result)]

        elif name == "list_notes":
            files = [str(p.relative_to(VAULT_PATH)) for p in VAULT_PATH.rglob("*.md")]
            result = f"Found {len(files)} notes:\n" + "\n".join(f"- {f}" for f in sorted(files)[:100])
            if len(files) > 100:
                result += f"\n... and {len(files) - 100} more"
            return [types.TextContent(type="text", text=result)]

        elif name == "create_note":
            p = VAULT_PATH / arguments["path"]
            if p.exists():
                return [types.TextContent(type="text", text="Error: File already exists")]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(arguments["content"], encoding='utf-8')
            return [types.TextContent(type="text", text=f"Created note at {arguments['path']}")]

        elif name == "semantic_search":
            limit = arguments.get("limit", 10)
            result = semantic.search_vault(arguments["query"], VAULT_PATH, limit=limit)
            return [types.TextContent(type="text", text=result)]

        elif name == "show_folder_structure":
            result = f"### Vault Folder Structure:\n\n{VAULT_PATH.name}/\n"
            result += generate_folder_structure(VAULT_PATH)
            if not result.strip().endswith("/"):
                result += "\n(No subdirectories found)"
            return [types.TextContent(type="text", text=result)]

        elif name == "reindex_vault":
            try:
                logger.info("Manual reindexing requested...")
                db_path = Path(os.getenv("DB_PATH", "/app/chroma_data"))
                search_engine = semantic.get_search_engine(VAULT_PATH, db_path)
                search_engine.reindex()

                # Get stats from ChromaDB
                if search_engine.collection:
                    count = search_engine.collection.count()
                    result = f"✓ Vault reindexed successfully!\n\n"
                    result += f"- Total vectors in database: {count}\n"
                    result += f"- Database path: {db_path}\n"
                else:
                    result = f"✓ Reindex started in background\n"
                return [types.TextContent(type="text", text=result)]
            except Exception as e:
                logger.error(f"Reindex error: {e}", exc_info=True)
                return [types.TextContent(type="text", text=f"Error during reindexing: {str(e)}")]

        elif name == "list_bases":
            base_files = [str(p.relative_to(VAULT_PATH)) for p in VAULT_PATH.rglob("*.base")]
            result = f"Found {len(base_files)} Base files:\n"
            if base_files:
                result += "\n".join(f"- {f}" for f in sorted(base_files))
            else:
                result += "(No .base files found in vault)"
            return [types.TextContent(type="text", text=result)]

        elif name == "read_base":
            p = VAULT_PATH / arguments["path"]
            if not p.exists():
                return [types.TextContent(type="text", text=f"Error: Base file not found at {arguments['path']}")]

            try:
                content = p.read_text(encoding='utf-8')
                base_data = yaml.safe_load(content)

                result = f"# Base: {p.name}\n\n"

                # Global filters
                if "filters" in base_data:
                    result += "## Global Filters\n"
                    result += f"```yaml\n{yaml.dump(base_data['filters'], default_flow_style=False)}```\n\n"

                # Formulas
                if "formulas" in base_data:
                    result += "## Formulas (Computed Properties)\n"
                    for name, formula in base_data['formulas'].items():
                        result += f"- **{name}**: `{formula}`\n"
                    result += "\n"

                # Display mappings
                if "display" in base_data:
                    result += "## Display Mappings\n"
                    for prop, display_name in base_data['display'].items():
                        result += f"- `{prop}` → **{display_name}**\n"
                    result += "\n"

                # Views
                if "views" in base_data:
                    result += f"## Views ({len(base_data['views'])} total)\n\n"
                    for idx, view in enumerate(base_data['views'], 1):
                        view_name = view.get('name', f'View {idx}')
                        view_type = view.get('type', 'unknown')
                        result += f"### {idx}. {view_name} (type: {view_type})\n\n"

                        if 'filters' in view:
                            result += "**Filters:**\n"
                            result += f"```yaml\n{yaml.dump(view['filters'], default_flow_style=False)}```\n"

                        if 'order' in view:
                            result += f"**Sort order:** {', '.join(view['order'])}\n"

                        if 'group_by' in view:
                            result += f"**Group by:** {view['group_by']}\n"

                        if 'agg' in view:
                            result += f"**Aggregation:** {view['agg']}\n"

                        if 'limit' in view:
                            result += f"**Limit:** {view['limit']} items\n"

                        result += "\n"

                return [types.TextContent(type="text", text=result)]

            except yaml.YAMLError as e:
                return [types.TextContent(type="text", text=f"Error parsing YAML: {str(e)}")]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error reading base file: {str(e)}")]

        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

# 2. Transport Management
# SseServerTransport handles the session logic internally for us if we use it correctly.
# We will create one transport instance per request in the SSE endpoint,
# but we need a way to route the POST request to it.
# The SDK's SseServerTransport is designed for Starlette.

app = FastAPI()

# We need a custom class to bridge FastAPI and MCP Transport
class MCPServerApp:
    def __init__(self, mcp_server):
        self.mcp_server = mcp_server
        # The transport handles the /messages endpoint internally
        self.transport = SseServerTransport("/messages")

    async def handle_sse(self, scope, receive, send):
        async with self.transport.connect_sse(scope, receive, send) as streams:
            async with self.mcp_server.create_initialization_context() as init_ctx:
                await self.mcp_server.run(streams[0], streams[1], init_ctx)

    async def handle_messages(self, scope, receive, send):
        await self.transport.handle_post_message(scope, receive, send)

mcp_app = MCPServerApp(server)

import threading

@app.on_event("startup")
async def startup():
    if SKIP_AUTH_DEBUG: logger.warning("⚠️ AUTH DISABLED")

    # Initialize semantic search with file watcher in background thread
    logger.info("Starting semantic search engine with file watcher...")
    try:
        def start_watcher_thread():
            """Background thread to start the file watcher."""
            try:
                # Get DB path from environment or default
                db_path = Path(os.getenv("DB_PATH", "/app/chroma_data"))
                search_engine = semantic.get_search_engine(VAULT_PATH, db_path)
                search_engine.start_watcher()
                # Keep thread alive
                while True:
                    import time
                    time.sleep(5)
            except Exception as e:
                logger.error(f"FATAL error in watcher thread: {e}", exc_info=True)

        watcher_thread = threading.Thread(target=start_watcher_thread, daemon=True)
        watcher_thread.start()
        logger.info("✓ File watcher thread started")
    except Exception as e:
        logger.error(f"Failed to start file watcher: {e}")
        logger.warning("Semantic search will not be available")

@app.middleware("http")
async def auth(request: Request, call_next):
    if not SKIP_AUTH_DEBUG and request.url.path.startswith("/mcp"):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(status_code=401, headers={"WWW-Authenticate": "Bearer"}, content={})
        # Validate token
        token = auth_header.replace("Bearer ", "")
        if token not in oauth_tokens:
            return JSONResponse(status_code=401, headers={"WWW-Authenticate": "Bearer"}, content={"error": "invalid_token"})
    return await call_next(request)

# OAuth 2.0 Endpoints

@app.get("/.well-known/oauth-authorization-server")
async def oauth_metadata(request: Request):
    """OAuth 2.0 Authorization Server Metadata (RFC 8414)"""
    base_url = request.url.scheme + "://" + request.url.netloc
    return JSONResponse({
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "registration_endpoint": f"{base_url}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256", "plain"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic", "none"]
    })

@app.post("/register")
async def register_client(request: Request):
    """Dynamic Client Registration (RFC 7591)"""
    body = await request.json()
    client_id = str(uuid.uuid4())
    client_secret = secrets.token_urlsafe(32)

    oauth_clients[client_id] = {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uris": body.get("redirect_uris", []),
        "client_name": body.get("client_name", "Claude")
    }

    logger.info(f"Registered OAuth client: {client_id}")
    return JSONResponse({
        "client_id": client_id,
        "client_secret": client_secret,
        "client_id_issued_at": 1234567890,
        "redirect_uris": body.get("redirect_uris", [])
    })

@app.get("/authorize")
async def authorize(
    request: Request,
    response_type: str = None,
    client_id: str = None,
    redirect_uri: str = None,
    state: str = None,
    code_challenge: str = None,
    code_challenge_method: str = None
):
    """OAuth Authorization Endpoint - Auto-approve for simplicity"""
    if not client_id or client_id not in oauth_clients:
        return HTMLResponse("<h1>Invalid client</h1>", status_code=400)

    # Auto-approve (skip user consent for simplicity)
    code = secrets.token_urlsafe(32)
    oauth_codes[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method
    }

    logger.info(f"Issued authorization code for client {client_id}")

    # Redirect back to Claude with the code
    redirect_url = f"{redirect_uri}?code={code}&state={state}" if state else f"{redirect_uri}?code={code}"
    return RedirectResponse(redirect_url)

@app.post("/token")
async def token_endpoint(
    request: Request,
    grant_type: str = Form(None),
    code: str = Form(None),
    redirect_uri: str = Form(None),
    client_id: str = Form(None),
    client_secret: str = Form(None),
    code_verifier: str = Form(None)
):
    """OAuth Token Endpoint"""
    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    if code not in oauth_codes:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    code_data = oauth_codes[code]

    # Validate client
    if code_data["client_id"] != client_id:
        return JSONResponse({"error": "invalid_client"}, status_code=400)

    # Generate access token
    access_token = secrets.token_urlsafe(32)
    oauth_tokens[access_token] = {
        "client_id": client_id,
        "scope": "mcp"
    }

    # Clean up used code
    del oauth_codes[code]

    logger.info(f"Issued access token for client {client_id}")

    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": 3600
    })

@app.get("/")
async def root_info(request: Request):
    """MCP Server information endpoint"""
    base_url = str(request.base_url).rstrip("/")
    return JSONResponse({
        "name": "Obsidian Vault MCP Server",
        "version": "1.0.0",
        "protocol": "mcp",
        "protocolVersion": "2024-11-05",
        "transport": {
            "type": "sse",
            "sse_url": f"{base_url}/sse"
        },
        "capabilities": {
            "tools": {}
        }
    })

@app.post("/")
async def root_post(request: Request):
    """Handle MCP protocol POST requests via HTTP transport"""
    try:
        body = await request.json()
        method = body.get('method', 'unknown')
        logger.info(f"Received MCP request: {method}")

        # Handle notifications (no response needed)
        if method.startswith("notifications/"):
            logger.info(f"Processed notification: {method}")
            return JSONResponse({"jsonrpc": "2.0"}, status_code=200)

        # Handle MCP protocol messages
        if method == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "Obsidian Vault MCP Server",
                        "version": "1.0.0"
                    }
                }
            })
        elif body.get("method") == "tools/list":
            tools = await list_tools()
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.inputSchema
                        } for t in tools
                    ]
                }
            })
        elif body.get("method") == "tools/call":
            params = body.get("params", {})
            result = await call_tool(params.get("name"), params.get("arguments", {}))
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "result": {
                    "content": [
                        {
                            "type": r.type,
                            "text": r.text if hasattr(r, 'text') else str(r)
                        } for r in result
                    ]
                }
            })
        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {body.get('method')}"
                }
            }, status_code=400)
    except Exception as e:
        logger.error(f"Error handling MCP request: {e}")
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": body.get("id") if "body" in locals() else None,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }, status_code=500)

@app.get("/sse")
async def sse_root(request: Request):
    """MCP SSE endpoint at root level for Claude compatibility"""
    return await mcp_app.handle_sse(request.scope, request.receive, request._send)

@app.post("/messages")
async def messages_root(request: Request):
    """MCP messages endpoint at root level for Claude compatibility"""
    return await mcp_app.handle_messages(request.scope, request.receive, request._send)

@app.get("/mcp/sse")
async def sse(request: Request):
    # Pass raw ASGI scope to the transport
    return await mcp_app.handle_sse(request.scope, request.receive, request._send)

@app.post("/mcp/messages")
async def messages(request: Request):
    return await mcp_app.handle_messages(request.scope, request.receive, request._send)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)