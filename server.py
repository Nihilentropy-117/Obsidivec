"""
MCP Server for Obsidian Vault
Rewritten with security and architectural fixes.
"""

import os
import logging
import secrets
import hashlib
import base64
import threading
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types

from search import fuzzy, semantic
import base_engine
import tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-obsidian")

# Configuration
VAULT_PATH = Path(os.getenv("VAULT_PATH", "/vault"))
DB_PATH = Path(os.getenv("DB_PATH", "/app/chroma_data"))
SKIP_AUTH_DEBUG = os.getenv("SKIP_AUTH_DEBUG", "false").lower() == "true"

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY environment variable must be set. "
        "Get your API key from https://openrouter.ai"
    )

# OAuth credentials - MUST be set via environment in production
STATIC_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
STATIC_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")

if not SKIP_AUTH_DEBUG and (not STATIC_CLIENT_ID or not STATIC_CLIENT_SECRET):
    raise RuntimeError(
        "OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET must be set. "
        "Set SKIP_AUTH_DEBUG=true for development only."
    )

# Token lifetimes
AUTH_CODE_LIFETIME = timedelta(minutes=10)
ACCESS_TOKEN_LIFETIME = timedelta(hours=1)


@dataclass
class AuthCode:
    client_id: str
    redirect_uri: str
    code_challenge: Optional[str]
    code_challenge_method: Optional[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.created_at + AUTH_CODE_LIFETIME


@dataclass
class AccessToken:
    client_id: str
    scope: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.created_at + ACCESS_TOKEN_LIFETIME


class OAuthStore:
    """Thread-safe OAuth storage with expiration cleanup."""

    def __init__(self):
        self._lock = threading.Lock()
        self._codes: dict[str, AuthCode] = {}
        self._tokens: dict[str, AccessToken] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        def cleanup_loop():
            while True:
                time.sleep(self._cleanup_interval)
                self._cleanup_expired()

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    def _cleanup_expired(self):
        with self._lock:
            self._codes = {k: v for k, v in self._codes.items() if not v.is_expired()}
            self._tokens = {k: v for k, v in self._tokens.items() if not v.is_expired()}

    def store_code(self, code: str, data: AuthCode):
        with self._lock:
            self._codes[code] = data

    def pop_code(self, code: str) -> Optional[AuthCode]:
        with self._lock:
            return self._codes.pop(code, None)

    def store_token(self, token: str, data: AccessToken):
        with self._lock:
            self._tokens[token] = data

    def validate_token(self, token: str) -> bool:
        with self._lock:
            data = self._tokens.get(token)
            if data is None or data.is_expired():
                return False
            return True


oauth_store = OAuthStore()


def verify_pkce(code_verifier: str, code_challenge: str, method: str) -> bool:
    """Verify PKCE code_verifier against stored code_challenge."""
    if method == "plain":
        return secrets.compare_digest(code_verifier, code_challenge)
    elif method == "S256":
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return secrets.compare_digest(computed, code_challenge)
    return False


def generate_folder_structure(root_path: Path, prefix: str = "") -> str:
    """Generate a tree-like folder structure (directories only)."""
    output = ""
    subdirs = sorted(
        [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda x: x.name,
    )

    for idx, subdir in enumerate(subdirs):
        is_last = idx == len(subdirs) - 1
        connector = "└── " if is_last else "├── "
        output += f"{prefix}{connector}{subdir.name}/\n"
        extension = "    " if is_last else "│   "
        output += generate_folder_structure(subdir, prefix + extension)

    return output


class FileWatcher:
    """Manages semantic search file watcher with health monitoring."""

    def __init__(self, vault_path: Path, db_path: Path):
        self.vault_path = vault_path
        self.db_path = db_path
        self._search_engine: Optional[semantic.SemanticSearchEngine] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._healthy = False

    def start(self):
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self):
        try:
            self._search_engine = semantic.get_search_engine(self.vault_path, self.db_path)
            self._search_engine.start_watcher()
            self._healthy = True
            logger.info("File watcher started successfully")

            while not self._stop_event.wait(timeout=30):
                pass  # Could add health checks here

        except Exception as e:
            self._healthy = False
            logger.error(f"File watcher failed: {e}", exc_info=True)

    def is_healthy(self) -> bool:
        return self._healthy

    def get_search_engine(self):
        with self._lock:
            if self._search_engine is None:
                self._search_engine = semantic.get_search_engine(self.vault_path, self.db_path)
            return self._search_engine


file_watcher = FileWatcher(VAULT_PATH, DB_PATH)

# Initialize tool handler
tool_handler = tools.ToolHandler(
    vault_path=VAULT_PATH,
    file_watcher=file_watcher,
    folder_structure_generator=generate_folder_structure,
)


# MCP Server Setup
server = Server("Obsidian Vault")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return list of available MCP tools."""
    return tools.get_tool_definitions()


@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Execute a tool by name with given arguments."""
    return await tool_handler.execute_tool(name, arguments)


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    if SKIP_AUTH_DEBUG:
        logger.warning("⚠️  AUTH DISABLED - development mode only")

    logger.info("Starting file watcher...")
    file_watcher.start()

    yield

    logger.info("Shutting down...")
    file_watcher.stop()


app = FastAPI(lifespan=lifespan)


class MCPServerApp:
    def __init__(self, mcp_server: Server):
        self.mcp_server = mcp_server
        self.transport = SseServerTransport("/messages")

    async def handle_sse(self, scope, receive, send):
        async with self.transport.connect_sse(scope, receive, send) as streams:
            async with self.mcp_server.create_initialization_context() as init_ctx:
                await self.mcp_server.run(streams[0], streams[1], init_ctx)

    async def handle_messages(self, scope, receive, send):
        await self.transport.handle_post_message(scope, receive, send)


mcp_app = MCPServerApp(server)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Skip auth for non-MCP endpoints and OAuth flow
    path = request.url.path
    if SKIP_AUTH_DEBUG or not path.startswith("/mcp"):
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            content={"error": "missing_token"},
        )

    token = auth_header[7:]  # Strip "Bearer "
    if not oauth_store.validate_token(token):
        return JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            content={"error": "invalid_token"},
        )

    return await call_next(request)


# OAuth Endpoints


@app.get("/.well-known/oauth-authorization-server")
async def oauth_metadata(request: Request):
    base = f"{request.url.scheme}://{request.url.netloc}"
    return JSONResponse(
        {
            "issuer": base,
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256", "plain"],
            "token_endpoint_auth_methods_supported": [
                "client_secret_post",
                "client_secret_basic",
            ],
        }
    )


@app.post("/register")
async def register_client(request: Request):
    """Return static credentials. Dynamic registration disabled."""
    if not STATIC_CLIENT_ID or not STATIC_CLIENT_SECRET:
        raise HTTPException(503, "OAuth not configured")

    body = await request.json()
    return JSONResponse(
        {
            "client_id": STATIC_CLIENT_ID,
            "client_secret": STATIC_CLIENT_SECRET,
            "client_id_issued_at": int(time.time()),
            "redirect_uris": body.get("redirect_uris", []),
        }
    )


@app.get("/authorize")
async def authorize(
    request: Request,
    response_type: str = None,
    client_id: str = None,
    redirect_uri: str = None,
    state: str = None,
    code_challenge: str = None,
    code_challenge_method: str = None,
):
    if client_id != STATIC_CLIENT_ID:
        return HTMLResponse("<h1>Invalid client</h1>", status_code=400)

    if response_type != "code":
        return HTMLResponse("<h1>Invalid response_type</h1>", status_code=400)

    code = secrets.token_urlsafe(32)
    oauth_store.store_code(
        code,
        AuthCode(
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        ),
    )

    logger.info(f"Issued auth code for client {client_id}")

    sep = "&" if "?" in redirect_uri else "?"
    url = f"{redirect_uri}{sep}code={code}"
    if state:
        url += f"&state={state}"

    return RedirectResponse(url)


@app.post("/token")
async def token_endpoint(
    grant_type: str = Form(None),
    code: str = Form(None),
    client_id: str = Form(None),
    client_secret: str = Form(None),
    code_verifier: str = Form(None),
):
    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    code_data = oauth_store.pop_code(code)
    if code_data is None:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    if code_data.is_expired():
        return JSONResponse({"error": "invalid_grant", "error_description": "code expired"}, status_code=400)

    if code_data.client_id != client_id:
        return JSONResponse({"error": "invalid_client"}, status_code=400)

    # Validate client secret
    if not secrets.compare_digest(client_secret or "", STATIC_CLIENT_SECRET or ""):
        logger.warning(f"Invalid client secret for {client_id}")
        return JSONResponse({"error": "invalid_client"}, status_code=401)

    # Validate PKCE if code_challenge was provided
    if code_data.code_challenge:
        if not code_verifier:
            return JSONResponse(
                {"error": "invalid_request", "error_description": "code_verifier required"},
                status_code=400,
            )
        method = code_data.code_challenge_method or "plain"
        if not verify_pkce(code_verifier, code_data.code_challenge, method):
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "PKCE verification failed"},
                status_code=400,
            )

    # Issue token
    access_token = secrets.token_urlsafe(32)
    oauth_store.store_token(access_token, AccessToken(client_id=client_id, scope="mcp"))

    logger.info(f"Issued access token for {client_id}")

    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": int(ACCESS_TOKEN_LIFETIME.total_seconds()),
        }
    )


# Info Endpoints


@app.get("/")
async def root_info(request: Request):
    base = str(request.base_url).rstrip("/")
    return JSONResponse(
        {
            "name": "Obsidian Vault MCP Server",
            "version": "1.0.0",
            "protocol": "mcp",
            "protocolVersion": "2024-11-05",
            "transport": {"type": "sse", "sse_url": f"{base}/sse"},
            "capabilities": {"tools": {}},
        }
    )


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "status": "healthy" if file_watcher.is_healthy() else "degraded",
            "file_watcher": file_watcher.is_healthy(),
        }
    )


# MCP Protocol Endpoints


async def handle_mcp_request(body: dict) -> JSONResponse:
    """Handle MCP JSON-RPC requests."""
    method = body.get("method", "")
    request_id = body.get("id")

    # Notifications don't need responses
    if method.startswith("notifications/"):
        return JSONResponse({"jsonrpc": "2.0"})

    if method == "initialize":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "Obsidian Vault MCP Server", "version": "1.0.0"},
                },
            }
        )

    if method == "tools/list":
        tools = await list_tools()
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                        for t in tools
                    ]
                },
            }
        )

    if method == "tools/call":
        params = body.get("params", {})
        result = await call_tool(params.get("name"), params.get("arguments", {}))
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {"type": r.type, "text": r.text if hasattr(r, "text") else str(r)}
                        for r in result
                    ]
                },
            }
        )

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        },
        status_code=400,
    )


@app.post("/")
async def root_post(request: Request):
    try:
        body = await request.json()
        return await handle_mcp_request(body)
    except Exception as e:
        logger.error(f"MCP request failed: {e}", exc_info=True)
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}},
            status_code=500,
        )


@app.get("/sse")
async def sse_root(request: Request):
    scope = request.scope
    receive = request.receive

    async def send(message):
        # Proper ASGI send wrapper
        if hasattr(request, "_send"):
            await request._send(message)

    return await mcp_app.handle_sse(scope, receive, send)


@app.post("/messages")
async def messages_root(request: Request):
    scope = request.scope
    receive = request.receive

    async def send(message):
        if hasattr(request, "_send"):
            await request._send(message)

    return await mcp_app.handle_messages(scope, receive, send)


@app.get("/mcp/sse")
async def mcp_sse(request: Request):
    return await sse_root(request)


@app.post("/mcp/messages")
async def mcp_messages(request: Request):
    return await messages_root(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)