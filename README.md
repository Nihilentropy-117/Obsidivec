# Vaultkeeper

An MCP server that gives LLMs intelligent read/write access to an Obsidian vault. Semantic search, Bases support, fact extraction, and note mutation — all exposed as MCP tools over Streamable HTTP.

## What It Does

Connect Claude, GPT, Gemini, or any MCP-compatible client to your Obsidian vault. Ask questions in natural language. The server's internal agent figures out what to search, where to look, and synthesizes answers.

**Example queries it handles:**
- "What's John Smith's garage code?" → searches extracted facts table, finds the exact value
- "Show me restaurants in New Orleans I haven't visited" → queries the Places Base, filters by NL
- "Summarize my notes on Stoic philosophy" → semantic search across the vault
- "What's linked to my Project X note?" → graph traversal via wikilinks

## Architecture

```
MCP Client (Claude, etc.)
    │
    ▼ HTTPS + Bearer token
Reverse Proxy (your domain)
    │
    ▼
Docker: Vaultkeeper
    ├── FastMCP (Streamable HTTP)
    ├── Internal Agent (OpenRouter)
    │   ├── Router (query classification)
    │   ├── Retriever (semantic + text + facts + Bases)
    │   └── Synthesizer (answer generation)
    ├── LanceDB (embedded vector DB)
    └── /vault (mounted volume)
```

## Tools Exposed

### Navigation
| Tool | Description |
|------|-------------|
| `vault_ls` | List directory contents |
| `vault_read` | Read note with parsed frontmatter, links, tags |
| `vault_tree` | Recursive directory tree |
| `vault_tags` | List all tags or find notes by tag |

### Search (agent-backed)
| Tool | Description |
|------|-------------|
| `vault_query` | Natural language Q&A over the entire vault |
| `vault_search` | Hybrid search with optional strategy override |

### Bases
| Tool | Description |
|------|-------------|
| `base_list` | List all Obsidian Bases with schemas |
| `base_read` | Read a Base's raw schema |
| `base_query` | Query a Base with optional NL filtering |

### Mutation
| Tool | Description |
|------|-------------|
| `note_create` | Create a new note with optional frontmatter |
| `note_delete` | Delete a note |
| `note_patch` | Line-level insert/delete/replace operations |
| `note_append` | Append content to a note |
| `note_frontmatter_update` | Update specific frontmatter fields |
| `vault_undo` | Revert all changes from the last operation |

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url> vaultkeeper
cd vaultkeeper
cp .env.example .env
cp config.example.yml config.yml
```

Edit `.env`:
```bash
VAULT_PATH=/path/to/your/obsidian/vault
VAULTKEEPER_TOKEN=$(openssl rand -hex 32)
OPENROUTER_API_KEY=your-key-here
```

### 2. Run with Docker Compose

```bash
docker compose up -d
```

The server starts on port 8000. On first run, it indexes your entire vault in the background.

### 3. Connect from Claude

In Claude Desktop or claude.ai, add as a custom connector:
- **URL:** `https://your-domain.com/mcp`
- **Auth:** Bearer token (the `VAULTKEEPER_TOKEN` you generated)

Or via the API:
```json
{
  "mcp_servers": [{
    "type": "url",
    "url": "https://your-domain.com/mcp",
    "name": "vaultkeeper",
    "authorization_token": "your-token-here"
  }]
}
```

## Configuration

See `config.example.yml` for all options. Key settings:

- **models.provider**: `openrouter` (default) — swap to `ollama` for local inference
- **models.router/extractor/synthesizer**: Configure which model handles each role
- **embeddings.model**: Embedding model for vector search
- **vault.debounce_seconds**: How long to wait after file changes before re-indexing
- **index.fact_extraction**: Enable/disable LLM-based fact extraction at index time

## How Search Works

When you call `vault_query`, the internal agent:

1. **Routes** the query through a classifier that picks retrieval strategies
2. **Executes** strategies in parallel:
   - **Semantic**: Vector similarity search (LanceDB)
   - **Text**: Exact match via ripgrep
   - **Facts**: Key-value lookup in extracted facts table
   - **Frontmatter**: Structured property queries
   - **Bases**: Obsidian Base resolution with NL filtering
   - **Graph**: Wikilink traversal
3. **Merges** and deduplicates results
4. **Synthesizes** an answer using the synthesizer model

The client LLM never orchestrates this — it calls one tool and gets a clean answer.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run locally (requires vault path and API keys)
VAULT_PATH=/path/to/vault OPENROUTER_API_KEY=... python -m vaultkeeper.server

# Lint
ruff check src/

# Test
pytest
```

## License

MIT
