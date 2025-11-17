# Obsidian MCP Server

A comprehensive Model Context Protocol (MCP) server for interacting with Obsidian vaults. This server provides AI assistants with powerful tools to search, read, create, modify, and manage your Obsidian notes using both vector embeddings and text-based search.

## Features

### 🔍 Search Capabilities
- **Vector Search**: Semantic search using embeddings (powered by ChromaDB)
- **Text Search**: Regular and fuzzy text matching across all notes
- **Frontmatter Aware**: Parses and respects YAML frontmatter in notes

### 📝 Note Management
- **Create Notes**: Create new markdown files with optional frontmatter
- **Read Notes**: Access note contents with metadata
- **Update Notes**: Modify notes using unified diff format
- **Delete Notes**: Safe deletion with optional trash folder
- **List Notes**: Browse vault contents with filtering

### 📁 Vault Operations
- **Create Folders**: Organize notes with folder creation
- **Vault Statistics**: Get comprehensive vault and index stats
- **Reindex**: Full vault reindexing for vector search

### 🔐 Security
- **OAuth 2.1**: Full OAuth 2.1 resource server implementation
- **Token Validation**: JWT token validation with JWKS
- **Path Security**: Prevents directory traversal attacks

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Environment**:
Copy `example.env` to `.env` and configure:
```bash
cp example.env .env
```

Edit `.env`:
```env
# Required
VAULT_PATH = "./vault"
DB_PATH = "./chroma_data"

# Optional - customize embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Optional - OAuth 2.1
MCP_OAUTH_ENABLED = "false"
```

## Usage

### Running the Server

#### Development Mode
Test the server using the MCP Inspector:
```bash
mcp dev mcp_server.py
```

#### Install for Claude Desktop
Install the server for use with Claude Desktop:
```bash
mcp install mcp_server.py
```

#### Manual Configuration
Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):
```json
{
  "mcpServers": {
    "obsidian-vault": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "env": {
        "VAULT_PATH": "/path/to/your/vault",
        "DB_PATH": "/path/to/chroma_data"
      }
    }
  }
}
```

## Available Tools

### 1. `vector_search`
Search notes using semantic similarity via vector embeddings.

**Parameters**:
- `query` (string): Search query text
- `n_results` (int, optional): Number of results (default: 5, max: 20)

**Returns**:
```json
{
  "query": "machine learning",
  "results": [
    {
      "filepath": "AI/ML Basics.md",
      "chunk_index": 0,
      "document": "Content...",
      "distance": 0.234
    }
  ],
  "count": 1
}
```

**Example**:
```python
# Find notes about Python programming
result = await vector_search(
    query="Python programming best practices",
    n_results=10
)
```

### 2. `text_search`
Regular text search with optional fuzzy matching.

**Parameters**:
- `query` (string): Search query
- `fuzzy` (bool, optional): Enable fuzzy matching (default: true)
- `case_sensitive` (bool, optional): Case-sensitive search (default: false)

**Returns**:
```json
{
  "query": "TODO",
  "results": [
    {
      "filepath": "Projects/Website.md",
      "matches": [
        {
          "line_number": 15,
          "line": "- TODO: Fix navigation",
          "match_type": "exact"
        }
      ],
      "match_count": 1
    }
  ],
  "files_matched": 1
}
```

### 3. `create_note`
Create a new markdown note in the vault.

**Parameters**:
- `path` (string): Relative path for new note (e.g., "folder/note.md")
- `content` (string): Markdown content
- `frontmatter_data` (dict, optional): YAML frontmatter as dictionary

**Returns**:
```json
{
  "status": "success",
  "filepath": "Ideas/New Feature.md",
  "indexed": true,
  "message": "Note created and indexed"
}
```

**Example**:
```python
result = await create_note(
    path="Projects/New Project.md",
    content="# New Project\n\nProject details here...",
    frontmatter_data={
        "tags": ["project", "todo"],
        "created": "2025-01-17"
    }
)
```

### 4. `create_folder`
Create a new folder in the vault.

**Parameters**:
- `path` (string): Relative path for new folder

**Returns**:
```json
{
  "status": "success",
  "path": "Projects/2025",
  "message": "Folder created successfully"
}
```

### 5. `read_note`
Read the contents of a note.

**Parameters**:
- `path` (string): Relative path to note
- `include_frontmatter` (bool, optional): Include frontmatter (default: true)

**Returns**:
```json
{
  "filepath": "Daily/2025-01-17.md",
  "content": "Note content...",
  "frontmatter": {
    "tags": ["daily-note"],
    "date": "2025-01-17"
  },
  "stats": {
    "size_bytes": 1234,
    "modified": "2025-01-17T10:30:00",
    "created": "2025-01-17T08:00:00"
  }
}
```

### 6. `update_note_diff`
Update a note using unified diff format.

**Parameters**:
- `path` (string): Relative path to note
- `diff` (string): Unified diff to apply

**Returns**:
```json
{
  "status": "success",
  "filepath": "Projects/API.md",
  "backup_created": "/vault/Projects/API.md.backup",
  "reindexed": true,
  "message": "Note updated successfully"
}
```

**Example Diff**:
```diff
@@ -1,3 +1,3 @@
 # API Documentation

-Version: 1.0
+Version: 2.0
```

### 7. `delete_note`
Delete a note from the vault.

**Parameters**:
- `path` (string): Relative path to note
- `permanent` (bool, optional): Permanently delete (default: false)

**Returns**:
```json
{
  "status": "success",
  "filepath": "Old/Deprecated.md",
  "trash_location": ".trash/Deprecated.md",
  "permanent": false,
  "message": "Note moved to trash"
}
```

### 8. `list_notes`
List all notes in the vault or a specific folder.

**Parameters**:
- `folder` (string, optional): Folder to list (default: "" for root)
- `pattern` (string, optional): File pattern (default: "*.md")
- `recursive` (bool, optional): Recursive search (default: true)

**Returns**:
```json
{
  "folder": "/",
  "pattern": "*.md",
  "recursive": true,
  "notes": [
    {
      "path": "Projects/Website.md",
      "name": "Website.md",
      "size_bytes": 2048,
      "modified": "2025-01-17T12:00:00",
      "frontmatter": {"tags": ["project"]}
    }
  ],
  "count": 1
}
```

### 9. `get_vault_stats`
Get comprehensive vault statistics.

**Returns**:
```json
{
  "vault_path": "./vault",
  "total_notes": 150,
  "total_size_bytes": 512000,
  "total_size_mb": 0.49,
  "max_folder_depth": 3,
  "vector_db": {
    "total_chunks": 450,
    "indexed_files": 148
  },
  "embedding_model": "all-MiniLM-L6-v2",
  "collection_name": "obsidian_vault"
}
```

### 10. `reindex_vault`
Reindex all notes for vector search.

**Parameters**:
- `force` (bool, optional): Force reindex all files (default: false)

**Returns**:
```json
{
  "status": "success",
  "stats": {
    "processed": 150,
    "indexed": 148,
    "skipped": 2,
    "errors": 0,
    "total_chunks": 450
  },
  "message": "Reindexed 148 files with 450 total chunks"
}
```

## Available Resources

Resources provide read-only access to vault data through a URI scheme.

### `vault://note/{path}`
Access individual note contents.

**Example**: `vault://note/Projects/Website.md`

### `vault://list`
Get a list of all notes in the vault.

**Example**: `vault://list`

### `vault://stats`
Get vault statistics.

**Example**: `vault://stats`

## Available Prompts

Prompts are reusable templates for common LLM interactions.

### `search_and_summarize`
Search and summarize notes on a topic.

**Parameters**:
- `topic` (string): Topic to search for

### `create_note_with_context`
Create a new note with context from existing notes.

**Parameters**:
- `title` (string): Title for new note
- `related_topics` (string, optional): Topics to search for context

## OAuth 2.1 Authentication

The MCP server implements OAuth 2.1 as a resource server for secure authentication.

### Configuration

1. **Enable OAuth** in `.env`:
```env
MCP_OAUTH_ENABLED = "true"
```

2. **Configure OAuth Provider**:

For **Google OAuth**:
```env
MCP_OAUTH_ISSUER = "https://accounts.google.com"
MCP_OAUTH_AUDIENCE = "your-client-id.apps.googleusercontent.com"
MCP_OAUTH_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"
```

For **Auth0**:
```env
MCP_OAUTH_ISSUER = "https://your-domain.auth0.com/"
MCP_OAUTH_AUDIENCE = "your-api-identifier"
MCP_OAUTH_JWKS_URI = "https://your-domain.auth0.com/.well-known/jwks.json"
```

For **Azure AD**:
```env
MCP_OAUTH_ISSUER = "https://login.microsoftonline.com/{tenant-id}/v2.0"
MCP_OAUTH_AUDIENCE = "your-application-id"
MCP_OAUTH_JWKS_URI = "https://login.microsoftonline.com/{tenant-id}/discovery/v2.0/keys"
```

### Token Validation

The server validates JWT tokens using:
- **JWKS (JSON Web Key Set)**: Fetches public keys from provider
- **Claims Validation**: Verifies issuer, audience, and expiration
- **Signature Validation**: Ensures token authenticity with RS256

### Security Features

- ✅ OAuth 2.1 compliant
- ✅ PKCE (Proof Key for Code Exchange) support
- ✅ Resource Indicators (RFC 8707) compatible
- ✅ JWT token validation with JWKS
- ✅ Automatic token expiration handling
- ✅ Path traversal protection
- ✅ Secure file operations

## Advanced Configuration

### Custom Embedding Models

You can use any model from Hugging Face:

```env
# Fast and lightweight (default)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Better quality, slower
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Multilingual support
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# State-of-the-art (2025)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
```

### Custom Chunking

Control how notes are split for indexing:

```env
# Split on double newlines (default)
DEFAULT_SPLIT_TOKEN = "\n\n"

# Split on headers
DEFAULT_SPLIT_TOKEN = "\n#"

# Split on horizontal rules
DEFAULT_SPLIT_TOKEN = "\n---\n"
```

Or use per-note frontmatter:
```yaml
---
split_token: "\n## "
tags: [documentation]
---

Your note content...
```

## Troubleshooting

### Vector Search Returns No Results

1. **Check if vault is indexed**:
   - Use `get_vault_stats` to see indexed files
   - Run `reindex_vault` to rebuild index

2. **Verify model loaded**:
   - Check logs for "Model loaded successfully"
   - Ensure `EMBEDDING_MODEL` is valid

3. **Query too specific**:
   - Try broader search terms
   - Use `text_search` for exact matches

### OAuth Authentication Failing

1. **Verify JWKS URI is accessible**:
   ```bash
   curl https://www.googleapis.com/oauth2/v3/certs
   ```

2. **Check token claims**:
   - Ensure `iss` matches `MCP_OAUTH_ISSUER`
   - Ensure `aud` matches `MCP_OAUTH_AUDIENCE`
   - Verify token hasn't expired

3. **Check logs** for detailed error messages

### Path Not Found Errors

- All paths are relative to `VAULT_PATH`
- Use forward slashes (`/`) not backslashes (`\`)
- Don't include leading slash: use `folder/note.md` not `/folder/note.md`

## Performance Tips

1. **Embedding Model Selection**:
   - Use smaller models for faster indexing
   - Use larger models for better search quality

2. **Chunking Strategy**:
   - Smaller chunks = more precise results
   - Larger chunks = more context per result

3. **Reindexing**:
   - Only reindex when needed
   - Use `force=false` to skip already-indexed files

## Integration Examples

### With Claude Desktop

The MCP server integrates seamlessly with Claude Desktop:

1. Install the server using `mcp install mcp_server.py`
2. Ask Claude to search your vault:
   ```
   Search my Obsidian notes for information about machine learning
   ```
3. Create notes through conversation:
   ```
   Create a new note about Python asyncio in my Projects folder
   ```

### With Custom MCP Clients

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async with stdio_client(
    command="python",
    args=["mcp_server.py"],
    env={
        "VAULT_PATH": "/path/to/vault",
        "DB_PATH": "/path/to/chroma"
    }
) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize
        await session.initialize()

        # Call a tool
        result = await session.call_tool(
            "vector_search",
            arguments={"query": "Python", "n_results": 5}
        )
        print(result)
```

## Architecture

### Components

```
mcp_server.py
├── FastMCP Server (MCP Protocol)
├── Vector Database (ChromaDB)
│   ├── Sentence Transformers (Embeddings)
│   └── Persistent Storage
├── OAuth 2.1 (JWT Validation)
│   └── JWKS Client
└── File System Operations
    └── Obsidian Vault
```

### Data Flow

```
1. Tool Call → MCP Server
2. OAuth Validation (if enabled)
3. Tool Execution
   ├── Vector Search → ChromaDB Query
   ├── Text Search → File System Scan
   ├── CRUD Operations → File System + Reindex
   └── Vault Stats → Aggregate Data
4. Return Results → MCP Client
```

## Contributing

This MCP server is part of the Obsidivec project. For issues or improvements:

1. Check existing issues in the repository
2. Test changes with `mcp dev mcp_server.py`
3. Ensure OAuth tests pass
4. Update documentation

## License

Same license as the main Obsidivec project.

## Support

- **Documentation**: See main README.md
- **Issues**: GitHub Issues
- **MCP Specification**: https://modelcontextprotocol.io/

## Version History

### v1.0.0 (2025-01-17)
- Initial release
- Full MCP server implementation
- 10 comprehensive tools
- 3 resources
- 2 prompt templates
- OAuth 2.1 authentication
- Vector and text search
- Complete CRUD operations
- Vault management features
