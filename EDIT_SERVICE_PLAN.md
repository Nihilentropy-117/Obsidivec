# Obsidian Vault Edit Service - Implementation Plan

## Executive Summary

This document outlines the architecture and implementation plan for a separate edit service that provides read-write HTTP API access to the Obsidian vault. The service will run as an independent Docker container alongside the existing `obsidian-search` and `obsidian` containers.

## Current Architecture Analysis

### Existing Services
1. **obsidian-search** (Port 8000)
   - Python 3.11 + FastAPI
   - Read-only vault mount (`./vault:/vault:ro`)
   - X-API-Key authentication
   - Vector search + file watching

2. **obsidian** (Ports 3000-3001)
   - LinuxServer Obsidian GUI image
   - Read-write vault mount (`./vault:/vault`)
   - VNC-based web interface

### Design Constraints
- ✅ NO modifications to existing Python files
- ✅ Separate Docker container
- ✅ Can be added/removed at will
- ✅ Expose HTTP API inside docker stack
- ✅ Mount vault with read-write access

## Recommended Technology Stack

### Choice: Python + FastAPI

**Rationale:**
- Matches existing `obsidian-search` architecture
- Team already familiar with Python/FastAPI
- Consistent dependency management (requirements.txt)
- Excellent API documentation (OpenAPI/Swagger)
- Strong validation with Pydantic models
- Easy integration with existing patterns (X-API-Key auth)

**Alternatives Considered:**
- Node.js/Express: Good, but introduces new language
- Go: Fast but different ecosystem
- Rust: Safest but steeper learning curve

## Service Architecture

### Container Configuration

```yaml
# docker-compose.yml addition
obsidian-edit:
  build:
    context: ./edit-service
    dockerfile: Dockerfile
  ports:
    - "8001:8001"  # Edit API port
  volumes:
    - ./vault:/vault:rw  # READ-WRITE access
  environment:
    - VAULT_PATH=/vault
    - API_KEY=${API_KEY}
    - PORT=8001
  restart: unless-stopped
  networks:
    - obsidian-network  # Shared network for inter-service communication
```

### Directory Structure

```
Obsidivec/
├── edit-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── models.py
│   ├── file_operations.py
│   ├── diff_engine.py
│   └── README.md
├── vault/
├── main.py (existing - unchanged)
├── docker-compose.yml (modified)
└── README.md (update to document edit service)
```

## API Design

### Authentication
- Same pattern as existing service: `X-API-Key` header
- Shares `API_KEY` environment variable
- Optional (like search service)

### Base URL
- Internal: `http://obsidian-edit:8001`
- External: `http://localhost:8001`

### Endpoints

#### 1. File Operations

##### GET /api/v1/files
List all files in vault (recursive)

**Response:**
```json
{
  "files": [
    {
      "path": "/notes/example.md",
      "name": "example.md",
      "size": 1234,
      "modified": "2025-11-03T12:00:00Z",
      "is_directory": false
    }
  ],
  "total": 42
}
```

##### GET /api/v1/files/{path:path}
Read file contents by path

**Query Parameters:**
- `include_metadata` (bool): Include frontmatter separately

**Response:**
```json
{
  "path": "/notes/example.md",
  "content": "file contents here",
  "metadata": {
    "tags": ["example"],
    "split_token": "\\n\\n"
  },
  "size": 1234,
  "modified": "2025-11-03T12:00:00Z"
}
```

##### POST /api/v1/files
Create new file

**Request:**
```json
{
  "path": "/notes/new-note.md",
  "content": "# New Note\n\nContent here",
  "metadata": {
    "tags": ["new"]
  },
  "overwrite": false
}
```

**Response:**
```json
{
  "status": "created",
  "path": "/notes/new-note.md",
  "size": 42
}
```

##### PUT /api/v1/files/{path:path}
Replace entire file content

**Request:**
```json
{
  "content": "completely new content",
  "metadata": {
    "tags": ["updated"]
  },
  "create_if_missing": false
}
```

##### PATCH /api/v1/files/{path:path}
Apply diff/patch to file

**Request:**
```json
{
  "operation": "unified_diff",
  "patch": "--- a/file.md\n+++ b/file.md\n@@ -1,3 +1,3 @@\n-old line\n+new line"
}
```

OR

```json
{
  "operation": "line_replace",
  "line_number": 5,
  "old_content": "old text",
  "new_content": "new text"
}
```

OR

```json
{
  "operation": "search_replace",
  "search": "old text",
  "replace": "new text",
  "all_occurrences": false
}
```

##### DELETE /api/v1/files/{path:path}
Delete file

**Response:**
```json
{
  "status": "deleted",
  "path": "/notes/example.md"
}
```

##### POST /api/v1/files/{path:path}/append
Append content to file

**Request:**
```json
{
  "content": "text to append",
  "position": "end",  // "start" or "end"
  "newline": true    // add newline separator
}
```

#### 2. Folder Operations

##### GET /api/v1/folders
List all folders

**Query Parameters:**
- `recursive` (bool): Include subfolders

**Response:**
```json
{
  "folders": [
    {
      "path": "/notes",
      "name": "notes",
      "file_count": 10,
      "subfolder_count": 2
    }
  ]
}
```

##### POST /api/v1/folders
Create folder

**Request:**
```json
{
  "path": "/notes/new-folder",
  "parents": true  // create parent directories
}
```

##### DELETE /api/v1/folders/{path:path}
Delete folder

**Query Parameters:**
- `recursive` (bool): Delete contents

#### 3. Metadata Operations

##### GET /api/v1/metadata/{path:path}
Get file frontmatter metadata

**Response:**
```json
{
  "path": "/notes/example.md",
  "metadata": {
    "tags": ["example"],
    "split_token": "\\n\\n",
    "created": "2025-01-01"
  }
}
```

##### PUT /api/v1/metadata/{path:path}
Update file frontmatter

**Request:**
```json
{
  "metadata": {
    "tags": ["example", "updated"],
    "custom_field": "value"
  },
  "merge": true  // merge with existing or replace
}
```

#### 4. Batch Operations

##### POST /api/v1/batch/files
Create/update multiple files

**Request:**
```json
{
  "operations": [
    {
      "action": "create",
      "path": "/notes/file1.md",
      "content": "content 1"
    },
    {
      "action": "update",
      "path": "/notes/file2.md",
      "content": "content 2"
    }
  ]
}
```

##### POST /api/v1/batch/search-replace
Search and replace across multiple files

**Request:**
```json
{
  "search": "old term",
  "replace": "new term",
  "paths": ["/notes/*.md"],  // glob patterns
  "preview": false  // return preview without applying
}
```

#### 5. Utility Operations

##### GET /api/v1/health
Health check

**Response:**
```json
{
  "status": "ok",
  "vault_path": "/vault",
  "vault_writable": true,
  "file_count": 42
}
```

##### POST /api/v1/validate
Validate file path and content

**Request:**
```json
{
  "path": "/notes/example.md",
  "content": "# Example\n\nContent",
  "check_overwrite": true
}
```

##### GET /api/v1/search
Search for files by name/pattern

**Query Parameters:**
- `pattern` (string): Glob pattern or regex
- `content` (string): Search in file content

## Implementation Phases

### Phase 1: Core Infrastructure (MVP)
1. Docker container setup
2. FastAPI application skeleton
3. Authentication middleware
4. Basic file operations:
   - GET /files (list)
   - GET /files/{path} (read)
   - POST /files (create)
   - PUT /files/{path} (update)
   - DELETE /files/{path} (delete)

### Phase 2: Advanced File Operations
1. PATCH endpoint with diff support
2. POST /append endpoint
3. Metadata operations (frontmatter)
4. Folder operations

### Phase 3: Batch & Utility Operations
1. Batch file operations
2. Search and replace across files
3. Validation endpoints
4. File search by pattern

### Phase 4: Safety & Quality
1. Comprehensive error handling
2. Input validation and sanitization
3. Path traversal protection
4. File locking mechanisms
5. Unit tests
6. Integration tests

## Security Considerations

### Path Traversal Prevention
```python
import os

def sanitize_path(path: str, base_path: str) -> str:
    """Ensure path is within vault directory"""
    abs_path = os.path.abspath(os.path.join(base_path, path.lstrip('/')))
    if not abs_path.startswith(os.path.abspath(base_path)):
        raise ValueError("Path traversal attempt detected")
    return abs_path
```

### File Size Limits
- Max file size: 10MB (configurable)
- Max batch operations: 100 files

### Concurrency Safety
- File locking for write operations
- Atomic write operations (write to temp, then rename)
- Conflict detection for concurrent edits

### Authentication
- Required for all modification endpoints
- Optional for read-only endpoints (configurable)
- Rate limiting (optional, via nginx/traefik)

## Diff Engine Design

### Supported Diff Formats

1. **Unified Diff** (Git-style)
   - Standard patch format
   - Cross-platform compatible
   - Library: `difflib` (Python stdlib)

2. **Line-based Operations**
   - Replace specific line by number
   - Insert at line number
   - Delete line range

3. **Search and Replace**
   - Simple string replacement
   - Regex-based replacement
   - First occurrence or all occurrences

### Example Implementation

```python
import difflib
from typing import List

def apply_unified_diff(original: str, patch: str) -> str:
    """Apply unified diff patch to content"""
    original_lines = original.splitlines(keepends=True)
    patch_lines = patch.splitlines(keepends=True)

    # Parse patch and apply
    result = list(difflib.unified_diff(original_lines, patch_lines))
    return ''.join(result)

def apply_line_replace(content: str, line_num: int,
                       old_text: str, new_text: str) -> str:
    """Replace text at specific line"""
    lines = content.splitlines()
    if 0 <= line_num < len(lines):
        if old_text in lines[line_num]:
            lines[line_num] = lines[line_num].replace(old_text, new_text, 1)
    return '\n'.join(lines)
```

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "FILE_NOT_FOUND",
    "message": "File not found: /notes/missing.md",
    "details": {
      "path": "/notes/missing.md",
      "vault_path": "/vault"
    }
  }
}
```

### Error Codes
- `FILE_NOT_FOUND` (404)
- `FILE_ALREADY_EXISTS` (409)
- `INVALID_PATH` (400)
- `PATH_TRAVERSAL` (400)
- `FILE_TOO_LARGE` (413)
- `INVALID_DIFF` (400)
- `UNAUTHORIZED` (401)
- `INTERNAL_ERROR` (500)

## Integration with Existing Services

### File Watching
The existing `obsidian-search` service has a file watcher that will automatically detect changes made by the edit service and update the vector index.

**Flow:**
1. Client calls edit API → modifies file in `/vault`
2. Search service file watcher detects change
3. Search service reindexes the modified file
4. No additional coordination needed

### Shared Vault Access
- Search service: read-only mount
- Edit service: read-write mount
- Obsidian GUI: read-write mount
- No conflicts due to file system atomicity

## Testing Strategy

### Unit Tests
- Path sanitization
- Diff application
- Metadata parsing
- Error handling

### Integration Tests
- File CRUD operations
- Diff operations
- Batch operations
- Authentication

### End-to-End Tests
- Create file → Search service indexes it
- Edit file → Search service re-indexes
- Delete file → Search service removes from index

## Configuration

### Environment Variables

```bash
# edit-service/.env
VAULT_PATH=/vault
API_KEY=your_secret_key_here
PORT=8001
MAX_FILE_SIZE_MB=10
MAX_BATCH_SIZE=100
ENABLE_AUTH=true
LOG_LEVEL=INFO
```

## Deployment

### Docker Compose Update

```yaml
services:
  # Existing services...

  obsidian-edit:
    build:
      context: ./edit-service
    ports:
      - "8001:8001"
    volumes:
      - ./vault:/vault:rw
    environment:
      - VAULT_PATH=/vault
      - API_KEY=${API_KEY}
      - PORT=8001
    restart: unless-stopped
    depends_on:
      - obsidian-search
```

### Optional: Nginx Reverse Proxy

```nginx
# Group all services under /api/
location /api/search/ {
    proxy_pass http://obsidian-search:8000/;
}

location /api/edit/ {
    proxy_pass http://obsidian-edit:8001/api/v1/;
}
```

## Documentation Updates

### README.md Additions
- Edit service overview
- API endpoint documentation
- Usage examples
- Security considerations

### Example Client Code

```python
import requests

EDIT_URL = "http://localhost:8001/api/v1"
API_KEY = "your_api_key"

# Create a new note
response = requests.post(
    f"{EDIT_URL}/files",
    json={
        "path": "/notes/my-note.md",
        "content": "# My Note\n\nContent here",
        "metadata": {"tags": ["personal"]}
    },
    headers={"X-API-Key": API_KEY}
)

# Read a file
response = requests.get(
    f"{EDIT_URL}/files/notes/my-note.md",
    headers={"X-API-Key": API_KEY}
)

# Apply a diff
response = requests.patch(
    f"{EDIT_URL}/files/notes/my-note.md",
    json={
        "operation": "search_replace",
        "search": "old text",
        "replace": "new text"
    },
    headers={"X-API-Key": API_KEY}
)
```

## Future Enhancements

### Phase 5+ (Optional)
1. **Conflict Resolution**
   - Optimistic locking with ETags
   - Three-way merge support
   - Change history tracking

2. **Advanced Features**
   - File versioning
   - Backup/restore
   - Export to different formats (PDF, HTML)
   - Real-time collaboration (WebSocket)

3. **Performance**
   - Caching layer (Redis)
   - Async file operations
   - Streaming for large files
   - Compression support

4. **Monitoring**
   - Prometheus metrics
   - File operation statistics
   - Performance monitoring

## Timeline Estimate

- **Phase 1 (MVP)**: 2-3 days
- **Phase 2 (Advanced Ops)**: 2-3 days
- **Phase 3 (Batch/Utility)**: 1-2 days
- **Phase 4 (Safety/Testing)**: 2-3 days

**Total**: 7-11 days for full implementation

## Success Metrics

- All endpoints functional and documented
- 100% path sanitization coverage
- Integration tests passing
- Documentation complete
- No disruption to existing services
- Clean separation of concerns

## Conclusion

This edit service provides a complete HTTP API for vault management while maintaining clean separation from the existing codebase. The architecture is:

- ✅ Non-invasive (no changes to existing Python files)
- ✅ Modular (can be added/removed independently)
- ✅ Consistent (matches existing patterns and style)
- ✅ Secure (path sanitization, authentication, validation)
- ✅ Scalable (can add features without impacting search service)
- ✅ Well-documented (OpenAPI/Swagger auto-generated)

The service integrates seamlessly with the existing file watcher, ensuring vector search stays synchronized with all vault changes.
