# Obsidivec

A powerful vector search engine for Obsidian vaults with integrated Telegram bot support and AI-powered query responses.

## Features

- **Vector Search**: Semantic search across your Obsidian vault using sentence transformers
- **Real-time Sync**: Automatic file watching and indexing with ChromaDB
- **Telegram Bot**: Optional Telegram bot interface for querying your vault
- **AI-Powered Responses**: LLM integration via OpenRouter for intelligent summaries
- **Custom Chunking**: Flexible document chunking with frontmatter configuration
- **Docker Support**: Fully containerized deployment with Docker Compose
- **REST API**: FastAPI-based API with authentication support
- **Obsidian Integration**: Bundled with Obsidian web interface

## Architecture

The system consists of three main components running concurrently:

1. **FastAPI Server** (main thread): REST API for vector search operations
2. **File Watcher** (daemon thread): Monitors vault for changes and maintains index
3. **Telegram Bot** (separate process): Optional bot interface for queries

## Quick Start

### Prerequisites

- Docker & Docker Compose
- (Optional) Telegram Bot Token from [@BotFather](https://t.me/botfather)
- (Optional) OpenRouter API key for AI responses

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Obsidivec
   ```

2. **Configure environment variables**
   ```bash
   cp example.env .env
   ```

   Edit `.env` with your configuration:
   ```env
   OPENROUTER_KEY=your_openrouter_key_here
   API_KEY=your_fastapi_secret_key
   ENABLE_TELEGRAM_BOT=True
   TELEGRAM_API_KEY=your_telegram_bot_token
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

The system will now:
- Start the vector search API on `http://localhost:8000`
- Launch Obsidian web interface on `http://localhost:3000`
- Initialize the Telegram bot (if enabled)
- Begin indexing your vault

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `API_KEY` | Authentication key for FastAPI endpoints | No | None (auth disabled) |
| `VAULT_PATH` | Path to Obsidian vault directory | No | `/vault` |
| `DB_PATH` | Path to ChromaDB storage | No | `/app/db` |
| `EMBEDDING_MODEL` | Sentence transformer model name | No | `all-MiniLM-L6-v2` |
| `ENABLE_TELEGRAM_BOT` | Enable Telegram bot interface | No | `False` |
| `TELEGRAM_API_KEY` | Telegram bot token | Yes (if bot enabled) | None |
| `OPENROUTER_KEY` | OpenRouter API key for LLM responses | Yes (for AI features) | None |
| `DEFAULT_SPLIT_TOKEN` | Default document chunking delimiter | No | `\n\n` |

### Document Chunking

Control how documents are split for indexing using frontmatter:

```markdown
---
split_token: "\n\n"
---

Your document content here...
```

If no `split_token` is specified, the system uses `DEFAULT_SPLIT_TOKEN` from environment.

## API Endpoints

### Health Check
```bash
GET /health
```
Returns server initialization status and collection count.

**Response:**
```json
{
  "status": "ok",
  "collection_count": 1234
}
```

### Search
```bash
POST /search
Headers: X-API-Key: your_api_key
Content-Type: application/json

{
  "query": "your search query",
  "n_results": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "filepath": "/vault/notes/example.md",
      "chunk_index": 0,
      "document": "Matching text chunk..."
    }
  ]
}
```

### Reindex
```bash
POST /reindex
Headers: X-API-Key: your_api_key
```
Triggers a full vault re-index. Useful after bulk changes.

**Response:**
```json
{
  "status": "Reindex started in background."
}
```

## Usage Examples

### Python Client (LLMsearch.py)

Search your vault with AI-powered responses:

```bash
python LLMsearch.py "What are my notes about machine learning?"
```

The script will:
1. Query the vector search API
2. Retrieve relevant chunks
3. Send to OpenRouter for summarization
4. Return a coherent answer

### Programmatic Usage

```python
import requests

SERVER_URL = "http://localhost:8000"
API_KEY = "your_api_key"

# Search the vault
response = requests.post(
    f"{SERVER_URL}/search",
    json={"query": "machine learning", "n_results": 3},
    headers={"X-API-Key": API_KEY}
)

results = response.json()
for result in results['results']:
    print(f"File: {result['filepath']}")
    print(f"Text: {result['document']}\n")
```

### Trigger Reindex

```bash
python reindex.py
```

Or via curl:
```bash
curl -X POST http://localhost:8000/reindex \
  -H "X-API-Key: your_api_key"
```

### Telegram Bot

Once enabled, simply message your bot with any query:

**User:** "What are my notes about Python?"

**Bot:** *[AI-generated summary from your vault]*

## How It Works

### Indexing Pipeline

1. **File Detection**: Watchdog monitors vault for `.md` file changes
2. **Frontmatter Parsing**: Extracts metadata and chunking configuration
3. **Document Chunking**: Splits content based on `split_token`
4. **Embedding Generation**: Creates vector embeddings with filename prefix
5. **Storage**: Stores in ChromaDB with metadata (filepath, chunk_index)

### Search Pipeline

1. **Query Embedding**: Converts search query to vector
2. **Similarity Search**: ChromaDB finds nearest neighbors
3. **Result Formatting**: Returns documents with metadata
4. **LLM Enhancement**: (Optional) Sends to OpenRouter for summarization

### Telegram Bot Integration

- Runs in **separate process** to avoid asyncio conflicts
- Uses `multiprocessing.Process` for true isolation
- Shares vault access via Docker volume mounts
- Independent logging and error handling

## Directory Structure

```
Obsidivec/
├── main.py                 # FastAPI server + file watcher
├── telegram_bot.py         # Telegram bot implementation
├── LLMsearch.py           # CLI search client with AI
├── reindex.py             # Reindex trigger script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container image definition
├── docker-compose.yml     # Multi-service orchestration
├── example.env            # Environment template
├── .env                   # Your configuration (create from example.env)
├── vault/                 # Obsidian vault directory
├── chroma_data/           # ChromaDB persistence
└── obsidian_config/       # Obsidian web config
```

## Development

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload
```

### Running Without Docker

```bash
# Set environment variables
export VAULT_PATH=./vault
export DB_PATH=./chroma_data
export API_KEY=your_key

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Technical Details

### Embedding Strategy

Documents are embedded with filename prefixes for better context:

```
Format: "{filename}:part\n---\n{chunk_content}"
```

This helps the model understand document context during similarity search.

### Thread Safety

- **File Watcher**: Daemon thread with global model/collection access
- **Reindex Operation**: Pauses watcher, recreates collection, resumes
- **API Endpoints**: Thread-safe ChromaDB operations

### Process Architecture

```
uvicorn (main process)
├── FastAPI App (main thread)
├── File Watcher (daemon thread)
│   ├── Load embedding model
│   ├── Connect to ChromaDB
│   ├── Monitor vault changes
│   └── Process file events
└── Telegram Bot (separate process)
    └── asyncio event loop with signal handlers
```

## Troubleshooting

### Bot Not Starting

**Error:** `set_wakeup_fd only works in main thread`

**Solution:** Ensure you're using the latest version which runs the bot in a separate process, not a thread.

### No Search Results

1. Check if files are indexed: `GET /health` → `collection_count > 0`
2. Verify vault path is mounted correctly
3. Check file watcher logs for processing errors
4. Trigger manual reindex: `POST /reindex`

### Authentication Issues

**Error:** `401 Unauthorized`

**Solution:** Ensure `X-API-Key` header matches `API_KEY` environment variable.

### Slow Indexing

- Large vaults may take time on first index
- Check `MODEL_NAME` - smaller models are faster
- Monitor container resources (CPU/RAM)

## Performance Tuning

### Embedding Model Selection

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` | 80MB | Fast | Good (default) |
| `all-mpnet-base-v2` | 420MB | Medium | Better |
| `multi-qa-mpnet-base-dot-v1` | 420MB | Medium | Best for Q&A |

Set via `EMBEDDING_MODEL` environment variable.

### ChromaDB Optimization

ChromaDB automatically persists to disk in `DB_PATH`. For better performance:
- Use SSD storage for `DB_PATH`
- Separate vault and DB volumes
- Regular reindexing for large vaults

## Security Considerations

- **API Authentication**: Always set `API_KEY` in production
- **Network Isolation**: Use Docker networks to isolate services
- **Vault Access**: Mounted as read-only (`:ro`) in docker-compose
- **Secrets Management**: Use `.env` file, never commit secrets

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test your changes
4. Submit a pull request

## License

[Your License Here]

## Acknowledgments

- **ChromaDB**: Vector database backend
- **Sentence Transformers**: Embedding generation
- **FastAPI**: REST API framework
- **python-telegram-bot**: Telegram integration
- **Watchdog**: File system monitoring

## Support

For issues and questions:
- GitHub Issues: [Your repo URL]
- Documentation: [Your docs URL]

---

**Built with ❤️ for the Obsidian community**
