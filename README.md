# Obsidivec

A powerful vector search engine for Obsidian vaults with integrated Telegram bot support and AI-powered query responses.

## TL;DR - Quick Start

1. **Clone and configure**:
   ```bash
   git clone https://github.com/Nihilentropy-117/Obsidivec.git
   cd Obsidivec
   cp config.example.toml config.toml
   ```

2. **Edit `config.toml`** - Add your API keys:
   - `telegram.api_key` - From [@BotFather](https://t.me/botfather)
   - `llm.openrouter_key` - From [OpenRouter](https://openrouter.ai)
   - `server.api_key` - Any secret string for API authentication

3. **Start services**:
   ```bash
   docker-compose up --build -d
   ```

4. **Setup Obsidian Sync**:
   - Open `http://localhost:3000` in your browser
   - In Obsidian, open `/vault` as your vault
   - Login to your Obsidian account
   - Enable Sync plugin and connect to your cloud vault
   - Let it sync (your notes will auto-index)

5. **Query via Telegram**:
   - Message your Telegram bot
   - Ask questions about your notes
   - Get AI-powered summaries from your vault

**Done!** Your vault is now searchable via Telegram with AI assistance.

---

## Features

- **Vector Search**: Semantic search across your Obsidian vault using sentence transformers
- **Real-time Sync**: Automatic file watching and indexing with ChromaDB
- **Obsidian Web Interface**: Full Obsidian desktop app accessible via browser with VNC
- **Obsidian Sync Support**: Enable Obsidian Sync through the web interface for cloud synchronization
- **Telegram Bot**: Optional Telegram bot interface for querying your vault
- **AI-Powered Responses**: LLM integration via OpenRouter for intelligent summaries
- **Custom Chunking**: Flexible document chunking with frontmatter configuration
- **Docker Support**: Fully containerized deployment with Docker Compose
- **REST API**: FastAPI-based API with authentication support
- **Shared Vault**: Single vault directory shared between Obsidian app and vector search

## Architecture

The system consists of **two Docker containers** working together:

### Container 1: Vector Search Service (`obsidian-search`)
Running three concurrent components:
1. **FastAPI Server** (main thread): REST API for vector search operations
2. **File Watcher** (daemon thread): Monitors vault for changes and maintains index
3. **Telegram Bot** (separate process): Optional bot interface for queries

### Container 2: Obsidian Web Interface (`obsidian`)
- **Full Obsidian Desktop App**: Complete Obsidian application accessible via web browser
- **VNC Access**: Web-based VNC interface for GUI interaction
- **Shared Vault**: Mounts the same vault directory as the search service
- **Obsidian Sync**: Can enable official Obsidian Sync for cloud synchronization

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

## Using the Obsidian Web Interface

The docker-compose setup includes a full Obsidian desktop application accessible through your web browser.

### Accessing Obsidian

1. **Open your browser** and navigate to `http://localhost:3000`
2. You'll see the Obsidian desktop application running in a web-based VNC viewer
3. The vault is automatically mounted at `/vault` inside the container

### First-Time Setup

On first launch:
1. Obsidian will ask you to open or create a vault
2. Select **"Open folder as vault"**
3. Navigate to `/vault` directory
4. Click **"Open"** to open your vault

### Enabling Obsidian Sync

If you have an Obsidian Sync subscription, you can enable cloud synchronization:

1. **Login to Obsidian** through the web interface
2. Go to **Settings** (gear icon) → **Core plugins**
3. Enable **"Sync"** plugin
4. Click **"Sync"** in the left sidebar
5. **Sign in** with your Obsidian account
6. Choose to either:
   - **Connect to existing vault**: Sync with your existing cloud vault
   - **Create new remote vault**: Upload this vault to Obsidian Sync

### What Happens After Enabling Sync

Once Obsidian Sync is enabled and connected:

1. **Bidirectional Synchronization**:
   - Changes made in the web interface sync to Obsidian cloud
   - Changes from other devices sync down to the container
   - The local `/vault` directory stays in sync

2. **Automatic Vector Re-indexing**:
   - The file watcher detects all changes written to `/vault`
   - New/modified notes are automatically re-indexed
   - Vector search stays up-to-date with your synchronized vault

3. **Multi-Device Workflow**:
   - Edit on mobile → Syncs to cloud → Downloads to container → Auto-indexed
   - Edit in web interface → Syncs to cloud → Available on all devices
   - Query via Telegram bot → Searches the synchronized vault

### Shared Vault Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Obsidian Cloud                        │
│                  (if Sync enabled)                       │
└────────────────────┬────────────────────────────────────┘
                     │ Bidirectional Sync
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Host: ./vault/ directory                    │
│         (shared between both containers)                 │
└───────────┬─────────────────────────────┬───────────────┘
            │                             │
            ▼                             ▼
  ┌─────────────────┐         ┌──────────────────────┐
  │ Obsidian        │         │ Vector Search        │
  │ Container       │         │ Container            │
  │                 │         │                      │
  │ • Web UI        │         │ • File Watcher       │
  │ • VNC Access    │         │ • Auto-indexing      │
  │ • Sync Plugin   │         │ • ChromaDB           │
  │ • /vault mount  │         │ • /vault mount (ro)  │
  └─────────────────┘         └──────────────────────┘
```

**Key Points**:
- Both containers share the same `./vault/` directory from the host
- Obsidian container has **read/write** access (can modify files)
- Search container has **read-only** access (monitors for changes)
- File watcher automatically detects and indexes all changes
- Obsidian Sync keeps everything synchronized across devices

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

Choose the best model for your use case based on the [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard):

| Model | Parameters | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `BAAI/bge-small-en-v1.5` | 33M | Very Fast | Good | Best for speed & efficiency |
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | Very Fast | Good | Default, fastest option |
| `nomic-ai/nomic-embed-text-v1.5` | 137M | Medium | Excellent | Best accuracy, matryoshka embeddings |
| `BAAI/bge-base-en-v1.5` | 109M | Medium | Very Good | Balanced performance |
| `sentence-transformers/all-mpnet-base-v2` | 110M | Medium | Very Good | General purpose |
| `Alibaba-NLP/gte-base-en-v1.5` | 137M | Medium | Very Good | Strong retrieval performance |

**2025 Recommendations**:
- **Speed Priority**: `BAAI/bge-small-en-v1.5` (33M params, ~14ms latency)
- **Balanced**: `BAAI/bge-base-en-v1.5` (excellent accuracy/speed trade-off)
- **Maximum Accuracy**: `nomic-ai/nomic-embed-text-v1.5` (supports matryoshka embeddings, 768→128 dims)
- **Multilingual**: `Alibaba-NLP/gte-multilingual-base` (305M params, 100+ languages)

Set via `EMBEDDING_MODEL` environment variable in `.env` file.

**Note**: Larger models provide better semantic understanding but require more RAM and slower indexing. For most personal vaults (<10K notes), `bge-small` or `bge-base` offer the best balance.

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
