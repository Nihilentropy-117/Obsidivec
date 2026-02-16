import sys
import threading
import multiprocessing
import time
import glob
import json
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer
import frontmatter
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
import os

from bases_manager import (
    BasesConfig,
    parse_base_file,
    write_base_file,
    delete_base_file,
    discover_base_files,
    list_bases_summary,
    query_base,
    get_all_properties,
    get_all_vault_properties,
    create_base_from_search_results,
    evaluate_filter,
)

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Global Configuration & Setup ---
ENABLE_TELEGRAM_BOT = os.getenv("ENABLE_TELEGRAM_BOT", False)
telegram_bot_process = None


def start_telegram_bot():
    """Starts the Telegram bot in a separate process."""
    import asyncio
    from telegram_bot import main as telegram_bot_main

    try:
        logger.info("Telegram bot process: Starting...")
        asyncio.run(telegram_bot_main())
    except Exception as e:
        logger.error(f"FATAL error in Telegram bot process: {e}", exc_info=True)


API_KEY = os.getenv("API_KEY", None)
VAULT_PATH = os.getenv("VAULT_PATH", "/vault")
DB_PATH = os.getenv("DB_PATH", "/app/db")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "obsidian_vault"
TOKEN_WARNING_LIMIT = 1000

logger.info("--- Configuration ---")
logger.info(f"VAULT_PATH: {VAULT_PATH}")
logger.info(f"DB_PATH: {DB_PATH}")
logger.info(f"MODEL_NAME: {MODEL_NAME}")
logger.info("-----------------------")

app = FastAPI(
    title="Obsidian Vector Search Server",
    description="An API to search an Obsidian vault using vector embeddings with Obsidian Bases integration."
)

# --- Global Variables (to be populated on startup) ---
model = None
client = None
collection = None
observer = None
# In-memory cache of parsed .base configs, keyed by filepath
bases_cache: dict[str, BasesConfig] = {}


# --- Pydantic Models ---
class SearchQuery(BaseModel):
    query: str
    n_results: int = 5
    base_filter: Optional[str] = Field(
        default=None,
        description="Path to a .base file whose filters restrict search results."
    )


class SearchResult(BaseModel):
    filepath: str
    chunk_index: int
    document: str
    properties: Optional[dict] = Field(
        default=None,
        description="Structured note/file properties (populated when Bases filter is active)."
    )


class SearchResponse(BaseModel):
    results: list[SearchResult]


class BaseCreateRequest(BaseModel):
    filepath: str = Field(description="Path for the new .base file relative to vault root.")
    filters: Optional[str | dict] = None
    properties: Optional[dict] = None
    formulas: Optional[dict] = None
    summaries: Optional[dict] = None
    views: Optional[list] = None


class BaseUpdateRequest(BaseModel):
    filters: Optional[str | dict] = None
    properties: Optional[dict] = None
    formulas: Optional[dict] = None
    summaries: Optional[dict] = None
    views: Optional[list] = None


class BaseQueryRequest(BaseModel):
    base_filepath: str = Field(description="Path to the .base file to query.")


class BaseFromSearchRequest(BaseModel):
    query: str
    n_results: int = 10
    name: str = "Search Results"
    view_type: str = "table"
    output_path: Optional[str] = Field(
        default=None,
        description="Path to write the .base file. If omitted, the config is returned without writing."
    )


# --- Core Logic ---
def verify_api_key(x_api_key: str = Header(...)):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def process_file(filepath: str):
    """
    Deletes, chunks, embeds, and ingests a single file into ChromaDB.
    Stores frontmatter properties as metadata for Bases-aware search.
    """
    if not model or not collection:
        logger.error(f"Skipping {filepath}, model/DB not initialized.")
        return

    logger.info(f"Processing file: {filepath}")
    try:
        # 1. Delete all existing vectors for this file
        collection.delete(where={"filepath": filepath})
        logger.info(f"Cleared old vectors for: {filepath}")

        # 2. Load file and parse frontmatter
        with open(filepath, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)

        content = post.content
        metadata = post.metadata
        split_token = metadata.get('split_token')

        chunks = []

        # 3. Chunking Strategy
        if split_token:
            logger.info(f"Using split_token '{split_token}' for {filepath}")
            split_token = split_token.encode('utf-8').decode('unicode_escape')
            raw_chunks = [c.strip() for c in content.split(split_token) if c.strip()]
            chunks = raw_chunks
        else:
            default_split_token = os.getenv("DEFAULT_SPLIT_TOKEN", "\n\n").encode('utf-8').decode('unicode_escape')
            logger.info(f"Using default split_token '{repr(default_split_token)}' for {filepath}")
            raw_chunks = [c.strip() for c in content.split(default_split_token) if c.strip()]
            chunks = raw_chunks
            if not chunks:
                logger.info(f"No valid chunks found in {filepath} after default split, skipping ingestion.")
                return

        if not chunks:
            logger.info(f"No content found in {filepath}, skipping ingestion.")
            return

        # 4. Build metadata — include frontmatter properties for Bases-aware filtering
        # ChromaDB metadata values must be str, int, float, or bool.
        base_meta = {"filepath": filepath}
        for key, value in metadata.items():
            if key == "split_token":
                continue
            meta_key = f"note.{key}"
            if isinstance(value, (str, int, float, bool)):
                base_meta[meta_key] = value
            elif isinstance(value, list):
                base_meta[meta_key] = json.dumps(value)
            else:
                base_meta[meta_key] = str(value)

        doc_ids = [f"{filepath}_{i}" for i in range(len(chunks))]
        metadatas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]

        # Generate embeddings with filepath prefix
        filename = os.path.basename(filepath)
        prefixed_chunks = [f"{filename}:part\n---\n{chunk}" for chunk in chunks]
        embeddings = model.encode(prefixed_chunks, show_progress_bar=False).tolist()

        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=doc_ids
        )
        logger.info(f"Successfully added {len(chunks)} chunks for {filepath}")

    except Exception as e:
        logger.error(f"Failed to process file {filepath}: {e}", exc_info=True)


def delete_vectors(filepath: str):
    """Deletes all vectors associated with a given filepath."""
    if not collection:
        logger.error(f"Skipping delete for {filepath}, DB not initialized.")
        return
    try:
        collection.delete(where={"filepath": filepath})
        logger.info(f"Deleted vectors for file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to delete vectors for {filepath}: {e}")


# --- Bases Cache Management ---

def refresh_bases_cache():
    """Reload all .base files into the in-memory cache."""
    global bases_cache
    bases_cache.clear()
    for bp in discover_base_files(VAULT_PATH):
        config = parse_base_file(bp)
        bases_cache[bp] = config
    logger.info(f"Bases cache refreshed: {len(bases_cache)} base(s) loaded.")


def update_base_in_cache(filepath: str):
    """Parse a single .base file and update the cache."""
    if os.path.exists(filepath):
        bases_cache[filepath] = parse_base_file(filepath)
        logger.info(f"Bases cache updated for: {filepath}")
    else:
        bases_cache.pop(filepath, None)
        logger.info(f"Bases cache removed: {filepath}")


# --- File Watchdog ---

class VaultEventHandler(FileSystemEventHandler):
    """Handles file system events for the vault — both .md and .base files."""

    def dispatch(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if path.endswith('.md') or path.endswith('.base'):
            super().dispatch(event)

    def on_created(self, event):
        path = event.src_path
        logger.info(f"File created: {path}")
        if path.endswith('.md'):
            process_file(path)
        elif path.endswith('.base'):
            update_base_in_cache(path)

    def on_modified(self, event):
        path = event.src_path
        logger.info(f"File modified: {path}")
        if path.endswith('.md'):
            process_file(path)
        elif path.endswith('.base'):
            update_base_in_cache(path)

    def on_deleted(self, event):
        path = event.src_path
        logger.info(f"File deleted: {path}")
        if path.endswith('.md'):
            delete_vectors(path)
        elif path.endswith('.base'):
            bases_cache.pop(path, None)
            logger.info(f"Bases cache removed: {path}")


# --- Initial Scan & Watcher Thread ---

def get_vault_files():
    """Recursively get all .md files in the vault."""
    return set(glob.glob(os.path.join(VAULT_PATH, '**', '*.md'), recursive=True))


def get_db_files():
    """Get all unique filepaths currently in the DB."""
    if not collection:
        return set()
    try:
        all_metadata = collection.get(include=["metadatas"])['metadatas']
        if not all_metadata:
            return set()
        return set(meta['filepath'] for meta in all_metadata if 'filepath' in meta)
    except Exception as e:
        logger.error(f"Error getting DB files: {e}")
        return set()


def perform_initial_scan():
    """Scans vault on startup and syncs with the DB."""
    logger.info("--- Starting Initial Scan ---")
    if not collection:
        logger.error("Cannot perform initial scan, DB not initialized.")
        return

    vault_files = get_vault_files()
    db_files = get_db_files()

    files_to_add = vault_files - db_files
    files_to_delete = db_files - vault_files

    logger.info(f"Found {len(vault_files)} files in vault.")
    logger.info(f"Found {len(db_files)} files in DB.")

    for f in files_to_add:
        logger.info(f"[SCAN] New file found: {f}")
        process_file(f)

    for f in files_to_delete:
        logger.info(f"[SCAN] Stale file found: {f}")
        delete_vectors(f)

    logger.info(
        f"--- Initial Scan Complete. Processed {len(files_to_add)} new, deleted {len(files_to_delete)} stale. ---")

    # Also load .base files into cache
    refresh_bases_cache()


def start_watcher():
    """Starts the file system watcher in a background thread."""
    global model, client, collection, observer

    try:
        logger.info("Watcher thread: Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Watcher thread: Model loaded.")

        logger.info("Watcher thread: Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info("Watcher thread: DB connected.")

        # Now that DB is ready, run the initial scan
        perform_initial_scan()

        # Start the watchdog
        event_handler = VaultEventHandler()
        observer = Observer(timeout=30)
        observer.schedule(event_handler, VAULT_PATH, recursive=True)
        observer.start()
        logger.info(f"File watcher started on {VAULT_PATH}")

        while True:
            time.sleep(5)

    except Exception as e:
        logger.error(f"FATAL error in watcher thread: {e}", exc_info=True)


# --- FastAPI Startup ---

@app.on_event("startup")
def on_startup():
    """Starts the background file watcher thread on server startup."""
    global telegram_bot_process

    logger.info("Starting background file watcher thread...")
    watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    watcher_thread.start()

    if ENABLE_TELEGRAM_BOT:
        logger.info("Starting Telegram bot in separate process...")
        bot_process = multiprocessing.Process(target=start_telegram_bot, daemon=True)
        bot_process.start()
        telegram_bot_process = bot_process
        logger.info(f"Telegram bot process started with PID: {bot_process.pid}")


# ==========================================================================
# API Endpoints
# ==========================================================================

# --- Health & Utility ---

@app.get("/health")
def health_check():
    """Health check endpoint — includes Bases stats."""
    if not model or not collection:
        return {"status": "initializing"}
    return {
        "status": "ok",
        "collection_count": collection.count(),
        "bases_count": len(bases_cache),
    }


# --- Vector Search ---

@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
def search(query: SearchQuery):
    """Performs a vector search, optionally scoped by a .base file's filters."""
    if not model or not collection:
        raise HTTPException(status_code=503, detail="Server is not initialized. Try again in a moment.")

    try:
        query_embedding = model.encode([query.query], show_progress_bar=False).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=query.n_results,
            include=["metadatas", "documents"]
        )

        # Optionally apply Bases filter to restrict results
        base_config = None
        if query.base_filter:
            abs_path = os.path.join(VAULT_PATH, query.base_filter) if not os.path.isabs(query.base_filter) else query.base_filter
            base_config = bases_cache.get(abs_path) or parse_base_file(abs_path)

        formatted_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                doc = results['documents'][0][i]
                filepath = meta.get('filepath', 'Unknown')

                # When a base filter is active, post-filter results
                if base_config and base_config.filters:
                    props = get_all_properties(filepath)
                    if not evaluate_filter(base_config.filters, props):
                        continue
                    result_props = props
                else:
                    result_props = None

                formatted_results.append(
                    SearchResult(
                        filepath=filepath,
                        chunk_index=meta.get('chunk_index', -1),
                        document=doc,
                        properties=result_props,
                    )
                )

        return SearchResponse(results=formatted_results)

    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")


@app.post("/reindex", dependencies=[Depends(verify_api_key)])
def reindex():
    """Manually triggers a full re-index of the vault."""
    global collection, observer
    if not client:
        raise HTTPException(status_code=503, detail="Server is not initialized.")

    logger.info("--- Reindex requested ---")
    try:
        if observer and observer.is_alive():
            logger.info("Pausing file watcher...")
            observer.stop()
            observer.join(timeout=5)
            logger.info("File watcher paused.")

        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Deleted collection: {COLLECTION_NAME}")

        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info(f"Recreated collection: {COLLECTION_NAME}")

        if observer:
            logger.info("Resuming file watcher...")
            event_handler = VaultEventHandler()
            observer = Observer(timeout=30)
            observer.schedule(event_handler, VAULT_PATH, recursive=True)
            observer.start()
            logger.info("File watcher resumed.")

        reindex_thread = threading.Thread(target=perform_initial_scan, daemon=True)
        reindex_thread.start()

        return {"status": "Reindex started in background."}
    except Exception as e:
        logger.error(f"Error during reindex: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during reindex: {e}")


# ==========================================================================
# Bases API Endpoints
# ==========================================================================

@app.get("/bases", dependencies=[Depends(verify_api_key)])
def list_bases():
    """List all .base files in the vault with summary info."""
    return {"bases": list_bases_summary(VAULT_PATH)}


@app.get("/bases/properties", dependencies=[Depends(verify_api_key)])
def list_vault_properties():
    """Scan the vault and list all unique property keys with their observed value types.

    Useful for building Bases filters and understanding your vault's schema.
    """
    raw = get_all_vault_properties(VAULT_PATH)
    return {
        "properties": {k: sorted(v) for k, v in sorted(raw.items())}
    }


@app.get("/bases/{base_path:path}", dependencies=[Depends(verify_api_key)])
def get_base(base_path: str):
    """Return the parsed configuration of a specific .base file."""
    abs_path = os.path.join(VAULT_PATH, base_path) if not os.path.isabs(base_path) else base_path
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail=f"Base file not found: {base_path}")
    config = parse_base_file(abs_path)
    return {"filepath": abs_path, "config": config.to_dict()}


@app.post("/bases", dependencies=[Depends(verify_api_key)])
def create_base(req: BaseCreateRequest):
    """Create a new .base file in the vault."""
    abs_path = os.path.join(VAULT_PATH, req.filepath) if not os.path.isabs(req.filepath) else req.filepath
    if not abs_path.endswith(".base"):
        abs_path += ".base"

    if os.path.exists(abs_path):
        raise HTTPException(status_code=409, detail=f"Base file already exists: {req.filepath}")

    config = BasesConfig(
        filepath=abs_path,
        filters=req.filters,
        properties=req.properties or {},
        formulas=req.formulas or {},
        summaries=req.summaries or {},
        views=req.views or [],
    )
    write_base_file(abs_path, config)
    update_base_in_cache(abs_path)
    return {"status": "created", "filepath": abs_path, "config": config.to_dict()}


@app.put("/bases/{base_path:path}", dependencies=[Depends(verify_api_key)])
def update_base(base_path: str, req: BaseUpdateRequest):
    """Update an existing .base file (merges with existing config)."""
    abs_path = os.path.join(VAULT_PATH, base_path) if not os.path.isabs(base_path) else base_path
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail=f"Base file not found: {base_path}")

    existing = parse_base_file(abs_path)
    if req.filters is not None:
        existing.filters = req.filters
    if req.properties is not None:
        existing.properties.update(req.properties)
    if req.formulas is not None:
        existing.formulas.update(req.formulas)
    if req.summaries is not None:
        existing.summaries.update(req.summaries)
    if req.views is not None:
        existing.views = req.views

    write_base_file(abs_path, existing)
    update_base_in_cache(abs_path)
    return {"status": "updated", "filepath": abs_path, "config": existing.to_dict()}


@app.delete("/bases/{base_path:path}", dependencies=[Depends(verify_api_key)])
def delete_base(base_path: str):
    """Delete a .base file."""
    abs_path = os.path.join(VAULT_PATH, base_path) if not os.path.isabs(base_path) else base_path
    if delete_base_file(abs_path):
        bases_cache.pop(abs_path, None)
        return {"status": "deleted", "filepath": abs_path}
    raise HTTPException(status_code=404, detail=f"Base file not found: {base_path}")


@app.post("/bases/query", dependencies=[Depends(verify_api_key)])
def query_base_endpoint(req: BaseQueryRequest):
    """Execute a Base's filters and return all matching vault notes with their properties.

    This mimics what Obsidian Bases does internally — it evaluates the filter
    expression from the .base file against every markdown file in the vault and
    returns the ones that match, along with their structured properties.
    """
    abs_path = os.path.join(VAULT_PATH, req.base_filepath) if not os.path.isabs(req.base_filepath) else req.base_filepath
    config = bases_cache.get(abs_path)
    if not config:
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404, detail=f"Base file not found: {req.base_filepath}")
        config = parse_base_file(abs_path)

    results = query_base(config, VAULT_PATH)
    return {
        "base_filepath": abs_path,
        "filter": config.filters,
        "match_count": len(results),
        "results": results,
    }


@app.post("/bases/from-search", dependencies=[Depends(verify_api_key)])
def create_base_from_search(req: BaseFromSearchRequest):
    """Run a vector search and generate a .base file that targets the matching notes.

    This bridges vector search with Obsidian Bases — search semantically, then
    save the results as a Base that Obsidian can render as a table/list/cards view.
    """
    if not model or not collection:
        raise HTTPException(status_code=503, detail="Server is not initialized.")

    # 1. Run the vector search
    query_embedding = model.encode([req.query], show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=req.n_results,
        include=["metadatas"]
    )

    if not results['ids'] or not results['ids'][0]:
        raise HTTPException(status_code=404, detail="No search results found.")

    # Deduplicate filepaths (multiple chunks may come from the same file)
    seen = set()
    filepaths = []
    for meta in results['metadatas'][0]:
        fp = meta.get('filepath', '')
        if fp and fp not in seen:
            seen.add(fp)
            filepaths.append(fp)

    # 2. Generate the Bases config
    config = create_base_from_search_results(
        name=req.name,
        filepaths=filepaths,
        vault_path=VAULT_PATH,
        view_type=req.view_type,
    )

    # 3. Optionally write to disk
    if req.output_path:
        abs_path = os.path.join(VAULT_PATH, req.output_path) if not os.path.isabs(req.output_path) else req.output_path
        if not abs_path.endswith(".base"):
            abs_path += ".base"
        config.filepath = abs_path
        write_base_file(abs_path, config)
        update_base_in_cache(abs_path)

    return {
        "status": "generated",
        "matched_files": filepaths,
        "config": config.to_dict(),
        "written_to": config.filepath or None,
    }
