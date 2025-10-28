import sys
import threading
import multiprocessing
import time
import glob
import logging
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import frontmatter
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
import os

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
    description="An API to search an Obsidian vault using vector embeddings."
)

# --- Global Variables (to be populated on startup) ---
model = None
client = None
collection = None
observer = None


# --- Pydantic Models ---
class SearchQuery(BaseModel):
    query: str
    n_results: int = 5


class SearchResult(BaseModel):
    filepath: str
    chunk_index: int
    document: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


# --- Core Logic ---
def verify_api_key(x_api_key: str = Header(...)):
    if not API_KEY:
        # No key set on server, disable authentication
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def process_file(filepath: str):
    """
    Deletes, chunks, embeds, and ingests a single file into ChromaDB
    based on its frontmatter.
    """
    if not model or not collection:
        logger.error(f"Skipping {filepath}, model/DB not initialized.")
        return

    logger.info(f"Processing file: {filepath}")
    try:
        # 1. Delete all existing vectors for this file
        # This ensures atomicity and handles modifications perfectly.
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

        # 4. Ingestion
        doc_ids = [f"{filepath}_{i}" for i in range(len(chunks))]
        metadatas = [{"filepath": filepath, "chunk_index": i} for i in range(len(chunks))]

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


# --- File Watchdog ---

class VaultEventHandler(FileSystemEventHandler):
    """Handles file system events for the vault."""

    def dispatch(self, event):
        """Only process .md files and ignore directory events."""
        if event.is_directory or not event.src_path.endswith('.md'):
            return
        super().dispatch(event)

    def on_created(self, event):
        logger.info(f"File created: {event.src_path}")
        process_file(event.src_path)

    def on_modified(self, event):
        logger.info(f"File modified: {event.src_path}")
        # The process_file function handles deletion first, so this is correct
        process_file(event.src_path)

    def on_deleted(self, event):
        logger.info(f"File deleted: {event.src_path}")
        delete_vectors(event.src_path)


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
            time.sleep(5)  # Keep the thread alive

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

    # Start Telegram bot in a separate process if enabled
    if ENABLE_TELEGRAM_BOT:
        logger.info("Starting Telegram bot in separate process...")
        bot_process = multiprocessing.Process(target=start_telegram_bot, daemon=True)
        bot_process.start()
        telegram_bot_process = bot_process
        logger.info(f"Telegram bot process started with PID: {bot_process.pid}")


# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Health check endpoint."""
    if not model or not collection:
        return {"status": "initializing"}
    return {"status": "ok", "collection_count": collection.count()}


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
def search(query: SearchQuery):
    """Performs a vector search on the collection."""
    if not model or not collection:
        raise HTTPException(status_code=503, detail="Server is not initialized. Try again in a moment.")

    try:
        query_embedding = model.encode([query.query], show_progress_bar=False).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=query.n_results,
            include=["metadatas", "documents"]
        )

        # Format the response
        formatted_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                doc = results['documents'][0][i]
                formatted_results.append(
                    SearchResult(
                        filepath=meta.get('filepath', 'Unknown'),
                        chunk_index=meta.get('chunk_index', -1),
                        document=doc
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
        # 1. Pause the file watcher to prevent race conditions
        if observer and observer.is_alive():
            logger.info("Pausing file watcher...")
            observer.stop()
            observer.join(timeout=5)
            logger.info("File watcher paused.")

        # 2. Delete old collection
        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Deleted collection: {COLLECTION_NAME}")

        # 3. Recreate collection
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info(f"Recreated collection: {COLLECTION_NAME}")

        # 4. Resume the file watcher
        if observer:
            logger.info("Resuming file watcher...")
            event_handler = VaultEventHandler()
            observer = Observer(timeout=30)
            observer.schedule(event_handler, VAULT_PATH, recursive=True)
            observer.start()
            logger.info("File watcher resumed.")

        # 5. Rescan
        # We run this in a thread so the request can return immediately
        reindex_thread = threading.Thread(target=perform_initial_scan, daemon=True)
        reindex_thread.start()

        return {"status": "Reindex started in background."}
    except Exception as e:
        logger.error(f"Error during reindex: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during reindex: {e}")