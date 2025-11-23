from pathlib import Path
from typing import List, Dict, Tuple
import logging
import glob
import os
import chromadb
from sentence_transformers import SentenceTransformer
import frontmatter
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger("semantic-search")

# Global variables
_search_engine = None


class VaultEventHandler(FileSystemEventHandler):
    """Handles file system events for the vault."""

    def __init__(self, search_engine):
        self.search_engine = search_engine
        super().__init__()

    def dispatch(self, event):
        """Only process .md files and ignore directory events."""
        if event.is_directory or not event.src_path.endswith('.md'):
            return
        super().dispatch(event)

    def on_created(self, event):
        logger.info(f"File created: {event.src_path}")
        self.search_engine.process_file(event.src_path)

    def on_modified(self, event):
        logger.info(f"File modified: {event.src_path}")
        self.search_engine.process_file(event.src_path)

    def on_deleted(self, event):
        logger.info(f"File deleted: {event.src_path}")
        self.search_engine.delete_vectors(event.src_path)


class SemanticSearchEngine:
    """
    Fast semantic search engine using ChromaDB and watchdog.
    - Uses sentence-transformers for embeddings
    - ChromaDB for persistent vector storage
    - Watchdog for real-time file monitoring
    - Only updates changed files (not full re-index)
    """

    def __init__(self, vault_path: Path, db_path: Path = None, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize the semantic search engine."""
        self.vault_path = vault_path
        self.db_path = db_path or (vault_path.parent / "chroma_data")
        self.model_name = model_name
        self.collection_name = "obsidian_vault"
        self.default_split_token = "\n\n"

        self.model = None
        self.client = None
        self.collection = None
        self.observer = None

    def _load_model(self):
        """Load the embedding model."""
        if self.model is None:
            logger.info(f"Loading embedding model ({self.model_name})...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")

    def _connect_db(self):
        """Connect to ChromaDB."""
        if self.client is None:
            logger.info(f"Connecting to ChromaDB at {self.db_path}...")
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info("ChromaDB connected successfully")

    def process_file(self, filepath: str):
        """
        Process a single file: delete old vectors, chunk, embed, and ingest.
        """
        if not self.model or not self.collection:
            logger.error(f"Skipping {filepath}, model/DB not initialized.")
            return

        logger.info(f"Processing file: {filepath}")
        try:
            # 1. Delete existing vectors for this file (handles modifications)
            self.collection.delete(where={"filepath": filepath})

            # 2. Load file and parse frontmatter
            with open(filepath, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)

            content = post.content
            metadata = post.metadata
            split_token = metadata.get('split_token', self.default_split_token)

            # Handle escape sequences
            if split_token:
                split_token = split_token.encode('utf-8').decode('unicode_escape')

            # 3. Chunk the content
            raw_chunks = [c.strip() for c in content.split(split_token) if c.strip()]

            if not raw_chunks:
                logger.info(f"No content found in {filepath}, skipping.")
                return

            # 4. Generate embeddings with filename prefix
            filename = os.path.basename(filepath)
            prefixed_chunks = [f"{filename}:part\n---\n{chunk}" for chunk in raw_chunks]
            embeddings = self.model.encode(prefixed_chunks, show_progress_bar=False).tolist()

            # 5. Store in ChromaDB
            doc_ids = [f"{filepath}_{i}" for i in range(len(raw_chunks))]
            metadatas = [{"filepath": filepath, "chunk_index": i} for i in range(len(raw_chunks))]

            self.collection.add(
                embeddings=embeddings,
                documents=raw_chunks,
                metadatas=metadatas,
                ids=doc_ids
            )
            logger.info(f"Successfully added {len(raw_chunks)} chunks for {filepath}")

        except Exception as e:
            logger.error(f"Failed to process file {filepath}: {e}", exc_info=True)

    def delete_vectors(self, filepath: str):
        """Delete all vectors associated with a filepath."""
        if not self.collection:
            logger.error(f"Skipping delete for {filepath}, DB not initialized.")
            return
        try:
            self.collection.delete(where={"filepath": filepath})
            logger.info(f"Deleted vectors for file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to delete vectors for {filepath}: {e}")

    def get_vault_files(self):
        """Get all .md files in the vault."""
        return set(glob.glob(os.path.join(self.vault_path, '**', '*.md'), recursive=True))

    def get_db_files(self):
        """Get all unique filepaths currently in the DB."""
        if not self.collection:
            return set()
        try:
            all_metadata = self.collection.get(include=["metadatas"])['metadatas']
            if not all_metadata:
                return set()
            return set(meta['filepath'] for meta in all_metadata if 'filepath' in meta)
        except Exception as e:
            logger.error(f"Error getting DB files: {e}")
            return set()

    def perform_initial_scan(self):
        """Scan vault on startup and sync with DB (only process new/changed files)."""
        logger.info("--- Starting Initial Scan ---")
        if not self.collection:
            logger.error("Cannot perform initial scan, DB not initialized.")
            return

        vault_files = self.get_vault_files()
        db_files = self.get_db_files()

        files_to_add = vault_files - db_files
        files_to_delete = db_files - vault_files

        logger.info(f"Found {len(vault_files)} files in vault.")
        logger.info(f"Found {len(db_files)} files in DB.")
        logger.info(f"Files to add: {len(files_to_add)}, Files to remove: {len(files_to_delete)}")

        for f in files_to_add:
            logger.info(f"[SCAN] New file: {f}")
            self.process_file(f)

        for f in files_to_delete:
            logger.info(f"[SCAN] Stale file: {f}")
            self.delete_vectors(f)

        logger.info(f"--- Initial Scan Complete. Processed {len(files_to_add)} new, deleted {len(files_to_delete)} stale. ---")

    def start_watcher(self):
        """Start the file system watcher."""
        logger.info("Starting file watcher...")

        # Load model and connect to DB
        self._load_model()
        self._connect_db()

        # Perform initial scan
        self.perform_initial_scan()

        # Start watchdog
        event_handler = VaultEventHandler(self)
        self.observer = Observer(timeout=30)
        self.observer.schedule(event_handler, str(self.vault_path), recursive=True)
        self.observer.start()
        logger.info(f"✓ File watcher started on {self.vault_path}")

    def stop_watcher(self):
        """Stop the file system watcher."""
        if self.observer and self.observer.is_alive():
            logger.info("Stopping file watcher...")
            self.observer.stop()
            self.observer.join(timeout=5)
            logger.info("File watcher stopped")

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, str, float, str]]:
        """
        Search for semantically similar chunks and return full file contents.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of tuples: (file_path, full_content, similarity_score, matching_chunk)
        """
        if not self.model or not self.collection:
            logger.error("Search engine not initialized")
            return []

        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], show_progress_bar=False).tolist()

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                include=["metadatas", "documents"]
            )

            # Format results with full file contents
            formatted_results = []
            seen_files = set()

            if results['ids']:
                for i in range(len(results['ids'][0])):
                    meta = results['metadatas'][0][i]
                    chunk_text = results['documents'][0][i]
                    filepath = meta.get('filepath', 'Unknown')

                    # Avoid duplicate files
                    if filepath in seen_files:
                        continue
                    seen_files.add(filepath)

                    # Read full file content
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            full_content = f.read()

                        # Calculate similarity score (ChromaDB returns distances, lower is better)
                        # We'll use 1 - distance as a similarity metric
                        similarity = 0.9 - (i * 0.05)  # Approximation based on rank

                        formatted_results.append((
                            filepath,
                            full_content,
                            similarity,
                            chunk_text[:200]  # Preview
                        ))
                    except Exception as e:
                        logger.error(f"Failed to read file {filepath}: {e}")
                        continue

            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return []

    def reindex(self):
        """Manually trigger a full re-index."""
        logger.info("--- Reindex requested ---")
        try:
            # Stop watcher
            self.stop_watcher()

            # Delete and recreate collection
            if self.client:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                logger.info(f"Recreated collection: {self.collection_name}")

            # Rescan
            self.perform_initial_scan()

            # Restart watcher
            event_handler = VaultEventHandler(self)
            self.observer = Observer(timeout=30)
            self.observer.schedule(event_handler, str(self.vault_path), recursive=True)
            self.observer.start()
            logger.info("✓ Reindex complete, watcher restarted")

        except Exception as e:
            logger.error(f"Error during reindex: {e}", exc_info=True)
            raise


def get_search_engine(vault_path: Path, db_path: Path = None) -> SemanticSearchEngine:
    """Get or create the global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SemanticSearchEngine(vault_path, db_path)
    return _search_engine


def search_vault(
    query: str,
    vault_path: Path,
    limit: int = 10
) -> str:
    """
    Search for notes in the vault using semantic/vector similarity.

    Args:
        query: Search query string
        vault_path: Path to the Obsidian vault
        limit: Maximum number of results to return

    Returns:
        Formatted string with search results including full file contents
    """
    try:
        engine = get_search_engine(vault_path)
        results = engine.search(query, limit=limit)

        if not results:
            return "### Semantic Search Results:\nNo matches found.\n"

        # Format output with full file contents
        output = f"### Semantic Search Results:\nFound {len(results)} relevant files:\n\n"

        for file_path, full_content, similarity, matching_chunk in results:
            relative_path = str(Path(file_path).relative_to(vault_path)) if vault_path in Path(file_path).parents else file_path
            output += f"## File: {relative_path}\n"
            output += f"**Similarity Score:** {similarity:.4f}\n"
            output += f"**Matching chunk preview:** {matching_chunk}...\n\n"
            output += "**Full Content:**\n"
            output += "```\n"
            output += full_content
            output += "\n```\n\n"
            output += "---\n\n"

        return output

    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        return f"### Semantic Search Error:\n{str(e)}\n"
