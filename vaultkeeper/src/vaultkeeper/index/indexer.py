"""Indexing pipeline for the vault.

Handles full vault indexing, incremental updates, and LanceDB storage
for chunks, embeddings, facts, and file metadata.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from vaultkeeper.config import Config
from vaultkeeper.index.chunker import Chunk, chunk_note, _hash_content
from vaultkeeper.llm.client import EmbeddingClient, LLMClient
from vaultkeeper.vault.reader import VaultReader

logger = logging.getLogger(__name__)

# --- LanceDB Schema Definitions ---

CHUNKS_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("source_path", pa.string()),
    pa.field("note_title", pa.string()),
    pa.field("section_header", pa.string()),
    pa.field("chunk_type", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("tags", pa.string()),           # JSON-encoded list
    pa.field("links", pa.string()),          # JSON-encoded list
    pa.field("content", pa.string()),
    pa.field("content_hash", pa.string()),
    pa.field("vector", pa.list_(pa.float32())),
])

FACTS_SCHEMA = pa.schema([
    pa.field("fact_id", pa.string()),
    pa.field("source_path", pa.string()),
    pa.field("note_title", pa.string()),
    pa.field("entity", pa.string()),
    pa.field("key", pa.string()),
    pa.field("value", pa.string()),
    pa.field("context", pa.string()),
])

METADATA_SCHEMA = pa.schema([
    pa.field("path", pa.string()),
    pa.field("mtime", pa.float64()),
    pa.field("size", pa.int64()),
    pa.field("content_hash", pa.string()),
])


@dataclass
class IndexStats:
    total_notes: int = 0
    indexed: int = 0
    skipped: int = 0
    chunks_created: int = 0
    facts_extracted: int = 0
    errors: int = 0
    duration_seconds: float = 0.0


class VaultIndexer:
    """Manages the vault search index in LanceDB."""

    def __init__(
        self,
        config: Config,
        vault_reader: VaultReader,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient | None = None,
    ):
        self.config = config
        self.vault_reader = vault_reader
        self.embedding_client = embedding_client
        self.llm_client = llm_client

        # Open/create LanceDB database
        self.db = lancedb.connect(config.index.db_path)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        existing = set(self.db.table_names())

        if "chunks" not in existing:
            self.db.create_table("chunks", schema=CHUNKS_SCHEMA)

        if "facts" not in existing:
            self.db.create_table("facts", schema=FACTS_SCHEMA)

        if "metadata" not in existing:
            self.db.create_table("metadata", schema=METADATA_SCHEMA)

    async def full_index(self) -> IndexStats:
        """Index the entire vault from scratch."""
        start = time.time()
        stats = IndexStats()

        all_notes = self.vault_reader.list_all_notes()
        stats.total_notes = len(all_notes)
        logger.info(f"Starting full index of {stats.total_notes} notes")

        for note_path in all_notes:
            try:
                await self._index_note(note_path, stats)
            except Exception as e:
                logger.error(f"Error indexing {note_path}: {e}")
                stats.errors += 1

        stats.duration_seconds = time.time() - start
        logger.info(
            f"Full index complete: {stats.indexed} indexed, {stats.skipped} skipped, "
            f"{stats.chunks_created} chunks, {stats.facts_extracted} facts, "
            f"{stats.errors} errors in {stats.duration_seconds:.1f}s"
        )
        return stats

    async def incremental_index(self, changed_paths: list[str]) -> IndexStats:
        """Re-index specific files that have changed."""
        start = time.time()
        stats = IndexStats()
        stats.total_notes = len(changed_paths)

        for note_path in changed_paths:
            try:
                if not self.vault_reader.exists(note_path):
                    # File was deleted
                    await self._remove_note(note_path)
                    stats.indexed += 1
                else:
                    await self._index_note(note_path, stats, force=True)
            except Exception as e:
                logger.error(f"Error re-indexing {note_path}: {e}")
                stats.errors += 1

        stats.duration_seconds = time.time() - start
        return stats

    async def _index_note(self, note_path: str, stats: IndexStats, force: bool = False) -> None:
        """Index a single note, with change detection."""
        # Check if note has changed
        if not force:
            file_stats = self.vault_reader.file_stats(note_path)
            stored_meta = self._get_metadata(note_path)

            if stored_meta is not None:
                if (
                    stored_meta["mtime"] == file_stats["mtime"]
                    and stored_meta["size"] == file_stats["size"]
                ):
                    stats.skipped += 1
                    return

        # Read and parse the note
        note = self.vault_reader.read_note(note_path)
        content_hash = _hash_content(note.content)

        # Check content hash (catches false positives from mtime/size check)
        if not force:
            stored_meta = self._get_metadata(note_path)
            if stored_meta is not None and stored_meta["content_hash"] == content_hash:
                # Update mtime/size but skip re-embedding
                self._update_metadata(note_path, note.content)
                stats.skipped += 1
                return

        # Remove old chunks and facts for this note
        await self._remove_note(note_path)

        # Chunk the note
        chunks = chunk_note(note, max_tokens=self.config.index.chunk_max_tokens)

        if chunks:
            # Generate embeddings for all chunks in batch
            texts = [c.content for c in chunks]
            embeddings = await self.embedding_client.embed(texts)

            # Store chunks with embeddings
            import json
            chunk_records = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_records.append({
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "note_title": chunk.note_title,
                    "section_header": chunk.section_header,
                    "chunk_type": chunk.chunk_type,
                    "chunk_index": chunk.chunk_index,
                    "tags": json.dumps(chunk.tags),
                    "links": json.dumps(chunk.links),
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "vector": embedding,
                })

            chunks_table = self.db.open_table("chunks")
            chunks_table.add(chunk_records)
            stats.chunks_created += len(chunk_records)

        # Extract facts (if enabled and LLM client available)
        if self.config.index.fact_extraction and self.llm_client:
            try:
                facts = await self._extract_facts(note_path, note.title, note.content)
                if facts:
                    facts_table = self.db.open_table("facts")
                    facts_table.add(facts)
                    stats.facts_extracted += len(facts)
            except Exception as e:
                logger.warning(f"Fact extraction failed for {note_path}: {e}")

        # Update metadata
        self._update_metadata(note_path, note.content)
        stats.indexed += 1

    async def _extract_facts(
        self, note_path: str, note_title: str, content: str
    ) -> list[dict[str, str]]:
        """Extract key-value facts from a note using the extractor LLM."""
        prompt = f"""Extract all key-value facts, codes, numbers, dates, addresses, 
relationships, and important details from this note. Focus on specific, concrete 
information that someone might search for later.

Note title: {note_title}
Note path: {note_path}

Content:
{content}

Return a JSON array of objects, each with these fields:
- "entity": the person, place, or thing this fact is about
- "key": what the fact is (e.g., "garage code", "birthday", "phone number")
- "value": the actual value
- "context": brief context sentence

Return ONLY the JSON array, nothing else. If no facts found, return [].
"""
        messages = [{"role": "user", "content": prompt}]

        try:
            result = await self.llm_client.complete_json("extractor", messages)
            if not isinstance(result, list):
                return []

            import hashlib
            facts = []
            for i, fact in enumerate(result):
                if not isinstance(fact, dict):
                    continue
                fact_id = hashlib.sha256(
                    f"{note_path}:{i}:{fact.get('key', '')}".encode()
                ).hexdigest()[:16]
                facts.append({
                    "fact_id": fact_id,
                    "source_path": note_path,
                    "note_title": note_title,
                    "entity": str(fact.get("entity", "")),
                    "key": str(fact.get("key", "")),
                    "value": str(fact.get("value", "")),
                    "context": str(fact.get("context", "")),
                })
            return facts
        except Exception as e:
            logger.warning(f"Fact extraction LLM error for {note_path}: {e}")
            return []

    async def _remove_note(self, note_path: str) -> None:
        """Remove all chunks, facts, and metadata for a note."""
        try:
            chunks_table = self.db.open_table("chunks")
            chunks_table.delete(f'source_path = "{note_path}"')
        except Exception:
            pass

        try:
            facts_table = self.db.open_table("facts")
            facts_table.delete(f'source_path = "{note_path}"')
        except Exception:
            pass

        try:
            meta_table = self.db.open_table("metadata")
            meta_table.delete(f'path = "{note_path}"')
        except Exception:
            pass

    def _get_metadata(self, note_path: str) -> dict | None:
        """Get stored metadata for a note."""
        try:
            meta_table = self.db.open_table("metadata")
            results = meta_table.search().where(f'path = "{note_path}"').limit(1).to_list()
            if results:
                return results[0]
        except Exception:
            pass
        return None

    def _update_metadata(self, note_path: str, content: str) -> None:
        """Update or insert metadata for a note."""
        file_stats = self.vault_reader.file_stats(note_path)
        content_hash = _hash_content(content)

        # Remove old entry
        try:
            meta_table = self.db.open_table("metadata")
            meta_table.delete(f'path = "{note_path}"')
        except Exception:
            pass

        meta_table = self.db.open_table("metadata")
        meta_table.add([{
            "path": note_path,
            "mtime": file_stats["mtime"],
            "size": file_stats["size"],
            "content_hash": content_hash,
        }])

    # --- Search Methods (used by retrieval layer) ---

    def semantic_search(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        """Search chunks by vector similarity."""
        try:
            chunks_table = self.db.open_table("chunks")
            results = (
                chunks_table.search(query_embedding)
                .limit(limit)
                .to_list()
            )
            return results
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def search_facts(
        self,
        entity: str | None = None,
        key: str | None = None,
        value: str | None = None,
    ) -> list[dict]:
        """Search the facts table by entity, key, or value."""
        try:
            facts_table = self.db.open_table("facts")

            conditions = []
            if entity:
                conditions.append(f'entity LIKE "%{entity}%"')
            if key:
                conditions.append(f'key LIKE "%{key}%"')
            if value:
                conditions.append(f'value LIKE "%{value}%"')

            if conditions:
                where_clause = " AND ".join(conditions)
                results = facts_table.search().where(where_clause).limit(50).to_list()
            else:
                results = facts_table.search().limit(50).to_list()

            return results
        except Exception as e:
            logger.error(f"Facts search error: {e}")
            return []
