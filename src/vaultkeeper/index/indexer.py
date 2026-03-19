"""Indexing pipeline for the vault.

Handles full vault indexing, incremental updates, and ChromaDB storage
for chunks/embeddings, with JSON sidecars for facts and file metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb

from vaultkeeper.config import Config
from vaultkeeper.index.chunker import Chunk, chunk_note, _hash_content
from vaultkeeper.llm.client import EmbeddingClient, LLMClient
from vaultkeeper.vault.reader import VaultReader

logger = logging.getLogger(__name__)


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
    """Manages the vault search index in ChromaDB."""

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

        db_path = Path(config.index.db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._chunks = self._client.get_or_create_collection(
            "chunks",
            metadata={"hnsw:space": "cosine"},
        )

        # Facts and metadata need no vector search — plain JSON is sufficient.
        self._facts_path = db_path / "facts.json"
        self._meta_path = db_path / "metadata.json"
        self._facts: dict[str, dict] = self._load_json(self._facts_path)
        self._metadata: dict[str, dict] = self._load_json(self._meta_path)

    @staticmethod
    def _load_json(path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {}

    def _save_facts(self) -> None:
        self._facts_path.write_text(json.dumps(self._facts))

    def _save_metadata(self) -> None:
        self._meta_path.write_text(json.dumps(self._metadata))

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

        note = self.vault_reader.read_note(note_path)
        content_hash = _hash_content(note.content)

        if not force:
            stored_meta = self._get_metadata(note_path)
            if stored_meta is not None and stored_meta["content_hash"] == content_hash:
                self._update_metadata(note_path, note.content)
                stats.skipped += 1
                return

        if len(note.body.strip()) < 10:
            stats.skipped += 1
            return

        await self._remove_note(note_path)

        chunks = chunk_note(note, max_tokens=self.config.index.chunk_max_tokens)

        if chunks:
            texts = [c.content for c in chunks]
            embeddings = await self.embedding_client.embed(texts)

            ids, vecs, docs, metas = [], [], [], []
            for chunk, embedding in zip(chunks, embeddings):
                ids.append(chunk.chunk_id)
                vecs.append(embedding)
                docs.append(chunk.content)
                metas.append({
                    "source_path": chunk.source_path,
                    "note_title": chunk.note_title,
                    "section_header": chunk.section_header,
                    "chunk_type": chunk.chunk_type,
                    "chunk_index": chunk.chunk_index,
                    "tags": json.dumps(chunk.tags),
                    "links": json.dumps(chunk.links),
                    "content_hash": chunk.content_hash,
                })

            self._chunks.add(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)
            stats.chunks_created += len(ids)

        if self.config.index.fact_extraction and self.llm_client:
            try:
                facts = await self._extract_facts(note_path, note.title, note.content)
                if facts:
                    for fact in facts:
                        self._facts[fact["fact_id"]] = fact
                    self._save_facts()
                    stats.facts_extracted += len(facts)
            except Exception as e:
                logger.warning(f"Fact extraction failed for {note_path}: {e}")

        self._update_metadata(note_path, note.content)
        stats.indexed += 1

    async def _extract_facts(
        self, note_path: str, note_title: str, content: str
    ) -> list[dict[str, str]]:
        """Extract key-value facts from a note using the extractor LLM."""
        system_prompt = """You are a fact extraction engine for a personal knowledge base.

Your job: extract ONLY specific, concrete, searchable facts from the note body.

EXTRACT these kinds of facts:
- Personal details: phone numbers, addresses, codes, PINs, passwords, birthdays
- Relationships: "X is Y's brother", "X works at Y"
- Specific numbers: prices, measurements, quantities, account numbers
- Dates of events: "moved to Cincinnati in 2023", "wedding is June 15"
- Named references: "favorite restaurant is X", "dentist is Dr. Y"
- Opinions/ratings: "rated 4/5", "didn't like the service"
- Action items: "need to call X", "wants to visit Y"

DO NOT extract:
- Frontmatter metadata (category, status, domain, created_time) — this is already indexed separately
- General knowledge (definitions, Wikipedia-style facts about famous people/concepts)
- The note's file path or title as a fact
- Vague thematic observations or essay-level ideas
- Information that only restates what the note title already says

If the note is purely conceptual (an essay, a topic overview, philosophical musings) with no personal/specific facts, return an empty array [].

Respond with ONLY a JSON array. No markdown fences, no explanation.

EXAMPLES:

Note: "John Smith"
Body contains: "garage code is 12345" and "birthday March 15" and "works at Kroger on Vine St"
Correct output:
[
  {"entity": "John Smith", "key": "garage code", "value": "12345", "context": "garage code is 12345"},
  {"entity": "John Smith", "key": "birthday", "value": "March 15", "context": "John's birthday"},
  {"entity": "John Smith", "key": "employer", "value": "Kroger on Vine St", "context": "works at Kroger on Vine St"}
]

Note: "Stoic Philosophy"
Body contains: an essay about Marcus Aurelius and Epictetus with no personal facts
Correct output:
[]

Note: "House Stuff"
Body contains: "WiFi password is BlueCat99!" and "furnace filter size 20x25x1" and "landlord Mike 513-555-0100"
Correct output:
[
  {"entity": "house", "key": "WiFi password", "value": "BlueCat99!", "context": "home WiFi password"},
  {"entity": "house", "key": "furnace filter size", "value": "20x25x1", "context": "furnace filter dimensions"},
  {"entity": "Mike", "key": "phone number", "value": "513-555-0100", "context": "landlord Mike's phone number"}
]

Note about a car or vehicle:
[
  {"entity": "Civic", "key": "oil type", "value": "0W-20 synthetic", "context": "takes 0W-20 full synthetic"},
  {"entity": "Civic", "key": "tire pressure", "value": "35 PSI", "context": "recommended tire pressure 35 PSI"},
  {"entity": "Civic", "key": "next service", "value": "March 2026 at 87,000 miles", "context": "next oil change due March 2026 or 87k miles"}
]

Note about a trip or travel plan:
[
  {"entity": "Tokyo trip", "key": "flight", "value": "Delta DL47 departing March 12 at 1:35pm", "context": "outbound flight Delta DL47 on March 12"},
  {"entity": "Tokyo trip", "key": "hotel", "value": "Shinjuku Granbell, confirmation #GRN-88412", "context": "staying at Shinjuku Granbell hotel"},
  {"entity": "Tokyo trip", "key": "recommendation", "value": "Ichiran Ramen in Shibuya", "context": "Jared recommended Ichiran Ramen in Shibuya"}
]

Note about a meeting or conversation:
[
  {"entity": "Lauren", "key": "new job", "value": "started at Deloitte in January", "context": "Lauren mentioned she started at Deloitte in January"},
  {"entity": "Kas", "key": "moving to", "value": "Denver", "context": "Kas is thinking about moving to Denver this summer"},
  {"entity": "game night", "key": "next date", "value": "Saturday the 22nd at Jared's place", "context": "next game night is Saturday the 22nd at Jared's"}
]

Remember: ALWAYS return a JSON array. Never a bare object. Extract ALL matching facts, not just the first one."""

        user_prompt = f"""Note title: {note_title}
Note path: {note_path}

Content:
{content}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result = await self.llm_client.complete_json("extractor", messages)
            if not isinstance(result, list):
                return []

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
            self._chunks.delete(where={"source_path": note_path})
        except Exception:
            pass

        self._facts = {
            fid: f for fid, f in self._facts.items()
            if f.get("source_path") != note_path
        }
        self._save_facts()

        if note_path in self._metadata:
            del self._metadata[note_path]
            self._save_metadata()

    def _get_metadata(self, note_path: str) -> dict | None:
        return self._metadata.get(note_path)

    def _update_metadata(self, note_path: str, content: str) -> None:
        file_stats = self.vault_reader.file_stats(note_path)
        content_hash = _hash_content(content)
        self._metadata[note_path] = {
            "path": note_path,
            "mtime": file_stats["mtime"],
            "size": file_stats["size"],
            "content_hash": content_hash,
        }
        self._save_metadata()

    # --- Search Methods (used by retrieval layer) ---

    def semantic_search(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        """Search chunks by vector similarity."""
        try:
            count = self._chunks.count()
            if count == 0:
                return []
            results = self._chunks.query(
                query_embeddings=[query_embedding],
                n_results=min(limit, count),
                include=["documents", "metadatas", "distances"],
            )
            # ChromaDB returns list-of-lists for batch queries; unwrap the single query.
            out = []
            for chunk_id, dist, doc, meta in zip(
                results["ids"][0],
                results["distances"][0],
                results["documents"][0],
                results["metadatas"][0],
            ):
                row = dict(meta)
                row["chunk_id"] = chunk_id
                row["content"] = doc
                row["_distance"] = dist
                out.append(row)
            return out
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def search_facts(
        self,
        entity: str | None = None,
        key: str | None = None,
        value: str | None = None,
    ) -> list[dict]:
        """Search the facts store by entity, key, or value (client-side filtering)."""
        results = []
        for fact in self._facts.values():
            if entity and entity.lower() not in fact.get("entity", "").lower():
                continue
            if key and key.lower() not in fact.get("key", "").lower():
                continue
            if value and value.lower() not in fact.get("value", "").lower():
                continue
            results.append(fact)
        return results[:50]
