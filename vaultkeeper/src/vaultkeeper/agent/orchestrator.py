"""Query orchestrator.

Takes routing decisions and executes retrieval strategies in parallel,
then merges and deduplicates results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vaultkeeper.agent.router import QueryRouter, RoutingResult, StrategySelection
from vaultkeeper.bases.parser import BasesParser
from vaultkeeper.config import Config
from vaultkeeper.index.indexer import VaultIndexer
from vaultkeeper.llm.client import EmbeddingClient, LLMClient
from vaultkeeper.vault.reader import VaultReader

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single result from any retrieval strategy."""
    source: str                 # strategy name
    path: str                   # note path
    title: str = ""
    content: str = ""           # relevant text
    score: float = 0.0          # relevance score (0-1)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResponse:
    """Combined response from all strategies."""
    query: str
    classification: str
    results: list[RetrievalResult]
    synthesized_answer: str = ""
    sources: list[str] = field(default_factory=list)


SYNTHESIZER_PROMPT = """You are an assistant answering questions about the user's Obsidian vault.
Given the user's query and retrieved context from their notes, provide a clear, concise answer.

Rules:
- Answer ONLY from the provided context. Do not make up information.
- If the context doesn't contain enough information, say so clearly.
- Cite the source note path for each piece of information.
- Be concise. The user wants facts, not fluff.
- If the answer is a simple value (a code, a date, a name), just state it directly.

Query: {query}

Retrieved context:
{context}

Answer:"""


class Orchestrator:
    """Executes retrieval strategies and synthesizes answers."""

    def __init__(
        self,
        config: Config,
        vault_reader: VaultReader,
        indexer: VaultIndexer,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
        bases_parser: BasesParser,
        router: QueryRouter,
    ):
        self.config = config
        self.vault_reader = vault_reader
        self.indexer = indexer
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.bases_parser = bases_parser
        self.router = router

    async def query(self, user_query: str) -> OrchestratorResponse:
        """Process a natural language query end-to-end."""
        # Step 1: Route the query
        routing = await self.router.route(user_query)
        logger.info(
            f"Query routed: classification={routing.classification}, "
            f"strategies={[s.name for s in routing.strategies]}"
        )

        # Step 2: Execute strategies in parallel
        tasks = [
            self._execute_strategy(strategy, user_query)
            for strategy in routing.strategies
        ]
        strategy_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Flatten and deduplicate results
        all_results: list[RetrievalResult] = []
        for result in strategy_results:
            if isinstance(result, Exception):
                logger.warning(f"Strategy failed: {result}")
                continue
            if isinstance(result, list):
                all_results.extend(result)

        all_results = self._deduplicate(all_results)

        # Step 4: Sort by relevance
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Limit to top results
        top_results = all_results[:15]

        # Step 5: Synthesize answer if we have results
        synthesized = ""
        if top_results:
            synthesized = await self._synthesize(user_query, top_results)

        sources = list(dict.fromkeys(r.path for r in top_results if r.path))

        return OrchestratorResponse(
            query=user_query,
            classification=routing.classification,
            results=top_results,
            synthesized_answer=synthesized,
            sources=sources,
        )

    async def search(self, user_query: str, strategy: str | None = None) -> list[RetrievalResult]:
        """Search without synthesis — just return results."""
        if strategy:
            strategies = [StrategySelection(name=strategy, params={"query": user_query})]
        else:
            routing = await self.router.route(user_query)
            strategies = routing.strategies

        tasks = [self._execute_strategy(s, user_query) for s in strategies]
        strategy_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[RetrievalResult] = []
        for result in strategy_results:
            if isinstance(result, list):
                all_results.extend(result)

        return self._deduplicate(all_results)

    async def _execute_strategy(
        self, strategy: StrategySelection, query: str
    ) -> list[RetrievalResult]:
        """Execute a single retrieval strategy."""
        if strategy.name == "semantic":
            return await self._strategy_semantic(query)
        elif strategy.name == "text":
            return self._strategy_text(strategy.params.get("pattern", query))
        elif strategy.name == "facts":
            return self._strategy_facts(strategy.params, query)
        elif strategy.name == "frontmatter":
            return self._strategy_frontmatter(strategy.params)
        elif strategy.name == "base":
            return await self._strategy_base(strategy.params, query)
        elif strategy.name == "graph":
            return self._strategy_graph(strategy.params)
        else:
            logger.warning(f"Unknown strategy: {strategy.name}")
            return []

    async def _strategy_semantic(self, query: str) -> list[RetrievalResult]:
        """Vector similarity search."""
        query_embedding = await self.embedding_client.embed_single(query)
        raw_results = self.indexer.semantic_search(query_embedding, limit=10)

        results = []
        for r in raw_results:
            results.append(RetrievalResult(
                source="semantic",
                path=r.get("source_path", ""),
                title=r.get("note_title", ""),
                content=r.get("content", ""),
                score=1.0 - r.get("_distance", 0.5),  # convert distance to similarity
                metadata={
                    "chunk_type": r.get("chunk_type", ""),
                    "section_header": r.get("section_header", ""),
                },
            ))
        return results

    def _strategy_text(self, pattern: str) -> list[RetrievalResult]:
        """Full-text search using ripgrep."""
        vault_path = str(self.config.vault.path)
        try:
            proc = subprocess.run(
                [
                    "rg",
                    "--ignore-case",
                    "--line-number",
                    "--max-count", "5",
                    "--glob", "*.md",
                    "--no-heading",
                    pattern,
                    vault_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            # ripgrep not installed, fall back to grep
            try:
                proc = subprocess.run(
                    ["grep", "-rn", "-i", "--include=*.md", "-m", "5", pattern, vault_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception:
                return []
        except Exception:
            return []

        results = []
        for line in proc.stdout.strip().split("\n"):
            if not line:
                continue
            # Format: /path/to/file.md:line_num:content
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue

            file_path = parts[0]
            line_num = parts[1]
            content = parts[2].strip()

            # Convert to relative path
            try:
                relative = str(Path(file_path).relative_to(vault_path))
            except ValueError:
                continue

            results.append(RetrievalResult(
                source="text",
                path=relative,
                title=Path(relative).stem,
                content=f"Line {line_num}: {content}",
                score=0.9,  # text matches are high confidence
                metadata={"line_number": int(line_num)},
            ))

        return results

    def _strategy_facts(self, params: dict, query: str) -> list[RetrievalResult]:
        """Search the extracted facts table."""
        entity = params.get("entity")
        key = params.get("key")
        value = params.get("value")

        # If no specific params, try to extract from the query string
        if not entity and not key:
            # Simple heuristic: use the full query as search terms
            raw_results = self.indexer.search_facts(entity=query, key=query)
        else:
            raw_results = self.indexer.search_facts(entity=entity, key=key, value=value)

        results = []
        for r in raw_results:
            results.append(RetrievalResult(
                source="facts",
                path=r.get("source_path", ""),
                title=r.get("note_title", ""),
                content=f"{r.get('entity', '')}: {r.get('key', '')} = {r.get('value', '')}",
                score=0.95,  # fact matches are very high confidence
                metadata={
                    "entity": r.get("entity", ""),
                    "key": r.get("key", ""),
                    "value": r.get("value", ""),
                    "context": r.get("context", ""),
                },
            ))
        return results

    def _strategy_frontmatter(self, params: dict) -> list[RetrievalResult]:
        """Search notes by frontmatter field values."""
        field_name = params.get("field", "")
        field_value = params.get("value", "")
        op = params.get("operator", "==")

        if not field_name:
            return []

        results = []
        for note_path in self.vault_reader.list_all_notes():
            try:
                note = self.vault_reader.read_note(note_path)
                fm_value = note.frontmatter.get(field_name)

                if fm_value is None:
                    continue

                match = False
                if op == "==" and str(fm_value) == str(field_value):
                    match = True
                elif op == "!=" and str(fm_value) != str(field_value):
                    match = True
                elif op == "contains" and str(field_value).lower() in str(fm_value).lower():
                    match = True

                if match:
                    results.append(RetrievalResult(
                        source="frontmatter",
                        path=note_path,
                        title=note.title,
                        content=f"{field_name}: {fm_value}",
                        score=0.9,
                        metadata={"frontmatter": dict(note.frontmatter)},
                    ))
            except Exception:
                continue

        return results

    async def _strategy_base(self, params: dict, query: str) -> list[RetrievalResult]:
        """Query an Obsidian Base."""
        base_path = params.get("base_path")

        if not base_path:
            # Try to find the right base from the query
            bases = self.bases_parser.list_bases()
            if not bases:
                return []
            # Use the first matching base (the router should have been more specific)
            base_path = bases[0]["path"]

        try:
            base = self.bases_parser.parse_base(base_path)
            rows = self.bases_parser.resolve_base(base)

            results = []
            for row in rows:
                path = row.get("_path", "")
                # Format row as readable content
                content_parts = [f"{k}: {v}" for k, v in row.items() if k != "_path" and v]
                content = " | ".join(content_parts)

                results.append(RetrievalResult(
                    source="base",
                    path=path,
                    title=row.get("file.name", Path(path).stem if path else ""),
                    content=content,
                    score=0.85,
                    metadata=row,
                ))

            return results
        except Exception as e:
            logger.warning(f"Base query failed: {e}")
            return []

    def _strategy_graph(self, params: dict) -> list[RetrievalResult]:
        """Follow wikilinks from a note."""
        start_path = params.get("path", "")
        if not start_path:
            return []

        try:
            note = self.vault_reader.read_note(start_path)
            results = []
            for link_target in note.wikilinks:
                # Try to resolve the wikilink to a file
                for note_path in self.vault_reader.list_all_notes():
                    if Path(note_path).stem == link_target:
                        linked_note = self.vault_reader.read_note(note_path)
                        results.append(RetrievalResult(
                            source="graph",
                            path=note_path,
                            title=linked_note.title,
                            content=linked_note.body[:500],
                            score=0.7,
                            metadata={"linked_from": start_path},
                        ))
                        break

            return results
        except Exception:
            return []

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Deduplicate results, keeping highest-scoring version per path+content."""
        seen: dict[str, RetrievalResult] = {}
        for r in results:
            key = f"{r.path}:{r.content[:100]}"
            if key not in seen or r.score > seen[key].score:
                seen[key] = r
        return list(seen.values())

    async def _synthesize(self, query: str, results: list[RetrievalResult]) -> str:
        """Use the synthesizer LLM to generate an answer from retrieved context."""
        context_parts = []
        for r in results:
            source_label = f"[{r.path}]" if r.path else f"[{r.source}]"
            context_parts.append(f"{source_label}\n{r.content}")

        context = "\n\n---\n\n".join(context_parts)
        prompt = SYNTHESIZER_PROMPT.format(query=query, context=context)

        try:
            answer = await self.llm_client.complete(
                "synthesizer",
                [{"role": "user", "content": prompt}],
            )
            return answer
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Return the best raw result as fallback
            if results:
                return f"From {results[0].path}: {results[0].content}"
            return "Unable to synthesize an answer from the available context."
