"""Query router for the internal agent.

Classifies incoming queries and selects retrieval strategies.
Uses the router LLM model for classification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from vaultkeeper.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class StrategySelection:
    """A selected retrieval strategy with parameters."""
    name: str               # "semantic", "text", "facts", "frontmatter", "base", "graph"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Result of query classification and strategy selection."""
    classification: str     # "factual", "conceptual", "structural", "base_query"
    strategies: list[StrategySelection]
    reasoning: str = ""


ROUTER_SYSTEM_PROMPT = """You are a query router for an Obsidian vault search system.
Your job is to classify the user's query and select the best retrieval strategies.

Available strategies:
- "semantic": Vector similarity search across note chunks. Best for conceptual, thematic queries.
- "text": Exact text/regex search across raw files. Best for specific phrases, codes, numbers.
- "facts": Search extracted key-value facts table. Best for specific attributes of entities (e.g., someone's phone number, a code, a date).
- "frontmatter": Query notes by frontmatter properties. Best for structured data (type, status, rating).
- "base": Query an Obsidian Base (structured view over notes). Best when the query maps to a known Base.
- "graph": Follow wikilinks from a note. Best for finding related/connected notes.

Classification types:
- "factual": Specific fact retrieval (a code, a date, a name, a number)
- "conceptual": Thematic/topical search (ideas, summaries, related content)
- "structural": Navigation (what folders exist, what's linked to X)
- "base_query": Query that maps to a Base view (places, media, etc.)

Rules:
- For specific facts about a person/thing, ALWAYS include "facts" AND "text" strategies
- For conceptual queries, prefer "semantic"
- For queries mentioning "places", "media", "books", "movies", or similar collections, include "base"
- You can select multiple strategies (they run in parallel)
- Be specific in strategy params: include entity names, search terms, frontmatter fields

Respond with ONLY a JSON object, no markdown, no explanation:
{
  "classification": "factual|conceptual|structural|base_query",
  "strategies": [
    {"name": "strategy_name", "params": {"key": "value"}}
  ],
  "reasoning": "brief explanation"
}"""


class QueryRouter:
    """Routes queries to appropriate retrieval strategies."""

    def __init__(self, llm_client: LLMClient, base_names: list[str] | None = None):
        self.llm_client = llm_client
        self.base_names = base_names or []

    async def route(self, query: str) -> RoutingResult:
        """Classify a query and select retrieval strategies."""
        base_context = ""
        if self.base_names:
            base_context = f"\n\nKnown Bases in the vault: {', '.join(self.base_names)}"

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT + base_context},
            {"role": "user", "content": query},
        ]

        try:
            result = await self.llm_client.complete_json("router", messages)

            strategies = []
            for s in result.get("strategies", []):
                if isinstance(s, dict) and "name" in s:
                    strategies.append(StrategySelection(
                        name=s["name"],
                        params=s.get("params", {}),
                    ))

            return RoutingResult(
                classification=result.get("classification", "conceptual"),
                strategies=strategies,
                reasoning=result.get("reasoning", ""),
            )

        except Exception as e:
            logger.warning(f"Router LLM failed, falling back to defaults: {e}")
            return self._fallback_route(query)

    def _fallback_route(self, query: str) -> RoutingResult:
        """Simple heuristic routing when the LLM fails."""
        query_lower = query.lower()

        strategies = []

        # Always include semantic as a baseline
        strategies.append(StrategySelection(name="semantic", params={"query": query}))

        # Check for fact-like queries
        fact_keywords = [
            "code", "number", "phone", "address", "email", "birthday",
            "password", "pin", "key", "date", "price", "cost",
        ]
        if any(kw in query_lower for kw in fact_keywords):
            strategies.append(StrategySelection(name="facts", params={"query": query}))
            strategies.append(StrategySelection(name="text", params={"pattern": query}))

        # Check for Base-related queries
        base_keywords = ["places", "media", "books", "movies", "restaurants", "music"]
        if any(kw in query_lower for kw in base_keywords):
            strategies.append(StrategySelection(name="base", params={"query": query}))

        return RoutingResult(
            classification="conceptual",
            strategies=strategies,
            reasoning="Fallback heuristic routing",
        )
