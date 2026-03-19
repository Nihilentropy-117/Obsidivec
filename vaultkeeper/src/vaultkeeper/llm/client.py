"""Unified LLM client abstraction.

Supports OpenRouter (and future Ollama) for chat completions and embeddings.
Each internal role (router, extractor, synthesizer) gets its own config
but shares the same provider connection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from vaultkeeper.config import ModelsConfig, EmbeddingsConfig, ModelConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client for chat completions via OpenRouter."""

    def __init__(self, config: ModelsConfig):
        self.config = config
        self._http = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/vaultkeeper",
                "X-Title": "Vaultkeeper",
            },
            timeout=60.0,
        )

    async def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request using the specified role's model config.

        Args:
            role: One of "router", "extractor", "synthesizer"
            messages: Chat messages in OpenAI format
            json_mode: If True, request JSON response format

        Returns:
            The assistant's response text.
        """
        model_config: ModelConfig = getattr(self.config, role)

        payload: dict[str, Any] = {
            "model": model_config.model,
            "messages": messages,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = await self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API error ({role}): {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"LLM request failed ({role}): {e}")
            raise

    async def complete_json(self, role: str, messages: list[dict[str, str]]) -> dict | list:
        """Send a completion request and parse the response as JSON.

        Handles markdown code fences in the response.
        """
        raw = await self.complete(role, messages, json_mode=True)

        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines[1:] if not l.strip() == "```"]
            cleaned = "\n".join(lines)

        return json.loads(cleaned)

    async def close(self) -> None:
        await self._http.aclose()


class EmbeddingClient:
    """Client for generating embeddings via OpenRouter."""

    def __init__(self, config: EmbeddingsConfig, api_key: str):
        self.config = config
        self._http = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/vaultkeeper",
                "X-Title": "Vaultkeeper",
            },
            timeout=120.0,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Handles batching internally based on config.batch_size.
        """
        all_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch."""
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": texts,
        }
        if self.config.dimensions:
            payload["dimensions"] = self.config.dimensions

        try:
            response = await self._http.post("/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()

            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed([text])
        return results[0]

    async def close(self) -> None:
        await self._http.aclose()
