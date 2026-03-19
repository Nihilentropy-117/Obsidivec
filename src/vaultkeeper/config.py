"""Configuration loader for Vaultkeeper.

Reads config.yml and resolves environment variable references like ${VAR_NAME}.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} references in config values."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")
        match = pattern.search(value)
        if match:
            env_var = match.group(1)
            env_val = os.environ.get(env_var, "")
            # If the entire string is one env var, return the resolved value directly
            if match.group(0) == value:
                return env_val
            return pattern.sub(lambda m: os.environ.get(m.group(1), ""), value)
        return value
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


@dataclass
class ModelConfig:
    model: str
    max_tokens: int = 1000
    temperature: float = 0.0


@dataclass
class ModelsConfig:
    provider: str = "openrouter"
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    router: ModelConfig = field(default_factory=lambda: ModelConfig(
        model="google/gemini-flash-2.0-lite",
        max_tokens=500,
        temperature=0.0,
    ))
    extractor: ModelConfig = field(default_factory=lambda: ModelConfig(
        model="google/gemini-flash-2.0",
        max_tokens=2000,
        temperature=0.0,
    ))
    synthesizer: ModelConfig = field(default_factory=lambda: ModelConfig(
        model="google/gemini-flash-2.0",
        max_tokens=1000,
        temperature=0.1,
    ))


@dataclass
class EmbeddingsConfig:
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64


@dataclass
class VaultConfig:
    path: str = "/vault"
    index_on_startup: bool = True
    watch: bool = True
    debounce_seconds: float = 3.0


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    transport: str = "streamable-http"
    auth_token: str = ""


@dataclass
class IndexConfig:
    db_path: str = "/data/chromadb"
    fact_extraction: bool = True
    chunk_max_tokens: int = 1500


@dataclass
class UndoConfig:
    max_operations: int = 50


@dataclass
class Config:
    vault: VaultConfig = field(default_factory=VaultConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    undo: UndoConfig = field(default_factory=UndoConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from a YAML file, resolving env vars."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        raw = _resolve_env_vars(raw)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, data: dict) -> Config:
        config = cls()

        if "vault" in data:
            v = data["vault"]
            config.vault = VaultConfig(
                path=v.get("path", config.vault.path),
                index_on_startup=v.get("index_on_startup", config.vault.index_on_startup),
                watch=v.get("watch", config.vault.watch),
                debounce_seconds=v.get("debounce_seconds", config.vault.debounce_seconds),
            )

        if "server" in data:
            s = data["server"]
            config.server = ServerConfig(
                host=s.get("host", config.server.host),
                port=s.get("port", config.server.port),
                transport=s.get("transport", config.server.transport),
                auth_token=s.get("auth", {}).get("token", "")
                if isinstance(s.get("auth"), dict)
                else config.server.auth_token,
            )

        if "models" in data:
            m = data["models"]
            config.models = ModelsConfig(
                provider=m.get("provider", config.models.provider),
                api_key=m.get("api_key", config.models.api_key),
                base_url=m.get("base_url", config.models.base_url),
                router=_parse_model_config(m.get("router"), config.models.router),
                extractor=_parse_model_config(m.get("extractor"), config.models.extractor),
                synthesizer=_parse_model_config(m.get("synthesizer"), config.models.synthesizer),
            )

        if "embeddings" in data:
            e = data["embeddings"]
            config.embeddings = EmbeddingsConfig(
                provider=e.get("provider", config.embeddings.provider),
                base_url=e.get("base_url", config.embeddings.base_url),
                model=e.get("model", config.embeddings.model),
                dimensions=e.get("dimensions", config.embeddings.dimensions),
                batch_size=e.get("batch_size", config.embeddings.batch_size),
            )

        if "index" in data:
            i = data["index"]
            config.index = IndexConfig(
                db_path=i.get("db_path", config.index.db_path),
                fact_extraction=i.get("fact_extraction", config.index.fact_extraction),
                chunk_max_tokens=i.get("chunk_max_tokens", config.index.chunk_max_tokens),
            )

        if "undo" in data:
            u = data["undo"]
            config.undo = UndoConfig(
                max_operations=u.get("max_operations", config.undo.max_operations),
            )

        return config


def _parse_model_config(data: dict | None, default: ModelConfig) -> ModelConfig:
    if data is None:
        return default
    return ModelConfig(
        model=data.get("model", default.model),
        max_tokens=data.get("max_tokens", default.max_tokens),
        temperature=data.get("temperature", default.temperature),
    )
