"""Wipe the Vaultkeeper index and reprocess all notes from scratch.

Run inside the container:
    docker exec obsidivec-vaultkeeper-1 python /app/reindex.py
"""

import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

sys.path.insert(0, "/app/src")

from vaultkeeper.config import Config
from vaultkeeper.index.indexer import VaultIndexer
from vaultkeeper.llm.client import EmbeddingClient, LLMClient
from vaultkeeper.vault.reader import VaultReader


async def main() -> None:
    config_path = os.environ.get("VAULTKEEPER_CONFIG", "/etc/vaultkeeper/config.yml")
    config = Config.from_yaml(config_path)

    vault_reader = VaultReader(config.vault.path)
    llm_client = LLMClient(config.models)
    embedding_client = EmbeddingClient(config.embeddings, api_key=config.models.api_key)

    indexer = VaultIndexer(
        config=config,
        vault_reader=vault_reader,
        embedding_client=embedding_client,
        llm_client=llm_client,
    )

    logging.info("Wiping existing index...")
    indexer._client.delete_collection("chunks")
    indexer._facts = {}
    indexer._metadata = {}
    indexer._save_facts()
    indexer._save_metadata()
    logging.info("Done. Index will be rebuilt on next container start.")

    await llm_client.close()
    await embedding_client.close()


if __name__ == "__main__":
    asyncio.run(main())
