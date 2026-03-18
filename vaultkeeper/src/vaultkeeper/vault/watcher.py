"""Filesystem watcher for the vault.

Uses watchfiles (Rust-based) for efficient change detection.
Events are debounced to handle Obsidian Sync burst writes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Awaitable

from watchfiles import awatch, Change

logger = logging.getLogger(__name__)


class VaultWatcher:
    """Watches the vault directory for file changes.

    Debounces events so rapid writes (e.g., Obsidian Sync) are batched
    into single index updates.
    """

    def __init__(
        self,
        vault_path: str | Path,
        debounce_seconds: float = 3.0,
        on_changes: Callable[[list[tuple[str, str]]], Awaitable[None]] | None = None,
    ):
        self.vault_path = Path(vault_path)
        self.debounce_seconds = debounce_seconds
        self.on_changes = on_changes
        self._task: asyncio.Task | None = None
        self._pending: dict[str, str] = {}  # path -> change_type
        self._debounce_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start watching the vault directory."""
        logger.info(f"Starting vault watcher on {self.vault_path}")
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop the watcher."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._debounce_task:
            self._debounce_task.cancel()
        logger.info("Vault watcher stopped")

    async def _watch_loop(self) -> None:
        """Main watch loop."""
        try:
            async for changes in awatch(
                self.vault_path,
                recursive=True,
                step=500,  # check every 500ms
            ):
                for change_type, path_str in changes:
                    path = Path(path_str)

                    # Skip hidden files/dirs and non-markdown
                    if any(part.startswith(".") for part in path.parts):
                        continue
                    if path.suffix not in (".md", ".base"):
                        continue

                    try:
                        relative = str(path.relative_to(self.vault_path))
                    except ValueError:
                        continue

                    change_name = _change_type_name(change_type)
                    self._pending[relative] = change_name

                # Reset the debounce timer
                if self._debounce_task:
                    self._debounce_task.cancel()
                self._debounce_task = asyncio.create_task(
                    self._flush_after_debounce()
                )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Watcher error: {e}")

    async def _flush_after_debounce(self) -> None:
        """Wait for the debounce period, then flush pending changes."""
        try:
            await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            return

        if not self._pending:
            return

        # Snapshot and clear pending
        changes = list(self._pending.items())
        self._pending.clear()

        logger.info(f"Flushing {len(changes)} file changes after debounce")

        if self.on_changes:
            try:
                await self.on_changes(changes)
            except Exception as e:
                logger.error(f"Error processing file changes: {e}")


def _change_type_name(change_type: Change) -> str:
    """Convert watchfiles Change enum to a string."""
    if change_type == Change.added:
        return "created"
    elif change_type == Change.modified:
        return "modified"
    elif change_type == Change.deleted:
        return "deleted"
    return "unknown"
