"""Undo system for vault mutations.

Tracks all mutations grouped by message_id (one per user message).
vault_undo reverts all mutations from the most recent message.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from vaultkeeper.vault.writer import MutationResult, VaultWriter


@dataclass
class Operation:
    """A recorded mutation operation."""
    timestamp: float
    message_id: str
    path: str
    action: str
    before_content: str | None
    after_content: str


class UndoManager:
    """Manages an in-memory journal of mutations for undo support."""

    def __init__(self, writer: VaultWriter, max_operations: int = 50):
        self.writer = writer
        self.max_operations = max_operations
        self._journal: deque[Operation] = deque(maxlen=max_operations)
        self._current_message_id: str = ""

    def set_message_id(self, message_id: str) -> None:
        """Set the current message ID. All subsequent mutations are grouped under it."""
        self._current_message_id = message_id

    def record(self, result: MutationResult) -> None:
        """Record a mutation in the journal."""
        self._journal.append(Operation(
            timestamp=time.time(),
            message_id=self._current_message_id,
            path=result.path,
            action=result.action,
            before_content=result.before_content,
            after_content=result.after_content,
        ))

    def undo_last_message(self) -> list[str]:
        """Undo all mutations from the most recent message_id.

        Returns a list of descriptions of what was undone.
        """
        if not self._journal:
            return ["Nothing to undo."]

        # Find the most recent message_id
        last_message_id = self._journal[-1].message_id

        # Collect all operations from that message, in reverse order
        ops_to_undo: list[Operation] = []
        while self._journal and self._journal[-1].message_id == last_message_id:
            ops_to_undo.append(self._journal.pop())

        undone: list[str] = []

        for op in ops_to_undo:
            if op.action == "create":
                # Undo create = delete the file
                self.writer.remove_file(op.path)
                undone.append(f"Deleted created file: {op.path}")

            elif op.action == "delete":
                # Undo delete = recreate with original content
                if op.before_content is not None:
                    self.writer.restore_content(op.path, op.before_content)
                    undone.append(f"Restored deleted file: {op.path}")

            elif op.action in ("patch", "append", "frontmatter_update"):
                # Undo edit = restore previous content
                if op.before_content is not None:
                    self.writer.restore_content(op.path, op.before_content)
                    undone.append(f"Reverted {op.action} on: {op.path}")

        return undone if undone else ["Nothing to undo."]

    @property
    def journal_size(self) -> int:
        return len(self._journal)

    def get_recent_operations(self, count: int = 10) -> list[dict]:
        """Return recent operations for inspection."""
        ops = list(self._journal)[-count:]
        return [
            {
                "timestamp": op.timestamp,
                "message_id": op.message_id,
                "path": op.path,
                "action": op.action,
            }
            for op in ops
        ]
