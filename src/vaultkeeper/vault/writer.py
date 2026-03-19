"""Vault filesystem writer.

Handles creating, deleting, patching, and appending to notes.
All mutations go through here so the undo system can intercept them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
import yaml


@dataclass
class PatchOperation:
    """A single line-level edit operation."""
    action: str         # "insert", "delete", "replace"
    line: int           # 1-indexed line number
    content: str = ""   # new content (for insert/replace)


@dataclass
class MutationResult:
    """Result of a mutation operation, used by the undo system."""
    path: str
    action: str             # "create", "delete", "patch", "append", "frontmatter_update"
    before_content: str | None   # file content before mutation (None for create)
    after_content: str           # file content after mutation


class VaultWriter:
    """Writes and mutates files in an Obsidian vault."""

    def __init__(self, vault_path: str | Path):
        self.vault_path = Path(vault_path)

    def _resolve(self, relative_path: str) -> Path:
        resolved = (self.vault_path / relative_path).resolve()
        if not str(resolved).startswith(str(self.vault_path.resolve())):
            raise ValueError(f"Path escapes vault root: {relative_path}")
        return resolved

    def create_note(
        self,
        relative_path: str,
        content: str = "",
        note_frontmatter: dict[str, Any] | None = None,
    ) -> MutationResult:
        """Create a new note. Fails if the file already exists."""
        full_path = self._resolve(relative_path)
        if full_path.exists():
            raise FileExistsError(f"Note already exists: {relative_path}")

        # Ensure parent directories exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Build content with frontmatter
        if note_frontmatter:
            post = frontmatter.Post(content, **note_frontmatter)
            file_content = frontmatter.dumps(post)
        else:
            file_content = content

        full_path.write_text(file_content, encoding="utf-8")

        return MutationResult(
            path=relative_path,
            action="create",
            before_content=None,
            after_content=file_content,
        )

    def delete_note(self, relative_path: str) -> MutationResult:
        """Delete a note. Returns former content for undo."""
        full_path = self._resolve(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Note not found: {relative_path}")

        before_content = full_path.read_text(encoding="utf-8")
        full_path.unlink()

        return MutationResult(
            path=relative_path,
            action="delete",
            before_content=before_content,
            after_content="",
        )

    def patch_note(self, relative_path: str, operations: list[PatchOperation]) -> MutationResult:
        """Apply line-level patch operations to a note.

        Operations are applied in order. Line numbers refer to the state
        of the file *before any operations in this batch*.

        To handle this correctly, operations are sorted and applied from
        bottom to top so line numbers remain stable.
        """
        full_path = self._resolve(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Note not found: {relative_path}")

        before_content = full_path.read_text(encoding="utf-8")
        lines = before_content.splitlines(keepends=True)

        # Sort operations by line number descending so earlier ops don't shift later ones
        sorted_ops = sorted(operations, key=lambda op: op.line, reverse=True)

        for op in sorted_ops:
            idx = op.line - 1  # convert to 0-indexed

            if op.action == "insert":
                # Insert before the given line
                new_line = op.content if op.content.endswith("\n") else op.content + "\n"
                lines.insert(idx, new_line)

            elif op.action == "delete":
                if 0 <= idx < len(lines):
                    lines.pop(idx)

            elif op.action == "replace":
                if 0 <= idx < len(lines):
                    new_line = op.content if op.content.endswith("\n") else op.content + "\n"
                    lines[idx] = new_line

            else:
                raise ValueError(f"Unknown patch action: {op.action}")

        after_content = "".join(lines)
        full_path.write_text(after_content, encoding="utf-8")

        return MutationResult(
            path=relative_path,
            action="patch",
            before_content=before_content,
            after_content=after_content,
        )

    def append_note(self, relative_path: str, content: str) -> MutationResult:
        """Append content to the end of a note."""
        full_path = self._resolve(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Note not found: {relative_path}")

        before_content = full_path.read_text(encoding="utf-8")

        # Ensure there's a newline before appended content
        separator = "" if before_content.endswith("\n") else "\n"
        after_content = before_content + separator + content

        full_path.write_text(after_content, encoding="utf-8")

        return MutationResult(
            path=relative_path,
            action="append",
            before_content=before_content,
            after_content=after_content,
        )

    def update_frontmatter(
        self, relative_path: str, updates: dict[str, Any]
    ) -> MutationResult:
        """Update specific frontmatter fields without touching the body.

        Set a value to None to remove that field.
        """
        full_path = self._resolve(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Note not found: {relative_path}")

        before_content = full_path.read_text(encoding="utf-8")
        post = frontmatter.loads(before_content)

        for key, value in updates.items():
            if value is None:
                post.metadata.pop(key, None)
            else:
                post.metadata[key] = value

        after_content = frontmatter.dumps(post)
        full_path.write_text(after_content, encoding="utf-8")

        return MutationResult(
            path=relative_path,
            action="frontmatter_update",
            before_content=before_content,
            after_content=after_content,
        )

    def restore_content(self, relative_path: str, content: str) -> None:
        """Restore a file to exact content. Used by undo system."""
        full_path = self._resolve(relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def remove_file(self, relative_path: str) -> None:
        """Remove a file without recording. Used by undo system."""
        full_path = self._resolve(relative_path)
        if full_path.exists():
            full_path.unlink()
