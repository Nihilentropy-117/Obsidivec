"""Vault filesystem reader.

Handles reading notes, parsing frontmatter, extracting wikilinks and tags,
and listing directory contents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter


# Patterns for extracting Obsidian-specific constructs
WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z0-9_][a-zA-Z0-9_/\-]*)", re.MULTILINE)
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class NoteHeader:
    level: int
    text: str
    line_number: int


@dataclass
class ParsedNote:
    """A fully parsed Obsidian note."""
    path: str                               # relative to vault root
    title: str                              # from filename (without .md)
    content: str                            # full raw content including frontmatter
    body: str                               # content without frontmatter
    frontmatter: dict[str, Any]             # parsed YAML frontmatter
    tags: list[str]                         # tags from frontmatter + inline #tags
    wikilinks: list[str]                    # [[link]] targets (without aliases)
    headers: list[NoteHeader]               # all markdown headers
    line_count: int


@dataclass
class FileInfo:
    """Basic file information for directory listings."""
    name: str
    path: str                               # relative to vault root
    is_dir: bool
    size: int = 0
    extension: str = ""


class VaultReader:
    """Reads and parses files from an Obsidian vault."""

    def __init__(self, vault_path: str | Path):
        self.vault_path = Path(vault_path)
        if not self.vault_path.is_dir():
            raise ValueError(f"Vault path does not exist or is not a directory: {vault_path}")

    def _resolve(self, relative_path: str) -> Path:
        """Resolve a relative path to an absolute path within the vault.

        Raises ValueError if the path escapes the vault root.
        """
        resolved = (self.vault_path / relative_path).resolve()
        if not str(resolved).startswith(str(self.vault_path.resolve())):
            raise ValueError(f"Path escapes vault root: {relative_path}")
        return resolved

    def _relative(self, absolute_path: Path) -> str:
        """Convert an absolute path to a vault-relative path."""
        return str(absolute_path.relative_to(self.vault_path))

    def read_note(self, relative_path: str) -> ParsedNote:
        """Read and parse a markdown note."""
        full_path = self._resolve(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Note not found: {relative_path}")
        if not full_path.suffix == ".md":
            raise ValueError(f"Not a markdown file: {relative_path}")

        raw_content = full_path.read_text(encoding="utf-8")
        return self._parse_note(relative_path, raw_content)

    def _parse_note(self, relative_path: str, raw_content: str) -> ParsedNote:
        """Parse raw markdown content into a structured note."""
        # Parse frontmatter
        post = frontmatter.loads(raw_content)
        fm = dict(post.metadata) if post.metadata else {}
        body = post.content

        # Title from filename
        title = Path(relative_path).stem

        # Extract tags from frontmatter
        fm_tags = []
        if "tags" in fm:
            raw_tags = fm["tags"]
            if isinstance(raw_tags, list):
                fm_tags = [str(t).lstrip("#") for t in raw_tags]
            elif isinstance(raw_tags, str):
                fm_tags = [t.strip().lstrip("#") for t in raw_tags.split(",")]

        # Extract inline tags from body
        inline_tags = TAG_PATTERN.findall(body)

        # Deduplicate, preserve order
        seen = set()
        all_tags = []
        for tag in fm_tags + inline_tags:
            if tag not in seen:
                seen.add(tag)
                all_tags.append(tag)

        # Extract wikilinks
        wikilinks = list(dict.fromkeys(WIKILINK_PATTERN.findall(body)))

        # Extract headers
        headers = []
        for match in HEADER_PATTERN.finditer(body):
            # Calculate line number in the full content
            line_num = raw_content[:raw_content.find(match.group(0))].count("\n") + 1
            headers.append(NoteHeader(
                level=len(match.group(1)),
                text=match.group(2).strip(),
                line_number=line_num,
            ))

        return ParsedNote(
            path=relative_path,
            title=title,
            content=raw_content,
            body=body,
            frontmatter=fm,
            tags=all_tags,
            wikilinks=wikilinks,
            headers=headers,
            line_count=raw_content.count("\n") + 1,
        )

    def list_dir(self, relative_path: str = "") -> list[FileInfo]:
        """List contents of a directory in the vault."""
        dir_path = self._resolve(relative_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {relative_path}")

        entries = []
        for item in sorted(dir_path.iterdir()):
            # Skip hidden files/dirs and .obsidian
            if item.name.startswith("."):
                continue
            entries.append(FileInfo(
                name=item.name,
                path=self._relative(item),
                is_dir=item.is_dir(),
                size=item.stat().st_size if item.is_file() else 0,
                extension=item.suffix if item.is_file() else "",
            ))
        return entries

    def tree(self, relative_path: str = "", depth: int = 3) -> dict:
        """Recursive directory tree.

        Returns a nested dict representing the tree structure.
        """
        dir_path = self._resolve(relative_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {relative_path}")

        return self._build_tree(dir_path, depth, current_depth=0)

    def _build_tree(self, dir_path: Path, max_depth: int, current_depth: int) -> dict:
        result = {
            "name": dir_path.name or str(self.vault_path),
            "path": self._relative(dir_path) if dir_path != self.vault_path else "",
            "type": "directory",
            "children": [],
        }

        if current_depth >= max_depth:
            return result

        for item in sorted(dir_path.iterdir()):
            if item.name.startswith("."):
                continue

            if item.is_dir():
                child = self._build_tree(item, max_depth, current_depth + 1)
                result["children"].append(child)
            else:
                result["children"].append({
                    "name": item.name,
                    "path": self._relative(item),
                    "type": "file",
                    "extension": item.suffix,
                    "size": item.stat().st_size,
                })

        return result

    def list_all_notes(self) -> list[str]:
        """Return relative paths of all .md files in the vault."""
        notes = []
        for md_file in self.vault_path.rglob("*.md"):
            if any(part.startswith(".") for part in md_file.relative_to(self.vault_path).parts):
                continue
            notes.append(self._relative(md_file))
        return sorted(notes)

    def list_all_bases(self) -> list[str]:
        """Return relative paths of all .base files in the vault."""
        bases = []
        for base_file in self.vault_path.rglob("*.base"):
            if any(part.startswith(".") for part in base_file.relative_to(self.vault_path).parts):
                continue
            bases.append(self._relative(base_file))
        return sorted(bases)

    def read_raw(self, relative_path: str) -> str:
        """Read raw file content."""
        full_path = self._resolve(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        return full_path.read_text(encoding="utf-8")

    def exists(self, relative_path: str) -> bool:
        """Check if a file or directory exists in the vault."""
        try:
            full_path = self._resolve(relative_path)
            return full_path.exists()
        except ValueError:
            return False

    def file_stats(self, relative_path: str) -> dict:
        """Get file stats (mtime, size) for change detection."""
        full_path = self._resolve(relative_path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        stat = full_path.stat()
        return {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }
