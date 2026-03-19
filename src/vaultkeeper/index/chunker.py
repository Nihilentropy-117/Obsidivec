"""Obsidian-aware markdown chunker.

Splits notes into semantically meaningful chunks respecting the document structure:
  Level 0: Frontmatter as structured text
  Level 1: Sections (split on markdown headers)
  Level 2: Paragraphs (fallback for oversized sections)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import tiktoken

from vaultkeeper.vault.reader import ParsedNote


HEADER_SPLIT_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Use cl100k_base tokenizer for token counting (GPT-4/embedding model tokenizer)
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


@dataclass
class Chunk:
    """A single indexed chunk from a note."""
    chunk_id: str                   # deterministic hash of path + chunk_index
    source_path: str                # relative path to the note
    note_title: str                 # from filename
    section_header: str             # nearest parent header, or ""
    chunk_type: str                 # "frontmatter", "section", "paragraph"
    chunk_index: int                # position within the note
    tags: list[str]                 # inherited from note
    links: list[str]                # wikilinks found in this chunk
    content: str                    # the actual text
    content_hash: str               # blake3 or sha256 of content for diffing


def _hash_content(text: str) -> str:
    """Hash chunk content for change detection."""
    try:
        import blake3
        return blake3.blake3(text.encode()).hexdigest()[:32]
    except ImportError:
        return hashlib.sha256(text.encode()).hexdigest()[:32]


def _make_chunk_id(path: str, index: int) -> str:
    """Deterministic chunk ID from path and index."""
    raw = f"{path}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_links(text: str) -> list[str]:
    """Extract wikilinks from a chunk of text."""
    from vaultkeeper.vault.reader import WIKILINK_PATTERN
    return list(dict.fromkeys(WIKILINK_PATTERN.findall(text)))


def chunk_note(note: ParsedNote, max_tokens: int = 1500) -> list[Chunk]:
    """Split a parsed note into chunks.

    Strategy:
    1. Frontmatter becomes its own chunk (if non-empty)
    2. Body is split on markdown headers into sections
    3. Sections exceeding max_tokens are sub-split on paragraphs
    4. Paragraphs exceeding max_tokens are split on sentences (rare)
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    # Chunk 0: Frontmatter
    if note.frontmatter:
        fm_text = _format_frontmatter(note)
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(note.path, chunk_index),
            source_path=note.path,
            note_title=note.title,
            section_header="",
            chunk_type="frontmatter",
            chunk_index=chunk_index,
            tags=note.tags,
            links=[],
            content=fm_text,
            content_hash=_hash_content(fm_text),
        ))
        chunk_index += 1

    # Split body into sections by headers
    sections = _split_into_sections(note.body)

    for section_header, section_text in sections:
        if not section_text.strip():
            continue

        # Prepend note title context to each chunk for better embedding
        context_prefix = f"Note: {note.title}\n"
        if section_header:
            context_prefix += f"Section: {section_header}\n"

        token_count = _count_tokens(section_text)

        if token_count <= max_tokens:
            # Section fits in one chunk
            full_text = context_prefix + section_text
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(note.path, chunk_index),
                source_path=note.path,
                note_title=note.title,
                section_header=section_header,
                chunk_type="section",
                chunk_index=chunk_index,
                tags=note.tags,
                links=_extract_links(section_text),
                content=full_text,
                content_hash=_hash_content(section_text),
            ))
            chunk_index += 1
        else:
            # Section too large — sub-split on paragraphs
            paragraphs = _split_into_paragraphs(section_text)
            for para in paragraphs:
                if not para.strip():
                    continue

                para_tokens = _count_tokens(para)
                if para_tokens <= max_tokens:
                    full_text = context_prefix + para
                    chunks.append(Chunk(
                        chunk_id=_make_chunk_id(note.path, chunk_index),
                        source_path=note.path,
                        note_title=note.title,
                        section_header=section_header,
                        chunk_type="paragraph",
                        chunk_index=chunk_index,
                        tags=note.tags,
                        links=_extract_links(para),
                        content=full_text,
                        content_hash=_hash_content(para),
                    ))
                    chunk_index += 1
                else:
                    # Paragraph still too large — split on sentences
                    sentences = _split_into_sentences(para)
                    current_batch = ""
                    for sentence in sentences:
                        test = current_batch + sentence
                        if _count_tokens(test) > max_tokens and current_batch:
                            full_text = context_prefix + current_batch
                            chunks.append(Chunk(
                                chunk_id=_make_chunk_id(note.path, chunk_index),
                                source_path=note.path,
                                note_title=note.title,
                                section_header=section_header,
                                chunk_type="paragraph",
                                chunk_index=chunk_index,
                                tags=note.tags,
                                links=_extract_links(current_batch),
                                content=full_text,
                                content_hash=_hash_content(current_batch),
                            ))
                            chunk_index += 1
                            current_batch = sentence
                        else:
                            current_batch = test

                    if current_batch.strip():
                        full_text = context_prefix + current_batch
                        chunks.append(Chunk(
                            chunk_id=_make_chunk_id(note.path, chunk_index),
                            source_path=note.path,
                            note_title=note.title,
                            section_header=section_header,
                            chunk_type="paragraph",
                            chunk_index=chunk_index,
                            tags=note.tags,
                            links=_extract_links(current_batch),
                            content=full_text,
                            content_hash=_hash_content(current_batch),
                        ))
                        chunk_index += 1

    return chunks


def _format_frontmatter(note: ParsedNote) -> str:
    """Format frontmatter as readable text for embedding."""
    lines = [f"Note: {note.title}", "Properties:"]
    for key, value in note.frontmatter.items():
        if isinstance(value, list):
            lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (header, content) tuples.

    Content before the first header gets an empty header string.
    """
    sections: list[tuple[str, str]] = []
    parts = HEADER_SPLIT_PATTERN.split(body)

    # parts will be: [before_first_header, level1, text1, between1, level2, text2, ...]
    # But re.split with groups gives: [pre, group1, group2, mid, group1, group2, ...]

    # Simpler approach: find header positions and slice
    headers = list(HEADER_SPLIT_PATTERN.finditer(body))

    if not headers:
        return [("", body)]

    # Content before first header
    pre_header = body[: headers[0].start()].strip()
    if pre_header:
        sections.append(("", pre_header))

    for i, match in enumerate(headers):
        header_text = match.group(2).strip()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        section_content = body[start:end].strip()
        sections.append((header_text, section_content))

    return sections


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text on double newlines (paragraph boundaries)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """Basic sentence splitting. Not perfect, but good enough for chunking."""
    # Split on period/exclamation/question followed by space or newline
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
