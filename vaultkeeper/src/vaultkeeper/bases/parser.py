"""Obsidian Bases parser.

Parses .base YAML files and resolves their filters against vault notes.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from vaultkeeper.vault.reader import VaultReader, ParsedNote


@dataclass
class BaseView:
    """A single view definition within a Base."""
    name: str
    view_type: str                          # "table", "board", etc.
    columns: list[str]                      # ordered column names from 'order'
    sort: list[dict[str, str]] = field(default_factory=list)
    column_sizes: dict[str, int] = field(default_factory=dict)


@dataclass
class ParsedBase:
    """A fully parsed .base file."""
    path: str
    name: str                               # from filename
    filters: list[str]                      # raw filter expressions
    filter_logic: str                       # "and" or "or"
    views: list[BaseView]
    raw: dict                               # the original YAML


class FilterEngine:
    """Evaluates Base filter expressions against notes."""

    # Pattern: property operator "value" or property operator value
    EXPR_PATTERN = re.compile(
        r'^([\w.]+)\s*(==|!=|contains|>|<|>=|<=)\s*"?([^"]*)"?$'
    )

    OPERATORS: dict[str, Callable] = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "contains": lambda a, b: b.lower() in str(a).lower() if a else False,
    }

    def __init__(self, vault_reader: VaultReader):
        self.vault_reader = vault_reader

    def evaluate_expression(self, expr: str, note: ParsedNote) -> bool:
        """Evaluate a single filter expression against a note."""
        match = self.EXPR_PATTERN.match(expr.strip())
        if not match:
            return True  # Can't parse = don't filter

        prop, op_str, value = match.groups()
        op_func = self.OPERATORS.get(op_str)
        if not op_func:
            return True

        note_value = self._get_property(prop, note)

        try:
            return op_func(note_value, value)
        except (TypeError, ValueError):
            return False

    def _get_property(self, prop: str, note: ParsedNote) -> Any:
        """Resolve a property reference to its value on a note."""
        if prop == "file.folder":
            return str(Path(note.path).parent)
        elif prop == "file.name":
            return note.title
        elif prop == "file.path":
            return note.path
        elif prop == "file.ext":
            return Path(note.path).suffix
        elif prop == "file.tags" or prop == "tags":
            return note.tags
        else:
            # Frontmatter field
            return note.frontmatter.get(prop)


class BasesParser:
    """Parses and resolves Obsidian Base files."""

    def __init__(self, vault_reader: VaultReader):
        self.vault_reader = vault_reader
        self.filter_engine = FilterEngine(vault_reader)

    def parse_base(self, relative_path: str) -> ParsedBase:
        """Parse a .base file into a structured representation."""
        raw_content = self.vault_reader.read_raw(relative_path)
        data = yaml.safe_load(raw_content) or {}

        # Extract filters
        filter_logic = "and"
        filter_exprs = []
        filters = data.get("filters", {})
        if "and" in filters:
            filter_logic = "and"
            filter_exprs = filters["and"]
        elif "or" in filters:
            filter_logic = "or"
            filter_exprs = filters["or"]

        # Ensure filter_exprs is a list of strings
        if not isinstance(filter_exprs, list):
            filter_exprs = []
        filter_exprs = [str(f) for f in filter_exprs]

        # Extract views
        views = []
        for view_data in data.get("views", []):
            views.append(BaseView(
                name=view_data.get("name", "Untitled"),
                view_type=view_data.get("type", "table"),
                columns=view_data.get("order", []),
                sort=[s for s in view_data.get("sort", []) if isinstance(s, dict)],
                column_sizes={
                    k: v for k, v in view_data.get("columnSize", {}).items()
                    if isinstance(v, (int, float))
                },
            ))

        name = Path(relative_path).stem

        return ParsedBase(
            path=relative_path,
            name=name,
            filters=filter_exprs,
            filter_logic=filter_logic,
            views=views,
            raw=data,
        )

    def resolve_base(self, base: ParsedBase) -> list[dict[str, Any]]:
        """Resolve a Base's filters to matching notes, returning rows.

        Each row is a dict with the columns defined in the first view.
        """
        all_notes = self.vault_reader.list_all_notes()

        matching_rows = []
        # Determine columns from the first view (if any)
        columns = base.views[0].columns if base.views else ["file.name"]

        for note_path in all_notes:
            try:
                note = self.vault_reader.read_note(note_path)
            except Exception:
                continue

            if self._matches_filters(note, base):
                row = self._extract_row(note, columns)
                matching_rows.append(row)

        # Apply sorting from the first view
        if base.views and base.views[0].sort:
            matching_rows = self._sort_rows(matching_rows, base.views[0].sort)

        return matching_rows

    def _matches_filters(self, note: ParsedNote, base: ParsedBase) -> bool:
        """Check if a note matches the Base's filter expressions."""
        if not base.filters:
            return True

        results = [
            self.filter_engine.evaluate_expression(expr, note)
            for expr in base.filters
        ]

        if base.filter_logic == "and":
            return all(results)
        else:
            return any(results)

    def _extract_row(self, note: ParsedNote, columns: list[str]) -> dict[str, Any]:
        """Extract a row of values from a note for the given columns."""
        row = {"_path": note.path}  # always include path for reference
        for col in columns:
            row[col] = self.filter_engine._get_property(col, note)
        return row

    def _sort_rows(
        self, rows: list[dict[str, Any]], sort_specs: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Sort rows according to Base sort specifications."""
        for spec in reversed(sort_specs):
            prop = spec.get("property", "")
            direction = spec.get("direction", "ASC")
            reverse = direction.upper() == "DESC"

            def sort_key(row: dict, p: str = prop) -> Any:
                val = row.get(p, "")
                if val is None:
                    return ""
                return str(val)

            rows.sort(key=sort_key, reverse=reverse)

        return rows

    def list_bases(self) -> list[dict[str, Any]]:
        """List all Base files with their parsed metadata."""
        base_paths = self.vault_reader.list_all_bases()
        results = []
        for bp in base_paths:
            try:
                parsed = self.parse_base(bp)
                results.append({
                    "path": parsed.path,
                    "name": parsed.name,
                    "filter_logic": parsed.filter_logic,
                    "filters": parsed.filters,
                    "views": [
                        {"name": v.name, "type": v.view_type, "columns": v.columns}
                        for v in parsed.views
                    ],
                })
            except Exception:
                continue
        return results
