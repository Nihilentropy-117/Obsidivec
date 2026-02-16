"""
Obsidian Bases Manager

Handles parsing, creating, querying, and managing .base files in an Obsidian vault.
Bases are YAML configuration files that define database-like views over vault notes.

.base file format (from Obsidian 1.9+):
    filters: <filter expression>
    properties:
      <property_id>:
        displayName: <string>
    formulas:
      <name>: <expression>
    summaries:
      <name>: <expression>
    views:
      - type: table|list|cards
        name: <string>
        filters: <filter expression>
        order: [<property_id>, ...]

Property ID format:
    note.<property>   - YAML frontmatter property
    file.<property>   - Intrinsic file metadata (name, ext, size, ctime, mtime, path, folder, tags, links)
    formula.<name>    - Computed value defined in the .base file
"""

import glob
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import frontmatter
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models (plain dicts/dataclasses kept lightweight)
# ---------------------------------------------------------------------------

class BasesConfig:
    """Parsed representation of a .base file."""

    def __init__(
        self,
        filepath: str,
        filters: Any = None,
        properties: dict | None = None,
        formulas: dict | None = None,
        summaries: dict | None = None,
        views: list | None = None,
        raw: dict | None = None,
    ):
        self.filepath = filepath
        self.filters = filters
        self.properties = properties or {}
        self.formulas = formulas or {}
        self.summaries = summaries or {}
        self.views = views or []
        self.raw = raw or {}

    def to_dict(self) -> dict:
        d: dict[str, Any] = {}
        if self.filters:
            d["filters"] = self.filters
        if self.properties:
            d["properties"] = self.properties
        if self.formulas:
            d["formulas"] = self.formulas
        if self.summaries:
            d["summaries"] = self.summaries
        if self.views:
            d["views"] = self.views
        return d

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# File property helpers
# ---------------------------------------------------------------------------

def _get_file_properties(filepath: str) -> dict:
    """Return the ``file.*`` properties for *filepath* (mirrors Obsidian Bases)."""
    p = Path(filepath)
    try:
        stat = p.stat()
    except OSError:
        stat = None

    props: dict[str, Any] = {
        "file.name": p.stem,
        "file.ext": p.suffix.lstrip("."),
        "file.path": str(p),
        "file.folder": str(p.parent),
    }
    if stat:
        props["file.size"] = stat.st_size
        props["file.ctime"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        props["file.mtime"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

    return props


def _get_note_properties(filepath: str) -> dict:
    """Return the ``note.*`` frontmatter properties for *filepath*."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        return {f"note.{k}": v for k, v in post.metadata.items()}
    except Exception:
        return {}


def get_all_properties(filepath: str) -> dict:
    """Return merged ``file.*`` and ``note.*`` properties for *filepath*."""
    props = _get_file_properties(filepath)
    if filepath.endswith(".md"):
        props.update(_get_note_properties(filepath))
    return props


# ---------------------------------------------------------------------------
# Simple filter evaluator
# ---------------------------------------------------------------------------

# Obsidian Bases filters use a custom expression language. We support a useful
# subset that covers the most common patterns:
#
#   note.status == "active"
#   contains(note.tags, "python")
#   startsWith(file.name, "2024")
#   endsWith(file.path, ".md")
#   file.ext == "md"
#   not <expr>
#   <expr> and <expr>
#   <expr> or <expr>

_COMPARISON_RE = re.compile(
    r"""^([\w.]+)\s*(==|!=|>=|<=|>|<)\s*(?:"([^"]*)"|'([^']*)'|([\d.]+))$"""
)
_FUNC_CALL_RE = re.compile(
    r"""^(\w+)\(\s*([\w.]+)\s*,\s*(?:"([^"]*)"|'([^']*)')\s*\)$"""
)


def _coerce(val: Any) -> Any:
    """Best-effort coerce a property value to something comparable."""
    if isinstance(val, list):
        return val
    if val is None:
        return ""
    return val


def _eval_atom(expr: str, props: dict) -> bool:
    """Evaluate a single comparison or function-call expression."""
    expr = expr.strip()

    # Simple comparison: note.status == "active"
    m = _COMPARISON_RE.match(expr)
    if m:
        prop_id, op, str_val, str_val2, num_val = m.groups()
        expected = str_val if str_val is not None else (str_val2 if str_val2 is not None else num_val)
        actual = _coerce(props.get(prop_id, ""))

        # Numeric comparison when possible
        try:
            actual_n = float(actual)
            expected_n = float(expected)
            if op == "==":
                return actual_n == expected_n
            if op == "!=":
                return actual_n != expected_n
            if op == ">":
                return actual_n > expected_n
            if op == "<":
                return actual_n < expected_n
            if op == ">=":
                return actual_n >= expected_n
            if op == "<=":
                return actual_n <= expected_n
        except (ValueError, TypeError):
            pass

        actual_s = str(actual).lower()
        expected_s = str(expected).lower()
        if op == "==":
            return actual_s == expected_s
        if op == "!=":
            return actual_s != expected_s
        if op in (">", "<", ">=", "<="):
            ops = {">": str.__gt__, "<": str.__lt__, ">=": str.__ge__, "<=": str.__le__}
            return ops[op](actual_s, expected_s)
        return False

    # Function call: contains(note.tags, "python")
    m = _FUNC_CALL_RE.match(expr)
    if m:
        func_name, prop_id, str_arg, str_arg2 = m.groups()
        arg = str_arg if str_arg is not None else str_arg2
        actual = _coerce(props.get(prop_id, ""))

        if func_name == "contains":
            if isinstance(actual, list):
                return any(arg.lower() in str(item).lower() for item in actual)
            return arg.lower() in str(actual).lower()
        if func_name == "startsWith":
            return str(actual).lower().startswith(arg.lower())
        if func_name == "endsWith":
            return str(actual).lower().endswith(arg.lower())
        if func_name == "linksTo":
            if isinstance(actual, list):
                return any(arg.lower() in str(item).lower() for item in actual)
            return arg.lower() in str(actual).lower()

        logger.warning(f"Unknown filter function: {func_name}")
        return True  # Unknown functions pass by default

    # Boolean literal
    if expr.lower() == "true":
        return True
    if expr.lower() == "false":
        return False

    logger.warning(f"Could not parse filter atom: {expr!r}")
    return True  # Unparseable atoms pass by default (permissive)


def evaluate_filter(filter_expr: Any, props: dict) -> bool:
    """Evaluate a Bases filter expression against a set of properties.

    Supports:
    - String expressions: ``'note.status == "active"'``
    - Compound dict expressions: ``{"and": [...]}``, ``{"or": [...]}``, ``{"not": [...]}``
    """
    if filter_expr is None:
        return True

    # Dict-based compound filters
    if isinstance(filter_expr, dict):
        if "and" in filter_expr:
            return all(evaluate_filter(sub, props) for sub in filter_expr["and"])
        if "or" in filter_expr:
            return any(evaluate_filter(sub, props) for sub in filter_expr["or"])
        if "not" in filter_expr:
            return not any(evaluate_filter(sub, props) for sub in filter_expr["not"])
        return True

    # String expression — may contain ``and`` / ``or`` / ``not`` keywords
    if isinstance(filter_expr, str):
        expr = filter_expr.strip()

        # Handle ``not <expr>``
        if expr.lower().startswith("not "):
            return not evaluate_filter(expr[4:].strip(), props)

        # Split on `` or `` (lowest precedence)
        or_parts = re.split(r"\s+or\s+", expr, flags=re.IGNORECASE)
        if len(or_parts) > 1:
            return any(evaluate_filter(part, props) for part in or_parts)

        # Split on `` and ``
        and_parts = re.split(r"\s+and\s+", expr, flags=re.IGNORECASE)
        if len(and_parts) > 1:
            return all(evaluate_filter(part, props) for part in and_parts)

        return _eval_atom(expr, props)

    return True


# ---------------------------------------------------------------------------
# .base file I/O
# ---------------------------------------------------------------------------

def parse_base_file(filepath: str) -> BasesConfig:
    """Parse a ``.base`` file and return a ``BasesConfig`` object."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to parse .base file {filepath}: {e}")
        return BasesConfig(filepath=filepath)

    return BasesConfig(
        filepath=filepath,
        filters=raw.get("filters"),
        properties=raw.get("properties", {}),
        formulas=raw.get("formulas", {}),
        summaries=raw.get("summaries", {}),
        views=raw.get("views", []),
        raw=raw,
    )


def write_base_file(filepath: str, config: BasesConfig) -> None:
    """Write a ``BasesConfig`` to a ``.base`` file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(config.to_yaml())
    logger.info(f"Wrote .base file: {filepath}")


def delete_base_file(filepath: str) -> bool:
    """Delete a ``.base`` file. Returns True if the file existed."""
    try:
        os.remove(filepath)
        logger.info(f"Deleted .base file: {filepath}")
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Vault-level operations
# ---------------------------------------------------------------------------

def discover_base_files(vault_path: str) -> list[str]:
    """Recursively find all ``.base`` files in the vault."""
    return sorted(glob.glob(os.path.join(vault_path, "**", "*.base"), recursive=True))


def discover_md_files(vault_path: str) -> list[str]:
    """Recursively find all ``.md`` files in the vault."""
    return sorted(glob.glob(os.path.join(vault_path, "**", "*.md"), recursive=True))


def query_base(base_config: BasesConfig, vault_path: str) -> list[dict]:
    """Execute a Base's filter against all vault markdown files.

    Returns a list of dicts, each with ``file`` (path) and ``properties`` (dict of all
    property values for the matching note).
    """
    md_files = discover_md_files(vault_path)
    results = []

    for md_path in md_files:
        props = get_all_properties(md_path)
        if evaluate_filter(base_config.filters, props):
            # Only include properties that are configured in the base, or all if none specified
            if base_config.properties:
                visible = {k: v for k, v in props.items() if k in base_config.properties or k.startswith("file.")}
            else:
                visible = props
            results.append({"file": md_path, "properties": visible})

    return results


def create_base_from_search_results(
    name: str,
    filepaths: list[str],
    vault_path: str,
    view_type: str = "table",
    extra_properties: dict | None = None,
) -> BasesConfig:
    """Generate a ``BasesConfig`` that filters to a specific set of files.

    Uses an ``or`` filter with ``file.path`` equality checks to target the exact
    files returned from a vector search.
    """
    # Build a filter that matches these specific files
    if len(filepaths) == 1:
        filter_expr = f'file.path == "{filepaths[0]}"'
    else:
        filter_expr = {
            "or": [f'file.path == "{fp}"' for fp in filepaths]
        }

    # Gather all unique note properties from the matching files
    all_props: set[str] = set()
    for fp in filepaths:
        props = _get_note_properties(fp)
        all_props.update(props.keys())

    properties = {}
    for prop in sorted(all_props):
        prop_name = prop.split(".", 1)[1] if "." in prop else prop
        properties[prop] = {"displayName": prop_name.replace("_", " ").title()}

    if extra_properties:
        properties.update(extra_properties)

    views = [
        {
            "type": view_type,
            "name": name,
            "order": sorted(properties.keys()),
        }
    ]

    return BasesConfig(
        filepath="",
        filters=filter_expr,
        properties=properties,
        views=views,
    )


def list_bases_summary(vault_path: str) -> list[dict]:
    """Return a summary of every ``.base`` file in the vault."""
    summaries = []
    for bp in discover_base_files(vault_path):
        config = parse_base_file(bp)
        view_types = [v.get("type", "unknown") for v in config.views] if config.views else []
        summaries.append({
            "filepath": bp,
            "filters": config.filters,
            "property_count": len(config.properties),
            "formula_count": len(config.formulas),
            "view_count": len(config.views),
            "view_types": view_types,
        })
    return summaries


def get_all_vault_properties(vault_path: str) -> dict[str, set]:
    """Scan the vault and return all unique property keys with their observed types.

    Returns a dict mapping property IDs to sets of Python type names.
    """
    md_files = discover_md_files(vault_path)
    prop_types: dict[str, set] = {}

    for md_path in md_files:
        props = get_all_properties(md_path)
        for key, value in props.items():
            if key not in prop_types:
                prop_types[key] = set()
            prop_types[key].add(type(value).__name__)

    return prop_types
