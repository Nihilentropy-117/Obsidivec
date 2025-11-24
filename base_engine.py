import os
import yaml
import frontmatter
import sys


# ==========================================
# INTERNAL HELPERS (Filter & Render Logic)
# ==========================================

def _check_condition(condition, post, file_path, vault_root):
    """Internal: Evaluates a single filter condition against a file."""
    rel_path = os.path.relpath(file_path, vault_root)

    # Handle Equality Check (==)
    if "==" in condition:
        prop, target_val = [x.strip() for x in condition.split("==", 1)]
        target_val = target_val.strip('"').strip("'")

        # 1. Handle IMPLICIT properties (file system data)
        if prop == "file.folder":
            actual_val = os.path.dirname(rel_path)
        elif prop == "file.name":
            actual_val = os.path.basename(file_path)
        elif prop == "file.path":
            actual_val = rel_path
        # 2. Handle EXPLICIT properties (frontmatter data)
        else:
            actual_val = str(post.metadata.get(prop, ""))

        return actual_val == target_val

    return True


def _matches_filters(post, file_path, filter_config, vault_root):
    """Internal: Recursive filter matcher."""
    if not filter_config:
        return True

    if isinstance(filter_config, dict):
        if 'and' in filter_config:
            return all(_matches_filters(post, file_path, f, vault_root) for f in filter_config['and'])
        if 'or' in filter_config:
            return any(_matches_filters(post, file_path, f, vault_root) for f in filter_config['or'])
        if 'not' in filter_config:
            return not _matches_filters(post, file_path, filter_config['not'], vault_root)

    if isinstance(filter_config, list):
        return all(_check_condition(cond, post, file_path, vault_root) for cond in filter_config)

    if isinstance(filter_config, str):
        return _check_condition(filter_config, post, file_path, vault_root)

    return True


def _render_table(headers, rows):
    """Internal: Prints the Markdown table to stdout."""
    if not rows:
        print("No matching records found.")
        return

    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            val_str = str(val) if val is not None else ""
            widths[i] = max(widths[i], len(val_str))

    # Header
    header_row = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    separator = "| " + " | ".join("-" * w for w in widths) + " |"

    print(header_row)
    print(separator)

    # Data
    for row in rows:
        line_parts = []
        for val, w in zip(row, widths):
            val_str = str(val) if val is not None else ""
            line_parts.append(val_str.ljust(w))
        print("| " + " | ".join(line_parts) + " |")


# ==========================================
# PUBLIC API
# ==========================================

def list_bases(vault_root):
    """
    Scans the vault and returns a list of absolute paths to .base files.
    """
    base_files = []
    for root, _, files in os.walk(vault_root):
        for file in files:
            if file.endswith(".base"):
                base_files.append(os.path.join(root, file))
    return base_files


def view_base(vault_root, base_path):
    """
    Parses a specific .base file, scans the vault for matches, and prints a table.
    """
    if not os.path.exists(base_path):
        print(f"Error: Base file not found at {base_path}")
        return

    # 1. Load Config
    print(f"\n--- Loading View from {os.path.basename(base_path)} ---\n")
    with open(base_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # 2. Determine View and Columns
    views = base_config.get('views', [])
    active_view = views[0] if views else base_config

    # Logic: Prefer 'columns' dict keys, fallback to 'order' list
    if 'columns' in active_view:
        target_columns = list(active_view['columns'].keys())
    elif 'order' in active_view:
        target_columns = active_view['order']
    else:
        target_columns = ['file.name']

    # 3. Merge filters (Global + View specific)
    global_filters = base_config.get('filters', {})
    view_filters = active_view.get('filters', {})

    if global_filters and view_filters:
        combined_filters = {'and': [global_filters, view_filters]}
    else:
        combined_filters = global_filters or view_filters

    # 4. Scan Vault
    rows = []
    scanned_count = 0

    for root, dirs, files in os.walk(vault_root):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                scanned_count += 1

                try:
                    post = frontmatter.load(full_path)

                    if _matches_filters(post, full_path, combined_filters, vault_root):
                        row = []
                        for col in target_columns:
                            # Handle implicit columns
                            if col == 'file.name':
                                row.append(file)
                            elif col == 'file.path':
                                row.append(os.path.relpath(full_path, vault_root))
                            elif col == 'file.folder':
                                row.append(os.path.dirname(os.path.relpath(full_path, vault_root)))
                            else:
                                # Handle standard frontmatter properties
                                val = post.metadata.get(col, "")
                                if isinstance(val, list):
                                    val = ", ".join(str(v) for v in val)
                                row.append(val)
                        rows.append(row)
                except Exception:
                    continue

    print(f"Scanned {scanned_count} files.")
    print(f"Found {len(rows)} matches.\n")

    _render_table(target_columns, rows)


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    root_path = "/home/nihil/Wanderland/WanderlandReX/"

    # 1. List
    bases = list_bases(root_path)

    if not bases:
        print("No .base files found.")
        sys.exit(0)

    print(f"\nFound {len(bases)} Bases:")
    for idx, path in enumerate(bases):
        print(f"[{idx}] {os.path.relpath(path, root_path)}")

    # 2. Select
    try:
        selection = int(input("\nSelect a base (number): "))
        selected_base_path = bases[selection]
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)

    # 3. View
    view_base(root_path, selected_base_path)
