from pathlib import Path
from typing import Literal
from rapidfuzz import process, fuzz


def search_vault(
    query: str,
    vault_path: Path,
    search_mode: Literal["filename", "content", "both"] = "filename",
    limit: int = 10,
    score_threshold: int = 50
) -> str:
    """
    Search for notes in the vault using fuzzy matching.

    Args:
        query: Search query string
        vault_path: Path to the Obsidian vault
        search_mode: Where to search - "filename", "content", or "both"
        limit: Maximum number of results to return
        score_threshold: Minimum fuzzy match score (0-100)

    Returns:
        Formatted string with search results
    """
    results = []

    if search_mode in ("filename", "both"):
        # Search filenames
        files = [str(p.relative_to(vault_path)) for p in vault_path.rglob("*.md")]
        filename_matches = process.extract(query, files, limit=limit, scorer=fuzz.partial_ratio)

        for filename, score, _ in filename_matches:
            if score > score_threshold:
                results.append({
                    "filename": filename,
                    "score": score,
                    "match_type": "filename"
                })

    if search_mode in ("content", "both"):
        # Search file contents
        file_paths = list(vault_path.rglob("*.md"))
        content_matches = []

        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding='utf-8')
                # Get best matching line/snippet
                lines = content.split('\n')
                line_matches = process.extract(query, lines, limit=1, scorer=fuzz.partial_ratio)

                if line_matches and line_matches[0][1] > score_threshold:
                    relative_path = str(file_path.relative_to(vault_path))
                    content_matches.append({
                        "filename": relative_path,
                        "score": line_matches[0][1],
                        "match_type": "content",
                        "snippet": line_matches[0][0][:100]  # First 100 chars of matching line
                    })
            except Exception:
                continue

        # Sort by score and take top results
        content_matches.sort(key=lambda x: x["score"], reverse=True)
        results.extend(content_matches[:limit])

    # Sort all results by score
    results.sort(key=lambda x: x["score"], reverse=True)

    # Format output
    result = f"### Fuzzy Search Results (mode: {search_mode}):\n"

    if not results:
        result += "No matches found.\n"
    else:
        for match in results[:limit]:
            if match["match_type"] == "filename":
                result += f"- {match['filename']} (Score: {match['score']}, matched: filename)\n"
            else:
                result += f"- {match['filename']} (Score: {match['score']}, matched: content)\n"
                result += f"  Snippet: {match['snippet']}...\n"

    return result
