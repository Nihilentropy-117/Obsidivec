"""Bearer token authentication middleware for FastMCP."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def validate_bearer_token(expected_token: str, authorization_header: str | None) -> bool:
    """Validate a Bearer token from the Authorization header.

    Returns True if valid, False otherwise.
    """
    if not expected_token:
        # No token configured = auth disabled
        return True

    if not authorization_header:
        return False

    if not authorization_header.startswith("Bearer "):
        return False

    provided_token = authorization_header.removeprefix("Bearer ").strip()
    return provided_token == expected_token
