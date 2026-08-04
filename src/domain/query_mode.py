"""Domain-level query mode parsing.

Recognised slash-command prefixes (e.g. ``/summary``) that control the answer
style.  ``parse_mode`` extracts the mode and returns the cleaned query.
"""

from __future__ import annotations

import re

QUERY_MODES: tuple[str, ...] = ("explain", "summary", "commands", "analyze")

_MODE_PATTERN = re.compile(
    r"^/(" + "|".join(QUERY_MODES) + r")\s+",
    re.IGNORECASE,
)


def parse_mode(query: str) -> tuple[str | None, str]:
    """Extract an optional ``/mode`` prefix from *query*.

    Returns ``(mode, cleaned_query)``.  If no mode prefix is found,
    *mode* is ``None`` and the query is returned unchanged.
    """
    m = _MODE_PATTERN.match(query)
    if m:
        mode = m.group(1).lower()
        return mode, query[m.end():].strip()
    return None, query
