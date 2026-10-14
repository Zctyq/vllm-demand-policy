"""Strict request/session identity decoder; no eviction policy."""

from __future__ import annotations

import base64
import json
import re
from typing import Any


def parse_request_id(request_id: str) -> dict[str, Any] | None:
    """Decode the replay marker while allowing vLLM's ID prefix and suffix."""
    marker = re.search(r"pe1\.([A-Za-z0-9_-]+)", request_id)
    if marker is None:
        return None
    encoded = marker.group(1)
    # Completions add -<prompt-index>; InputProcessor then adds -<8 hex>.
    for length in range(len(encoded), max(0, len(encoded) - 128), -1):
        candidate = encoded[:length]
        try:
            raw = base64.urlsafe_b64decode(candidate + "=" * (-len(candidate) % 4))
            value = json.loads(raw)
            if (
                isinstance(value, dict)
                and set(value) == {"r", "s", "c"}
                and base64.urlsafe_b64encode(raw).decode().rstrip("=") == candidate
                and isinstance(value.get("r"), str)
                and isinstance(value.get("s"), str)
                and isinstance(value.get("c"), int)
                and not isinstance(value["c"], bool)
                and value["c"] >= 0
            ):
                return {
                    "run_id": value["r"],
                    "session_id": value["s"],
                    "call_index": value["c"],
                    "request_id": "pe1." + candidate,
                }
        except (ValueError, UnicodeError, TypeError):
            continue
    return None
