"""Internal helpers — shared across modules."""
from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating markdown fences."""
    s = text.strip()
    if s.startswith("```"):
        after = s[3:]
        if after.lstrip().lower().startswith("json"):
            after = after.lstrip()[4:]
        if "```" in after:
            after = after[: after.rfind("```")]
        s = after.strip()
    return json.loads(s)
