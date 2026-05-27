"""Small reusable helpers (hashing, JSON parsing, IDs, truncation)."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


SIMULATION_YEARS = [2026, 2027, 2028, 2029, 2030, 2031]


def new_run_id() -> str:
    return "run_" + uuid.uuid4().hex[:12]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(*parts: Any) -> str:
    """Deterministic hash for cache keys."""
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parse.

    Tolerates code fences and surrounding prose. Returns None if no JSON
    object can be recovered.
    """
    if text is None:
        return None
    candidates = []
    fenced = _FENCE_RE.findall(text)
    candidates.extend(fenced)
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = cand.find("{")
        end = cand.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cand[start : end + 1]
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def estimate_tokens(text: str) -> int:
    """Cheap heuristic: ~4 chars per token. Used only for metrics."""
    if not text:
        return 0
    return max(1, len(text) // 4)
