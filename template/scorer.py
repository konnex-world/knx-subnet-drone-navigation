"""External / local scoring switch for drone-navigation validator rewards."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def scorer_mode() -> str:
    mode = (os.getenv("SCORER_MODE") or "hash").strip().lower()
    if mode not in {"hash", "api"}:
        raise ValueError(f"SCORER_MODE must be 'hash' or 'api', got {mode!r}")
    return mode


def call_scorer_api(payload: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    url = (os.getenv("SCORER_API") or "").strip()
    if not url:
        raise RuntimeError("SCORER_MODE=api requires SCORER_API (HTTP URL of the scorer service)")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SCORER_API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"SCORER_API unreachable: {exc}") from exc
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("SCORER_API must return a JSON object")
    return data
