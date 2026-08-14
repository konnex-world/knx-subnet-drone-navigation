#!/usr/bin/env python3
"""
HTTP wrapper around run_drone_probe: POST /v1/drone-probe, GET /health.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING — LOCAL SMOKE-TEST ONLY. NOT A PRODUCTION SERVICE.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

This process is an internal localnet / developer smoke helper. It was never
intended to be exposed on the public internet or to run as a long-lived prod
API. It loads a local Bittensor wallet and signs dendrite traffic.

Hardening (do not weaken without a security review):
  - Listens on 127.0.0.1 by default (override only via PROBE_BIND).
  - wallet_name / hotkey / chain_endpoint come ONLY from environment variables;
    JSON body cannot choose which key material or chain endpoint to use.
  - If PROBE_AUTH_TOKEN is set, POST requires Authorization: Bearer <token>.

Do not publish this port in Docker as 0.0.0.0:… — use 127.0.0.1:hostPort:… if
you must map it at all.

Run from /app root (example):

  PROBE_PORT=8091 NETUID=2 WALLET_NAME=validator \\
    PROBE_AUTH_TOKEN=dev-only python scripts/subnet_probe_http.py
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from subnet_probe_lib import run_drone_probe  # noqa: E402

# Client JSON must never supply these — they select key material / chain target.
_ENV_ONLY_KEYS = frozenset({"wallet_name", "hotkey", "chain_endpoint"})


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return float(v)


DEFAULTS = {
    "netuid": lambda: _env_int("NETUID", 1),
    "wallet_name": lambda: os.environ.get("WALLET_NAME", "validator"),
    "hotkey": lambda: os.environ.get("HOTKEY", "default"),
    "chain_endpoint": lambda: os.environ.get(
        "SUBTENSOR_CHAIN",
        "ws://127.0.0.1:9944",
    ),
    "sample_size": lambda: _env_int("PROBE_SAMPLE_SIZE", 4),
    "instruction": lambda: os.environ.get(
        "PROBE_INSTRUCTION",
        "Proceed toward the most salient building ahead.",
    ),
    "task_id": lambda: os.environ.get("PROBE_TASK_ID", "http-probe-task"),
    "timeout": lambda: _env_float("PROBE_TIMEOUT", 30.0),
}


def _merge_body(body: dict | None) -> dict:
    """Build probe kwargs. wallet / hotkey / chain_endpoint are env-only."""
    b = dict(body) if isinstance(body, dict) else {}
    rejected = sorted(k for k in _ENV_ONLY_KEYS if k in b)
    if rejected:
        raise ValueError(
            "refusing client-supplied "
            + ", ".join(rejected)
            + " — set WALLET_NAME / HOTKEY / SUBTENSOR_CHAIN in the environment only"
        )
    out = {}
    out["netuid"] = int(b.get("netuid", DEFAULTS["netuid"]()))
    out["wallet_name"] = str(DEFAULTS["wallet_name"]())
    out["hotkey"] = str(DEFAULTS["hotkey"]())
    out["chain_endpoint"] = str(DEFAULTS["chain_endpoint"]())
    out["sample_size"] = int(b.get("sample_size", DEFAULTS["sample_size"]()))
    out["instruction"] = str(b.get("instruction", DEFAULTS["instruction"]()))
    out["task_id"] = str(b.get("task_id", DEFAULTS["task_id"]()))
    out["timeout"] = float(b.get("timeout", DEFAULTS["timeout"]()))
    if "miner_uids" in b and b["miner_uids"] is not None:
        out["miner_uids"] = [int(x) for x in b["miner_uids"]]
    else:
        out["miner_uids"] = None
    return out


def _auth_token() -> str:
    return (os.environ.get("PROBE_AUTH_TOKEN") or "").strip()


def _bearer_ok(header_val: str | None) -> bool:
    expected = _auth_token()
    if not expected:
        return True
    if header_val is None:
        return False
    h = header_val.strip()
    if not h.lower().startswith("bearer "):
        return False
    supplied = h[7:].strip()
    return bool(supplied) and secrets.compare_digest(supplied, expected)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.client_address[0], self.log_date_time_string(), fmt % args)
        )

    def _send_json(self, status: int, obj: object) -> None:
        try:
            raw = json.dumps(
                obj,
                ensure_ascii=False,
                indent=2,
                default=str,
            ).encode("utf-8")
        except Exception as e:
            raw = json.dumps(
                {"ok": False, "error": f"json_encode: {e}", "type": type(e).__name__},
                ensure_ascii=False,
            ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/health", "/"):
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "subnet-drone-probe",
                    "warning": (
                        "local smoke-test only; not a production API; "
                        "binds loopback by default"
                    ),
                    "auth_required_for_post": bool(_auth_token()),
                    "routes": ["GET /health", "POST /v1/drone-probe"],
                },
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path != "/v1/drone-probe":
            self.send_error(404)
            return
        if not _bearer_ok(self.headers.get("Authorization")):
            self._send_json(
                401,
                {
                    "ok": False,
                    "error": "Unauthorized — set Authorization: Bearer <PROBE_AUTH_TOKEN>",
                },
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                text = (raw.decode("utf-8-sig") or "{}").strip()
                body = json.loads(text or "{}")
            except json.JSONDecodeError as e:
                self._send_json(400, {"ok": False, "error": f"invalid JSON: {e}"})
                return
            if not isinstance(body, dict):
                self._send_json(
                    400,
                    {"ok": False, "error": "JSON body must be an object"},
                )
                return
            try:
                kw = _merge_body(body)
            except ValueError as e:
                self._send_json(400, {"ok": False, "error": str(e)})
                return
            try:
                result = asyncio.run(run_drone_probe(**kw))
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                self._send_json(
                    500,
                    {"ok": False, "error": str(e), "type": type(e).__name__},
                )
                return
            status = 200 if result.get("ok") else 400
            self._send_json(status, result)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            try:
                self._send_json(
                    500,
                    {
                        "ok": False,
                        "error": str(e),
                        "type": type(e).__name__,
                    },
                )
            except Exception:
                pass


def main() -> None:
    port = _env_int("PROBE_PORT", 8091)
    # Default loopback for bare-metal. In Docker set PROBE_BIND=0.0.0.0 and
    # PROBE_ALLOW_ALL_INTERFACES=1, and publish the host port as 127.0.0.1:… only.
    bind = (os.environ.get("PROBE_BIND") or "127.0.0.1").strip() or "127.0.0.1"
    allow_all = (os.environ.get("PROBE_ALLOW_ALL_INTERFACES") or "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    )
    if bind in ("0.0.0.0", "::", "[::]") and not allow_all:
        print(
            "FATAL: refusing to bind PROBE_BIND="
            + bind
            + " without PROBE_ALLOW_ALL_INTERFACES=1. "
            "For bare metal use PROBE_BIND=127.0.0.1. "
            "For Docker: PROBE_BIND=0.0.0.0 PROBE_ALLOW_ALL_INTERFACES=1 "
            "and publish ports as 127.0.0.1:hostPort:containerPort only.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    print(
        "WARNING: subnet_probe_http is a LOCAL SMOKE-TEST only — "
        "not for production / public exposure.",
        file=sys.stderr,
        flush=True,
    )
    if bind in ("0.0.0.0", "::", "[::]"):
        print(
            "WARNING: listening on all interfaces inside this process/network "
            "namespace. Ensure the Docker host publish is 127.0.0.1:… "
            "(never 0.0.0.0:… on the host).",
            file=sys.stderr,
            flush=True,
        )
    if not _auth_token():
        print(
            "WARNING: PROBE_AUTH_TOKEN is unset — POST /v1/drone-probe has no auth. "
            "Set PROBE_AUTH_TOKEN for any shared or long-lived use.",
            file=sys.stderr,
            flush=True,
        )

    server = HTTPServer((bind, port), Handler)
    print(f"subnet-drone-probe listening on {bind}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
