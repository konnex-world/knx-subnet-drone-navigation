# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import json
import os
import typing

import bittensor as bt
import numpy as np

from template.openfly_policy_io import canonicalize_exit_action_id
from template.protocol import ALLOWED_ACTION_IDS
from template.scorer import call_scorer_api, scorer_mode


def _expected_action_heuristic(instruction: str) -> int:
    t = (instruction or "").lower()
    if "stop" in t or "hold" in t or "wait" in t:
        return 0
    if "left" in t and "strafe" not in t:
        return 2
    if "right" in t and "strafe" not in t:
        return 3
    if "strafe left" in t:
        return 6
    if "strafe right" in t:
        return 7
    if "up" in t or "ascend" in t:
        return 4
    if "down" in t or "descend" in t:
        return 5
    if "photo" in t or "snapshot" in t:
        return 10
    return 1


def _score_single_response(
    instruction: str,
    response: typing.Any,
) -> tuple[float, dict[str, typing.Any]]:
    expected_action = _expected_action_heuristic(instruction)
    detail: dict[str, typing.Any] = {
        "expected_action_id": expected_action,
        "status_code": None,
        "valid_json": False,
        "action_id": None,
        "confidence": None,
        "score_raw": 0.0,
        "scorer_mode": scorer_mode(),
    }
    if response is None:
        detail["error"] = "none_response"
        return 0.0, detail
    dendrite = getattr(response, "dendrite", None)
    if dendrite is not None:
        code = getattr(dendrite, "status_code", None)
        detail["status_code"] = int(code) if code is not None else None
        if code is not None and int(code) != 200:
            detail["error"] = f"status_{code}"
            return 0.0, detail

    raw = getattr(response, "miner_response_json", None)
    if not raw:
        detail["error"] = "empty_miner_response_json"
        return 0.0, detail
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        detail["error"] = "bad_json"
        return 0.0, detail
    if not isinstance(obj, dict):
        detail["error"] = "json_not_object"
        return 0.0, detail
    detail["valid_json"] = True
    try:
        aid = canonicalize_exit_action_id(int(obj.get("action_id")))
    except (TypeError, ValueError):
        detail["error"] = "bad_action_id"
        return 0.05, detail
    detail["action_id"] = aid
    conf_raw = obj.get("confidence", 0.0)
    try:
        conf = max(0.0, min(1.0, float(conf_raw)))
    except (TypeError, ValueError):
        conf = 0.0
    detail["confidence"] = conf

    mode = detail["scorer_mode"]
    if mode == "api":
        try:
            data = call_scorer_api(
                {
                    "kind": "drone_navigation",
                    "instruction": instruction,
                    "action_id": aid,
                    "confidence": conf,
                    "expected_action_id_heuristic": expected_action,
                    "miner_response": obj,
                },
                timeout=float(os.getenv("SCORER_TIMEOUT_SECONDS", "30")),
            )
            score = float(data["score"])
            detail["score_raw"] = round(score, 4)
            detail["scorer_api"] = True
            return score, detail
        except Exception as exc:  # noqa: BLE001
            detail["error"] = f"scorer_api_failed:{exc}"
            return 0.0, detail

    # SCORER_MODE=hash: keyword heuristic match (local beta proxy).
    score = 0.10  # minimal reward for parseable output
    if aid in ALLOWED_ACTION_IDS:
        score += 0.20
    if aid == expected_action:
        score += 0.60
    else:
        score += 0.10
    score += 0.10 * conf
    detail["score_raw"] = round(score, 4)
    return score, detail


def get_rewards(
    self,
    instruction: str,
    responses: typing.List[typing.Any],
) -> tuple[np.ndarray, list[dict[str, typing.Any]]]:
    """
    Validator-side verification of mined instruction responses.

    SCORER_MODE=hash (default): heuristic keyword match + confidence.
    SCORER_MODE=api: POST each response to SCORER_API and use returned `score`.
    """
    raw_scores: list[float] = []
    details: list[dict[str, typing.Any]] = []
    for r in responses:
        s, d = _score_single_response(instruction, r)
        raw_scores.append(float(s))
        details.append(d)
    arr = np.array(raw_scores, dtype=np.float32)
    ssum = float(np.sum(arr))
    if ssum <= 1e-9:
        bt.logging.warning("No valid miner responses for reward.")
        return np.zeros(len(raw_scores), dtype=np.float32), details
    arr = arr / ssum
    return arr.astype(np.float32), details
