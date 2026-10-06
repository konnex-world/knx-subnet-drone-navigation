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
import math
import os
import typing

import bittensor as bt
import numpy as np

from template.openfly_policy_io import canonicalize_exit_action_id
from template.protocol import ALLOWED_ACTION_IDS, expected_action_id
from template.scorer import call_scorer_api, scorer_mode

# The scorer contract is a normalized score in [0, 1]. Values outside that range
# are clamped rather than rejected outright so a slightly-out-of-range scorer
# still contributes, while non-finite values are always rejected because they
# would poison the whole round during normalization (see get_rewards).
SCORER_SCORE_MIN = 0.0
SCORER_SCORE_MAX = 1.0

# Maximum raw score a hash-mode response can earn when it fails to pick a valid
# action, or picks a valid action that does not match the expected one. Without
# this cap, round normalization handed a confidently wrong miner roughly 29% of
# the round's emissions, which is unsafe for a flight-safety subnet.
WRONG_ACTION_SCORE_CAP = 0.05


def _coerce_scorer_score(raw: typing.Any) -> float:
    """Validate and clamp a score returned by the external SCORER_API.

    SECURITY/INTEGRITY: the scorer API is an untrusted trust boundary. A raw
    `float(data["score"])` accepted "NaN", "Infinity", booleans, and arbitrary
    magnitudes, all of which corrupt the reward round: NaN propagates to every
    miner's normalized reward, +inf zeroes out every peer with no diagnostic, and
    a negative score reaches `update_scores` and later raises on set_weights.

    Raises:
        ValueError: if the value is missing, non-numeric, boolean, or non-finite.
    """
    if isinstance(raw, bool):
        # bool is a subclass of int; True would otherwise score as 1.0.
        raise ValueError(f"scorer score must be numeric, got bool {raw!r}")
    if raw is None:
        raise ValueError("scorer response is missing the 'score' field")
    try:
        score = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scorer score is not numeric: {raw!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"scorer returned a non-finite score: {raw!r}")
    # Clamp to the documented [0, 1] contract so an out-of-range value can never
    # make the normalized reward negative or unbounded.
    return max(SCORER_SCORE_MIN, min(SCORER_SCORE_MAX, score))


def _expected_action_heuristic(instruction: str) -> int:
    """Return the expected action id for an instruction.

    Delegates to the single shared implementation in template.protocol so the
    validator, the miner fallback and the mock dendrite cannot drift apart.

    FLIGHT SAFETY: the previous local copy used bare substring matching, which
    mapped "Take a photo of the roundup ceremony." to ascend because "up" appears
    inside "roundup", and rewarded a miner for commanding the wrong flight action.
    See template.protocol.expected_action_id for the full rationale.
    """
    return expected_action_id(instruction)


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
            score = _coerce_scorer_score(data.get("score"))
            detail["score_raw"] = round(score, 4)
            detail["scorer_api"] = True
            return score, detail
        except Exception as exc:  # noqa: BLE001
            detail["error"] = f"scorer_api_failed:{exc}"
            return 0.0, detail

    # SCORER_MODE=hash: keyword heuristic match (local beta proxy).
    score = 0.10  # minimal reward for parseable output
    is_valid_action = aid in ALLOWED_ACTION_IDS
    is_correct_action = aid == expected_action
    if is_valid_action:
        score += 0.20
    if is_correct_action:
        score += 0.60
    else:
        score += 0.10
    score += 0.10 * conf

    # Absolute quality gate. Because get_rewards normalizes the round to sum to
    # 1.0, a wrong action previously retained roughly 29% of the emissions: a
    # confidently wrong or malformed flight command was economically almost
    # viable. That is unsafe for a flight-safety subnet, where commanding the
    # wrong action is materially worse than returning nothing usable.
    # Only a response that names a valid action AND matches the expected action
    # keeps its score; anything else is capped near zero.
    if not (is_valid_action and is_correct_action):
        score = min(score, WRONG_ACTION_SCORE_CAP)
        detail["score_gated"] = True
        detail["gate_reason"] = (
            "invalid_action" if not is_valid_action else "action_mismatch"
        )

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

    INTEGRITY: the returned array is guaranteed to be finite and non-negative.
    A single non-finite or out-of-range score must never be able to poison the
    whole round -- previously one NaN turned every miner's reward into NaN, and
    one +inf silently zeroed every honest peer while still marking their nonce as
    consumed.
    """
    raw_scores: list[float] = []
    details: list[dict[str, typing.Any]] = []
    for r in responses:
        s, d = _score_single_response(instruction, r)
        raw_scores.append(float(s))
        details.append(d)

    # Defence in depth: _coerce_scorer_score already rejects non-finite values at
    # the API boundary, but hash-mode and any future scorer path must not be able
    # to poison the round either. float64 is used for accumulation because
    # float32 loses precision once individual scores are small.
    arr = np.asarray(raw_scores, dtype=np.float64)
    if arr.size:
        if not np.all(np.isfinite(arr)):
            bt.logging.warning(
                "Non-finite reward scores received; coercing those responses to 0.0."
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp to the documented [0, 1] score contract. This guarantees the
        # emitted reward is never negative, which would otherwise flow into the
        # validator's moving average and later raise on set_weights.
        arr = np.clip(arr, 0.0, 1.0)

    ssum = float(np.sum(arr)) if arr.size else 0.0
    if not math.isfinite(ssum) or ssum <= 1e-9:
        bt.logging.warning("No valid miner responses for reward.")
        return np.zeros(len(raw_scores), dtype=np.float32), details
    arr = arr / ssum
    return arr.astype(np.float32), details
