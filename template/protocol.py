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
import re
import typing

import bittensor as bt

from template.openfly_policy_io import ACTION_LABELS_DRONE_NAV

# Action space aligned with the existing OpenFly dashboard flow.
ACTION_LABELS: dict[int, str] = dict(ACTION_LABELS_DRONE_NAV)
ALLOWED_ACTION_IDS: typing.Tuple[int, ...] = tuple(sorted(ACTION_LABELS.keys()))

# --- Shared expected-action heuristic ---------------------------------------
#
# FLIGHT SAFETY: this heuristic decides the ground-truth action id used for the
# +0.60 correctness bonus, and under SCORER_MODE=hash (the shipped default) it is
# therefore the only source of truth for on-chain emission weights.
#
# It previously used bare substring tests (`"up" in t`), which mislabelled real
# instructions:
#   "Take a photo of the roundup ceremony."  -> matched "up" inside "roundup"
#                                              => ascend (4) instead of make_photo (10)
#   "Pick up the backup plan and download it." -> "up" in "pickup"/"backup",
#                                              "down" in "download"
#                                              => ascend (4) instead of forward (1)
#   "Group the shots and upload them."        -> "up" in "group"/"upload"
# Because the photo/snapshot check sat AFTER the up/down checks, the photo branch
# was unreachable for any instruction containing the substring "up".
#
# The implementation below matches whole words only and checks the more specific
# action labels before the directional ones. It lives here so the validator
# (template/validator/reward.py), the miner fallback (neurons/miner.py) and the
# mock dendrite (template/mock.py) all share one correct implementation instead of
# three divergent copies.
#
# Ordering below is deliberate: photo/snapshot and strafe are more specific than
# the bare up/down/left/right directions and must be evaluated first.
_EXPECTED_ACTION_KEYWORDS: typing.Tuple[typing.Tuple[typing.Tuple[str, ...], int], ...] = (
    # id 0: stop / hold / wait
    (("stop", "hold", "wait", "hover", "halt"), 0),
    # id 10: photo / snapshot must precede up/down (e.g. "take a photo on the rooftop")
    (("photo", "picture", "snapshot", "selfie"), 10),
    # id 6/7: strafes must precede left/right
    (("strafe left", "strafe-left", "sidestep left"), 6),
    (("strafe right", "strafe-right", "sidestep right"), 7),
    # id 2/3: turn left / turn right
    (("turn left", "rotate left", "yaw left", "bank left", "orbit left"), 2),
    (("turn right", "rotate right", "yaw right", "bank right", "orbit right"), 3),
    # id 4/5: vertical movement, whole words only
    (("ascend", "climb", "rise", "higher", "upward", "go up"), 4),
    (("descend", "lower", "fall", "downward", "go down"), 5),
    # id 12: return home, kept last so "return to base" is not shadowed above
    (("return to base", "return home", "rtb"), 12),
)

# Movement keywords that must NOT match when they are only part of an unrelated
# word. Word boundaries alone are not sufficient here, because these appear as
# standalone English words in non-flight contexts:
#   "Pick up the backup plan"  -> "up" is a real word, but not a flight command
#   "download the file"       -> "down" is part of a different verb
#   "upload the shots"        -> "up" starts a different compound
#   "pickup zone" / "backup"  -> "up" is embedded, already excluded by \b
#
# Each entry is a regex fragment matched against the lowercased instruction and
# must not contain its own word boundaries, so it composes with the \b wrapper.
_NON_FLIGHT_UP_PATTERNS: typing.Tuple[str, ...] = (
    r"pick\s+up",
    r"pickup",
    r"back\s*up",
    r"backup",
    r"upload",
    r"group",
    r"top\s*up",
    r"line\s*up",
    r"fill\s*up",
    r"up\s*load",
)

_NON_FLIGHT_DOWN_PATTERNS: typing.Tuple[str, ...] = (
    r"down\s*load",
    r"download",
    r"down\s*stairs",
    r"downtown",
    r"shut\s*down",
    r"break\s*down",
    r"down\s*time",
)

# Precompiled, since these run for every miner response.
_NON_FLIGHT_UP_RE = re.compile(
    r"\b(?:" + "|".join(_NON_FLIGHT_UP_PATTERNS) + r")\b", re.IGNORECASE
)
_NON_FLIGHT_DOWN_RE = re.compile(
    r"\b(?:" + "|".join(_NON_FLIGHT_DOWN_PATTERNS) + r")\b", re.IGNORECASE
)

# Bare direction keywords, evaluated only when no non-flight usage is present.
_UP_KEYWORDS: typing.Tuple[str, ...] = ("up",)
_DOWN_KEYWORDS: typing.Tuple[str, ...] = ("down",)


def expected_action_id(instruction: str) -> int:
    """Return the expected action id for a natural-language instruction.

    Matching is whole-word (and multi-word-phrase) based, so substrings inside
    unrelated words such as "roundup", "group" and "upload" no longer select a
    movement action.

    The bare direction words "up" and "down" are additionally suppressed when they
    appear in a known non-flight usage ("pick up", "download", "back up", ...).
    Word boundaries alone cannot catch these, because "Pick up the backup plan"
    contains "up" as a complete word yet does not describe a flight command.

    Unrecognised instructions default to 1 (forward), the safe neutral action for
    a drone.

    Args:
        instruction: The natural-language instruction shown to the miner.

    Returns:
        The expected action id, defaulting to 1 (forward) when nothing matches.
    """
    text = (instruction or "").lower()
    if not text.strip():
        return 1

    # Detect non-flight usages of the bare direction words before matching them.
    up_is_flight = not _NON_FLIGHT_UP_RE.search(text)
    down_is_flight = not _NON_FLIGHT_DOWN_RE.search(text)

    for keywords, action_id in _EXPECTED_ACTION_KEYWORDS:
        for keyword in keywords:
            # \b word boundaries prevent matching inside a longer word.
            # re.escape is used so multi-word keywords are matched literally.
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                return action_id

    # The bare direction words are evaluated last, and only when they are not
    # part of a recognised non-flight usage.
    if up_is_flight:
        for keyword in _UP_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                return 4
    if down_is_flight:
        for keyword in _DOWN_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                return 5

    return 1


DEFAULT_SYNTHETIC_INSTRUCTIONS: typing.Tuple[str, ...] = (
    "Proceed toward the most salient building ahead of the drone.",
    "Move forward and keep the rectangular gray building centered.",
    "Turn left and align with the nearest road corridor.",
    "Strafe right to avoid the obstacle and continue forward.",
    "Stop if the target landmark is reached or motion is unsafe.",
)


class DroneNavSynapse(bt.Synapse):
    """
    Subnet protocol for instruction mining and validator-side verification.

    Validator -> miner:
      - instruction: natural-language navigation instruction
      - task_id: deterministic per-round id
      - synthetic_context_json: optional context (pose/socket/meta)
      - frame_jpeg_b64: optional visual context

    Miner -> validator:
      - action_id: selected action id from ALLOWED_ACTION_IDS
      - confidence: model confidence [0..1], optional
      - miner_response_json: full structured response (competition/debug fields)
      - miner_error: non-empty when miner failed
    """

    version: str = "drone-nav-v1"
    instruction: str
    task_id: str
    synthetic_context_json: typing.Optional[str] = None
    frame_jpeg_b64: typing.Optional[str] = None

    action_id: typing.Optional[int] = None
    confidence: typing.Optional[float] = None
    miner_response_json: typing.Optional[str] = None
    miner_error: typing.Optional[str] = None

    def deserialize(self) -> typing.Dict[str, typing.Any]:
        raw = self.miner_response_json
        if not raw:
            return {
                "action_id": self.action_id,
                "confidence": self.confidence,
                "error": self.miner_error,
            }
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except (TypeError, ValueError):
            pass
        return {
            "action_id": self.action_id,
            "confidence": self.confidence,
            "error": self.miner_error,
            "raw": str(raw),
        }
