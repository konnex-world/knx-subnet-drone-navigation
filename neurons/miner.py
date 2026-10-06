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

import base64
import json
import os
import random
import sys
import time
import typing
import urllib.error
import urllib.request
from pathlib import Path

import bittensor as bt

import template
from template.base.miner import BaseMinerNeuron
from template.openfly_policy_io import (
    canonicalize_exit_action_id,
    normalize_user_instruction,
    structured_explain_discrete,
)
from template.protocol import ACTION_LABELS, ALLOWED_ACTION_IDS, expected_action_id


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name, str(default)) or str(default)).strip())
    except ValueError:
        return default


def _resolve_miner_model_mode() -> str:
    raw = (os.environ.get("OPENFLY_SUBNET_MINER_MODEL", "openai") or "openai").strip().lower()
    if raw == "openfly":
        return "openfly"
    if raw != "openai":
        bt.logging.warning(
            f"OPENFLY_SUBNET_MINER_MODEL={raw!r} is not openai|openfly; using openai"
        )
    return "openai"


class Miner(BaseMinerNeuron):
    """
    Policy backends (``OPENFLY_SUBNET_MINER_MODEL``):

    - ``openai`` — Chat Completions in-process (urllib), no PyTorch.
    - ``openfly`` — local HuggingFace VLM using ``OpenFly-Platform/train/eval.py:get_action`` (same stack
      as the parent repo ``docker/openfly-dashboard`` image). Requires the **CUDA miner** Dockerfile
      (``docker/subnet-miner/Dockerfile``), GPU, and ``OPENFLY_MODEL`` weights mounted at runtime.

    This class inherits from BaseMinerNeuron → BaseNeuron (wallet, subtensor, metagraph, config).

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self._policy_backend = _resolve_miner_model_mode()
        self._openfly_policy = None
        self._openfly_processor = None
        self._openfly_get_action = None
        self._openfly_load_error: str | None = None
        bt.logging.info(f"OPENFLY_SUBNET_MINER_MODEL -> policy backend: {self._policy_backend}")
        if self._policy_backend == "openfly":
            self._ensure_openfly_local_model()

    def _openai_key(self) -> str:
        return os.environ.get("OPENAI_API_TOKEN", "").strip()

    def _openai_model(self) -> str:
        return (
            os.environ.get("OPENFLY_SUBNET_MINER_OPENAI_MODEL", "").strip()
            or os.environ.get("OPENAI_GPT_POLICY_MODEL", "").strip()
            or "gpt-4.1-mini"
        )

    def _expected_action_heuristic(self, instruction: str) -> int:
        """Return the expected action id for an instruction.

        Delegates to the shared implementation in template.protocol.

        FLIGHT SAFETY: the previous local copy used bare substring matching, so
        "Take a photo of the roundup ceremony." resolved to ascend because "up"
        appears inside "roundup". The miner then trained toward the wrong flight
        action and disagreed with the validator's ground truth.
        """
        return expected_action_id(instruction)

    def _rule_based_candidate(self, instruction: str, *, tag: str) -> dict[str, typing.Any]:
        ins = normalize_user_instruction(instruction)
        aid = self._expected_action_heuristic(ins)
        lab = ACTION_LABELS.get(aid, f"unknown_{aid}")
        return {
            "action_id": aid,
            "label": lab,
            "confidence": 0.55,
            "explain": structured_explain_discrete(
                backend="heuristic",
                instruction=ins,
                action_id=aid,
                label_semantic=lab,
                note=tag,
            ),
        }

    def _call_openai_candidate(
        self,
        *,
        instruction: str,
        synthetic_context_json: str | None,
        frame_jpeg_b64: str | None,
        temperature: float,
    ) -> dict[str, typing.Any]:
        ins = normalize_user_instruction(instruction)
        key = self._openai_key()
        if not key:
            return self._rule_based_candidate(ins, tag=f"temp={temperature:.1f}")
        url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        sys_prompt = (
            "You are an OpenFly drone policy miner. Return strict JSON object only with keys: "
            "action_id (int), confidence (0..1), explain (short string). "
            f"Allowed action_id values: {list(ALLOWED_ACTION_IDS)}."
        )
        user_blob = {
            "instruction": ins,
            "synthetic_context_json": synthetic_context_json or "",
            "has_frame": bool(frame_jpeg_b64),
        }
        messages: list[dict[str, typing.Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_blob, ensure_ascii=False)},
        ]
        if frame_jpeg_b64:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Optional vision context frame."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame_jpeg_b64}"},
                        },
                    ],
                }
            )
        body = {
            "model": self._openai_model(),
            "temperature": float(temperature),
            "max_tokens": 220,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            raw = (
                payload.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            obj = json.loads(raw) if raw else {}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as e:
            bt.logging.warning(f"openai miner candidate failed: {type(e).__name__}: {e}")
            return self._rule_based_candidate(ins, tag=f"temp={temperature:.1f}:fallback")
        aid = canonicalize_exit_action_id(
            int(obj.get("action_id", self._expected_action_heuristic(ins)))
        )
        if aid not in ALLOWED_ACTION_IDS:
            aid = self._expected_action_heuristic(ins)
        conf_raw = obj.get("confidence", 0.5)
        try:
            conf = max(0.0, min(1.0, float(conf_raw)))
        except (TypeError, ValueError):
            conf = 0.5
        exp = str(obj.get("explain", "") or "").strip()[:500]
        lab = ACTION_LABELS.get(aid, f"unknown_{aid}")
        if exp:
            explain_out = exp
        else:
            explain_out = structured_explain_discrete(
                backend="openai",
                instruction=ins,
                action_id=aid,
                label_semantic=lab,
                note="model returned empty explain",
            )
        return {
            "action_id": aid,
            "label": lab,
            "confidence": conf,
            "explain": explain_out,
        }

    def _openfly_train_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "OpenFly-Platform" / "train"

    def _ensure_openfly_local_model(self) -> bool:
        if self._openfly_load_error is not None:
            return False
        if self._openfly_policy is not None and self._openfly_processor is not None:
            return True
        train = self._openfly_train_dir()
        if not train.is_dir():
            self._openfly_load_error = f"missing OpenFly train dir: {train}"
            bt.logging.error(self._openfly_load_error)
            return False
        try:
            import numpy as np
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            if str(train) not in sys.path:
                sys.path.insert(0, str(train))
            from eval import get_action  # type: ignore  # noqa: E402

            model_id = os.environ.get("OPENFLY_MODEL", "IPEC-COMMUNITY/openfly-agent-7b").strip()
            hf_tok = os.environ.get("HF_TOKEN", "").strip() or None
            attn = os.environ.get("OPENFLY_ATTN_IMPLEMENTATION", "eager").strip()
            load_kw: dict[str, typing.Any] = dict(
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )
            if hf_tok:
                load_kw["token"] = hf_tok
            bt.logging.info(f"openfly: loading VLM from {model_id!r} …")
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, token=hf_tok
            )
            try:
                policy = AutoModelForVision2Seq.from_pretrained(
                    model_id, attn_implementation=attn, **load_kw
                )
            except Exception as e:
                bt.logging.warning(f"openfly: retry AutoModel with eager attn ({e!r})")
                policy = AutoModelForVision2Seq.from_pretrained(
                    model_id, attn_implementation="eager", **load_kw
                )
            policy = policy.to("cuda:0")
            self._openfly_processor = processor
            self._openfly_policy = policy
            self._openfly_get_action = get_action
            bt.logging.info("openfly: VLM loaded on cuda:0")
            _ = np  # noqa: F841 — keep import used for type/runtime consistency
            return True
        except Exception as e:
            self._openfly_load_error = f"{type(e).__name__}: {e}"
            bt.logging.error(f"openfly: failed to load VLM: {self._openfly_load_error}")
            return False

    def _decode_frame_or_blank(self, frame_jpeg_b64: str | None) -> typing.Any:
        import numpy as np
        import cv2

        if frame_jpeg_b64:
            try:
                raw = base64.standard_b64decode(frame_jpeg_b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except (ValueError, OSError, TypeError) as e:
                bt.logging.warning(f"openfly: bad frame_jpeg_b64 ({type(e).__name__}), using blank frame")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _call_openfly_local_candidate(
        self,
        *,
        instruction: str,
        frame_jpeg_b64: str | None,
    ) -> dict[str, typing.Any]:
        ins = normalize_user_instruction(instruction)
        if not self._ensure_openfly_local_model():
            return self._rule_based_candidate(ins, tag=f"openfly:{self._openfly_load_error or 'load_failed'}")
        assert self._openfly_policy is not None
        assert self._openfly_processor is not None
        assert self._openfly_get_action is not None
        bgr = self._decode_frame_or_blank(frame_jpeg_b64)
        try:
            aid_raw = int(
                self._openfly_get_action(
                    self._openfly_policy,
                    self._openfly_processor,
                    [bgr],
                    ins,
                    [],
                    if_his=True,
                    his_step=2,
                )
            )
        except Exception as e:
            bt.logging.warning(f"openfly get_action failed: {type(e).__name__}: {e}")
            return self._rule_based_candidate(ins, tag="openfly:get_action_error")
        aid = canonicalize_exit_action_id(aid_raw)
        if aid not in ALLOWED_ACTION_IDS:
            aid = self._expected_action_heuristic(ins)
        lab = ACTION_LABELS.get(aid, f"unknown_{aid}")
        explain_out = structured_explain_discrete(
            backend="openfly",
            instruction=ins,
            action_id=aid,
            label_semantic=lab,
            note="local VLM",
        )
        return {
            "action_id": aid,
            "label": lab,
            "confidence": 0.82,
            "explain": explain_out,
        }

    def _mine_instruction_openai(
        self,
        instruction: str,
        synthetic_context_json: str | None,
        frame_jpeg_b64: str | None,
    ) -> dict[str, typing.Any]:
        temps = (0.2, 0.8, 0.8)
        miners: list[dict[str, typing.Any]] = []
        for i, t in enumerate(temps):
            cand = self._call_openai_candidate(
                instruction=instruction,
                synthetic_context_json=synthetic_context_json,
                frame_jpeg_b64=frame_jpeg_b64,
                temperature=t,
            )
            cand["miner_index"] = i
            cand["temperature"] = t
            miners.append(cand)
        winner = max(miners, key=lambda x: float(x.get("confidence", 0.0)))
        return {
            "mode": "competition",
            "winner_miner_index": int(winner["miner_index"]),
            "miners": miners,
            "action_id": int(winner["action_id"]),
            "label": str(winner["label"]),
            "confidence": float(winner.get("confidence", 0.0)),
            "explain": str(winner.get("explain", "")),
        }

    def _mine_instruction_openfly(
        self,
        instruction: str,
        synthetic_context_json: str | None,
        frame_jpeg_b64: str | None,
    ) -> dict[str, typing.Any]:
        _ = synthetic_context_json
        cand = self._call_openfly_local_candidate(
            instruction=instruction,
            frame_jpeg_b64=frame_jpeg_b64,
        )
        return {
            "mode": "openfly",
            "action_id": int(cand["action_id"]),
            "label": str(cand["label"]),
            "confidence": float(cand.get("confidence", 0.0)),
            "explain": str(cand.get("explain", "")),
            "openfly": cand,
        }

    def _mine_instruction(
        self,
        instruction: str,
        synthetic_context_json: str | None,
        frame_jpeg_b64: str | None,
    ) -> dict[str, typing.Any]:
        if self._policy_backend == "openfly":
            return self._mine_instruction_openfly(
                instruction, synthetic_context_json, frame_jpeg_b64
            )
        return self._mine_instruction_openai(
            instruction, synthetic_context_json, frame_jpeg_b64
        )

    async def forward(
        self, synapse: template.protocol.DroneNavSynapse
    ) -> template.protocol.DroneNavSynapse:
        instruction = str(synapse.instruction or "").strip()
        if not instruction:
            synapse.miner_error = "empty instruction"
            synapse.action_id = None
            synapse.confidence = 0.0
            synapse.miner_response_json = json.dumps(
                {"ok": False, "error": "empty instruction"},
                ensure_ascii=False,
            )
            return synapse
        pack = self._mine_instruction(
            instruction=instruction,
            synthetic_context_json=synapse.synthetic_context_json,
            frame_jpeg_b64=synapse.frame_jpeg_b64,
        )
        synapse.action_id = int(pack["action_id"])
        synapse.confidence = float(pack["confidence"])
        synapse.miner_error = None
        synapse.miner_response_json = json.dumps(pack, ensure_ascii=False)
        bt.logging.info(
            f"DRONE_MINER task={synapse.task_id} action_id={synapse.action_id} "
            f"conf={float(synapse.confidence or 0.0):.2f}"
        )
        return synapse

    async def blacklist(
        self, synapse: template.protocol.DroneNavSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.DroneNavSynapse): A synapse object constructed from request headers.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: template.protocol.DroneNavSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.DroneNavSynapse): The synapse object with incoming request metadata.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
