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

import asyncio
import copy
import json
import os
import time

import bittensor as bt

from template.validator.reward import get_rewards
from template.validator.evidence import (
    build_drone_evidence_bundle,
    mirror_evidence_to_control_plane,
    publish_evidence_remark,
)
from template.validator.synthetic_context import build_synthetic_drone_nav_synapse
from template.validator.synthetic_debug import (
    is_enabled as synthetic_debug_enabled,
    new_run_dir,
    write_round_artifacts,
)
from template.validator.ue_synthetic import maybe_teleport_and_frame
from template.utils.uids import get_random_uids

_BIND_LOCAL_AXON_IPS = frozenset({"0.0.0.0", "::", "[::]"})


def _ue_synthetic_timeout_sec() -> float:
    raw = (os.environ.get("OPENFLY_SYNTHETIC_UE_TOTAL_TIMEOUT_SEC") or "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            bt.logging.warning(f"Bad OPENFLY_SYNTHETIC_UE_TOTAL_TIMEOUT_SEC={raw!r}; using default")
    try:
        settle = float(os.environ.get("OPENFLY_SYNTHETIC_POST_TELEPORT_SLEEP_SEC", "10") or "10")
    except ValueError:
        settle = 10.0
    return max(30.0, settle + 45.0)


def _axons_for_dendrite(validator_self, miner_uids):
    """Route dendrite to miners when running in Docker / network_mode: service:openfly-ue.

    - Chain axon ip ``0.0.0.0`` (or ::): replace with OPENFLY_VALIDATOR_MINER_AXON_HOST.
    - Same public IP as ``dendrite.external_ip``: bittensor maps that to ``0.0.0.0:port`` (loopback), which
      does not reach a host-published miner from this netns — replace with the same host.
    """
    metagraph = validator_self.metagraph
    fallback = (os.environ.get("OPENFLY_VALIDATOR_MINER_AXON_HOST") or "").strip()
    ext_ip = str(getattr(validator_self.dendrite, "external_ip", "") or "").strip()
    out = []
    for uid in miner_uids:
        axon = metagraph.axons[uid]
        if not fallback:
            out.append(axon)
            continue
        ip = getattr(axon, "ip", None)
        ip_s = str(ip).strip() if ip is not None else ""
        bind_like = ip is None or ip_s in _BIND_LOCAL_AXON_IPS
        same_public = bool(ext_ip) and ip_s == ext_ip
        if bind_like or same_public:
            patched = copy.copy(axon)
            patched.ip = fallback
            out.append(patched)
        else:
            out.append(axon)
    return out


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    synapse, synthetic_context = build_synthetic_drone_nav_synapse(
        validator_step=int(getattr(self, "step", 0)),
    )
    # UnrealCV client is synchronous and can wedge on a stale UDS/TCP connection.
    # Keep the validator loop alive: a stuck UE round should degrade one round, not block the subnet.
    ue_timeout = _ue_synthetic_timeout_sec()
    try:
        ue_extra = await asyncio.wait_for(
            asyncio.to_thread(maybe_teleport_and_frame, synapse),
            timeout=ue_timeout,
        )
    except asyncio.TimeoutError:
        bt.logging.error(f"UE synthetic teleport/capture timed out after {ue_timeout:.1f}s")
        ue_extra = {
            "ue_synthetic_ok": False,
            "ue_synthetic_error": f"timeout after {ue_timeout:.1f}s",
            "ue_synthetic_timeout_sec": ue_timeout,
        }
    try:
        merged = json.loads(synapse.synthetic_context_json or "{}")
        if isinstance(merged, dict):
            synthetic_context = merged
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    if isinstance(ue_extra, dict) and ue_extra.get("ue_synthetic_ok") is False:
        synthetic_context = {**synthetic_context, **ue_extra}
    instruction = str(synapse.instruction)

    axons = _axons_for_dendrite(self, miner_uids)
    responses = await self.dendrite(
        axons=axons,
        synapse=synapse,
        deserialize=False,
        timeout=float(self.config.neuron.timeout),
    )
    rewards, details = get_rewards(self, instruction=instruction, responses=responses)

    scoreboard = {
        "task_id": str(synapse.task_id),
        "instruction": instruction,
        "synthetic_context": synthetic_context,
        "uids": [int(u) for u in miner_uids],
        "responses": [
            {
                "action_id": getattr(r, "action_id", None),
                "confidence": getattr(r, "confidence", None),
                "error": getattr(r, "miner_error", None),
            }
            for r in responses
        ],
        "verification": details,
        "rewards": [float(x) for x in rewards.tolist()],
    }
    bt.logging.info("DRONE_SCOREBOARD " + json.dumps(scoreboard, ensure_ascii=False))

    run_dir = None
    if synthetic_debug_enabled():
        try:
            run_dir = new_run_dir()
            write_round_artifacts(
                run_dir,
                synapse=synapse,
                synthetic_context=dict(synthetic_context) if isinstance(synthetic_context, dict) else {},
                miner_uids=miner_uids,
                axons=axons,
                responses=responses,
                scoreboard=scoreboard,
                validator_self=self,
                ue_extra=ue_extra if isinstance(ue_extra, dict) else None,
            )
        except Exception as e:
            bt.logging.warning(f"VALIDATOR_SYNTHETIC_DEBUG dump failed: {e!r}")

    try:
        bundle, ev_hash = build_drone_evidence_bundle(
            validator_self=self,
            synapse=synapse,
            synthetic_context=dict(synthetic_context) if isinstance(synthetic_context, dict) else {},
            miner_uids=[int(u) for u in miner_uids],
            responses=list(responses),
            scoreboard=scoreboard,
            run_dir=run_dir,
        )
        remark_result = publish_evidence_remark(
            self,
            evidence_hash_value=ev_hash,
            job_id=str(synapse.task_id),
        )
        mirror_evidence_to_control_plane(
            validator_self=self,
            bundle=bundle,
            evidence_remark=remark_result,
        )
        bt.logging.info(
            "AI_EVIDENCE "
            + json.dumps(
                {
                    "job_id": str(synapse.task_id),
                    "evidence_hash": ev_hash,
                    "evidence_extrinsic_hash": remark_result.get("extrinsic_hash"),
                    "remark_ok": remark_result.get("ok"),
                    "remark_error": remark_result.get("error"),
                },
                ensure_ascii=False,
            )
        )
    except Exception as e:
        bt.logging.warning(f"AI evidence publish failed: {type(e).__name__}: {e}")

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
    time.sleep(float(self.config.neuron.forward_sleep))
