"""
One dendrite pass with DroneNavSynapse over several UIDs + validator verification.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING — used by scripts/subnet_probe_http.py (LOCAL SMOKE-TEST ONLY).
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

run_drone_probe() loads a real local wallet and signs dendrite requests. Callers
must pass wallet_name / hotkey / chain_endpoint from trusted configuration
(environment), never from untrusted HTTP JSON. See subnet_probe_http.py.
"""

from __future__ import annotations

import random
import time
from typing import Any, List, Optional

import bittensor as bt

from template.protocol import DroneNavSynapse, DEFAULT_SYNTHETIC_INSTRUCTIONS
from template.validator.reward import get_rewards


def _pick_miner_uids(
    mg: bt.metagraph,
    miner_uids: Optional[List[int]],
    sample_size: int,
) -> List[int]:
    n = int(mg.n)
    if miner_uids is not None and len(miner_uids) > 0:
        uids = [int(u) for u in miner_uids if 0 <= int(u) < n]
        if not uids:
            raise ValueError("miner_uids empty or out of metagraph range")
        return uids
    k = max(1, min(int(sample_size), n))
    candidates = [
        u
        for u in range(n)
        if mg.axons[u].is_serving and not bool(mg.validator_permit[u])
    ]
    if len(candidates) < k:
        candidates = [u for u in range(n) if mg.axons[u].is_serving]
    if len(candidates) < k:
        candidates = list(range(n))
    k = min(k, len(candidates))
    return random.sample(candidates, k)


def _extract_float_response(item: Any) -> Optional[float]:
    # legacy helper kept for backwards compatibility with callers; not used for drone protocol
    try:
        return float(item)
    except Exception:
        return None


def _dendrite_meta(synapse: Any) -> Any:
    d = getattr(synapse, "dendrite", None)
    if d is None:
        return None
    meta: dict = {}
    for attr in ("status_code", "status_message", "process_time", "ip", "port"):
        if hasattr(d, attr):
            v = getattr(d, attr)
            try:
                meta[attr] = v
            except Exception:
                meta[attr] = str(v)
    return meta if meta else str(d)


def _axon_meta(mg: bt.metagraph, uid: int) -> dict:
    ax = mg.axons[uid]
    hk = getattr(ax, "hotkey", None)
    return {
        "ip": getattr(ax, "ip", None),
        "port": getattr(ax, "port", None),
        "is_serving": getattr(ax, "is_serving", None),
        "hotkey": str(hk) if hk is not None else None,
    }


def _synapse_payload(item: Any) -> Any:
    if item is None:
        return None
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        return {"deserialized_only": float(item)}
    out: dict = {
        "action_id": getattr(item, "action_id", None),
        "confidence": getattr(item, "confidence", None),
        "miner_error": getattr(item, "miner_error", None),
        "miner_response_json": getattr(item, "miner_response_json", None),
    }
    for k in ("instruction", "task_id", "synthetic_context_json"):
        if hasattr(item, k):
            out[k] = getattr(item, k)
    out["dendrite"] = _dendrite_meta(item)
    return out


def _hotkey_at(mg: bt.metagraph, uid: int) -> Optional[str]:
    try:
        h = mg.hotkeys[uid]
        return str(h)
    except Exception:
        return None


async def run_drone_probe(
    *,
    netuid: int,
    chain_endpoint: str,
    wallet_name: str,
    hotkey: str = "default",
    miner_uids: Optional[List[int]] = None,
    sample_size: int = 4,
    instruction: str = DEFAULT_SYNTHETIC_INSTRUCTIONS[0],
    task_id: str = "probe-task",
    timeout: float = 30.0,
) -> dict:
    t0 = time.perf_counter()

    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
    subtensor = bt.Subtensor(network=chain_endpoint)
    mg = subtensor.metagraph(netuid)

    uids_list = _pick_miner_uids(mg, miner_uids, sample_size)
    synapse = DroneNavSynapse(instruction=instruction, task_id=task_id)
    axons = [mg.axons[u] for u in uids_list]

    dendrite = bt.Dendrite(wallet)
    t_call = time.perf_counter()
    raw_out = None
    used_deserialize_true = False
    try:
        raw_out = await dendrite(
            axons,
            synapse=synapse,
            deserialize=False,
            timeout=timeout,
        )
    except TypeError:
        raw_out = await dendrite(
            axons,
            synapse=synapse,
            deserialize=True,
            timeout=timeout,
        )
        used_deserialize_true = True
    t_after = time.perf_counter()

    if raw_out is None:
        raw_list: List[Any] = []
    elif isinstance(raw_out, (list, tuple)):
        raw_list = list(raw_out)
    else:
        raw_list = [raw_out]

    miners: List[dict] = []
    for i, uid in enumerate(uids_list):
        item = raw_list[i] if i < len(raw_list) else None
        miners.append(
            {
                "uid": int(uid),
                "hotkey_ss58": _hotkey_at(mg, uid),
                "axon": _axon_meta(mg, uid),
                "synapse": _synapse_payload(item),
            }
        )

    rewards, verification = get_rewards(None, instruction=instruction, responses=raw_list)
    rew_list = [float(x) for x in rewards.tolist()]
    for i, r in enumerate(rew_list):
        if i < len(miners):
            miners[i]["reward_share"] = r
            miners[i]["verification"] = verification[i] if i < len(verification) else None

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    call_ms = int((t_after - t_call) * 1000)

    return {
        "ok": True,
        "protocol": "drone-nav DroneNavSynapse",
        "bittensor_sdk_pinned": "9.7.0 (see xsubnet-template/requirements.txt)",
        "netuid": netuid,
        "chain_endpoint": chain_endpoint,
        "wallet_coldkey": wallet_name,
        "dendrite_deserialize_flag": not used_deserialize_true,
        "request": {
            "instruction": instruction,
            "task_id": task_id,
        },
        "miner_uids_queried": uids_list,
        "miners": miners,
        "rewards": rew_list,
        "timing_ms": {"total": elapsed_ms, "dendrite_call": call_ms},
    }
