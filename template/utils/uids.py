import random
import bittensor as bt
import numpy as np
from typing import FrozenSet, List


def parse_validator_axon_ports(s: str) -> FrozenSet[int]:
    raw = (s or "").strip()
    if not raw:
        return frozenset()
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return frozenset(out)


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph",
    uid: int,
    vpermit_tao_limit: int,
    validator_axon_ports: FrozenSet[int],
) -> bool:
    """Serving axons only; optional exclusion of validator axon ports (Docker 9101/9102)."""
    if not metagraph.axons[uid].is_serving:
        return False
    if validator_axon_ports:
        try:
            p = int(metagraph.axons[uid].port)
        except (TypeError, ValueError):
            p = None
        if p is not None and p in validator_axon_ports:
            return False
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    vports = parse_validator_axon_ports(
        getattr(self.config.neuron, "validator_axon_ports", "") or ""
    )
    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph,
            uid,
            self.config.neuron.vpermit_tao_limit,
            vports,
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = np.array(random.sample(available_uids, k))
    return uids
