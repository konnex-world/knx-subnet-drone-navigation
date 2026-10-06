import time

import asyncio
import random
import json
import bittensor as bt

from typing import List

from template.protocol import ALLOWED_ACTION_IDS, ACTION_LABELS, expected_action_id


class MockWallet:
    """
    Minimal stand-in for tests / --mock. Public SDK 9.7 does not ship bt.MockWallet on the
    top-level module; only hotkey/coldkey with ss58 are required for MockSubtensor.
    """

    def __init__(self, config=None):
        self.hotkey = bt.Keypair.create_from_uri("//subnet-math-mock-hotkey")
        self.coldkey = bt.Keypair.create_from_uri("//subnet-math-mock-coldkey")

    def __str__(self) -> str:
        return f"MockWallet({self.hotkey.ss58_address})"


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)

        if not self.subnet_exists(netuid):
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                balance=100000,
                stake=100000,
            )

        # Register n mock neurons who will be miners
        for i in range(1, n + 1):
            self.force_register_neuron(
                netuid=netuid,
                hotkey=f"miner-hotkey-{i}",
                coldkey="mock-coldkey",
                balance=100000,
                stake=100000,
            )


class MockMetagraph(bt.Metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "127.0.0.0"
            axon.port = 8091

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")


class MockDendrite(bt.Dendrite):
    """
    Replaces a real bittensor network request with a mock request that just returns some static response for all axons that are passed and adds some random delay.
    """

    def __init__(self, wallet):
        super().__init__(wallet)

    async def forward(
        self,
        axons: List[bt.Axon],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")

        async def query_all_axons(streaming: bool):
            """Queries all axons for responses."""

            async def single_axon_response(i, axon):
                """Queries a single axon for a response."""

                start_time = time.time()
                s = synapse.copy()
                # Attach some more required data so it looks real
                s = self.preprocess_synapse_for_request(axon, s, timeout)
                # We just want to mock the response, so we'll just fill in some data
                process_time = random.random()
                if process_time < timeout:
                    s.dendrite.process_time = str(time.time() - start_time)
                    # TODO (developer): replace with your own expected synapse data
                    # Use the shared heuristic so the mock produces the same
                    # ground truth as the validator. The previous inline version
                    # used bare substring matching and ordered "up" before
                    # "photo", so "roundup" selected ascend and the mock trained
                    # miners toward the wrong flight action.
                    aid = expected_action_id(str(getattr(s, "instruction", "") or ""))
                    if aid not in ALLOWED_ACTION_IDS:
                        aid = 1
                    conf = max(0.0, min(1.0, 0.5 + random.uniform(-0.2, 0.2)))
                    s.action_id = aid
                    s.confidence = conf
                    s.miner_error = None
                    s.miner_response_json = json.dumps(
                        {
                            "mode": "mock",
                            "action_id": aid,
                            "label": ACTION_LABELS.get(aid, "forward"),
                            "confidence": conf,
                            "explain": "mock dendrite response",
                        },
                        ensure_ascii=False,
                    )
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                    synapse.dendrite.process_time = str(process_time)
                else:
                    s.action_id = None
                    s.confidence = 0.0
                    s.miner_error = "timeout"
                    s.miner_response_json = json.dumps({"ok": False, "error": "timeout"})
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"
                    synapse.dendrite.process_time = str(timeout)

                # Return the updated synapse object after deserializing if requested
                if deserialize:
                    return s.deserialize()
                else:
                    return s

            return await asyncio.gather(
                *(
                    single_axon_response(i, target_axon)
                    for i, target_axon in enumerate(axons)
                )
            )

        return await query_all_axons(streaming)

    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.

        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return "MockDendrite({})".format(self.keypair.ss58_address)
