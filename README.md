# Konnex Drone Navigation Subnet

Konnex drone-navigation runtime package with:
- `subnet-miner`
- `subnet-validator`
- UnrealEngine sidecar (`openfly-ue`)
- one-shot assets bootstrap (`assets-init`)

**Default:** root **`docker-compose.yml`** includes validator + miner stacks so **`docker compose up -d --build`** starts everything in one project (no orphan containers). Split files **`docker-compose.validator.yml`** / **`docker-compose.miner.yml`** stay for partial runs.

## WARNING: `scripts/subnet_probe_http.py` is a local smoke-test only

**Do not run this as a production / internet-facing service.**

`POST /v1/drone-probe` is an internal developer helper: one dendrite pass against localnet miners so you can smoke-test without a full validator loop. It loads a real wallet from disk and signs traffic.

- **Not part of the default compose stack** in this repo (miner / validator / UE only).
- When used in a localnet Docker setup elsewhere, the host port must be published as **`127.0.0.1:…` only** — never `0.0.0.0`.
- The script binds **`127.0.0.1` by default** and **refuses** `PROBE_BIND=0.0.0.0`.
- `wallet_name` / `hotkey` / `chain_endpoint` are taken **only from env** (`WALLET_NAME`, `HOTKEY`, `SUBTENSOR_CHAIN`); the JSON body cannot choose them.
- Set **`PROBE_AUTH_TOKEN`** and send `Authorization: Bearer …` on POST for any shared machine.

See the big warning banner at the top of `scripts/subnet_probe_http.py` and `scripts/subnet_probe_lib.py`.

## Environment (set this first)

Create `.env` from the template (this file is **only for this repo**; it does not replace `.env` in the parent `drone-navigation` tree):

```bash
cd xsubnet-template
cp .env.example .env
```

**Everyone must set (chain + identities):**
- `SUBTENSOR_CHAIN_ENDPOINT` — WebSocket RPC (e.g. `ws://127.0.0.1:9944` for localnet).
- `NETUID` — subnet id you are registered on.
- `MINER_WALLET_NAME`, `MINER_WALLET_HOTKEY`, `MINER_AXON_PORT` — miner axon identity.
- `VALIDATOR_WALLET_NAME`, `VALIDATOR_WALLET_HOTKEY`, `VALIDATOR_AXON_PORT` — validator axon identity.

**Wallets (project-local, no JSON paths in `.env`):** `MINER_*` / `VALIDATOR_*` are the **coldkey name** and **hotkey name** from `btcli` (e.g. `btcli wallet new_coldkey --wallet.name miner`). The SDK loads `coldkey` / `hotkeys/` files from a wallet **tree** on disk; you only pass those logical names in `.env`, not paths to `keyfile.json`.

This template keeps keys next to the repo under **`./wallets/`** (listed in **`.gitignore`** — never commit it). Docker Compose bind-mounts **`./wallets` → `/root/.bittensor/wallets`** inside miner and validator containers. Create/populate that directory on the host first, for example:

```bash
cd xsubnet-template
mkdir -p wallets
export BT_WALLET_PATH="$(pwd)/wallets"
btcli wallet new_coldkey --wallet.name miner
# …repeat for validator, or copy an existing tree: rsync -a ~/.bittensor/wallets/<name>/ ./wallets/<name>/
```

**Expected wallet directory structure (`./wallets`):**

```text
wallets/
  miner/
    coldkey
    coldkeypub.txt
    hotkeys/
      default
  validator/
    coldkey
    coldkeypub.txt
    hotkeys/
      default
```

If your `.env` uses different names (for example `MINER_WALLET_HOTKEY=myhk`), the file must exist as `wallets/<wallet_name>/hotkeys/<hotkey_name>`.

For **host-run** `python neurons/…` or **`offchain_validator_smoke.py`**, set the same `BT_WALLET_PATH` (or symlink `./wallets` to your usual location) so the SDK sees the same files as Docker.

**Security (same as any bind mount):** keys under `./wallets` are visible to any code in the container with that mount; treat `./wallets` like a **secret directory** on disk. For **mainnet / serious stake**, prefer a **hotkey-only** host, **coldkey offline**, separate keys for labs vs production, trusted images, and optionally a **read-only** mount (`:ro`) in a compose override if signing still works.

**UnrealCV / UE port:** `OPENFLY_UNREALCV_PORT` is the port **inside** the UE container (`unrealcv.ini`, TCP, `/tmp/unrealcv_<port>.socket`). The validator shares that network namespace and connects to `127.0.0.1` on that same port. UnrealCV is **not** published to host in this subnet compose (internal only).

**HF / assets (when you use auto-download):**
- `HF_TOKEN` — **required** if Hugging Face assets are gated (UE zip / weights). Same role as `HUGGINGFACE_HUB_TOKEN`.
- For gated Hugging Face assets, users must first open the asset pages in a browser, sign in, and complete/accept the required access form/terms; otherwise downloads will fail even with a token.
  - Model page: `https://huggingface.co/IPEC-COMMUNITY/openfly-agent-7b`
  - UE dataset page: `https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen`
- `OPENFLY_ASSET_AUTO_DOWNLOAD` — `1` (default): `assets-init` downloads model + UE before UE starts; `0`: you place `models/` and `OpenFly-Platform/envs/ue/...` yourself.
- `OPENFLY_UE_ARCHIVE_URL` — optional; if set, UE is taken from this URL instead of the HF dataset.
- `OPENFLY_UE_DATASET_REPO`, `OPENFLY_UE_DATASET_SUBDIR` — only matter when `OPENFLY_UE_ARCHIVE_URL` is empty; defaults match OpenFly_DataGen layout.
- `OPENFLY_DOCKER_UID` / `OPENFLY_DOCKER_GID` — user inside `openfly-ue`; `assets-init` `chown`s the UE tree to this uid/gid on the bind mount so UnrealCV ini edits work (repeat runs skip `chown` if ownership already matches).

**Where assets are downloaded (important):**
- Downloads are performed by the **`assets-init`** service (not by `openfly-ue` / validator runtime code).
- UE env is stored on the host at `./OpenFly-Platform/envs/ue/env_ue_smallcity`.
- Model weights are stored on the host at `./models/openfly-agent-7b`.
- Typical disk usage in this setup: UE env ~**19 GB**, OpenFly model ~**15 GB** (total ~**34 GB**; leave extra free space for caches/logs).
- These are bind mounts, so data persists across container restarts/recreates.
- To avoid auto-download entirely (for example, pre-seeded disks or internal artifact storage), set `OPENFLY_ASSET_AUTO_DOWNLOAD=0` and place files manually in the paths above.
- To host UE elsewhere, set `OPENFLY_UE_ARCHIVE_URL` to your own HTTP(S) archive URL.

**Miner policy (`neurons/miner.py`):**
- `OPENFLY_SUBNET_MINER_MODEL` — `openai` (default) or `openfly`.
  - **`openai`** — three sampled Chat Completions “competition” winners (same temperatures as before). Set **`OPENAI_API_TOKEN`** in `.env`. Without it the miner falls back to a small heuristic.
  - **`openfly`** — loads the **OpenFly HF VLM inside the miner process** (same dependency pattern as the parent repo `docker/openfly-dashboard/Dockerfile`: CUDA base + `OpenFly-Platform/requirements.txt` + torch/transformers extras). Uses `OpenFly-Platform/train/eval.py:get_action`. Compose **`subnet-miner`** (see `docker-compose.miner.yml` / root `docker-compose.yml`) builds **`docker/subnet-miner/Dockerfile`**, requests a **GPU**, and bind-mounts **`./OpenFly-Platform`** and **`./models`**. Set **`OPENFLY_MODEL`** to a HF id or `/app/models/...` path; optional **`OPENFLY_ATTN_IMPLEMENTATION`**, **`HF_TOKEN`** for gated weights. Validator image stays slim (`Dockerfile`); only the miner stack carries PyTorch.
- When `OPENFLY_SUBNET_MINER_MODEL=openai`: optional `OPENAI_API_BASE`, `OPENFLY_SUBNET_MINER_OPENAI_MODEL` / `OPENAI_GPT_POLICY_MODEL`.

**Validator synthetic rounds (on-chain `template/validator/forward.py`):**
- **Cadence:** `OPENFLY_VALIDATOR_FORWARD_SLEEP` (seconds) is passed as `--neuron.forward_sleep` (default **1200** = 20 minutes in `template/utils/config.py`; override in `.env`, e.g. `180` for 3 minutes while testing).
- **Single “AI Step” only:** each round is **one** dendrite query with one `DroneNavSynapse` — there is **no** multi-step “AI Steps” loop in the validator (unlike the main-repo dashboard’s **AI Steps** button). `synthetic_context_json` includes `ai_mode: "single_step"` and `ai_steps_loop: false` when UE prep runs.
- **UE + teleport (when `OPENFLY_SYNTHETIC_UE_ENABLED=1` in compose):** before querying miners, the validator picks a **random spot** from `OPENFLY_TELEPORT_SPOTS_JSON` (same schema as main repo `web/openfly_dashboard/teleport_spots.json`, shipped under `./data/teleport_spots.json`), applies the pose via **UnrealCV** (same remap as `OpenFly-Platform/train/eval.py`), waits **`OPENFLY_SYNTHETIC_POST_TELEPORT_SLEEP_SEC`** (default **10** in compose / `ue_synthetic` fallback), captures **one** lit JPEG, sets **`instruction`** from the spot’s `instruction_preview`, and attaches **`frame_jpeg_b64`** for miners.
- The validator image installs **`unrealcv`** + **`opencv-python-headless`** for that path; it does **not** load the OpenFly HF VLM (only miners do, when configured).

## Quick Start

```bash
cd xsubnet-template
mkdir -p wallets logs/ue-dashboard logs logs/miner-hf-cache logs/miner-torch-cache
git submodule update --init --recursive
```

**Full stack (recommended):**

```bash
docker compose up -d --build
```

**Partial:** validator + UE only, or miner only:

```bash
docker compose -f docker-compose.validator.yml up -d --build
docker compose -f docker-compose.miner.yml up -d --build
```

Check status:

```bash
docker compose logs assets-init --tail 120
docker compose logs openfly-ue --tail 120
docker compose logs subnet-validator --tail 120
docker compose logs subnet-miner --tail 120
```

## Offchain smoke (validator → miners, no weights)

This is **not** a separate chain mode: the script uses your **normal** subtensor RPC and metagraph, sends the same kind of `DroneNavSynapse` a validator would send, and prints rewards — **without** `set_weights` or running the full validator neuron.

**Prerequisites:** subtensor reachable; **miner** registered on `NETUID`, axon serving (e.g. `docker compose up -d` or miner-only compose file); **validator** cold/hotkey exists on disk (`--wallet-name` / `--wallet-hotkey` are those **names**, same as in `.env`). Miner UID(s) must exist on the metagraph (`btcli subnet list` / wallet overview).

From the repo host (needs local `bittensor` + this package on `PYTHONPATH`; or run inside a dev container with the same deps):

```bash
cd xsubnet-template
export BT_WALLET_PATH="$(pwd)/wallets"
PYTHONPATH=. python scripts/offchain_validator_smoke.py \
  --netuid 1 --miner-uids 0 --rounds 1 --timeout 30 \
  --wallet-name validator --wallet-hotkey default \
  --subtensor.chain_endpoint ws://127.0.0.1:9944
```

Useful flags: `--rounds`, `--sleep`, `--instruction "..."`, `--tag-offchain` (marks `synthetic_context_json` for miners that log it). `OPENAI_API_TOKEN` belongs in the **miner** `.env` / environment if you use `OPENFLY_SUBNET_MINER_MODEL=openai`; the smoke script host does not need it.

## Onchain Runtime

**Subnet registration (once per coldkey/hotkey + `NETUID`):** costs **recycle burn** on that coldkey; needs enough **TAO** balance. Use the same `NETUID` and `SUBTENSOR_CHAIN_ENDPOINT` as in `.env`, and the same wallet **names** as `MINER_WALLET_*` / `VALIDATOR_WALLET_*`.

**Quick register checklist (miner + validator):**
1. Create/import wallets under `./wallets` (coldkey + hotkey names from `.env`).
2. Ensure coldkeys have enough TAO for recycle on selected `NETUID`.
3. Register miner hotkey.
4. Register validator hotkey.
5. Start stack and verify both appear on metagraph.

### Concrete commands: create/import + register miner and validator

```bash
cd xsubnet-template
mkdir -p wallets
export BT_WALLET_PATH="$(pwd)/wallets"

# 1) Create miner wallet + hotkey (skip if already imported)
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# 2) Create validator wallet + hotkey (skip if already imported)
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default

# 3) (Optional) verify files exist
ls -la wallets/miner wallets/miner/hotkeys wallets/validator wallets/validator/hotkeys
```

Set your runtime values (can also be read from `.env`):

```bash
export NETUID=4
export SUBTENSOR_CHAIN_ENDPOINT=wss://testnet-rpc1.konnex.world:39944
export MINER_WALLET_NAME=miner
export MINER_WALLET_HOTKEY=default
export VALIDATOR_WALLET_NAME=validator
export VALIDATOR_WALLET_HOTKEY=default
```

Register both hotkeys on subnet:

```bash
# Register miner hotkey
btcli subnet register \
  --wallet-name "$MINER_WALLET_NAME" \
  --hotkey "$MINER_WALLET_HOTKEY" \
  --netuid "$NETUID" \
  --network "$SUBTENSOR_CHAIN_ENDPOINT" \
  --no-prompt -y

# Register validator hotkey
btcli subnet register \
  --wallet-name "$VALIDATOR_WALLET_NAME" \
  --hotkey "$VALIDATOR_WALLET_HOTKEY" \
  --netuid "$NETUID" \
  --network "$SUBTENSOR_CHAIN_ENDPOINT" \
  --no-prompt -y
```

Verify both are visible on subnet:

```bash
btcli wallet overview --wallet-name "$MINER_WALLET_NAME" --network "$SUBTENSOR_CHAIN_ENDPOINT"
btcli wallet overview --wallet-name "$VALIDATOR_WALLET_NAME" --network "$SUBTENSOR_CHAIN_ENDPOINT"
btcli subnet list --network "$SUBTENSOR_CHAIN_ENDPOINT"
```

```bash
cd xsubnet-template
# export NETUID=… SUBTENSOR_CHAIN_ENDPOINT=… MINER_WALLET_NAME=… MINER_WALLET_HOTKEY=… VALIDATOR_WALLET_NAME=… VALIDATOR_WALLET_HOTKEY=…
docker run --rm -v "$(pwd)/wallets:/root/.bittensor/wallets:rw" xsubnet-drone-validator:local \
  btcli subnet register --wallet-name "$MINER_WALLET_NAME" --hotkey "$MINER_WALLET_HOTKEY" \
  --netuid "$NETUID" --network "$SUBTENSOR_CHAIN_ENDPOINT" --no-prompt -y
docker run --rm -v "$(pwd)/wallets:/root/.bittensor/wallets:rw" xsubnet-drone-validator:local \
  btcli subnet register --wallet-name "$VALIDATOR_WALLET_NAME" --hotkey "$VALIDATOR_WALLET_HOTKEY" \
  --netuid "$NETUID" --network "$SUBTENSOR_CHAIN_ENDPOINT" --no-prompt -y
```

Host `btcli` instead of Docker: `export BT_WALLET_PATH="$(pwd)/wallets"` and run the same `btcli subnet register …` lines (no `docker run` wrapper).

After registration:

```bash
docker compose up -d --build
```

## Reset OpenFly Submodule

```bash
cd xsubnet-template
git submodule deinit -f OpenFly-Platform
rm -rf OpenFly-Platform
git submodule update --init --recursive OpenFly-Platform
```
