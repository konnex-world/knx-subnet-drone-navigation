#!/usr/bin/env python3
from __future__ import annotations

import ntpath
import os
import shutil
import ssl
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env(name: str, default: str) -> str:
    raw = (os.environ.get(name) or "").strip()
    return raw if raw else default


def _log(msg: str) -> None:
    print(f"[assets-init] {msg}", flush=True)


def _parse_asset_owner_uid_gid() -> tuple[int | None, int | None]:
    raw_u = (os.environ.get("OPENFLY_ASSET_OWNER_UID") or "").strip()
    raw_g = (os.environ.get("OPENFLY_ASSET_OWNER_GID") or "").strip()
    if not raw_u or not raw_g:
        return None, None
    try:
        return int(raw_u), int(raw_g)
    except ValueError:
        _log(f"warn: invalid OPENFLY_ASSET_OWNER_UID/GID: {raw_u!r} / {raw_g!r}")
        return None, None


def _ue_tree_owned_by(ue_dir: Path, uid: int, gid: int) -> bool:
    """Fast path for repeat `docker compose up` (skip huge recursive chown)."""
    probe = ue_dir / "City_UE52/Binaries/Linux/unrealcv.ini"
    if not probe.is_file():
        probe = ue_dir / "CitySample.sh"
    if not probe.is_file():
        return False
    st = probe.stat()
    return st.st_uid == uid and st.st_gid == gid


def _maybe_chown_ue_for_openfly(ue_dir: Path) -> None:
    owner_u, owner_g = _parse_asset_owner_uid_gid()
    if owner_u is None or owner_g is None:
        return
    if _ue_tree_owned_by(ue_dir, owner_u, owner_g):
        _log(f"UE bind-mount ownership ok ({owner_u}:{owner_g}), skip chown")
        return
    _log(f"chown UE tree to {owner_u}:{owner_g} for openfly-ue bind mount")
    _chown_tree(ue_dir, owner_u, owner_g)


def _chown_tree(path: Path, uid: int, gid: int) -> None:
    """So bind-mounted UE is writable by openfly-ue (non-root user)."""
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            p = Path(root) / name
            try:
                os.chown(p, uid, gid, follow_symlinks=False)
            except OSError as e:
                _log(f"warn: chown {p}: {e}")
        for name in dirs:
            p = Path(root) / name
            try:
                os.chown(p, uid, gid, follow_symlinks=False)
            except OSError as e:
                _log(f"warn: chown {p}: {e}")
    try:
        os.chown(path, uid, gid, follow_symlinks=False)
    except OSError as e:
        _log(f"warn: chown {path}: {e}")


# Upper bound on a downloaded UE asset archive. A CitySample-style content tree
# is large but finite; capping the download prevents a malicious or hijacked
# source from filling the host disk (unbounded `shutil.copyfileobj` was the
# previous behaviour).
_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024 * 1024  # 64 GiB
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def _download_file(url: str, dst: Path) -> None:
    """Download `url` to `dst` with an explicit TLS context and a size cap.

    SECURITY/HARDENING: the previous implementation passed no TLS context, used
    no size limit, and copied the whole body in one call. It now:
      * verifies TLS with the system CA bundle (never an unverified context),
      * streams in chunks and aborts once `_MAX_DOWNLOAD_BYTES` is exceeded,
      * only writes to `dst` once the download has been read into place.
    """
    _log(f"download: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "xsubnet-assets-init/1.0"})
    # Explicitly request certificate and hostname verification. Relying on the
    # default is fragile across Python patch versions; being explicit documents
    # the intent and prevents a regression to an unverified context.
    ssl_context = ssl.create_default_context()
    tmp_dst = dst.with_suffix(dst.suffix + ".part")
    total = 0
    try:
        with (
            urllib.request.urlopen(req, timeout=1800, context=ssl_context) as resp,
            open(tmp_dst, "wb") as out,
        ):
            content_length = resp.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError:
                    declared = None
                if declared is not None and declared > _MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(
                        f"UE archive declares {declared} bytes, exceeding the "
                        f"{_MAX_DOWNLOAD_BYTES} byte cap"
                    )
            while True:
                chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(
                        f"UE archive exceeds the {_MAX_DOWNLOAD_BYTES} byte cap"
                    )
                out.write(chunk)
        # Rename only after a complete, in-limit download, so a partial or
        # oversized response never leaves a truncated file that later looks valid.
        os.replace(tmp_dst, dst)
    finally:
        if tmp_dst.exists():
            try:
                tmp_dst.unlink()
            except OSError:
                pass


def _is_within(base: Path, target: Path) -> bool:
    """Return True if `target` resolves to a location inside `base`.

    `base` must already be resolved. Both sides are fully resolved first so that
    symlinks and `..` components cannot be used to escape the destination.
    """
    try:
        target.relative_to(base)
    except ValueError:
        return False
    return True


def _safe_members_zip(zf: zipfile.ZipFile, root: Path) -> None:
    """Raise if any zip member would be written outside `root`.

    Guards against zip-slip ("../" traversal) and against absolute or drive-letter
    member names. CPython's zipfile already strips ".." from the destination it
    computes, so this is defence in depth rather than the sole control.
    """
    for info in zf.infolist():
        name = info.filename
        # Reject absolute paths and Windows drive-qualified names outright.
        if name.startswith(("/", "\\")) or ntpath.isabs(name):
            raise RuntimeError(f"unsafe zip member (absolute path): {name!r}")
        if os.path.splitdrive(name)[0]:
            raise RuntimeError(f"unsafe zip member (drive-qualified): {name!r}")
        # Reject traversal that survives normalisation.
        dest = (root / name).resolve()
        if not _is_within(root, dest):
            raise RuntimeError(f"unsafe zip member (path traversal): {name!r}")


def _safe_members_tar(tf: tarfile.TarFile, root: Path) -> list[tarfile.TarInfo]:
    """Return the tar members that are safe to extract into `root`.

    Rejects absolute paths, ".." traversal, device/FIFO nodes, and symlink or
    hardlink members. Links are rejected because a link inside the destination
    can be used to make a subsequent member write outside `root` even when that
    member's own name looks safe.

    This is CVE-2007-4559 class behaviour: Python 3.12 added `tarfile`'s
    `filter=` parameter, but on 3.11 and earlier `extractall()` writes members
    verbatim and silently, which is the version this project targets.
    """
    safe: list[tarfile.TarInfo] = []
    for member in tf.getmembers():
        name = member.name
        if name.startswith(("/", "\\")) or ntpath.isabs(name) or os.path.splitdrive(name)[0]:
            raise RuntimeError(f"unsafe archive member (absolute path): {name!r}")
        dest = (root / name).resolve()
        if not _is_within(root, dest):
            raise RuntimeError(f"unsafe archive member (path traversal): {name!r}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"unsafe archive member (link): {name!r}")
        if member.isdev() or member.isfifo():
            raise RuntimeError(f"unsafe archive member (device/FIFO): {name!r}")
        safe.append(member)
    return safe


def _extract_archive(archive: Path, out_dir: Path) -> None:
    """Extract a UE asset archive into `out_dir`, rejecting unsafe members.

    SECURITY: this function runs as root inside the assets-init container and its
    output directory is bind-mounted from the host. Previously it called
    `extractall()` with no member validation, so a malicious archive (or a
    hijacked upstream dataset) could write arbitrary files anywhere the container
    can reach -- including the host's ./OpenFly-Platform tree, which is later
    executed. Every member is now validated before anything is written.
    """
    name = archive.name.lower()
    # Resolve the destination so containment checks compare real paths.
    root = out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            _safe_members_zip(zf, root)
            zf.extractall(out_dir)
        return
    if name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar"):
        with tarfile.open(archive, "r:*") as tf:
            safe_members = _safe_members_tar(tf, root)
            # Pass `filter="data"` as an additional layer where supported
            # (CPython >= 3.12); the explicit validation above is authoritative
            # and works on the 3.11 runtime this project targets.
            try:
                tf.extractall(out_dir, members=safe_members, filter="data")
            except TypeError:
                tf.extractall(out_dir, members=safe_members)
        return
    raise RuntimeError(f"unsupported UE archive type: {archive.name}")


def _find_ue_root(extracted_root: Path) -> Path | None:
    if (extracted_root / "CitySample.sh").is_file():
        return extracted_root
    matches = list(extracted_root.rglob("CitySample.sh"))
    if not matches:
        return None
    # pick first and use its parent directory as env root
    return matches[0].parent


def _copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _resolve_ue_dataset_source(
    *,
    repo_id: str,
    requested_path: str,
    token: str | None,
) -> tuple[str, str]:
    """
    Resolve actual UE source path in HF dataset.

    Returns tuple:
      - ("dir", "<path>") for extracted directory that should contain CitySample.sh
      - ("archive", "<path>") for zip/tar artifact to download and extract

    Handles dataset layouts where UE is published as:
      - folder tree with CitySample.sh
      - zip/tar files (e.g. ue/env_ue_smallcity.zip)
    """
    req = requested_path.strip().strip("/")
    req_base = Path(req).name
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    except Exception as e:
        _log(f"warn: list_repo_files failed for dataset {repo_id}: {e!r}; fallback to dir={req!r}")
        return ("dir", req)

    citysample_paths = [p for p in files if p.endswith("/CitySample.sh")]
    roots = [p[: -len("/CitySample.sh")] for p in citysample_paths]
    root_set = set(roots)

    # First, prefer directory layouts when they exist.
    if req in root_set:
        return ("dir", req)
    prefixed = f"ue/{req}"
    if prefixed in root_set:
        return ("dir", prefixed)
    if req.startswith("ue/"):
        unpref = req[len("ue/") :]
        if unpref in root_set:
            return ("dir", unpref)
    for r in roots:
        if r.endswith("/" + req_base) or r == req_base:
            _log(f"resolved UE dir by basename: requested={req!r} -> resolved={r!r}")
            return ("dir", r)

    # Then resolve archive layouts.
    archive_exts = (".zip", ".tar.gz", ".tgz", ".tar")
    archive_files = [p for p in files if p.lower().endswith(archive_exts)]
    archive_candidates: list[str] = []
    archive_candidates.extend([req + ext for ext in archive_exts])
    archive_candidates.extend([f"ue/{req_base}{ext}" for ext in archive_exts])
    archive_candidates.extend([f"{req_base}{ext}" for ext in archive_exts])
    if req.startswith("ue/"):
        unpref = req[len("ue/") :]
        archive_candidates.extend([unpref + ext for ext in archive_exts])

    for cand in archive_candidates:
        if cand in archive_files:
            _log(f"resolved UE archive: requested={req!r} -> {cand!r}")
            return ("archive", cand)

    # Last resort: pick archive with matching basename, then any UE archive.
    for p in archive_files:
        stem = Path(p).stem
        if stem == req_base or req_base in stem:
            _log(f"resolved UE archive by stem: requested={req!r} -> {p!r}")
            return ("archive", p)
    for p in archive_files:
        if "/ue/" in ("/" + p) or p.startswith("ue/"):
            _log(f"resolved UE archive fallback: requested={req!r} -> {p!r}")
            return ("archive", p)

    _log(f"warn: cannot resolve UE source in dataset; fallback to dir={req!r}")
    return ("dir", req)


def _ensure_model_assets() -> None:
    model_dir = Path(_env("OPENFLY_MODEL_DIR", "/workspace/models/openfly-agent-7b"))
    model_repo = _env("OPENFLY_MODEL_REPO", "IPEC-COMMUNITY/openfly-agent-7b")
    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip() or None

    marker = model_dir / "config.json"
    if marker.is_file():
        _log(f"model already present: {model_dir}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    _log(f"downloading model repo={model_repo} -> {model_dir}")
    snapshot_download(
        repo_id=model_repo,
        local_dir=str(model_dir),
        token=hf_token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if not marker.is_file():
        raise RuntimeError(f"model download finished but marker missing: {marker}")
    _log("model ready")


def _ensure_ue_assets() -> None:
    ue_dir = Path(_env("OPENFLY_UE_DIR", "/workspace/OpenFly-Platform/envs/ue/env_ue_smallcity"))
    citysample = ue_dir / "CitySample.sh"
    if citysample.is_file():
        _log(f"ue env already present: {ue_dir}")
        _maybe_chown_ue_for_openfly(ue_dir)
        _log("ue env ready")
        return

    archive_url = (os.environ.get("OPENFLY_UE_ARCHIVE_URL") or "").strip()
    dataset_repo = _env("OPENFLY_UE_DATASET_REPO", "IPEC-COMMUNITY/OpenFly_DataGen")
    dataset_subdir_req = _env("OPENFLY_UE_DATASET_SUBDIR", "ue/env_ue_smallcity")
    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip() or None

    with tempfile.TemporaryDirectory(prefix="openfly-ue-init-") as td:
        tmp = Path(td)
        if archive_url:
            basename = Path(archive_url.split("?")[0]).name or "ue_asset.tar.gz"
            archive = tmp / basename
            _download_file(archive_url, archive)
            extracted = tmp / "ue_extracted"
            extracted.mkdir(parents=True, exist_ok=True)
            _extract_archive(archive, extracted)
            src_root = _find_ue_root(extracted)
            if src_root is None:
                raise RuntimeError("cannot locate CitySample.sh in extracted UE archive")
            _log(f"copy UE from archive: {src_root} -> {ue_dir}")
            _copytree_replace(src_root, ue_dir)
        else:
            kind, source_path = _resolve_ue_dataset_source(
                repo_id=dataset_repo,
                requested_path=dataset_subdir_req,
                token=hf_token,
            )
            _log(
                "OPENFLY_UE_ARCHIVE_URL is empty; trying HF dataset "
                f"{dataset_repo}:{source_path} kind={kind} (requested={dataset_subdir_req})"
            )
            local_dir = tmp / "hf_dataset"
            snapshot_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=hf_token,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=[
                    source_path,
                    f"{source_path}/**",
                    f"{source_path}/*",
                ],
            )
            if kind == "archive":
                archive = local_dir / source_path
                if not archive.is_file():
                    raise RuntimeError(
                        "dataset archive was not downloaded: "
                        f"{archive}; set OPENFLY_UE_ARCHIVE_URL or adjust OPENFLY_UE_DATASET_SUBDIR"
                    )
                extracted = tmp / "ue_from_dataset_extracted"
                extracted.mkdir(parents=True, exist_ok=True)
                _extract_archive(archive, extracted)
                src_root = _find_ue_root(extracted)
                if src_root is None:
                    raise RuntimeError(
                        "dataset archive extracted but CitySample.sh not found; "
                        f"archive={archive}"
                    )
                _log(f"copy UE from dataset archive: {src_root} -> {ue_dir}")
                _copytree_replace(src_root, ue_dir)
            else:
                src_root = local_dir / source_path
                if not (src_root / "CitySample.sh").is_file():
                    raise RuntimeError(
                        "dataset download finished but CitySample.sh not found under "
                        f"{src_root}; set OPENFLY_UE_ARCHIVE_URL or adjust OPENFLY_UE_DATASET_SUBDIR"
                    )
                _log(f"copy UE from dataset dir: {src_root} -> {ue_dir}")
                _copytree_replace(src_root, ue_dir)

    required = [
        ue_dir / "CitySample.sh",
        ue_dir / "City_UE52/Binaries/Linux/CitySample",
        ue_dir / "City_UE52/Binaries/Linux/unrealcv.ini",
    ]
    for p in required:
        if not p.is_file():
            raise RuntimeError(f"UE asset validation failed, missing: {p}")
    _maybe_chown_ue_for_openfly(ue_dir)
    _log("ue env ready")


def main() -> None:
    auto = _env_bool("OPENFLY_ASSET_AUTO_DOWNLOAD", True)
    if not auto:
        _log("OPENFLY_ASSET_AUTO_DOWNLOAD=0, skipping download")
        return
    _ensure_model_assets()
    _ensure_ue_assets()
    _log("all assets ready")


if __name__ == "__main__":
    main()
