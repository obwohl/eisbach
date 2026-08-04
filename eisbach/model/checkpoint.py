"""Checkpoint resolution for the vendored DUET-Prob model.

The trained weights (~10.8 MB) are not stored in this repository. This module
resolves them from a local cache, and downloads them once from the pinned
upstream commit if the cache is empty. Every file that is used -- cached,
user-supplied or freshly downloaded -- is verified against a hardcoded SHA256
before it is handed back.

Resolution order:

1. ``$EISBACH_CHECKPOINT`` -- either a ``.pt`` file to use directly, or a
   directory to use as the cache dir instead of ``data/model/``.
2. ``<repo>/data/model/best_model.pt`` (the default cache location).
3. ``<repo>/ts_proba_cuda/checkpoints/best_model.pt`` -- legacy location from
   the git submodule, kept only so that checkouts which still have the
   submodule do not need to download anything. Goes away with the submodule.
4. Download from ``CHECKPOINT_URL`` into the cache dir.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

__all__ = [
    "CHECKPOINT_SHA256",
    "CHECKPOINT_URL",
    "CHECKPOINT_FILENAME",
    "ChecksumError",
    "resolve_checkpoint",
    "sha256_of",
]

#: SHA256 of ``checkpoints/best_model.pt`` at commit a8de694266a629124687a8f2b9fcfdba15a3590c.
#: Computed from the submodule working tree on 2026-08-04; size 10_843_610 bytes.
CHECKPOINT_SHA256 = "1c7a531768d883af0c70aea1d7fe62fe59638000bf70097d61fb90f2bc4309b0"

CHECKPOINT_SIZE_BYTES = 10_843_610

CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/obwohl/ts_proba_cuda/"
    "a8de694266a629124687a8f2b9fcfdba15a3590c/checkpoints/best_model.pt"
)

CHECKPOINT_FILENAME = "best_model.pt"

#: Repository root: eisbach/model/checkpoint.py -> eisbach/model -> eisbach -> <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "model"

_LEGACY_SUBMODULE_PATH = _REPO_ROOT / "ts_proba_cuda" / "checkpoints" / CHECKPOINT_FILENAME

_ENV_VAR = "EISBACH_CHECKPOINT"


class ChecksumError(RuntimeError):
    """Raised when a checkpoint file does not match :data:`CHECKPOINT_SHA256`."""


def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA256 digest of ``path``, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path) -> Path:
    actual = sha256_of(path)
    if actual != CHECKPOINT_SHA256:
        raise ChecksumError(
            f"Checkpoint at {path} has SHA256 {actual}, expected {CHECKPOINT_SHA256}. "
            f"Refusing to use it. Delete the file to force a fresh download from "
            f"{CHECKPOINT_URL}"
        )
    return path


def _download(destination: Path) -> None:
    """Download the checkpoint to ``destination`` atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=destination.name + ".", suffix=".part"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        # requests is already a hard dependency of this repo; urllib is the fallback.
        try:
            import requests
        except ImportError:  # pragma: no cover - requests is in requirements.txt
            requests = None

        if requests is not None:
            with requests.get(CHECKPOINT_URL, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        else:  # pragma: no cover
            from urllib.request import urlopen

            with urlopen(CHECKPOINT_URL, timeout=120) as response, open(tmp_path, "wb") as fh:
                shutil.copyfileobj(response, fh)

        _verify(tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def resolve_checkpoint(
    path: str | Path | None = None,
    *,
    allow_download: bool = True,
) -> Path:
    """Return a verified path to the model checkpoint.

    Parameters
    ----------
    path:
        Explicit checkpoint path. If given it must exist and match the expected
        SHA256; nothing is downloaded.
    allow_download:
        If ``False``, raise instead of fetching a missing checkpoint from the
        network (useful in tests and offline environments).

    Raises
    ------
    FileNotFoundError
        If an explicit ``path`` does not exist, or if the checkpoint is missing
        and ``allow_download`` is ``False``.
    ChecksumError
        If the resolved file's SHA256 does not match :data:`CHECKPOINT_SHA256`.
    """
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return _verify(candidate)

    cache_dir = DEFAULT_CACHE_DIR
    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.is_dir():
            cache_dir = env_path
        elif env_path.is_file():
            return _verify(env_path)
        elif env_path.suffix == ".pt":
            # Points at a file that does not exist yet: download to it.
            cache_dir = env_path.parent
            cached = env_path
            if allow_download:
                _download(cached)
                return _verify(cached)
            raise FileNotFoundError(
                f"Checkpoint not found at ${_ENV_VAR}={cached} and downloads are disabled."
            )
        else:
            cache_dir = env_path

    cached = cache_dir / CHECKPOINT_FILENAME
    if cached.is_file():
        return _verify(cached)

    if _LEGACY_SUBMODULE_PATH.is_file():
        return _verify(_LEGACY_SUBMODULE_PATH)

    if not allow_download:
        raise FileNotFoundError(
            f"Checkpoint not found at {cached} and downloads are disabled. "
            f"Fetch it manually from {CHECKPOINT_URL}"
        )

    _download(cached)
    return _verify(cached)
