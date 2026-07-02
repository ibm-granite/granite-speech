from __future__ import annotations

import os
from pathlib import Path

ENV_CACHE = "GRANITE_SPEECH_CACHE"


def resolve_cache_dir(download_root: str | os.PathLike[str] | None = None) -> Path:
    if download_root is not None:
        return Path(download_root).expanduser()

    custom = os.environ.get(ENV_CACHE)
    if custom:
        return Path(custom).expanduser()

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "granite-speech"
