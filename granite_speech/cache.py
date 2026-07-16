# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

ENV_CACHE = "GRANITE_SPEECH_CACHE"


def resolve_cache_dir(download_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the directory model weights are cached in.

    Checks sources in priority order and returns the first that is set:
    the explicit ``download_root`` argument, then ``$GRANITE_SPEECH_CACHE``,
    then the Hugging Face ``$HF_HUB_CACHE`` and ``$HF_HOME/hub`` locations,
    falling back to ``~/.cache/granite-speech``. The path is expanded (``~``)
    but not created here.
    """
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
