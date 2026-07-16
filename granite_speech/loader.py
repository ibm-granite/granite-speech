# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._backends import BackendCapabilities
from ._backends.llama_cpp import LlamaCppBackend
from ._backends.transformers import TransformersBackend
from ._models import DEFAULT_MODEL, ModelSpec, resolve_model_spec
from .cache import resolve_cache_dir
from .errors import InvalidArgumentError, ModelLoadError, TransformersVersionError
from .model import GraniteSpeechModel

TRANSFORMERS_MIN_VERSION = "5.8.0"
TRANSFORMERS_TESTED_MAX_VERSION = "5.12.1"
DEFAULT_LLAMA_CPP_QUANT = "Q4_K_M"
LLAMA_CPP_DEFAULT_MAX_AUDIO_SECONDS = 120.0


def load_model(
    name: str = DEFAULT_MODEL,
    *,
    device: str | None = None,
    download_root: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    llama_cpp_binary: str | None = None,
    llama_cpp_quant: str = DEFAULT_LLAMA_CPP_QUANT,
    llama_cpp_mmproj: str | None = None,
    llama_cpp_extra_args: Sequence[str] | None = None,
    llama_cpp_timeout: float | None = None,
) -> GraniteSpeechModel:
    spec = resolve_model_spec(name)

    return _load_llama_cpp_model(
        spec,
        device=device,
        download_root=download_root,
        revision=revision,
        local_files_only=local_files_only,
        llama_cpp_binary=llama_cpp_binary,
        llama_cpp_quant=llama_cpp_quant,
        llama_cpp_mmproj=llama_cpp_mmproj,
        llama_cpp_extra_args=llama_cpp_extra_args,
        llama_cpp_timeout=llama_cpp_timeout,
    )


def _load_transformers_model(
    spec: ModelSpec,
    *,
    device: str | None,
    dtype: str,
    download_root: str | None,
    revision: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> GraniteSpeechModel:
    """Load a Granite Speech model through the transformers backend.

    Reserved for future use; not yet reachable from ``load_model``, which routes
    every model to the llama.cpp backend. It will be wired in when a model that
    requires transformers (and cannot run on llama.cpp) ships. Its dependencies
    live in the optional ``granite-speech[transformers]`` extra.
    """
    _check_transformers_version()

    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except TransformersVersionError:
        raise
    except Exception as exc:
        raise ModelLoadError(
            "loading Granite Speech through the transformers backend requires torch and "
            "transformers; install the optional extra with "
            "`pip install granite-speech[transformers]`"
        ) from exc

    resolved_device = _resolve_device(device, torch)
    torch_dtype = _resolve_dtype(dtype, resolved_device, torch)
    model_path = download_model(
        spec.name,
        download_root=download_root,
        revision=revision,
        local_files_only=local_files_only,
    )

    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model.to(resolved_device)
        model.eval()
    except Exception as exc:
        raise ModelLoadError(f"failed to load Granite Speech model {spec.name!r}: {exc}") from exc

    capabilities = _backend_capabilities(spec)
    backend = TransformersBackend(
        processor=processor,
        model=model,
        device=resolved_device,
        capabilities=capabilities,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    return GraniteSpeechModel(
        backend=backend,
        processor=processor,
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        spec=spec,
    )


def download_model(
    name: str = DEFAULT_MODEL,
    *,
    download_root: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str:
    spec = resolve_model_spec(name)
    path = Path(spec.repo_id).expanduser()
    if path.exists():
        return str(path)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise ModelLoadError(
            "huggingface_hub is required to download Granite Speech models"
        ) from exc

    cache_dir = resolve_cache_dir(download_root)
    try:
        return snapshot_download(
            repo_id=spec.repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise ModelLoadError(
            f"failed to download Granite Speech model {spec.name!r}: {exc}"
        ) from exc


def download_llama_cpp_model(
    name: str = DEFAULT_MODEL,
    *,
    quant: str = DEFAULT_LLAMA_CPP_QUANT,
    mmproj_path: str | None = None,
    download_root: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> tuple[str, str]:
    spec = resolve_model_spec(name)
    local_paths = _resolve_local_llama_cpp_paths(spec, quant=quant, mmproj_path=mmproj_path)
    if local_paths is not None:
        return local_paths

    if (
        spec.llama_cpp_repo_id is None
        or spec.llama_cpp_model_file_template is None
        or spec.llama_cpp_mmproj_file is None
    ):
        raise ModelLoadError(f"model {spec.name!r} does not have a llama.cpp GGUF variant")

    model_file = spec.llama_cpp_model_file_template.format(quant=quant)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise ModelLoadError("huggingface_hub is required to download llama.cpp models") from exc

    cache_dir = resolve_cache_dir(download_root)
    allow_patterns = [model_file]
    if mmproj_path is None:
        allow_patterns.append(spec.llama_cpp_mmproj_file)

    try:
        snapshot = snapshot_download(
            repo_id=spec.llama_cpp_repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        raise ModelLoadError(
            f"failed to download llama.cpp model {spec.llama_cpp_repo_id!r}: {exc}"
        ) from exc

    model_path = Path(snapshot) / model_file
    resolved_mmproj_path = (
        Path(mmproj_path).expanduser()
        if mmproj_path is not None
        else Path(snapshot) / spec.llama_cpp_mmproj_file
    )
    if not model_path.exists():
        raise ModelLoadError(
            f"llama.cpp download for {spec.name!r} did not contain {model_file!r}"
        )
    if not resolved_mmproj_path.exists():
        raise ModelLoadError(f"llama.cpp mmproj file does not exist: {resolved_mmproj_path}")
    return str(model_path), str(resolved_mmproj_path)


def _load_llama_cpp_model(
    spec: ModelSpec,
    *,
    device: str | None,
    download_root: str | None,
    revision: str | None,
    local_files_only: bool,
    llama_cpp_binary: str | None,
    llama_cpp_quant: str,
    llama_cpp_mmproj: str | None,
    llama_cpp_extra_args: Sequence[str] | None,
    llama_cpp_timeout: float | None,
) -> GraniteSpeechModel:
    if not _spec_supports_llama_cpp(spec):
        raise ModelLoadError(f"model {spec.name!r} does not have a llama.cpp GGUF variant")

    binary = _resolve_llama_cpp_binary(llama_cpp_binary)
    model_path, mmproj_path = download_llama_cpp_model(
        spec.name,
        quant=llama_cpp_quant,
        mmproj_path=llama_cpp_mmproj,
        download_root=download_root,
        revision=revision,
        local_files_only=local_files_only,
    )

    extra_args = list(llama_cpp_extra_args or ())
    if device == "cpu":
        extra_args.extend(["--gpu-layers", "0"])
    elif device is not None:
        extra_args.extend(["--device", device])

    capabilities = _backend_capabilities(spec, backend="llama.cpp")
    backend = LlamaCppBackend(
        binary=binary,
        model_path=model_path,
        mmproj_path=mmproj_path,
        capabilities=capabilities,
        extra_args=extra_args,
        timeout=llama_cpp_timeout,
    )
    return GraniteSpeechModel(
        backend=backend,
        processor=None,
        model=model_path,
        tokenizer=None,
        device=device or "auto",
        spec=spec,
    )


def _backend_capabilities(spec: ModelSpec, *, backend: str = "transformers") -> BackendCapabilities:
    return BackendCapabilities(
        max_reliable_audio_seconds=(
            LLAMA_CPP_DEFAULT_MAX_AUDIO_SECONDS if backend == "llama.cpp" else None
        ),
        supports_word_timing_output=spec.supports_word_timing_output,
        supports_speaker_attribution_output=spec.supports_speaker_attribution_output,
        supports_batch=False,
        supports_translation=spec.supports_translation,
    )


def _resolve_device(device: str | None, torch: Any) -> Any:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def _resolve_dtype(dtype: str, device: Any, torch: Any) -> Any:
    _validate_dtype(dtype)
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32

    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _validate_dtype(dtype: str) -> None:
    if dtype not in {"auto", "bf16", "fp32"}:
        raise InvalidArgumentError("dtype must be one of {'auto', 'bf16', 'fp32'}")


def _spec_supports_llama_cpp(spec: ModelSpec) -> bool:
    return (
        spec.llama_cpp_repo_id is not None
        and spec.llama_cpp_model_file_template is not None
        and spec.llama_cpp_mmproj_file is not None
    )


def _resolve_llama_cpp_binary(binary: str | None) -> str:
    resolved = _find_llama_cpp_binary(binary)
    if resolved is None:
        requested = binary or "llama-cli"
        raise ModelLoadError(
            f"llama.cpp backend requires {requested!r} on PATH; install llama.cpp or pass "
            "llama_cpp_binary="
        )
    return resolved


def _find_llama_cpp_binary(binary: str | None) -> str | None:
    if binary is not None:
        expanded = Path(binary).expanduser()
        if expanded.exists():
            return str(expanded)
        return shutil.which(binary)
    return shutil.which("llama-cli")


def _resolve_local_llama_cpp_paths(
    spec: ModelSpec,
    *,
    quant: str,
    mmproj_path: str | None,
) -> tuple[str, str] | None:
    path = Path(spec.repo_id).expanduser()
    if not path.exists():
        return None

    if path.is_file():
        model_path = path
        mmproj = (
            Path(mmproj_path).expanduser()
            if mmproj_path is not None
            else path.with_name("mmproj-model-f16.gguf")
        )
    else:
        if spec.llama_cpp_model_file_template is None:
            return None
        model_path = path / spec.llama_cpp_model_file_template.format(quant=quant)
        mmproj_file = spec.llama_cpp_mmproj_file or "mmproj-model-f16.gguf"
        mmproj = Path(mmproj_path).expanduser() if mmproj_path is not None else path / mmproj_file

    if not model_path.exists():
        raise ModelLoadError(f"llama.cpp model file does not exist: {model_path}")
    if not mmproj.exists():
        raise ModelLoadError(f"llama.cpp mmproj file does not exist: {mmproj}")
    return str(model_path), str(mmproj)


def _check_transformers_version() -> None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        installed = version("transformers")
    except PackageNotFoundError as exc:
        raise ModelLoadError(
            "the transformers backend requires transformers; install the optional extra "
            "with `pip install granite-speech[transformers]`"
        ) from exc

    try:
        from packaging.version import Version
    except Exception:
        return

    parsed = Version(installed)
    minimum = Version(TRANSFORMERS_MIN_VERSION)
    tested_max = Version(TRANSFORMERS_TESTED_MAX_VERSION)
    if parsed < minimum:
        raise TransformersVersionError(
            f"transformers>={TRANSFORMERS_MIN_VERSION} is required for Granite Speech; "
            f"found {installed}"
        )
    if parsed > tested_max:
        warnings.warn(
            f"transformers {installed} is newer than the granite-speech tested ceiling "
            f"{TRANSFORMERS_TESTED_MAX_VERSION}; continuing without an install-time upper bound",
            RuntimeWarning,
            stacklevel=2,
        )
