# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import shutil
import warnings
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path

import pytest

import granite_speech

pytestmark = pytest.mark.real_weights

SMOKE_AUDIO_REPO_ID = "ibm-granite/granite-speech-4.1-2b"
SMOKE_AUDIO_FILENAME = "multilingual_sample.wav"
SMOKE_AUDIO_REVISION = "de575db64086f84fdc79da4932d1076e965bc546"
SMOKE_AUDIO_SHA256 = "91d243650809c1274141ec20ff23045315eaf27567694002ea3ef390048b7058"
LOCAL_SMOKE_AUDIO_PATH = Path(__file__).with_name("fixtures") / SMOKE_AUDIO_FILENAME
DEFAULT_SIMILARITY_THRESHOLD = 0.75
TRANSLATION_SMOKE_CLIP_TIMESTAMPS = "12"
TRANSLATION_SMOKE_EXPECTED_TEXT = (
    "Dinarzade, the following night, called her sister when it was time. "
    '"If you do not sleep, my sister," she said, "I will pray you to continue '
    'the story of the fisherman."'
)


@pytest.fixture(scope="module")
def smoke_context():
    load_kwargs = {
        "download_root": os.environ.get("GRANITE_SPEECH_SMOKE_DOWNLOAD_ROOT"),
        "local_files_only": _env_flag("GRANITE_SPEECH_SMOKE_LOCAL_FILES_ONLY"),
        "revision": os.environ.get("GRANITE_SPEECH_SMOKE_MODEL_REVISION"),
        "llama_cpp_binary": os.environ.get("GRANITE_SPEECH_SMOKE_LLAMA_CPP_BINARY"),
        "llama_cpp_quant": os.environ.get("GRANITE_SPEECH_SMOKE_LLAMA_CPP_QUANT", "Q4_K_M"),
        "llama_cpp_timeout": _optional_float("GRANITE_SPEECH_SMOKE_TIMEOUT"),
    }
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}

    if load_kwargs.get("llama_cpp_binary") is None and shutil.which("llama-cli") is None:
        pytest.skip(
            "llama-cli is required for the real-weights smoke; install llama.cpp or set "
            "GRANITE_SPEECH_SMOKE_LLAMA_CPP_BINARY"
        )

    audio_path = _smoke_audio_path(
        cache_dir=load_kwargs.get("download_root"),
        local_files_only=bool(load_kwargs.get("local_files_only")),
    )
    return load_kwargs, audio_path


@pytest.fixture(scope="module")
def smoke_model(smoke_context):
    load_kwargs, _audio_path = smoke_context
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = granite_speech.load_model("granite-speech-4.1-2b", **load_kwargs)

    _assert_no_package_warnings(caught)
    assert model.backend.name == "llama.cpp"
    return model


@pytest.fixture(scope="module")
def smoke_audio_path(smoke_context) -> Path:
    _load_kwargs, audio_path = smoke_context
    return audio_path


def test_base_model_llama_cpp_real_weights_smoke(smoke_model, smoke_audio_path):
    expected_text = os.environ.get("GRANITE_SPEECH_SMOKE_EXPECTED_TEXT")

    result = _transcribe_without_package_warnings(smoke_model, smoke_audio_path)

    _assert_successful_smoke_result(result)
    _report("base ASR", result)
    if expected_text:
        assert _close_enough(
            result["text"],
            expected_text,
            threshold=_similarity_threshold(),
        )


def test_base_model_llama_cpp_translation_real_weights_smoke(smoke_model, smoke_audio_path):
    result = _transcribe_without_package_warnings(
        smoke_model,
        smoke_audio_path,
        task="translate",
        language="fr",
        clip_timestamps=TRANSLATION_SMOKE_CLIP_TIMESTAMPS,
    )

    _assert_successful_smoke_result(result)
    _report("base translate (fr->en)", result)
    assert result["language"] == "fr"
    assert result["target_language"] == "en"
    expected_text = os.environ.get(
        "GRANITE_SPEECH_TRANSLATION_SMOKE_EXPECTED_TEXT",
        TRANSLATION_SMOKE_EXPECTED_TEXT,
    )
    assert _close_enough(
        result["text"],
        expected_text,
        threshold=_similarity_threshold(),
    )


def _transcribe_without_package_warnings(model, audio_path: Path, **kwargs) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.transcribe(str(audio_path), **kwargs)

    _assert_no_package_warnings(caught)
    return result


def _assert_successful_smoke_result(result: dict) -> None:
    assert result["warnings"] == []
    assert result["segments"]
    assert result["text"].strip()


def _report(label: str, result: dict) -> None:
    # Print the transcription so a manual smoke run (pytest -s) shows what the model produced,
    # not just pass/fail. Kept to stdout rather than logging so it is visible without extra flags.
    print(f"\n--- {label} ---")
    language = result.get("language")
    target = result.get("target_language")
    if language:
        print(f"language: {language}" + (f" -> {target}" if target else ""))
    print(f"segments: {len(result.get('segments') or [])}")
    print(f"text: {result['text'].strip()}")


def _smoke_audio_path(*, cache_dir: str | None, local_files_only: bool) -> Path:
    value = os.environ.get("GRANITE_SPEECH_SMOKE_AUDIO")
    if value:
        path = Path(value).expanduser()
        if not path.exists():
            pytest.fail(f"GRANITE_SPEECH_SMOKE_AUDIO does not exist: {path}")
        return path

    if (
        os.environ.get("GRANITE_SPEECH_SMOKE_AUDIO_REVISION") is None
        and LOCAL_SMOKE_AUDIO_PATH.exists()
    ):
        expected_sha = os.environ.get("GRANITE_SPEECH_SMOKE_AUDIO_SHA256", SMOKE_AUDIO_SHA256)
        if expected_sha:
            _assert_sha256(LOCAL_SMOKE_AUDIO_PATH, expected_sha)
        return LOCAL_SMOKE_AUDIO_PATH

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise AssertionError(
            "huggingface_hub is required to download the default smoke audio"
        ) from exc

    revision = os.environ.get("GRANITE_SPEECH_SMOKE_AUDIO_REVISION", SMOKE_AUDIO_REVISION)
    path = Path(
        hf_hub_download(
            repo_id=SMOKE_AUDIO_REPO_ID,
            filename=SMOKE_AUDIO_FILENAME,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
    expected_sha = os.environ.get(
        "GRANITE_SPEECH_SMOKE_AUDIO_SHA256",
        SMOKE_AUDIO_SHA256 if revision == SMOKE_AUDIO_REVISION else "",
    )
    if expected_sha:
        _assert_sha256(path, expected_sha)
    return path


def _close_enough(actual: str, expected: str, *, threshold: float) -> bool:
    actual_normalized = _normalize_text(actual)
    expected_normalized = _normalize_text(expected)
    if not actual_normalized or not expected_normalized:
        return False
    if actual_normalized in expected_normalized or expected_normalized in actual_normalized:
        return True
    return SequenceMatcher(None, actual_normalized, expected_normalized).ratio() >= threshold


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _is_package_warning(filename: str) -> bool:
    return "granite_speech" in Path(filename).parts


def _assert_no_package_warnings(caught: list[warnings.WarningMessage]) -> None:
    package_warnings = [warning for warning in caught if _is_package_warning(warning.filename)]
    assert package_warnings == []


def _assert_sha256(path: Path, expected_sha: str) -> None:
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha:
        pytest.fail(
            f"smoke audio SHA-256 mismatch for {path}: expected {expected_sha}, got {digest}"
        )


def _similarity_threshold() -> float:
    raw_value = os.environ.get("GRANITE_SPEECH_SMOKE_SIMILARITY")
    if raw_value is None:
        return DEFAULT_SIMILARITY_THRESHOLD
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise AssertionError("GRANITE_SPEECH_SMOKE_SIMILARITY must be a float") from exc
    if not 0 <= threshold <= 1:
        pytest.fail("GRANITE_SPEECH_SMOKE_SIMILARITY must be between 0 and 1")
    return threshold


def _optional_float(name: str) -> float | None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except ValueError as exc:
        raise AssertionError(f"{name} must be a float") from exc


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
