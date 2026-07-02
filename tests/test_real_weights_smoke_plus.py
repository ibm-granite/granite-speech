from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from hashlib import sha256
from pathlib import Path

import pytest

import granite_speech

pytestmark = pytest.mark.real_weights

PLUS_MODEL_NAME = "granite-speech-4.1-2b-plus"
# Support for granite-speech-4.1-2b-plus landed in llama.cpp via
# https://github.com/ggml-org/llama.cpp/pull/24818. Builds before this crash (SIGABRT) when
# encoding the plus model's audio features. 9850 is the earliest build verified to pass this smoke;
# override with GRANITE_SPEECH_PLUS_SMOKE_MIN_LLAMA_CPP_BUILD if an earlier fixed build is known.
MIN_LLAMA_CPP_BUILD = 9850
_LLAMA_CPP_VERSION_RE = re.compile(r"version:\s*(\d+)")
# The plus model card exercises all four tasks against the AMI meeting clip (sample 0, the 5:00-6:00
# window of the ihm test split). It is genuinely multi-speaker, so it drives speaker attribution and
# incremental decoding faithfully. Unlike the base smoke's multilingual_sample.wav, this clip is not
# hosted in a HF model repo; regenerate it with scripts/fetch_ami_smoke_audio.py.
SMOKE_AUDIO_FILENAME = "ami_ihm_sample0_5m-6m.wav"
SMOKE_AUDIO_SHA256 = "e248864aa8ac4fb145a874ef8efe86c6278dcb584f91989e8693c0f81df61ff4"
LOCAL_SMOKE_AUDIO_PATH = Path(__file__).with_name("fixtures") / SMOKE_AUDIO_FILENAME


@pytest.fixture(scope="module")
def smoke_context():
    load_kwargs = {
        "download_root": os.environ.get("GRANITE_SPEECH_PLUS_SMOKE_DOWNLOAD_ROOT"),
        "local_files_only": _env_flag("GRANITE_SPEECH_PLUS_SMOKE_LOCAL_FILES_ONLY"),
        "revision": os.environ.get("GRANITE_SPEECH_PLUS_SMOKE_MODEL_REVISION"),
        "llama_cpp_binary": os.environ.get("GRANITE_SPEECH_PLUS_SMOKE_LLAMA_CPP_BINARY"),
        "llama_cpp_quant": os.environ.get("GRANITE_SPEECH_PLUS_SMOKE_LLAMA_CPP_QUANT", "Q4_K_M"),
        "llama_cpp_timeout": _optional_float("GRANITE_SPEECH_PLUS_SMOKE_TIMEOUT"),
    }
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}

    binary = load_kwargs.get("llama_cpp_binary") or shutil.which("llama-cli")
    if binary is None:
        pytest.skip(
            "llama-cli is required for the real-weights plus smoke; install llama.cpp or set "
            "GRANITE_SPEECH_PLUS_SMOKE_LLAMA_CPP_BINARY"
        )
    _skip_if_llama_cpp_too_old(binary)

    audio_path = _smoke_audio_path()
    return load_kwargs, audio_path


@pytest.fixture(scope="module")
def smoke_model(smoke_context):
    load_kwargs, _audio_path = smoke_context
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = granite_speech.load_model(PLUS_MODEL_NAME, **load_kwargs)

    _assert_no_package_warnings(caught)
    assert model.backend.name == "llama.cpp"
    return model


@pytest.fixture(scope="module")
def smoke_audio_path(smoke_context) -> Path:
    _load_kwargs, audio_path = smoke_context
    return audio_path


def test_plus_model_llama_cpp_real_weights_smoke(smoke_model, smoke_audio_path):
    result = _transcribe_without_package_warnings(smoke_model, smoke_audio_path)

    _assert_successful_smoke_result(result)
    _report("plus ASR", result)


def test_plus_model_word_timestamps_real_weights_smoke(smoke_model, smoke_audio_path):
    # word_timestamps=True with the default prompt_mode emits a whisper-compat warning from the
    # package; select the prompt mode directly so the no-package-warnings assertion still holds.
    result = _transcribe_without_package_warnings(
        smoke_model,
        smoke_audio_path,
        prompt_mode="word_timestamps",
    )

    _assert_successful_smoke_result(result)
    _report("plus word timestamps", result)
    words = result.get("words")
    assert words, "expected word-timestamp output for the plus model"
    # Timestamps are window-relative, so starts reset at each window boundary; assert per-word
    # validity rather than global monotonicity across the whole transcript.
    for entry in words:
        assert set(entry) >= {"word", "start", "end"}
        assert entry["word"].strip()
        assert 0.0 <= entry["start"] <= entry["end"]


def test_plus_model_speaker_attribution_real_weights_smoke(smoke_model, smoke_audio_path):
    result = _transcribe_without_package_warnings(
        smoke_model,
        smoke_audio_path,
        prompt_mode="speaker_attributed",
    )

    _assert_successful_smoke_result(result)
    _report("plus speaker attribution", result)
    assert "speakers" in result
    for turn in result["speakers"]:
        assert set(turn) >= {"speaker", "text"}
        assert turn["speaker"].strip()
        assert turn["text"].strip()
    # The AMI clip is genuinely multi-speaker, so speaker attribution should surface at least two
    # distinct speaker labels.
    distinct_speakers = {turn["speaker"] for turn in result["speakers"]}
    assert len(distinct_speakers) >= 2, f"expected multiple speakers, got {distinct_speakers}"


def test_plus_model_incremental_decoding_real_weights_smoke(smoke_model, smoke_audio_path):
    # Mirror the model card's incremental loop: transcribe an early window, then transcribe a later
    # cumulative window while passing the earlier transcript as prefix_text so speaker numbering
    # stays consistent across segments.
    seed = _transcribe_without_package_warnings(
        smoke_model,
        smoke_audio_path,
        prompt_mode="speaker_attributed",
        clip_timestamps="0,30",
    )
    _assert_successful_smoke_result(seed)
    _report("plus incremental seed (0-30s)", seed)
    assert seed["text"].strip()

    continued = _transcribe_without_package_warnings(
        smoke_model,
        smoke_audio_path,
        prompt_mode="speaker_attributed",
        clip_timestamps="0,60",
        prefix_text=seed["text"],
    )

    _assert_successful_smoke_result(continued)
    _report("plus incremental continued (0-60s)", continued)
    assert "speakers" in continued
    for turn in continued["speakers"]:
        assert set(turn) >= {"speaker", "text"}
        assert turn["speaker"].strip()
        assert turn["text"].strip()

    # Speaker numbering should carry over from the prefix rather than restart: the continued run's
    # speaker labels should overlap the seed's, not be a disjoint relabeling. Exact transcript text
    # is nondeterministic, so assert on label-set overlap rather than content.
    seed_speakers = {turn["speaker"] for turn in seed.get("speakers", [])}
    continued_speakers = {turn["speaker"] for turn in continued["speakers"]}
    if seed_speakers:
        assert seed_speakers & continued_speakers, (
            f"expected continued speakers {continued_speakers} to overlap seed {seed_speakers}"
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
    print(f"segments: {len(result.get('segments') or [])}")
    print(f"text: {result['text'].strip()}")
    speakers = result.get("speakers")
    if speakers:
        print("speakers:")
        for turn in speakers:
            print(f"  [{turn.get('speaker')}] {turn.get('text', '').strip()}")
    words = result.get("words")
    if words:
        formatted = " ".join(
            f"{entry.get('word', '').strip()}({entry.get('start')}-{entry.get('end')})"
            for entry in words
        )
        print(f"words: {formatted}")


def _smoke_audio_path() -> Path:
    value = os.environ.get("GRANITE_SPEECH_PLUS_SMOKE_AUDIO")
    if value:
        path = Path(value).expanduser()
        if not path.exists():
            pytest.fail(f"GRANITE_SPEECH_PLUS_SMOKE_AUDIO does not exist: {path}")
        return path

    # The AMI clip is committed under tests/fixtures/ rather than fetched from a HF model repo, so
    # require the local fixture instead of falling back to a download.
    if not LOCAL_SMOKE_AUDIO_PATH.exists():
        pytest.fail(
            f"plus smoke audio fixture is missing: {LOCAL_SMOKE_AUDIO_PATH}. Regenerate it with "
            "scripts/fetch_ami_smoke_audio.py, or point GRANITE_SPEECH_PLUS_SMOKE_AUDIO at a clip."
        )
    expected_sha = os.environ.get("GRANITE_SPEECH_PLUS_SMOKE_AUDIO_SHA256", SMOKE_AUDIO_SHA256)
    if expected_sha:
        _assert_sha256(LOCAL_SMOKE_AUDIO_PATH, expected_sha)
    return LOCAL_SMOKE_AUDIO_PATH


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


def _skip_if_llama_cpp_too_old(binary: str) -> None:
    minimum = _min_llama_cpp_build()
    if minimum is None:
        return
    build = _llama_cpp_build(binary)
    if build is None:
        return
    if build < minimum:
        pytest.skip(
            f"llama.cpp build {build} predates granite-speech-4.1-2b-plus support; build "
            f"{minimum} or newer is required (see llama.cpp PR #24818). Upgrade llama.cpp or set "
            "GRANITE_SPEECH_PLUS_SMOKE_MIN_LLAMA_CPP_BUILD to override."
        )


def _llama_cpp_build(binary: str) -> int | None:
    try:
        completed = subprocess.run(
            [binary, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _LLAMA_CPP_VERSION_RE.search(completed.stdout or "")
    return int(match.group(1)) if match else None


def _min_llama_cpp_build() -> int | None:
    raw_value = os.environ.get("GRANITE_SPEECH_PLUS_SMOKE_MIN_LLAMA_CPP_BUILD")
    if raw_value is None:
        return MIN_LLAMA_CPP_BUILD
    stripped = raw_value.strip()
    if stripped == "" or stripped == "0":
        return None
    try:
        return int(stripped)
    except ValueError as exc:
        raise AssertionError(
            "GRANITE_SPEECH_PLUS_SMOKE_MIN_LLAMA_CPP_BUILD must be an integer build number"
        ) from exc


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
