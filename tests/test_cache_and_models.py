from __future__ import annotations

import inspect

import pytest

from granite_speech._models import (
    available_models,
    resolve_model_spec,
    validate_source_language,
    validate_translation_pair,
)
from granite_speech.cache import resolve_cache_dir
from granite_speech.errors import InvalidArgumentError, ModelLoadError
from granite_speech.loader import (
    _resolve_local_llama_cpp_paths,
    load_model,
)


def test_available_models_and_alias_resolution():
    assert "granite-speech-4.1-2b" in available_models()
    assert "granite-speech-4.1-2b-plus" in available_models()
    assert resolve_model_spec("ibm-granite/granite-speech-4.1-2b").name == "granite-speech-4.1-2b"
    assert (
        resolve_model_spec("ibm-granite/granite-speech-4.1-2b-GGUF").name
        == "granite-speech-4.1-2b"
    )
    assert resolve_model_spec("granite-speech-4.1-2b-GGUF").name == "granite-speech-4.1-2b"
    assert (
        resolve_model_spec("ibm-granite/granite-speech-4.1-2b-plus").name
        == "granite-speech-4.1-2b-plus"
    )
    assert (
        resolve_model_spec("ibm-granite/granite-speech-4.1-2b-plus-GGUF").name
        == "granite-speech-4.1-2b-plus"
    )
    assert (
        resolve_model_spec("granite-speech-4.1-2b-plus-GGUF").name
        == "granite-speech-4.1-2b-plus"
    )


def test_local_path_resolution_uses_path(tmp_path):
    spec = resolve_model_spec(tmp_path)
    assert spec.repo_id == str(tmp_path)


def test_language_validation():
    spec = resolve_model_spec("granite-speech-4.1-2b")
    validate_source_language(spec, "French")
    validate_translation_pair(spec, "fr", "en")

    with pytest.raises(InvalidArgumentError, match="source language"):
        validate_source_language(spec, "Klingon")
    with pytest.raises(InvalidArgumentError, match="does not support translation pair"):
        validate_translation_pair(spec, "fr", "de")


def test_cache_resolution_order(monkeypatch, tmp_path):
    monkeypatch.delenv("GRANITE_SPEECH_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    assert resolve_cache_dir(tmp_path) == tmp_path

    monkeypatch.setenv("GRANITE_SPEECH_CACHE", str(tmp_path / "granite"))
    assert resolve_cache_dir() == tmp_path / "granite"

    monkeypatch.delenv("GRANITE_SPEECH_CACHE")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-hub"))
    assert resolve_cache_dir() == tmp_path / "hf-hub"

    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    assert resolve_cache_dir() == tmp_path / "hf-home" / "hub"


def test_load_model_has_no_backend_or_transformers_only_arguments():
    parameters = inspect.signature(load_model).parameters

    assert "backend" not in parameters
    assert "dtype" not in parameters
    assert "trust_remote_code" not in parameters


def test_load_model_uses_llama_cpp_loader(monkeypatch):
    sentinel = object()
    seen: dict[str, object] = {}

    def fake_load_llama_cpp_model(spec, **kwargs):
        seen["spec"] = spec
        seen["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        "granite_speech.loader._load_llama_cpp_model",
        fake_load_llama_cpp_model,
    )

    loaded = load_model(
        "granite-speech-4.1-2b-plus",
        device="cpu",
        download_root="/tmp/granite-cache",
        revision="main",
        local_files_only=True,
        llama_cpp_binary="llama-cli",
        llama_cpp_quant="Q8_0",
        llama_cpp_mmproj="/tmp/mmproj-model-f16.gguf",
        llama_cpp_extra_args=["--threads", "4"],
        llama_cpp_timeout=30,
    )

    assert loaded is sentinel
    assert seen["spec"].name == "granite-speech-4.1-2b-plus"
    assert seen["kwargs"] == {
        "device": "cpu",
        "download_root": "/tmp/granite-cache",
        "revision": "main",
        "local_files_only": True,
        "llama_cpp_binary": "llama-cli",
        "llama_cpp_quant": "Q8_0",
        "llama_cpp_mmproj": "/tmp/mmproj-model-f16.gguf",
        "llama_cpp_extra_args": ["--threads", "4"],
        "llama_cpp_timeout": 30,
    }


def test_load_model_rejects_plain_local_model_dirs(tmp_path):
    (tmp_path / "config.json").write_text("{}")

    with pytest.raises(ModelLoadError, match="GGUF"):
        load_model(str(tmp_path))


def test_local_llama_cpp_paths_resolve_model_and_mmproj(tmp_path):
    (tmp_path / "granite-speech-4.1-2b-Q4_K_M.gguf").write_bytes(b"model")
    (tmp_path / "mmproj-model-f16.gguf").write_bytes(b"mmproj")
    spec = resolve_model_spec(tmp_path)

    model_path, mmproj_path = _resolve_local_llama_cpp_paths(
        spec,
        quant="Q4_K_M",
        mmproj_path=None,
    )

    assert model_path == str(tmp_path / "granite-speech-4.1-2b-Q4_K_M.gguf")
    assert mmproj_path == str(tmp_path / "mmproj-model-f16.gguf")
