# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import granite_speech
from granite_speech import cli
from granite_speech.model import GraniteSpeechModel


class StubModel:
    def transcribe(self, audio_path, **kwargs):
        return {
            "text": "hello world",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.25,
                    "text": "hello world",
                    "temperature": 0.0,
                }
            ],
            "language": kwargs.get("language"),
            "target_language": (
                kwargs.get("target_language") or "en"
                if kwargs.get("task") == "translate"
                else None
            ),
            "warnings": [],
        }


def test_readme_python_api_options_match_public_signatures():
    load_params = inspect.signature(granite_speech.load_model).parameters
    for name in ["device", "llama_cpp_quant"]:
        assert name in load_params

    transcribe_params = inspect.signature(GraniteSpeechModel.transcribe).parameters
    for name in [
        "task",
        "language",
        "target_language",
        "keyword_biases",
        "max_new_tokens",
        "chunk_length",
        "chunk_overlap",
        "segmentation",
        "clip_timestamps",
        "prompt_mode",
        "prefix_text",
    ]:
        assert name in transcribe_params


@pytest.mark.parametrize(
    "args",
    [
        ["audio.wav", "--model", "granite-speech-4.1-2b", "--output_format", "txt"],
        ["audio.wav", "--llama_cpp_quant", "Q4_K_M"],
        ["audio.wav", "--task", "translate", "--language", "fr", "--output_format", "json"],
        ["audio.wav", "--keyword", "Granite Speech", "--keyword", "watsonx.ai"],
        [
            "meeting.wav",
            "--model",
            "granite-speech-4.1-2b-plus",
            "--prompt_mode",
            "speaker_attributed",
        ],
        [
            "meeting.wav",
            "--model",
            "granite-speech-4.1-2b-plus",
            "--prompt_mode",
            "word_timestamps",
            "--max_new_tokens",
            "10000",
        ],
        ["audio.wav", "--segmentation", "vad", "--chunk_length", "30"],
        ["audio.wav", "--clip_timestamps", "10,20"],
        ["audio.wav", "--output_format", "srt", "--max_line_width", "42", "--max_line_count", "2"],
        ["audio.wav", "--output_format", "all", "--output_dir", "transcripts/"],
    ],
)
def test_readme_cli_examples_are_accepted(monkeypatch, args):
    captured: dict[str, object] = {}

    def fake_load_model(name, **kwargs):
        captured["model_name"] = name
        captured["load_kwargs"] = kwargs
        return StubModel()

    def fake_write_result(result, audio_path, **kwargs):
        captured["write_result"] = {
            "result": result,
            "audio_path": audio_path,
            "kwargs": kwargs,
        }
        return [Path(kwargs["output_dir"]) / f"{Path(audio_path).stem}.{kwargs['output_format']}"]

    monkeypatch.setattr(cli, "load_model", fake_load_model)
    monkeypatch.setattr(cli, "write_result", fake_write_result)

    assert cli.main(args) == 0
    assert "write_result" in captured


def test_readme_default_cli_output_is_txt_in_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_model", lambda _name, **_kwargs: StubModel())

    assert cli.main(["audio.wav"]) == 0

    assert (tmp_path / "audio.txt").read_text(encoding="utf-8") == "hello world\n"


def test_readme_cli_output_dir_uses_input_stem_for_all_formats(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_model", lambda _name, **_kwargs: StubModel())

    assert cli.main(["audio.wav", "--output_format", "all", "--output_dir", "transcripts/"]) == 0

    output_dir = tmp_path / "transcripts"
    assert {path.name for path in output_dir.iterdir()} == {
        "audio.txt",
        "audio.srt",
        "audio.vtt",
        "audio.tsv",
        "audio.json",
    }


@pytest.mark.parametrize(
    "args",
    [
        ["download", "granite-speech-4.1-2b", "--download_root", "/models/granite-speech"],
        ["download", "granite-speech-4.1-2b", "--llama_cpp_quant", "Q4_K_M"],
        ["download", "granite-speech-4.1-2b-plus", "--llama_cpp_quant", "Q4_K_M"],
    ],
)
def test_readme_download_examples_are_accepted(monkeypatch, args):
    calls = []

    def fake_download_llama_cpp_model(name, **kwargs):
        calls.append((name, kwargs))
        return "/models/model.gguf", "/models/mmproj-model-f16.gguf"

    monkeypatch.setattr(cli, "download_llama_cpp_model", fake_download_llama_cpp_model)

    assert cli.main(args) == 0
    assert calls
