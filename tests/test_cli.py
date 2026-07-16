# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from granite_speech import cli


def test_cli_whisper_aliases_are_passed_to_transcribe(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class StubModel:
        def transcribe(self, audio_path, **kwargs):
            captured["audio_path"] = audio_path
            captured["transcribe_kwargs"] = kwargs
            return {
                "text": "ok",
                "segments": [],
                "language": None,
                "target_language": None,
                "warnings": [],
            }

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
        return []

    monkeypatch.setattr(cli, "load_model", fake_load_model)
    monkeypatch.setattr(cli, "write_result", fake_write_result)

    code = cli.main(
        [
            "clip.wav",
            "--model_dir",
            str(tmp_path / "cache"),
            "--initial_prompt",
            "IBM Granite",
            "--word_timestamps",
            "--clip_timestamps",
            "1,2",
            "--output_format",
            "json",
            "--max_line_width",
            "42",
            "--max_line_count",
            "2",
        ]
    )

    assert code == 0
    assert captured["audio_path"] == "clip.wav"
    assert captured["load_kwargs"]["download_root"] == str(tmp_path / "cache")
    assert captured["transcribe_kwargs"]["initial_prompt"] == "IBM Granite"
    assert captured["transcribe_kwargs"]["word_timestamps"] is True
    assert captured["transcribe_kwargs"]["clip_timestamps"] == "1,2"
    assert captured["write_result"]["kwargs"]["max_line_width"] == 42
    assert captured["write_result"]["kwargs"]["max_line_count"] == 2


def test_cli_rejects_conflicting_cache_aliases():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "clip.wav",
                "--download_root",
                "/tmp/native-cache",
                "--model_dir",
                "/tmp/whisper-cache",
            ]
        )

    assert exc_info.value.code == 2


def test_cli_rejects_beam_size_alias():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["clip.wav", "--beam_size", "5"])

    assert exc_info.value.code == 2
