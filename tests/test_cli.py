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

    monkeypatch.setattr(cli, "load_model", fake_load_model)
    monkeypatch.setattr(cli, "write_result", lambda *args, **kwargs: [])

    code = cli.main(
        [
            "clip.wav",
            "--model_dir",
            str(tmp_path / "cache"),
            "--initial_prompt",
            "IBM Granite",
            "--word_timestamps",
            "--output_format",
            "json",
        ]
    )

    assert code == 0
    assert captured["audio_path"] == "clip.wav"
    assert captured["load_kwargs"]["download_root"] == str(tmp_path / "cache")
    assert captured["transcribe_kwargs"]["initial_prompt"] == "IBM Granite"
    assert captured["transcribe_kwargs"]["word_timestamps"] is True


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
