from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

import granite_speech
from granite_speech import cli
from granite_speech._backends import GenerateResult
from granite_speech._backends.fake import FakeBackend
from granite_speech._models import MODELS
from granite_speech.model import GraniteSpeechModel


class FakeTokenizer:
    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT:{chat[-1]['content']}:GEN"


def write_silent_wav(path: Path, *, seconds: float = 1.0, sample_rate: int = 16000) -> Path:
    samples = np.zeros(int(seconds * sample_rate), dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return path


def fake_model(
    responses,
    *,
    model_name: str = "granite-speech-4.1-2b",
) -> GraniteSpeechModel:
    tokenizer = FakeTokenizer()
    return GraniteSpeechModel(
        backend=FakeBackend(responses),
        processor=tokenizer,
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        spec=MODELS[model_name],
    )


def test_public_load_model_then_model_transcribe_path_workflow(monkeypatch, tmp_path):
    audio_path = write_silent_wav(tmp_path / "clip.wav")
    loaded = fake_model([GenerateResult(text="path transcript", tokens=[101, 202])])
    captured: dict[str, object] = {}

    def fake_load_llama_cpp_model(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return loaded

    monkeypatch.setattr(
        "granite_speech.loader._load_llama_cpp_model",
        fake_load_llama_cpp_model,
    )

    model = granite_speech.load_model(
        "granite-speech-4.1-2b",
        device="cpu",
        local_files_only=True,
    )
    result = model.transcribe(
        audio_path,
        initial_prompt="transcribe this IBM Granite update",
        temperature=0.2,
    )

    assert model is loaded
    assert captured["spec"].name == "granite-speech-4.1-2b"
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["local_files_only"] is True
    assert result == {
        "text": "path transcript",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "path transcript",
                "temperature": 0.2,
                "tokens": [101, 202],
            }
        ],
        "language": None,
        "target_language": None,
        "warnings": [],
    }
    assert loaded.backend.calls[0].instruction == "transcribe this IBM Granite update"
    assert loaded.backend.calls[0].wav.shape == (1, 16000)


def test_package_transcribe_uses_cached_model_and_initial_prompt(monkeypatch, tmp_path):
    audio_path = write_silent_wav(tmp_path / "clip.wav")
    loaded = fake_model(["first transcript", "second transcript"])
    load_names: list[str] = []

    def fake_load_model(name):
        load_names.append(name)
        return loaded

    monkeypatch.setattr(granite_speech, "_MODEL_CACHE", {})
    monkeypatch.setattr(granite_speech, "load_model", fake_load_model)

    first = granite_speech.transcribe(
        audio_path,
        model="granite-speech-4.1-2b",
        initial_prompt="prefer IBM Granite terms",
    )
    second = granite_speech.transcribe(audio_path, model="granite-speech-4.1-2b")

    assert load_names == ["granite-speech-4.1-2b"]
    assert first["text"] == "first transcript"
    assert first["segments"][0] == {
        "id": 0,
        "start": 0.0,
        "end": 1.0,
        "text": "first transcript",
        "temperature": 0.0,
    }
    assert first["language"] is None
    assert first["target_language"] is None
    assert first["warnings"] == []
    assert second["text"] == "second transcript"
    assert loaded.backend.calls[0].instruction == "prefer IBM Granite terms"
    assert len(loaded.backend.calls) == 2


def test_package_transcribe_supports_plus_word_timestamps_alias(monkeypatch, tmp_path):
    audio_path = write_silent_wav(tmp_path / "clip.wav")
    loaded = fake_model(["hello [T:45] world [T:82]"], model_name="granite-speech-4.1-2b-plus")

    monkeypatch.setattr(granite_speech, "_MODEL_CACHE", {})
    monkeypatch.setattr(granite_speech, "load_model", lambda _name: loaded)

    with pytest.warns(UserWarning, match="word_timestamps=True maps"):
        result = granite_speech.transcribe(
            audio_path,
            model="granite-speech-4.1-2b-plus",
            word_timestamps=True,
        )

    assert result["text"] == "hello world"
    assert result["words"] == [
        {"word": "hello", "start": 0.0, "end": 0.45},
        {"word": "world", "start": 0.45, "end": 0.82},
    ]
    assert result["segments"][0]["id"] == 0
    assert result["segments"][0]["temperature"] == 0.0
    assert "Timestamps: Transcribe the speech" in loaded.backend.calls[0].instruction


def test_cli_writes_srt_and_passes_initial_prompt(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class StubModel:
        def transcribe(self, audio_path, **kwargs):
            captured["audio_path"] = audio_path
            captured["transcribe_kwargs"] = kwargs
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
                "language": None,
                "target_language": None,
                "warnings": [],
            }

    def fake_load_model(name, **kwargs):
        captured["model_name"] = name
        captured["load_kwargs"] = kwargs
        return StubModel()

    monkeypatch.setattr(cli, "load_model", fake_load_model)

    code = cli.main(
        [
            "clip.wav",
            "--initial_prompt",
            "prefer IBM Granite terms",
            "--output_format",
            "srt",
            "--output_dir",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert captured["audio_path"] == "clip.wav"
    assert captured["transcribe_kwargs"]["initial_prompt"] == "prefer IBM Granite terms"
    assert captured["transcribe_kwargs"]["word_timestamps"] is False
    assert (tmp_path / "clip.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,250\nhello world\n"
    )
