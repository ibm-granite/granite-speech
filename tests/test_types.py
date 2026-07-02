"""Drift guard for the public result TypedDicts (`granite_speech/types.py`).

No type checker runs in CI (only ruff, which does not validate TypedDict
conformance), so these tests substitute for it: they assert the public entry
points advertise `TranscriptionResult` and that a real result never grows a key
the TypedDicts do not declare — including that the deliberately-omitted Whisper
confidence fields stay absent.
"""

from __future__ import annotations

import typing

import numpy as np

import granite_speech
from granite_speech import Segment, Speaker, TranscriptionResult, Word
from granite_speech._backends.fake import FakeBackend
from granite_speech._models import MODELS
from granite_speech.model import GraniteSpeechModel


class _FakeTokenizer:
    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        return f"CHAT:{chat[-1]['content']}:GEN"


def _fake_model(responses, *, model_name: str = "granite-speech-4.1-2b") -> GraniteSpeechModel:
    tokenizer = _FakeTokenizer()
    return GraniteSpeechModel(
        backend=FakeBackend(responses),
        processor=tokenizer,
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        spec=MODELS[model_name],
    )


def _mono(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def _declared_keys(typed_dict) -> set[str]:
    return set(typed_dict.__required_keys__) | set(typed_dict.__optional_keys__)


# Whisper fields Granite Speech deliberately does not fabricate; must never appear.
_OMITTED_WHISPER_FIELDS = {
    "seek",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
    "duration",
}


def test_public_transcribe_advertises_transcription_result():
    for func in (granite_speech.transcribe, GraniteSpeechModel.transcribe):
        hints = typing.get_type_hints(func)
        assert hints["return"] is TranscriptionResult


def _assert_keys_declared(result: dict) -> None:
    assert set(result) <= _declared_keys(TranscriptionResult)
    assert not (_OMITTED_WHISPER_FIELDS & set(result))

    for segment in result["segments"]:
        assert set(segment) <= _declared_keys(Segment)
        assert not (_OMITTED_WHISPER_FIELDS & set(segment))
        for word in segment.get("words", []):
            assert set(word) <= _declared_keys(Word)
        for speaker in segment.get("speakers", []):
            assert set(speaker) <= _declared_keys(Speaker)

    for word in result.get("words", []):
        assert set(word) <= _declared_keys(Word)
    for speaker in result.get("speakers", []):
        assert set(speaker) <= _declared_keys(Speaker)


def test_default_result_only_uses_declared_keys():
    model = _fake_model(["hello world"])
    result = model.transcribe(_mono(5), sample_rate=16000)
    assert isinstance(result, dict)  # runtime is still a plain dict
    _assert_keys_declared(result)


def test_word_timestamp_result_only_uses_declared_keys():
    model = _fake_model(
        ["hello [T:45] world [T:82]"],
        model_name="granite-speech-4.1-2b-plus",
    )
    result = model.transcribe(
        _mono(5),
        sample_rate=16000,
        prompt_mode="word_timestamps",
    )
    assert result["words"]  # sanity: the mode actually produced words
    _assert_keys_declared(result)


def test_speaker_attributed_result_only_uses_declared_keys():
    model = _fake_model(
        ["[Speaker 1] hello [Speaker 2] world"],
        model_name="granite-speech-4.1-2b-plus",
    )
    result = model.transcribe(
        _mono(5),
        sample_rate=16000,
        prompt_mode="speaker_attributed",
    )
    assert result["speakers"]  # sanity: the mode actually produced speakers
    _assert_keys_declared(result)
