from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from granite_speech._backends.fake import FakeBackend
from granite_speech._models import MODELS, resolve_model_spec, validate_translation_pair
from granite_speech.audio import load_audio
from granite_speech.errors import InvalidArgumentError
from granite_speech.loader import download_llama_cpp_model
from granite_speech.model import GraniteSpeechModel
from granite_speech.prompts import (
    PLUS_SYSTEM_PROMPT,
    add_keyword_bias_instruction,
    build_prompt,
    default_instruction,
)


class FakeTokenizer:
    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT:{chat[0]['content']}:GEN"


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        self.calls.append({"chat": chat, "kwargs": kwargs})
        prefix = f":PREFIX:{kwargs['prefix_text']}" if "prefix_text" in kwargs else ""
        return f"CHAT:{chat[-1]['content']}:GEN{prefix}"


class LlamaCppPromptCaptureBackend(FakeBackend):
    name = "llama.cpp"


def fake_llama_cpp_model(responses) -> GraniteSpeechModel:
    return GraniteSpeechModel(
        backend=LlamaCppPromptCaptureBackend(responses),
        processor=None,
        model="/models/granite.gguf",
        tokenizer=None,
        device="auto",
        spec=MODELS["granite-speech-4.1-2b"],
    )


def fake_plus_model(responses, tokenizer=None) -> GraniteSpeechModel:
    tokenizer = tokenizer or RecordingTokenizer()
    return GraniteSpeechModel(
        backend=FakeBackend(responses),
        processor=tokenizer,
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        spec=MODELS["granite-speech-4.1-2b-plus"],
    )


def test_model_card_prompt_forms_are_rendered_by_defaults():
    assert (
        default_instruction("transcribe", None, None)
        == "transcribe the speech with proper punctuation and capitalization."
    )
    assert (
        default_instruction("transcribe", None, None, keyword_biased=True)
        == "transcribe the speech to text."
    )
    assert (
        add_keyword_bias_instruction(
            default_instruction("transcribe", None, None, keyword_biased=True),
            ["kw1", "kw2"],
        )
        == "transcribe the speech to text. Keywords: kw1, kw2"
    )
    assert (
        default_instruction("translate", "fr", "en")
        == "translate the speech to English with proper punctuation and capitalization."
    )
    assert (
        default_instruction("translate", "en", "zh")
        == "translate the speech to Mandarin with proper punctuation and capitalization."
    )
    assert (
        add_keyword_bias_instruction(
            default_instruction("translate", "en", "de", keyword_biased=True),
            ["kw1", "kw2"],
        )
        == "translate the speech to German. Keywords: kw1, kw2"
    )


def test_plus_model_card_prompt_forms_are_rendered_by_defaults():
    spec = resolve_model_spec("granite-speech-4.1-2b-plus")

    asr_prompt = "can you transcribe the speech into a written format?"
    saa_prompt = (
        "Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: "
        "and [Speaker 2]: tags before speaker turns."
    )
    timestamp_prompt = (
        "Timestamps: Transcribe the speech. After each word, add a timestamp tag showing "
        "the end time in centiseconds, e.g. hello [T:45] world [T:82]"
    )

    assert default_instruction("transcribe", None, None, model_spec=spec) == asr_prompt
    assert (
        add_keyword_bias_instruction(
            default_instruction("transcribe", None, None, model_spec=spec),
            ["kw1", "kw2"],
        )
        == "can you transcribe the speech into a written format? Keywords: kw1, kw2"
    )
    assert (
        default_instruction(
            "transcribe",
            None,
            None,
            model_spec=spec,
            prompt_mode="speaker_attributed",
        )
        == saa_prompt
    )
    assert (
        default_instruction(
            "transcribe",
            None,
            None,
            model_spec=spec,
            prompt_mode="word_timestamps",
        )
        == timestamp_prompt
    )


def test_model_card_prompt_forms_reach_llama_cpp_backend_exactly():
    model = fake_llama_cpp_model(
        ["raw asr", "punct asr", "kw asr", "raw ast", "punct ast", "kw ast"]
    )
    wav = np.zeros(16000, dtype=np.float32)

    cases = [
        (
            {"prompt": "can you transcribe the speech into a written format?"},
            "can you transcribe the speech into a written format?",
        ),
        (
            {},
            "transcribe the speech with proper punctuation and capitalization.",
        ),
        (
            {"keyword_biases": ["kw1", "kw2"]},
            "transcribe the speech to text. Keywords: kw1, kw2",
        ),
        (
            {
                "task": "translate",
                "language": "fr",
                "target_language": "en",
                "prompt": "translate the speech to English.",
            },
            "translate the speech to English.",
        ),
        (
            {"task": "translate", "language": "fr", "target_language": "en"},
            "translate the speech to English with proper punctuation and capitalization.",
        ),
        (
            {
                "task": "translate",
                "language": "en",
                "target_language": "de",
                "keyword_biases": ["kw1", "kw2"],
            },
            "translate the speech to German. Keywords: kw1, kw2",
        ),
    ]

    for kwargs, _expected_instruction in cases:
        model.transcribe(wav, sample_rate=16000, **kwargs)

    assert [call.instruction for call in model.backend.calls] == [
        expected_instruction for _kwargs, expected_instruction in cases
    ]
    assert [call.prompt for call in model.backend.calls] == [
        f"<|audio|>{expected_instruction}" for _kwargs, expected_instruction in cases
    ]


def test_plus_model_card_prompt_forms_reach_transformers_backend_exactly():
    tokenizer = RecordingTokenizer()
    model = fake_plus_model(["asr", "kw asr", "full asr", "saa", "ts"], tokenizer)
    wav = np.zeros(16000, dtype=np.float32)

    asr_prompt = "can you transcribe the speech into a written format?"
    saa_prompt = (
        "Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: "
        "and [Speaker 2]: tags before speaker turns."
    )
    timestamp_prompt = (
        "Timestamps: Transcribe the speech. After each word, add a timestamp tag showing "
        "the end time in centiseconds, e.g. hello [T:45] world [T:82]"
    )

    cases = [
        ({}, asr_prompt, f"<|audio|> {asr_prompt}"),
        (
            {"keyword_biases": ["kw1", "kw2"]},
            f"{asr_prompt} Keywords: kw1, kw2",
            f"<|audio|> {asr_prompt} Keywords: kw1, kw2",
        ),
        (
            {"prompt": f"<|audio|> {asr_prompt}"},
            asr_prompt,
            f"<|audio|> {asr_prompt}",
        ),
        (
            {"prompt_mode": "speaker_attributed"},
            saa_prompt,
            f"<|audio|> {saa_prompt}",
        ),
        (
            {"prompt_mode": "word_timestamps", "max_new_tokens": 10000},
            timestamp_prompt,
            f"<|audio|> {timestamp_prompt}",
        ),
    ]

    for kwargs, _expected_instruction, _expected_content in cases:
        model.transcribe(wav, sample_rate=16000, **kwargs)

    assert [call.instruction for call in model.backend.calls] == [
        expected_instruction for _kwargs, expected_instruction, _expected_content in cases
    ]
    assert [call.prompt for call in model.backend.calls] == [
        f"CHAT:{expected_content}:GEN"
        for _kwargs, _expected_instruction, expected_content in cases
    ]
    assert [call.max_new_tokens for call in model.backend.calls] == [200, 200, 200, 200, 10000]
    for template_call, (_kwargs, _expected_instruction, expected_content) in zip(
        tokenizer.calls,
        cases,
        strict=True,
    ):
        assert template_call["chat"] == [
            {"role": "system", "content": PLUS_SYSTEM_PROMPT},
            {"role": "user", "content": expected_content},
        ]
        assert template_call["kwargs"] == {}


def test_plus_model_card_incremental_prefix_reaches_chat_template():
    tokenizer = RecordingTokenizer()
    model = fake_plus_model(["incremental"], tokenizer)
    wav = np.zeros(16000, dtype=np.float32)

    model.transcribe(
        wav,
        sample_rate=16000,
        prompt_mode="speaker_attributed",
        prefix_text="[Speaker 1]: Hello how are you",
    )

    assert tokenizer.calls[0]["kwargs"] == {
        "prefix_text": "[Speaker 1]: Hello how are you",
    }
    assert model.backend.calls[0].prompt.endswith(":PREFIX:[Speaker 1]: Hello how are you")


def test_plus_speaker_tags_are_parsed_into_structured_turns():
    raw = "[Speaker 1]: Hello there.\n[Speaker 2]: Hi."
    model = fake_plus_model([raw])
    wav = np.zeros(16000, dtype=np.float32)

    result = model.transcribe(wav, sample_rate=16000, prompt_mode="speaker_attributed")

    assert result["text"] == "Hello there. Hi."
    assert result["speakers"] == [
        {"speaker": "Speaker 1", "text": "Hello there."},
        {"speaker": "Speaker 2", "text": "Hi."},
    ]
    assert result["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "Hello there. Hi.",
            "temperature": 0.0,
            "raw_text": raw,
            "speakers": result["speakers"],
        }
    ]


def test_plus_timestamp_tags_are_parsed_into_structured_words():
    raw = "hello [T:45] world [T:82]"
    model = fake_plus_model([raw])
    wav = np.zeros(16000, dtype=np.float32)

    result = model.transcribe(wav, sample_rate=16000, prompt_mode="word_timestamps")

    assert result["text"] == "hello world"
    assert result["words"] == [
        {"word": "hello", "start": 0.0, "end": 0.45},
        {"word": "world", "start": 0.45, "end": 0.82},
    ]
    assert result["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "hello world",
            "temperature": 0.0,
            "raw_text": raw,
            "words": result["words"],
        }
    ]


def test_word_timestamps_alias_maps_to_plus_prompt_mode():
    raw = "hello [T:45]"
    model = fake_plus_model([raw])
    wav = np.zeros(16000, dtype=np.float32)

    with pytest.warns(UserWarning, match="word_timestamps=True maps"):
        result = model.transcribe(wav, sample_rate=16000, word_timestamps=True)

    assert result["words"] == [{"word": "hello", "start": 0.0, "end": 0.45}]
    assert "Timestamps: Transcribe the speech" in model.backend.calls[0].instruction


def test_plus_prompt_modes_are_validated_against_model_capabilities():
    base_model = fake_llama_cpp_model(["unused"])
    plus_model = fake_plus_model(["unused"])
    wav = np.zeros(16000, dtype=np.float32)

    with pytest.raises(InvalidArgumentError, match="speaker-attributed"):
        base_model.transcribe(wav, sample_rate=16000, prompt_mode="speaker_attributed")
    with pytest.raises(InvalidArgumentError, match="word timestamp"):
        base_model.transcribe(wav, sample_rate=16000, prompt_mode="word_timestamps")
    with pytest.raises(InvalidArgumentError, match="prefix_text"):
        base_model.transcribe(wav, sample_rate=16000, prefix_text="previous")
    with pytest.raises(InvalidArgumentError, match="prompt_mode"):
        plus_model.transcribe(wav, sample_rate=16000, prompt_mode="bogus")


def test_model_card_raw_asr_prompt_can_be_passed_explicitly():
    prompt = build_prompt(
        FakeTokenizer(),
        task="transcribe",
        language=None,
        target_language=None,
        prompt="can you transcribe the speech into a written format?",
        keyword_biases=None,
    )

    assert prompt.instruction == "can you transcribe the speech into a written format?"
    assert (
        prompt.prompt
        == "CHAT:<|audio|>can you transcribe the speech into a written format?:GEN"
    )


def test_model_card_translation_pairs_are_registered():
    spec = resolve_model_spec("granite-speech-4.1-2b")

    for source, target in [
        ("fr", "en"),
        ("de", "en"),
        ("es", "en"),
        ("pt", "en"),
        ("ja", "en"),
        ("en", "fr"),
        ("en", "de"),
        ("en", "es"),
        ("en", "pt"),
        ("en", "ja"),
        ("en", "it"),
        ("en", "mandarin"),
    ]:
        validate_translation_pair(spec, source, target)


def test_model_card_torchaudio_shape_is_accepted_for_raw_audio():
    wav = np.zeros((1, 16000), dtype=np.float32)

    loaded = load_audio(wav, sample_rate=16000)

    assert loaded.wav.shape == (1, 16000)
    assert loaded.sample_rate == 16000


def test_llama_cpp_model_card_q8_quant_downloads_expected_files(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "granite-speech-4.1-2b-Q8_0.gguf").write_bytes(b"model")
    (snapshot / "mmproj-model-f16.gguf").write_bytes(b"mmproj")
    seen: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    model_path, mmproj_path = download_llama_cpp_model(
        "granite-speech-4.1-2b",
        quant="Q8_0",
        download_root=str(tmp_path / "cache"),
    )

    assert model_path == str(snapshot / "granite-speech-4.1-2b-Q8_0.gguf")
    assert mmproj_path == str(snapshot / "mmproj-model-f16.gguf")
    assert seen["repo_id"] == "ibm-granite/granite-speech-4.1-2b-GGUF"
    assert seen["allow_patterns"] == [
        "granite-speech-4.1-2b-Q8_0.gguf",
        "mmproj-model-f16.gguf",
    ]


def test_plus_llama_cpp_model_card_q8_quant_downloads_expected_files(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "granite-speech-4.1-2b-plus-Q8_0.gguf").write_bytes(b"model")
    (snapshot / "mmproj-model-f16.gguf").write_bytes(b"mmproj")
    seen: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    model_path, mmproj_path = download_llama_cpp_model(
        "granite-speech-4.1-2b-plus-GGUF",
        quant="Q8_0",
        download_root=str(tmp_path / "cache"),
    )

    assert model_path == str(snapshot / "granite-speech-4.1-2b-plus-Q8_0.gguf")
    assert mmproj_path == str(snapshot / "mmproj-model-f16.gguf")
    assert seen["repo_id"] == "ibm-granite/granite-speech-4.1-2b-plus-GGUF"
    assert seen["allow_patterns"] == [
        "granite-speech-4.1-2b-plus-Q8_0.gguf",
        "mmproj-model-f16.gguf",
    ]
