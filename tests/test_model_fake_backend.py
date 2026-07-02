from __future__ import annotations

import numpy as np
import pytest

from granite_speech._backends import GenerateResult
from granite_speech._backends.fake import FakeBackend
from granite_speech._models import MODELS
from granite_speech.errors import InvalidArgumentError, TranscriptionError
from granite_speech.model import GraniteSpeechModel


class FakeTokenizer:
    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT:{chat[0]['content']}:GEN"


def fake_model(responses) -> GraniteSpeechModel:
    tokenizer = FakeTokenizer()
    return GraniteSpeechModel(
        backend=FakeBackend(responses),
        processor=tokenizer,
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        spec=MODELS["granite-speech-4.1-2b"],
    )


def mono(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def test_short_audio_yields_one_window_and_result_contract():
    model = fake_model(["hello world"])

    result = model.transcribe(mono(5), sample_rate=16000)

    assert result["text"] == "hello world"
    assert result["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 5.0,
            "text": "hello world",
            "temperature": 0.0,
        }
    ]
    assert "tokens" not in result["segments"][0]
    assert "avg_logprob" not in result["segments"][0]
    assert "compression_ratio" not in result["segments"][0]
    assert "no_speech_prob" not in result["segments"][0]
    assert result["language"] is None
    assert result["target_language"] is None
    assert result["warnings"] == []
    assert model.backend.calls[0].wav.shape == (1, 5 * 16000)
    assert model.backend.calls[0].prompt.startswith("CHAT:<|audio|>")


def test_backend_tokens_are_preserved_when_available():
    model = fake_model([GenerateResult(text="tokenized transcript", tokens=[101, 202, 303])])

    result = model.transcribe(mono(1), sample_rate=16000, temperature=0.2)

    assert result["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "tokenized transcript",
            "temperature": 0.2,
            "tokens": [101, 202, 303],
        }
    ]


def test_empty_audio_is_successful_empty_result():
    model = fake_model([])

    result = model.transcribe(np.zeros(0, dtype=np.float32), sample_rate=16000)

    assert result["text"] == ""
    assert result["segments"] == []
    assert result["warnings"] == []


def test_vad_silence_skips_backend_and_returns_empty_result():
    model = fake_model(["unused"])

    result = model.transcribe(mono(10), sample_rate=16000, segmentation="vad")

    assert result["text"] == ""
    assert result["segments"] == []
    assert result["warnings"] == []
    assert model.backend.calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_length": 0}, "chunk_length"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        ({"chunk_length": 10, "chunk_overlap": 10}, "chunk_overlap"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 0.7, "num_beams": 2}, "num_beams"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"num_beams": 0}, "num_beams"),
        ({"keyword_biases": [" "]}, "keyword_biases"),
        ({"keyword_biases": [123]}, "keyword_biases"),
        ({"segmentation": "unknown"}, "segmentation"),
        ({"segmentation": "vad", "chunk_overlap": 1}, "chunk_overlap"),
        ({"segmentation": "vad", "vad_threshold": 0}, "vad_threshold"),
        ({"segmentation": "vad", "vad_threshold": 1.1}, "vad_threshold"),
        ({"segmentation": "vad", "vad_min_speech_duration": -0.1}, "vad_min_speech"),
        ({"segmentation": "vad", "vad_min_silence_duration": -0.1}, "vad_min_silence"),
        ({"segmentation": "vad", "vad_speech_pad": -0.1}, "vad_speech_pad"),
    ],
)
def test_invalid_generation_and_chunking_arguments(kwargs, message):
    model = fake_model(["unused"])

    with pytest.raises(InvalidArgumentError, match=message):
        model.transcribe(mono(1), sample_rate=16000, **kwargs)


def test_keyword_biases_are_rendered_and_passed_to_backend():
    model = fake_model(["Granite Speech mentions watsonx.ai"])

    result = model.transcribe(
        mono(1),
        sample_rate=16000,
        keyword_biases=["Granite Speech", "watsonx.ai", "Granite Speech"],
    )

    assert result["text"] == "Granite Speech mentions watsonx.ai"
    call = model.backend.calls[0]
    assert call.keyword_biases == ("Granite Speech", "watsonx.ai")
    assert (
        call.instruction
        == "transcribe the speech to text. Keywords: Granite Speech, watsonx.ai"
    )
    assert "Keywords: Granite Speech, watsonx.ai" in call.prompt


def test_keyword_biases_extend_custom_prompt():
    model = fake_model(["domain transcript"])

    model.transcribe(
        mono(1),
        sample_rate=16000,
        prompt="transcribe this quarterly earnings call",
        keyword_biases="Qiskit",
    )

    assert model.backend.calls[0].instruction == (
        "transcribe this quarterly earnings call. Keywords: Qiskit"
    )


def test_whisper_initial_prompt_alias_maps_to_native_prompt():
    model = fake_model(["domain transcript"])

    result = model.transcribe(
        mono(1),
        sample_rate=16000,
        initial_prompt="transcribe this quarterly earnings call",
    )

    assert result["text"] == "domain transcript"
    call = model.backend.calls[0]
    assert call.instruction == "transcribe this quarterly earnings call"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"prompt": "native prompt", "initial_prompt": "whisper prompt"},
            "prompt or initial_prompt",
        ),
        (
            {"prompt_mode": "speaker_attributed", "word_timestamps": True},
            "word_timestamps=True",
        ),
        ({"word_timestamps": "yes"}, "word_timestamps"),
    ],
)
def test_whisper_alias_conflicts_are_rejected(kwargs, message):
    model = fake_model(["unused"])

    with pytest.raises(InvalidArgumentError, match=message):
        model.transcribe(mono(1), sample_rate=16000, **kwargs)


def test_whisper_temperature_fallback_warns_and_uses_first_value():
    model = fake_model(["sampled transcript"])

    with pytest.warns(UserWarning, match="temperature fallback"):
        model.transcribe(mono(1), sample_rate=16000, temperature=(0.2, 0.4))

    assert model.backend.calls[0].temperature == 0.2


def test_harmless_unsupported_whisper_options_warn_and_continue():
    model = fake_model(["ok", "still ok", "fp32 ok"])

    with pytest.warns(UserWarning, match="condition_on_previous_text"):
        assert (
            model.transcribe(
                mono(1),
                sample_rate=16000,
                condition_on_previous_text=True,
            )["text"]
            == "ok"
        )

    with pytest.warns(UserWarning, match="no_speech_threshold"):
        assert model.transcribe(mono(1), sample_rate=16000, no_speech_threshold=0.6)[
            "text"
        ] == "still ok"

    with pytest.warns(UserWarning, match="fp16"):
        assert model.transcribe(mono(1), sample_rate=16000, fp16=False)["text"] == "fp32 ok"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip_timestamps": "10,20"},
        {"no_speech_threshold": 0.2},
        {"beam_size": 5},
        {"best_of": 5},
        {"suppress_tokens": "-1"},
        {"unknown_whisper_option": True},
    ],
)
def test_transcript_affecting_unsupported_whisper_options_are_rejected(kwargs):
    model = fake_model(["unused"])

    with pytest.raises(InvalidArgumentError, match="unsupported"):
        model.transcribe(mono(1), sample_rate=16000, **kwargs)


def test_keyword_biases_use_translation_keyword_prompt():
    model = fake_model(["translated domain transcript"])

    model.transcribe(
        mono(1),
        sample_rate=16000,
        task="translate",
        language="fr",
        target_language="en",
        keyword_biases=["Qiskit"],
    )

    assert model.backend.calls[0].instruction == (
        "translate the speech to English. Keywords: Qiskit"
    )


def test_transcribe_rejects_target_language():
    model = fake_model(["unused"])

    with pytest.raises(InvalidArgumentError, match="target_language"):
        model.transcribe(mono(1), sample_rate=16000, target_language="fr")


def test_translate_requires_source_and_distinct_supported_target():
    model = fake_model(["unused"])

    with pytest.raises(InvalidArgumentError, match="requires an explicit source"):
        model.transcribe(mono(1), sample_rate=16000, task="translate")

    with pytest.raises(InvalidArgumentError, match="same language"):
        model.transcribe(mono(1), sample_rate=16000, task="translate", language="en")

    result = model.transcribe(
        mono(1),
        sample_rate=16000,
        task="translate",
        language="fr",
        target_language=None,
    )
    assert result["language"] == "fr"
    assert result["target_language"] == "en"


def test_overlap_geometry_and_boundary_agreement_dedup():
    model = fake_model(["hello shared words", "shared words next", "next tail"])

    result = model.transcribe(
        mono(50),
        sample_rate=16000,
        chunk_length=30,
        chunk_overlap=10,
    )

    assert result["text"] == "hello shared words next tail"
    assert [(seg["start"], seg["end"]) for seg in result["segments"]] == [
        (0.0, 25.0),
        (25.0, 45.0),
        (45.0, 50.0),
    ]
    assert [seg["text"] for seg in result["segments"]] == [
        "hello shared words",
        "next",
        "tail",
    ]


def test_overlap_allows_single_word_boundary_agreement():
    model = fake_model(["the quick brown", "brown fox jumps"])

    result = model.transcribe(
        mono(35),
        sample_rate=16000,
        chunk_length=30,
        chunk_overlap=10,
    )

    assert result["text"] == "the quick brown fox jumps"
    assert result["segments"][1]["text"] == "fox jumps"


def test_overlap_reconciliation_drops_disagreed_boundary_edges():
    model = fake_model(["alpha noisy", "wrong omega"])

    result = model.transcribe(
        mono(35),
        sample_rate=16000,
        chunk_length=30,
        chunk_overlap=10,
    )

    assert result["text"] == "alpha omega"
    assert [seg["text"] for seg in result["segments"]] == ["alpha", "omega"]


def test_partial_window_failure_is_warning_and_segment_slot():
    model = fake_model(["ok", RuntimeError("boom"), "after"])

    result = model.transcribe(mono(65), sample_rate=16000, chunk_length=30)

    assert result["text"] == "ok after"
    assert result["segments"][1]["error"] == "boom"
    assert result["warnings"] == [{"type": "window_error", **result["segments"][1]}]


def test_every_window_failure_raises():
    model = fake_model([RuntimeError("boom")])

    with pytest.raises(TranscriptionError, match="every audio window failed"):
        model.transcribe(mono(1), sample_rate=16000)


def test_raw_audio_sample_rate_is_required():
    model = fake_model(["unused"])

    with pytest.raises(InvalidArgumentError, match="sample_rate is required"):
        model.transcribe(mono(1))
