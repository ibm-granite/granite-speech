# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from granite_speech._backends import BackendCapabilities, GenerateRequest
from granite_speech._backends.llama_cpp import LlamaCppBackend, _extract_transcript


def capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        max_reliable_audio_seconds=120.0,
        supports_word_timing_output=False,
        supports_speaker_attribution_output=False,
        supports_batch=False,
        supports_translation=True,
    )


def test_llama_cpp_backend_runs_single_turn_audio_command(monkeypatch):
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        audio_path = Path(cmd[cmd.index("--audio") + 1])
        assert audio_path.exists()
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="""
Loading model...
Loaded media from '/tmp/audio.wav'

> transcribe the speech with proper punctuation and capitalization.

This is a granite speech smoke test.

Exiting...
""",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = LlamaCppBackend(
        binary="llama-cli",
        model_path="/models/granite.gguf",
        mmproj_path="/models/mmproj.gguf",
        capabilities=capabilities(),
        extra_args=["--threads", "4"],
        timeout=30,
    )

    result = backend.generate(
        GenerateRequest(
            prompt="<|audio|>ignored when instruction is available",
            instruction="transcribe the speech with proper punctuation and capitalization.",
            wav=np.zeros((1, 1600), dtype=np.float32),
            sample_rate=16000,
            max_new_tokens=64,
            temperature=0.0,
        )
    )

    assert result.text == "This is a granite speech smoke test."
    cmd = seen["cmd"]
    assert cmd == [
        "llama-cli",
        "--single-turn",
        "--model",
        "/models/granite.gguf",
        "--mmproj",
        "/models/mmproj.gguf",
        "--audio",
        cmd[cmd.index("--audio") + 1],
        "--prompt",
        "transcribe the speech with proper punctuation and capitalization.",
        "--predict",
        "64",
        "--temperature",
        "0.0",
        "--no-warmup",
        "--log-disable",
        "--no-display-prompt",
        "--no-show-timings",
        "--simple-io",
        "--threads",
        "4",
    ]
    assert seen["kwargs"]["timeout"] == 30
    assert not Path(cmd[cmd.index("--audio") + 1]).exists()


def test_extract_transcript_ignores_llama_cli_framing():
    output = """
Loading model...

build      : b9630
model      : granite-speech-4.1-2b-Q8_0.gguf
modalities : text, audio

Loaded media from '/tmp/input.wav'

> transcribe the speech with proper punctuation and capitalization.

This is a granite speech smoke test.

[ Prompt: 512.5 t/s | Generation: 170.8 t/s ]

Exiting...
"""

    text = _extract_transcript(
        output,
        prompt="transcribe the speech with proper punctuation and capitalization.",
    )

    assert text == "This is a granite speech smoke test."
