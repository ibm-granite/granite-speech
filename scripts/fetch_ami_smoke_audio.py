#!/usr/bin/env python3
# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the AMI smoke-test fixture used by the plus-model real-weights smoke.

This reproduces the example-audio selection from the granite-speech-4.1-2b-plus model card
(https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus): test sample 0 of the AMI ``ihm``
config, the 5:00-6:00 minute window. The clip is genuinely multi-speaker, which is what makes it a
faithful fixture for the speaker-attribution and incremental-decoding tasks.

The output WAV is committed to ``tests/fixtures/`` and pinned by SHA-256 in
``tests/test_real_weights_smoke_plus.py`` and ``tests/fixtures/README.md``. This script is a
one-time author-machine helper: it is not imported by the package or the test suite, and its
dependencies (``datasets`` plus an audio decoder backend) are intentionally not runtime or
default-test deps.

Usage (in a scratch environment, not the package venv)::

    pip install "datasets>=3" soundfile torchcodec
    python scripts/fetch_ami_smoke_audio.py

Then copy the printed SHA-256 into the test module and the fixtures README.
"""

from __future__ import annotations

import argparse
import sys
from hashlib import sha256
from pathlib import Path

# Card verbatim: ds = load_dataset("diarizers-community/ami", "ihm", split="test")
AMI_REPO_ID = "diarizers-community/ami"
AMI_CONFIG = "ihm"
AMI_SPLIT = "test"
# Pin the dataset revision so the fixture is reproducible. main @ time of authoring.
AMI_REVISION = "main"

TEST_SAMPLE = 0
START_TIME, END_TIME = 5 * 60, 6 * 60  # seconds; the card's 5:00-6:00 window

# granite_speech.audio.SAMPLE_RATE resamples everything to 16 kHz anyway; writing the fixture at
# 16 kHz mono keeps it small and decode-stable.
TARGET_SAMPLE_RATE = 16000

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ami_ihm_sample0_5m-6m.wav"
)


def _require(module: str, pip_name: str | None = None):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - author-machine helper
        pip_name = pip_name or module
        raise SystemExit(
            f"{module} is required to regenerate the AMI smoke fixture but is not installed.\n"
            f"Install it into a scratch environment (not the package venv):\n"
            f'    pip install "datasets>=3" soundfile torchcodec\n'
            f"(missing: {pip_name})"
        ) from exc


def fetch(output: Path, *, revision: str, cache_dir: str | None) -> Path:
    _require("datasets")
    soundfile = _require("soundfile")
    import numpy as np
    from datasets import load_dataset

    print(f"Loading {AMI_REPO_ID} ({AMI_CONFIG}, split={AMI_SPLIT}, revision={revision})...")
    ds = load_dataset(
        AMI_REPO_ID, AMI_CONFIG, split=AMI_SPLIT, revision=revision, cache_dir=cache_dir
    )

    print(f"Selecting sample {TEST_SAMPLE}, window {START_TIME}-{END_TIME}s...")
    # Newer datasets Audio features return a torchcodec AudioDecoder exposing
    # get_samples_played_in_range; this mirrors the model card exactly.
    audio = ds["audio"][TEST_SAMPLE].get_samples_played_in_range(START_TIME, END_TIME)

    samples = np.asarray(audio.data)  # (channels, num_samples)
    source_sr = int(audio.sample_rate)

    # Downmix to mono.
    if samples.ndim == 2:
        mono = samples.mean(axis=0)
    else:
        mono = samples
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    if source_sr != TARGET_SAMPLE_RATE:
        librosa = _require("librosa")
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=TARGET_SAMPLE_RATE)
        mono = np.ascontiguousarray(mono, dtype=np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(output), mono, TARGET_SAMPLE_RATE, subtype="PCM_16")

    duration = len(mono) / TARGET_SAMPLE_RATE
    digest = sha256(output.read_bytes()).hexdigest()
    print()
    print(f"Wrote {output} ({duration:.2f}s @ {TARGET_SAMPLE_RATE} Hz mono)")
    print(f"SHA-256: {digest}")
    print()
    print(
        "Copy this SHA-256 into tests/test_real_weights_smoke_plus.py and tests/fixtures/README.md."
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output WAV path")
    parser.add_argument("--revision", default=AMI_REVISION, help="AMI dataset revision to pin")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face datasets cache dir")
    args = parser.parse_args(argv)

    fetch(args.output, revision=args.revision, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
