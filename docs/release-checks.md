# Release checks

Use these checks before publishing an alpha release.

## Unit, lint, build, and metadata

```bash
uv sync --extra dev
uv run pytest -m "not real_weights"
uv run ruff check .
uv build
uv run twine check dist/*
```

The GitHub Actions release gate runs the same lightweight checks and intentionally excludes
`real_weights` tests.

## Real-weights llama.cpp smoke

The real-weights smoke is opt-in because the GGUF model cache is several GB and requires
`llama-cli`.

By default, the smoke uses the committed `tests/fixtures/multilingual_sample.wav` fixture. The
fixture came from the official `ibm-granite/granite-speech-4.1-2b` Hugging Face repo and is pinned
to a known snapshot and SHA-256. Run:

```bash
uv run pytest tests/test_real_weights_smoke.py -m real_weights
```

The smoke loads the public `granite-speech-4.1-2b` model name through `granite_speech.load_model`,
asserts that it resolves to the default `llama.cpp` backend, transcribes the full fixture,
translates the French tail of the same fixture to English, requires at least one segment, and
requires no package or transcription warnings. Set `GRANITE_SPEECH_SMOKE_EXPECTED_TEXT` to
additionally compare the ASR output with a known transcript.

Useful environment overrides:

- `GRANITE_SPEECH_SMOKE_DOWNLOAD_ROOT`: model cache directory for the smoke.
- `GRANITE_SPEECH_SMOKE_LOCAL_FILES_ONLY=1`: require an already-populated cache.
- `GRANITE_SPEECH_SMOKE_AUDIO`: local WAV or FLAC fixture to use instead of the committed fixture.
- `GRANITE_SPEECH_SMOKE_AUDIO_REVISION`: Hugging Face revision for the fallback remote sample.
- `GRANITE_SPEECH_SMOKE_AUDIO_SHA256`: expected SHA-256 for the committed or fallback sample.
- `GRANITE_SPEECH_SMOKE_MODEL_REVISION`: Hugging Face revision for the GGUF model download.
- `GRANITE_SPEECH_SMOKE_LLAMA_CPP_BINARY`: path to `llama-cli` if it is not on `PATH`.
- `GRANITE_SPEECH_SMOKE_LLAMA_CPP_QUANT`: GGUF quantization, default `Q4_K_M`.
- `GRANITE_SPEECH_SMOKE_TIMEOUT`: llama.cpp subprocess timeout in seconds.
- `GRANITE_SPEECH_SMOKE_EXPECTED_TEXT`: optional transcript for close-enough text comparison.
- `GRANITE_SPEECH_TRANSLATION_SMOKE_EXPECTED_TEXT`: optional expected translation text.
- `GRANITE_SPEECH_SMOKE_SIMILARITY`: text similarity threshold from `0` to `1`, default `0.75`.

Expected first-run behavior:

- The base GGUF cache is approximately 4.6 GB.
- The committed audio fixture is approximately 1.6 MB.
- First run downloads model and mmproj files from Hugging Face unless the cache already exists.
- Device behavior is controlled by llama.cpp. Pass `--device` through the CLI or use CPU-only
  loading with the Python API's `device="cpu"` path when validating CPU.

## Built artifact validation

Run the package from a wheel in a clean environment:

```bash
scripts/check-built-artifacts.sh
```

This builds the wheel and sdist, installs the wheel into a temporary virtual environment, copies the
test suite outside the source tree, runs unit tests against the installed package, and runs the
real-weights smoke when `GRANITE_SPEECH_RUN_REAL_WEIGHTS=1` is set.

## Dependency policy

Runtime dependencies use lower bounds in `pyproject.toml`. Granite Speech does not currently set an
install-time upper bound for `transformers`; instead, the loader keeps a tested ceiling and warns
when the installed version is newer than release validation.

Known-good release validation versions are captured in
`constraints/release-validation.txt`.
