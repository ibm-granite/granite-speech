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

## Real-weights llama.cpp plus smoke

`granite-speech-4.1-2b-plus` adds word-timestamp and speaker-attribution output (and drops
translation). Its smoke lives in `tests/test_real_weights_smoke_plus.py` and reuses the same
committed `multilingual_sample.wav` fixture:

```bash
uv run pytest tests/test_real_weights_smoke_plus.py -m real_weights
```

It loads `granite-speech-4.1-2b-plus`, asserts the `llama.cpp` backend, transcribes the fixture,
exercises `prompt_mode="word_timestamps"` (asserting per-word `word`/`start`/`end` entries) and
`prompt_mode="speaker_attributed"` (asserting `speaker`/`text` turns), and requires no package or
transcription warnings. The speaker-attribution assertions are shape-only because the fixture is
not guaranteed to contain multiple speakers.

Plus support landed in llama.cpp via [PR #24818](https://github.com/ggml-org/llama.cpp/pull/24818);
earlier builds crash when encoding the plus audio features. The smoke reads `llama-cli --version`
and skips (rather than fails) when the build predates `MIN_LLAMA_CPP_BUILD` (currently `9850`, the
earliest build verified to pass).

Environment overrides mirror the base smoke under a `GRANITE_SPEECH_PLUS_SMOKE_*` prefix
(`_DOWNLOAD_ROOT`, `_LOCAL_FILES_ONLY`, `_MODEL_REVISION`, `_LLAMA_CPP_BINARY`, `_LLAMA_CPP_QUANT`,
`_TIMEOUT`, `_AUDIO`, `_AUDIO_REVISION`, `_AUDIO_SHA256`), plus
`GRANITE_SPEECH_PLUS_SMOKE_MIN_LLAMA_CPP_BUILD` to change or disable (set to `0`) the version gate.

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
