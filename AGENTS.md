# Repository Guidelines

## Project Structure & Module Organization

`granite_speech/` contains the package source and the `granite-speech` CLI. Public entry points live
in `__init__.py`, `model.py`, `loader.py`, `cli.py`, and `types.py`; keep stable API behavior there.
Model aliases and capability metadata are centralized in `_models.py`. Audio loading, clipping,
chunking, VAD segmentation, reconciliation, prompt rendering, plus-model output parsing, cache
resolution, and output writers are split across their matching modules.

Backend-specific code belongs in `granite_speech/_backends/`. The active default loader path is
llama.cpp/GGUF via `_backends/llama_cpp.py`; `_backends/transformers.py` is reserved for future
models and is not currently reachable from `load_model()`. Use `_backends/fake.py` for deterministic
tests.

Tests live in `tests/` and mirror behavior areas: audio, chunking, segmentation, reconciliation,
writers, CLI, model registry/cache handling, Whisper compatibility, model-card prompts, plus output,
and llama.cpp backend command construction. Opt-in real-weight smoke tests are marked
`real_weights`. Docs live in `README.md` and `docs/`; optional integrations and model-card examples
live in `examples/`, which has its own `pyproject.toml`.

## Build, Test, and Development Commands

- `uv sync --extra dev`: create or update the local development environment with pytest, Ruff, and
  Twine.
- `uv run pytest -m "not real_weights"`: run the normal unit test suite; this is the GitHub Actions
  release-gate test command.
- `uv run pytest tests/test_audio.py`: run a focused test file while iterating.
- `uv run ruff check .`: lint imports and Python style with the repo's Ruff settings.
- `uv build && uv run twine check dist/*`: build and validate package metadata.
- `scripts/check-built-artifacts.sh`: build a wheel, install it into a temporary venv, and run tests
  against the installed package.
- `scripts/smoke-testpypi.sh [version]`: install the published Test PyPI build into a temporary venv
  (deps resolve from real PyPI) and run the base + plus real-weights smoke tests against it. Requires
  `llama-cli` (build >= 9850 for the plus suite); set `GRANITE_SPEECH_SMOKE_SUITE=base|plus` to narrow.
- `uv run granite-speech audio.wav --output_format txt`: exercise the CLI from the checkout.
- `uv run granite-speech download granite-speech-4.1-2b --llama_cpp_quant Q4_K_M`: prefetch GGUF
  model and mmproj files for offline or container use.

Real-weight smoke tests require `llama-cli` and multi-GB model downloads. Run them only when needed,
for example `uv run pytest tests/test_real_weights_smoke.py -m real_weights` or
`uv run pytest tests/test_real_weights_smoke_plus.py -m real_weights`. See `docs/release-checks.md`
for cache, binary, revision, timeout, and expected-text environment overrides.

For the Pipecat example, run `cd examples && uv sync`, then
`uv run python live_audio_pipecat.py`. Pipecat requires Python 3.11 or newer.

## Coding Style & Naming Conventions

Use Python 3.10-compatible syntax, 4-space indentation, and Ruff's configured
`line-length = 100`. Prefer `snake_case` for modules, functions, variables, and CLI options; use
`PascalCase` for classes and exception types. Keep CLI options in the existing argparse style:
native underscore options may have hyphen aliases, and Whisper migration aliases should remain
explicitly tested.

Keep model capability rules in `_models.py` rather than scattering model-name checks through the
code. The base model supports ASR and validated translation pairs; the plus model supports ASR,
speaker-attributed output, word timestamp output, and `prefix_text`, but not translation. The
default llama.cpp quantization is `Q4_K_M`.

Preserve model-card prompt compatibility in `prompts.py` and `examples/model_card_examples.md`.
The plus system prompt contains fixed training-time dates; do not update them to the current date.
Parse model-generated plus tags in `plus_output.py` and keep user-facing results as plain dicts with
the `TypedDict` shapes in `types.py`.

Declare `__all__` only on import-boundary modules: the package `__init__.py`,
`_backends/__init__.py`, and concrete backend modules. Internal implementation modules are imported
by explicit name and intentionally omit `__all__`.

## Testing Guidelines

Use `pytest`. Name files `test_*.py` and test functions `test_*`. Prefer the fake backend for model
tests so they remain fast and deterministic. Add focused regression coverage for parser changes,
prompt rendering, chunk/window behavior, VAD option validation, cache resolution, writer output,
CLI argument mapping, backend selection, and warning/error behavior.

Do not make default tests depend on downloading model weights, `llama-cli`, GPU hardware, or private
audio. Put those checks behind the existing `real_weights` marker or an explicit script/env opt-in.
When changing packaging or public imports, run the built-artifact script in addition to unit tests.

## Commit & Pull Request Guidelines

Recent commit subjects are short and direct, such as `plus model smoke test`, `more tests`, and
`cleanup`. Keep commit summaries concise and behavior-focused. DCO is enabled for the repo, so use a
Signed-off-by trailer when creating commits unless maintainers say otherwise.

Pull requests should include a brief description, commands run (`uv run pytest -m "not real_weights"`,
`uv run ruff check .`, and relevant build or smoke checks), linked issues when relevant, and notes
for changes that affect model downloads, cache paths, audio limits, llama.cpp requirements, prompt
text, result shape, or CLI output formats.

## Security & Configuration Tips

Do not commit cached model weights, private audio, transcripts, local virtualenvs, build outputs, or
egg-info directories. `audio/`, `dist/`, `.venv/`, and `*.egg-info/` are local/generated paths.
Committed fixtures under `tests/fixtures/` must stay small, public, and documented.

Use `GRANITE_SPEECH_CACHE`, `HF_HUB_CACHE`, `HF_HOME`, or `download_root=` for controlled cache
locations. Use `local_files_only=True` or `--local_files_only` only after required model files are
already cached. Keep `GRANITE_SPEECH_MAX_AUDIO_SECONDS` safeguards in mind when changing audio
loading behavior; the default duration cap protects against decompression-bomb style inputs.
