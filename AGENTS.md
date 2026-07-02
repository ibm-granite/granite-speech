# Repository Guidelines

## Project Structure & Module Organization

`granite_speech/` contains the package source and CLI entry point. Core modules cover model loading, audio normalization, chunking, segmentation, reconciliation, cache handling, output writers, and backend adapters under `granite_speech/_backends/`. Tests live in `tests/` and mirror the main behaviors, for example `tests/test_audio.py`, `tests/test_chunking.py`, and `tests/test_model_fake_backend.py`. Documentation and specs are in `README.md`, `docs/`, and `examples/`; the Pipecat demo has its own `examples/pyproject.toml`.

## Build, Test, and Development Commands

- `uv sync`: create or update the local environment from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run the full test suite configured to discover `tests/`.
- `uv run pytest tests/test_audio.py`: run a focused test file while iterating.
- `uv run ruff check .`: lint the package and tests with the repo's Ruff settings.
- `uv build`: build source and wheel distributions.
- `uv run granite-speech audio.wav --output_format txt`: exercise the installed CLI locally.

For the Pipecat example, run `cd examples && uv sync`, then `uv run python live_audio_pipecat.py`.

## Coding Style & Naming Conventions

Use Python 3.10-compatible syntax, 4-space indentation, and Ruff's configured `line-length = 100`. Prefer `snake_case` for modules, functions, variables, and CLI options; use `PascalCase` for classes and exception types. Keep backend-specific behavior inside `granite_speech/_backends/` and expose stable public APIs through `granite_speech/__init__.py`, `model.py`, and `cli.py`.

## Testing Guidelines

Use `pytest`. Name files `test_*.py` and test functions `test_*`. Add focused regression tests for parsing, chunking, cache resolution, writer output, CLI-adjacent behavior, and backend selection changes. Prefer the fake backend for deterministic model tests; avoid tests that require downloading real model weights unless explicitly marked or isolated.

## Commit & Pull Request Guidelines

Recent commits use short, direct summaries such as `vad`, `long form asr`, and `model card tests`. Keep commit subjects concise and behavior-focused. Pull requests should include a brief description, the commands run (`uv run pytest`, `uv run ruff check .`), linked issues when relevant, and notes for changes that affect model downloads, cache paths, audio limits, or CLI output formats.

## Security & Configuration Tips

Do not commit cached model weights, private audio, transcripts, or local `.venv` contents. Use `GRANITE_SPEECH_CACHE`, `HF_HUB_CACHE`, or `download_root=` for controlled cache locations. Keep `GRANITE_SPEECH_MAX_AUDIO_SECONDS` safeguards in mind when changing audio loading behavior.
