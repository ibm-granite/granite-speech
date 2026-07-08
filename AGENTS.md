# Repository Guidelines

## Project Structure & Module Organization

`granite_speech/` contains the package source and the `granite-speech` CLI. Public entry points live
in `__init__.py`, `model.py`, `loader.py`, `cli.py`, and `types.py`; keep stable API behavior there.
Model aliases and capability metadata are centralized in `_models.py`, and `errors.py` holds the
public exception hierarchy (`GraniteSpeechError`, `InvalidArgumentError`). Audio loading and
clipping (`audio.py`), chunking (`chunking.py`), VAD segmentation (`segmenter.py`), reconciliation
(`reconciliation.py`), prompt rendering (`prompts.py`), plus-model output parsing
(`plus_output.py`), cache resolution (`cache.py`), and output writers (`writers.py`) each live in
their matching module.
`version.py` holds `__version__`, which `pyproject.toml` reads as the dynamic package version and
`publish-pypi.yml` verifies against the release git tag.

Backend-specific code belongs in `granite_speech/_backends/`. The active default loader path is
llama.cpp/GGUF via `_backends/llama_cpp.py`; `_backends/transformers.py` is wired into `loader.py`
but reserved for future models and not selected by `load_model()`. Use `_backends/fake.py` for
deterministic tests. The `mlx` optional extra (`mlx-audio`) is declared in `pyproject.toml` but
reserved: there is no MLX backend yet.

Tests live in `tests/` and mirror behavior areas: audio, chunking, segmentation, reconciliation,
writers, CLI, model registry/cache handling, Whisper compatibility, model-card prompts, README
examples, `types.py` shapes, fake-backend model behavior, plus output, and llama.cpp backend command
construction. Opt-in real-weight smoke tests are marked `real_weights`. Committed fixtures live under
`tests/fixtures/` (`multilingual_sample.wav` for the base smoke, `ami_ihm_sample0_5m-6m.wav` for the
plus smoke; see `tests/fixtures/README.md`). Docs live in `README.md` and `docs/`
(`docs/release-checks.md`, `docs/porting-from-whisper.md`); optional integrations and model-card
examples live in `examples/`, which has its own `pyproject.toml` and pinned `uv.lock`.

## Build, Test, and Development Commands

Use `uv` for all Python commands: run tools and scripts through `uv run` (e.g. `uv run pytest`,
`uv run python ...`) and manage the environment with `uv sync` / `uv pip` rather than a bare
`python`, `pip`, or a manually activated virtualenv. This keeps everyone on the locked dependency set.

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
- `scripts/fetch_ami_smoke_audio.py`: regenerate the committed AMI plus-smoke fixture
  `tests/fixtures/ami_ihm_sample0_5m-6m.wav`; run only when refreshing that fixture.
- `uv run granite-speech audio.wav --output_format txt`: exercise the CLI from the checkout.
- `uv run granite-speech download granite-speech-4.1-2b --llama_cpp_quant Q4_K_M`: prefetch GGUF
  model and mmproj files for offline or container use.

Real-weight smoke tests require `llama-cli` and multi-GB model downloads. Run them only when needed,
for example `uv run pytest tests/test_real_weights_smoke.py -m real_weights` or
`uv run pytest tests/test_real_weights_smoke_plus.py -m real_weights`. The plus suite skips (rather
than fails) on llama.cpp builds older than `9850`; override with
`GRANITE_SPEECH_PLUS_SMOKE_MIN_LLAMA_CPP_BUILD`. See `docs/release-checks.md` for the full set of
`GRANITE_SPEECH_SMOKE_*` / `GRANITE_SPEECH_PLUS_SMOKE_*` cache, binary, revision, timeout, and
expected-text environment overrides.

CI lives in `.github/workflows/`. `release-gates.yml` is the release gate that runs
`uv run pytest -m "not real_weights"`, `uv run ruff check .`, `uv build`, and
`uv run twine check dist/*`. `smoke-base.yml` and `smoke-plus.yml` are manual real-weights smoke
runs (`smoke-plus.yml` enforces the `>= 9850` llama.cpp build floor). `publish-testpypi.yml` and
`publish-pypi.yml` build and upload releases; `publish-pypi.yml` verifies the git tag matches
`granite_speech.version.__version__`.

For the Pipecat example, run `cd examples && uv sync`, then
`uv run python live_audio_pipecat.py`. Pipecat requires Python 3.11 or newer.

## Coding Style & Naming Conventions

Use Python 3.10-compatible syntax, 4-space indentation, and Ruff's configured
`line-length = 100`. Prefer `snake_case` for modules, functions, variables, and CLI options; use
`PascalCase` for classes and exception types. Keep CLI options in the existing argparse style:
native underscore options may have hyphen aliases, and Whisper migration aliases should remain
explicitly tested.

The CLI (`cli.py`) largely mirrors the `transcribe()` kwargs: a default transcribe path and a
`download` subcommand, plus segmentation controls (`--segmentation {fixed,vad}` and the `--vad_*`
tuning options), prompt/output controls (`--prompt_mode`, `--prefix_text`, `--keyword`/
`--keyword_bias`, `--clip_timestamps`, `--word_timestamps`, `--output_format`, subtitle
`--max_line_width`/`--max_line_count`), and llama.cpp passthrough (`--llama_cpp_binary`,
`--llama_cpp_quant`, `--llama_cpp_mmproj`, repeatable `--llama_cpp_arg`, `--llama_cpp_timeout`).
`--model_dir` and `--download_root` are mutually exclusive. The CLI uses per-file exit codes
(0 success, 1 partial/window errors, 2 usage) with a warning summary; keep that behavior covered
when changing CLI output.

Keep model capability rules in `_models.py` rather than scattering model-name checks through the
code. The default model is the base `granite-speech-4.1-2b`, which supports ASR and validated
translation pairs (source languages `en, fr, de, es, pt, ja`). The plus model supports ASR,
speaker-attributed output, word timestamp output, and `prefix_text`, but not translation, and has a
narrower source-language set (`en, fr, de, es, pt`). The default llama.cpp quantization is `Q4_K_M`.

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
