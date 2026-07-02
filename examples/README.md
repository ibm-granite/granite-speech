# Examples

These examples are optional integrations. They are not installed with the core
`granite-speech` package dependencies unless noted.

## Model Card Prompt Support

[model_card_examples.md](model_card_examples.md) maps the current Granite Speech 4.1 2B and 2B
Plus model-card prompt forms to repo-local Python and CLI calls.

## Live Audio With Pipecat

`live_audio_pipecat.py` serves Pipecat's prebuilt Small WebRTC browser UI and
uses a small `GraniteSpeechSTTService` adapter to transcribe microphone turns
with Granite Speech.

The example has its own `pyproject.toml`. It installs `granite-speech` from the
parent checkout in editable mode and installs Pipecat's WebRTC UI dependencies,
so no prior `granite-speech` install is required.

Pipecat requires Python 3.11 or newer.

```bash
cd examples
uv sync
```

Run from `examples/`, then open the printed browser URL:

```bash
uv run python live_audio_pipecat.py
```

The first run downloads the Granite Speech model unless it is already cached.
Use `--device`, `--download-root`, or `--local-files-only` the same way you would
for the library API:

```bash
uv run python live_audio_pipecat.py --device mps
```

This example is utterance-based rather than token-streaming: Pipecat captures
browser microphone audio and VAD boundaries, then Granite Speech transcribes
each completed speech segment. Final transcripts are printed in the server
terminal and emitted to the Pipecat UI as `user-transcription` messages.
