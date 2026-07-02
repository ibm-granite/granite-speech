# granite-speech

A Whisper-like Python library for IBM's Granite Speech models.

The goal is to make speech-to-text with Granite as simple as Whisper:

```bash
git clone https://github.com/ibm-granite/granite-speech.git
cd granite-speech
uv sync
```

```bash
uv run python - <<'PY'
import granite_speech

model = granite_speech.load_model()        # defaults to granite-speech-4.1-2b
result = model.transcribe("audio.wav")     # arbitrary-length audio
print(result["text"])
PY
```

…and from the command line:

```bash
uv run granite-speech audio.wav --model granite-speech-4.1-2b --output_format srt
```

## Install

```bash
git clone https://github.com/ibm-granite/granite-speech.git
cd granite-speech
uv sync
```

Run commands from the repository root with `uv run`, or activate `.venv` manually.

For a specific CUDA build, sync with the matching PyTorch wheel index:

```bash
uv sync --index https://download.pytorch.org/whl/cu124
```

## Python API

```python
import granite_speech

model = granite_speech.load_model(
    "granite-speech-4.1-2b",
    device=None,          # optional llama.cpp device override; use "cpu" for CPU-only
    llama_cpp_quant="Q4_K_M",
)

result = model.transcribe(
    "audio.wav",
    task="transcribe",
    language=None,        # source-language hint only, never detected
    keyword_biases=["Granite Speech", "watsonx.ai"],
    max_new_tokens=None,  # auto-size per window; pass an int to force a cap
    chunk_length=30.0,
    chunk_overlap=0.0,
    segmentation="fixed",  # use "vad" to skip silence and cut on speech activity
    clip_timestamps=None,  # e.g. "10,20" or "10,20,30,45" for selected ranges
)

print(result["text"])
```

`result` has stable keys:

```python
{
    "text": str,
    "segments": [
        {
            "id": int,
            "start": float,
            "end": float,
            "text": str,
            "temperature": float,
            # "tokens": list[int],  # only present when the backend provides token IDs
        }
    ],
    "language": str | None,
    "target_language": str | None,
    "warnings": list[dict],
}
```

The segment `id` and `temperature` fields are included as Whisper-familiar compatibility metadata.
Granite Speech does not fabricate Whisper confidence fields such as `avg_logprob`,
`compression_ratio`, or `no_speech_prob`.

For translation, pass an explicit source language. If `target_language` is omitted, it defaults to
English:

```python
result = model.transcribe("fr.wav", task="translate", language="fr")
```

Use the plus model for the richer transcription prompt modes from its model card:

```python
plus = granite_speech.load_model("granite-speech-4.1-2b-plus")

speakers = plus.transcribe("meeting.wav", prompt_mode="speaker_attributed")
timestamps = plus.transcribe(
    "meeting.wav",
    prompt_mode="word_timestamps",
    max_new_tokens=10000,
)
```

`prompt_mode="speaker_attributed"` renders the model-card speaker tag prompt, and
`prompt_mode="word_timestamps"` renders the word timestamp prompt. The returned `text` strips the
model-card tags, parsed segments include `raw_text`, and structured fields are attached as
`segments[].speakers` or `segments[].words` plus top-level `speakers` or `words`. Pass
`prefix_text=` with the plus model to use the incremental decoding hook shown in the model card.

The module-level convenience API is also available:

```python
result = granite_speech.transcribe("audio.wav")
```

Migrating existing Whisper code? See the [porting guide](docs/porting-from-whisper.md) for the
supported aliases and intentional differences.

## CLI

```bash
uv run granite-speech audio.wav --model granite-speech-4.1-2b --output_format txt
uv run granite-speech audio.wav --llama_cpp_quant Q4_K_M
uv run granite-speech audio.wav --task translate --language fr --output_format json
uv run granite-speech audio.wav --keyword "Granite Speech" --keyword watsonx.ai
uv run granite-speech meeting.wav --model granite-speech-4.1-2b-plus --prompt_mode speaker_attributed
uv run granite-speech meeting.wav --model granite-speech-4.1-2b-plus --prompt_mode word_timestamps --max_new_tokens 10000
uv run granite-speech audio.wav --segmentation vad --chunk_length 30
uv run granite-speech audio.wav --clip_timestamps 10,20
uv run granite-speech audio.wav --output_format srt --max_line_width 42 --max_line_count 2
uv run granite-speech audio.wav --output_format all --output_dir transcripts/
```

Supported output formats are `txt`, `srt`, `vtt`, `tsv`, `json`, and `all`.
For subtitle output, `--max_line_width` and `--max_line_count` wrap SRT/VTT cue text without
changing TXT, TSV, or JSON output.

Pass `keyword_biases=[...]` in Python or repeat `--keyword` in the CLI to use Granite Speech's
keyword list biasing. The library renders the model-card prompt form, for example
`transcribe the speech to text. Keywords: Granite Speech, watsonx.ai`.

Long audio is processed as fixed-size windows. By default, `max_new_tokens=None` auto-sizes the
generation budget from the window length: 30-second windows use 200 tokens, and longer windows scale
proportionally. Pass an explicit integer to force the same token cap on every window.

For long recordings with substantial silence, pass `segmentation="vad"` in Python or
`--segmentation vad` in the CLI. VAD mode uses built-in energy-based voice activity detection to
skip silent regions, pad detected speech, merge speech separated by short silences, and split any
speech span longer than `chunk_length`. Tune it with `vad_threshold`, `vad_min_speech_duration`,
`vad_min_silence_duration`, and `vad_speech_pad`.

To transcribe selected regions, pass `clip_timestamps=` in Python or `--clip_timestamps` in the CLI
as comma-separated seconds. Pairs select explicit ranges such as `10,20`; an odd final timestamp
selects through the end of the file, such as `30` or `10,20,30`. Segment timestamps stay relative to
the original audio file.

Exit codes:

- `0`: output produced with no per-window failures
- `1`: output produced, but one or more windows failed and are listed in `warnings`
- `2`: unrecoverable argument, model, audio, or download failure

Pre-download model weights for offline or container builds:

```bash
uv run granite-speech download granite-speech-4.1-2b --download_root /models/granite-speech
uv run granite-speech download granite-speech-4.1-2b --llama_cpp_quant Q4_K_M
uv run granite-speech download granite-speech-4.1-2b-plus --llama_cpp_quant Q4_K_M
```

## llama.cpp

Granite Speech uses the llama.cpp GGUF backend. The loader expects `llama-cli` to be installed and
the selected model to have an official GGUF variant.

The backend shells out to `llama-cli` because Granite Speech audio is exposed through
llama.cpp's multimodal CLI. Install llama.cpp separately, for example with Homebrew:

```bash
brew install llama.cpp
```

The default GGUF quant is `Q4_K_M`. Override it with `llama_cpp_quant="Q8_0"` in Python or
`--llama_cpp_quant Q8_0` in the CLI. Local GGUF paths are also supported when the matching
`mmproj-model-f16.gguf` file is in the same directory, or by passing `llama_cpp_mmproj=`.
The official base and Plus GGUF repos,
`ibm-granite/granite-speech-4.1-2b-GGUF` and
`ibm-granite/granite-speech-4.1-2b-plus-GGUF`, are accepted as model aliases.

## Audio Inputs

Path inputs infer sample rate from the file. Raw `numpy` arrays or `torch` tensors must pass
`sample_rate=`.

The loader normalizes audio to mono, 16 kHz, float32. WAV and FLAC are the primary supported file
paths through `soundfile`; MP3/M4A/AAC are best effort through the installed audio libraries.
For guaranteed container support, transcode to WAV or FLAC before calling the library.

`GRANITE_SPEECH_MAX_AUDIO_SECONDS` caps decoded duration as a decompression-bomb guard. The default
is 4 hours; set it to `0` to disable the cap.

## Cache and Offline Use

Cache resolution order:

1. `download_root=`
2. `GRANITE_SPEECH_CACHE`
3. `HF_HUB_CACHE`
4. `HF_HOME/hub`
5. `~/.cache/granite-speech`

Use `local_files_only=True` with `load_model()` for offline runs after weights are cached.

## Current Limitations

This is an early v1 implementation of the [original spec](docs/original-spec.md) contract. Fixed-window
mode uses window-granular timestamps and hard-cut boundaries may split speech. VAD mode can cut on
speech activity, but it is energy-based rather than a learned speech detector. The
`granite-speech-4.1-2b-plus` parser supports the model-card `[Speaker N]:` and `[T:N]` tag forms;
word starts are inferred from the previous word end, and speaker turns do not include intra-window
timing. Streaming is not implemented yet.
