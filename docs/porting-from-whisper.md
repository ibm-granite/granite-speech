# Porting from Whisper

Granite Speech is intentionally Whisper-like where that makes migration easier, but it is not a
drop-in replacement for the `whisper` package. Keep imports and model objects Granite-native, then
use the small set of compatibility aliases only at call sites that are expensive to change all at
once.

## Python API

Replace the package import and choose a Granite Speech model name:

```python
# Whisper
import whisper

model = whisper.load_model("large")
result = model.transcribe("audio.wav")
print(result["text"])
```

```python
# Granite Speech
import granite_speech

model = granite_speech.load_model("granite-speech-4.1-2b")
result = model.transcribe("audio.wav")
print(result["text"])
```

For one-shot transcription, use the module-level convenience function:

```python
import granite_speech

result = granite_speech.transcribe("audio.wav", model="granite-speech-4.1-2b")
```

## Model Names

Whisper shorthand names such as `tiny`, `base`, `small`, `medium`, `large`, and `turbo` are not
Granite Speech model names and are intentionally unsupported. Use Granite model names or supported
repository aliases instead:

| Use case | Granite Speech model |
| --- | --- |
| Default ASR and translation model | `granite-speech-4.1-2b` |
| Speaker-attributed ASR and word timestamp prompt modes | `granite-speech-4.1-2b-plus` |

The corresponding Hugging Face repository names are also accepted:
`ibm-granite/granite-speech-4.1-2b`, `ibm-granite/granite-speech-4.1-2b-plus`, and their official
GGUF repository aliases.

## Prompts

Whisper code often uses `initial_prompt=`. Granite Speech's native name is `prompt=`:

```python
result = model.transcribe(
    "audio.wav",
    prompt="transcribe this earnings call and preserve IBM product names",
)
```

During migration, `initial_prompt=` is accepted as an alias for `prompt=`:

```python
result = model.transcribe(
    "audio.wav",
    initial_prompt="transcribe this earnings call and preserve IBM product names",
)
```

Do not pass both `prompt=` and `initial_prompt=` in the same call.

For keyword biasing, prefer Granite Speech's native keyword API:

```python
result = model.transcribe(
    "audio.wav",
    keyword_biases=["Granite Speech", "watsonx.ai"],
)
```

## Options

| Whisper option | Granite Speech migration path |
| --- | --- |
| `initial_prompt` | Accepted as an alias for native `prompt`. |
| `word_timestamps=True` | Maps to `prompt_mode="word_timestamps"` on capable Plus models. This is model-generated timestamp text, not Whisper DTW alignment. |
| `temperature=(...)` fallback schedule | Accepted, but only the first value is used. Granite Speech does not implement Whisper fallback schedules. |
| `fp16` | Accepted and ignored with a warning. Precision is backend/load-time behavior, not a per-transcription option. |
| `clip_timestamps` | Accepted as comma-separated seconds or Python timestamp values. Selected clip timestamps stay relative to the original audio file. |
| Default `compression_ratio_threshold`, `logprob_threshold`, `no_speech_threshold`, and similar default-only kwargs | Accepted only when set to Whisper's default value; non-default transcript-affecting values are rejected. |
| `beam_size` | Unsupported. The default llama.cpp backend does not expose this as a compatibility alias. |
| `condition_on_previous_text`, `carry_initial_prompt` | Accepted and ignored with warnings. Granite Speech applies prompts independently to each audio window. |

Prefer Granite Speech native names in new code: `prompt`, `prompt_mode`, `keyword_biases`,
`max_new_tokens`, `chunk_length`, `chunk_overlap`, and `segmentation`.

## Language and Translation

Granite Speech does not auto-detect language. `language=` is a source-language hint and the returned
`result["language"]` is the language you supplied, or `None` if you did not supply one.

For translation, pass an explicit source language:

```python
result = model.transcribe("fr.wav", task="translate", language="fr")
```

If `target_language=` is omitted for translation, Granite Speech defaults to English.

## Results

Granite Speech returns a Whisper-familiar dictionary shape:

```python
{
    "text": "...",
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 30.0,
            "text": "...",
            "temperature": 0.0,
        }
    ],
    "language": None,
    "target_language": None,
    "warnings": [],
}
```

Segment `id` and `temperature` fields are included as compatibility metadata. Token IDs are included
only when the backend provides them. Granite Speech does not fabricate Whisper confidence metrics
such as `avg_logprob`, `compression_ratio`, or `no_speech_prob`.

## CLI

Replace the `whisper` command with `granite-speech` and use Granite model names:

```bash
granite-speech audio.wav --model granite-speech-4.1-2b --output_format srt
```

Common migration aliases are available:

```bash
granite-speech audio.wav \
  --model granite-speech-4.1-2b-plus \
  --initial_prompt "prefer IBM Granite terms" \
  --word_timestamps \
  --output_format json
```

`--model_dir` is accepted as an alias for `--download_root` when migrating cache/download scripts.
For new scripts, prefer `--download_root`.

For subtitles, `--max_line_width` and `--max_line_count` wrap SRT/VTT cue text:

```bash
granite-speech audio.wav --output_format srt --max_line_width 42 --max_line_count 2
```
