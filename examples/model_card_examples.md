# Model Card Prompt Support

This page maps the preferred prompts from the
[Granite Speech 4.1 2B model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)
to this repo's API and CLI. The goal is prompt compatibility, not reproducing every upstream
framework snippet from the model card.

## llama.cpp Setup

Install llama.cpp before running the examples:

```bash
brew install llama.cpp
```

```bash
uv run granite-speech audio.wav
```

The repo default quantization is `Q4_K_M`; use `--llama_cpp_quant Q8_0` when you want the same
quantization shown in the model card:

```bash
uv run granite-speech audio.wav --llama_cpp_quant Q8_0
```

## Prompt Mapping

The base model-card prompt forms are supported as follows.

### ASR Raw Transcripts

```python
model.transcribe(
    "audio.wav",
    prompt="can you transcribe the speech into a written format?",
)
```

```bash
uv run granite-speech audio.wav \
  --prompt "can you transcribe the speech into a written format?"
```

### ASR With Punctuation And Capitalization

This is the default transcription prompt:

```python
model.transcribe("audio.wav")
```

```bash
uv run granite-speech audio.wav
```

You can also pass it explicitly:

```bash
uv run granite-speech audio.wav \
  --prompt "transcribe the speech with proper punctuation and capitalization."
```

### ASR With Keyword Biasing

```python
model.transcribe("audio.wav", keyword_biases=["kw1", "kw2"])
```

```bash
uv run granite-speech audio.wav \
  --keyword kw1 \
  --keyword kw2
```

This renders `transcribe the speech to text. Keywords: kw1, kw2`.

### AST Raw Transcripts

```python
model.transcribe(
    "audio.wav",
    task="translate",
    language="fr",
    target_language="en",
    prompt="translate the speech to English.",
)
```

```bash
uv run granite-speech audio.wav \
  --task translate \
  --language fr \
  --target_language en \
  --prompt "translate the speech to English."
```

### AST With Punctuation And Capitalization

This is the default translation prompt for `task="translate"`:

```python
model.transcribe("audio.wav", task="translate", language="fr", target_language="en")
```

```bash
uv run granite-speech audio.wav \
  --task translate \
  --language fr \
  --target_language en
```

This renders `translate the speech to English with proper punctuation and capitalization.`

### AST With Keyword Biasing

```python
model.transcribe(
    "audio.wav",
    task="translate",
    language="en",
    target_language="de",
    keyword_biases=["kw1", "kw2"],
)
```

```bash
uv run granite-speech audio.wav \
  --task translate \
  --language en \
  --target_language de \
  --keyword kw1 \
  --keyword kw2
```

This renders `translate the speech to German. Keywords: kw1, kw2`.

## Supported Translation Targets

The model card lists English, French, German, Spanish, Japanese, Italian, and Mandarin as AST
target prompts. The repo validates the supported model-card pairs: non-English source languages
to English, and English to French, German, Spanish, Portuguese, Japanese, Italian, or Mandarin.

The API still requires `language=` for translation so it can validate that the requested pair is
supported before sending the prompt to the model.

## Plus Model Prompt Mapping

The plus model is available by name:

```python
model = granite_speech.load_model("granite-speech-4.1-2b-plus")
```

### Plus ASR

This is the default plus transcription prompt:

```python
model.transcribe("audio.wav")
```

It sends `<|audio|> can you transcribe the speech into a written format?`.

### Speaker Attributed ASR

```python
model.transcribe("audio.wav", prompt_mode="speaker_attributed")
```

```bash
uv run granite-speech audio.wav \
  --model granite-speech-4.1-2b-plus \
  --prompt_mode speaker_attributed
```

### Word-Level Timestamps

```python
model.transcribe("audio.wav", prompt_mode="word_timestamps", max_new_tokens=10000)
```

```bash
uv run granite-speech audio.wav \
  --model granite-speech-4.1-2b-plus \
  --prompt_mode word_timestamps \
  --max_new_tokens 10000
```

These plus modes return tag-stripped `text`, preserve the tagged transcript as segment `raw_text`,
and attach structured `segments[].speakers` or `segments[].words` fields plus top-level
`speakers` or `words`.

### Incremental Decoding

```python
model.transcribe(
    "audio.wav",
    prompt_mode="speaker_attributed",
    prefix_text="[Speaker 1]: Hello how are you",
)
```

`prefix_text=` is passed through the tokenizer chat template for the plus model.
