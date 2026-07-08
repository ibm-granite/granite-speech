# Test fixtures

`multilingual_sample.wav` is the release smoke-test audio fixture for Granite Speech.

- Source: `ibm-granite/granite-speech-4.1-2b` on Hugging Face
- File: `multilingual_sample.wav`
- Revision: `de575db64086f84fdc79da4932d1076e965bc546`
- SHA-256: `91d243650809c1274141ec20ff23045315eaf27567694002ea3ef390048b7058`

The full clip is used by the real-weights ASR smoke. The French tail, selected with
`clip_timestamps="12"`, is used by the translation smoke.

## `ami_ihm_sample0_5m-6m.wav`

The plus-model smoke fixture, mirroring the example audio on the
[granite-speech-4.1-2b-plus model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus).
It is genuinely multi-speaker, so it exercises speaker attribution and incremental decoding
faithfully.

- Source: `diarizers-community/ami` on Hugging Face (CC-BY-4.0)
- Config: `ihm`
- Split: `test`
- Sample index: `0`
- Time window: 5:00–6:00 (the card's `START_TIME, END_TIME = 5 * 60, 6 * 60`)
- Written as mono 16 kHz PCM WAV
- SHA-256: `e248864aa8ac4fb145a874ef8efe86c6278dcb584f91989e8693c0f81df61ff4`

This clip drives all four plus-model smoke tasks: ASR, speaker-attributed ASR, word-level
timestamps, and incremental decoding (`prefix_text`). Regenerate it with
`python scripts/fetch_ami_smoke_audio.py` (needs `datasets` plus an audio decoder backend such as
`soundfile`/`torchcodec`, installed into a scratch environment — not the package venv).
