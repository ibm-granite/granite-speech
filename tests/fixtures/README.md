# Test fixtures

`multilingual_sample.wav` is the release smoke-test audio fixture for Granite Speech.

- Source: `ibm-granite/granite-speech-4.1-2b` on Hugging Face
- File: `multilingual_sample.wav`
- Revision: `de575db64086f84fdc79da4932d1076e965bc546`
- SHA-256: `91d243650809c1274141ec20ff23045315eaf27567694002ea3ef390048b7058`

The full clip is used by the real-weights ASR smoke. The French tail, selected with
`clip_timestamps="12"`, is used by the translation smoke.
