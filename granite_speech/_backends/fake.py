from __future__ import annotations

from collections.abc import Callable, Iterable

from . import BackendCapabilities, GenerateRequest, GenerateResult


class FakeBackend:
    name = "fake"

    def __init__(
        self,
        responses: Iterable[str | GenerateResult | BaseException]
        | Callable[[GenerateRequest, int], str | GenerateResult | BaseException]
        | None = None,
        *,
        capabilities: BackendCapabilities | None = None,
    ) -> None:
        self._responses = responses
        self._response_list = (
            list(responses) if responses is not None and not callable(responses) else None
        )
        self.calls: list[GenerateRequest] = []
        self.capabilities = capabilities or BackendCapabilities(
            max_reliable_audio_seconds=None,
            supports_word_timing_output=False,
            supports_speaker_attribution_output=False,
            supports_batch=False,
            supports_translation=True,
        )

    def generate(self, req: GenerateRequest) -> GenerateResult:
        idx = len(self.calls)
        self.calls.append(req)

        if callable(self._responses):
            response = self._responses(req, idx)
        elif self._response_list is not None and idx < len(self._response_list):
            response = self._response_list[idx]
        else:
            seconds = req.wav.shape[-1] / req.sample_rate if req.sample_rate else 0.0
            response = f"window {idx} {seconds:.3f}s"

        if isinstance(response, BaseException):
            raise response
        if isinstance(response, GenerateResult):
            return response
        return GenerateResult(text=str(response))
