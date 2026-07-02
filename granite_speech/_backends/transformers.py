from __future__ import annotations

from typing import Any

from . import BackendCapabilities, GenerateRequest, GenerateResult


class TransformersBackend:
    name = "transformers"

    def __init__(
        self,
        *,
        processor: Any,
        model: Any,
        device: Any,
        capabilities: BackendCapabilities,
    ) -> None:
        self.processor = processor
        self.model = model
        self.device = device
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.capabilities = capabilities

    def generate(self, req: GenerateRequest) -> GenerateResult:
        if req.sample_rate != 16000:
            raise ValueError(f"GenerateRequest.sample_rate must be 16000, got {req.sample_rate}")

        import torch

        inputs = self.processor(req.prompt, req.wav, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": req.max_new_tokens,
            "num_beams": req.num_beams,
        }
        if req.temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = req.temperature
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
        if input_ids is not None and output_ids.ndim == 2:
            output_ids = output_ids[:, input_ids.shape[-1] :]

        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        tokens = output_ids[0].detach().cpu().tolist() if output_ids.ndim == 2 else None
        return GenerateResult(text=text, tokens=tokens)
