"""Hugging Face `transformers` native Decision Model backend (`AutoModelForDecision` / `DecisionPipeline`).

Provides forward-compatible support for late-2026 Hugging Face decision architectures
(Cloudflare `Clef`/`Clef-Flash`, Fastino `GLiDE`, Convai `Laya`, `Qwen3Guard`, etc.)
while staying completely optional in core installs.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from .base import BaseDecisionEngine, validate_question_spec
from .http_systemone_backend import (
    _contains_media,
    _extract_simplex_from_answer,
    _pack_canonical_answer,
    normalize_decision_questions,
)


class TransformersDecisionBackend(BaseDecisionEngine):
    """Adapter wrapping Hugging Face `DecisionPipeline` / `AutoModelForDecision`."""

    supports_multimodal: bool = False

    def __init__(
        self,
        model_id: str = "Cloudflare/clef-flash",
        *,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        text_only: bool = True,
        pipeline_fn: Optional[Callable[[Any, Dict[str, Dict[str, Any]]], Dict[str, Any]]] = None,
    ):
        super().__init__(model_id=model_id)
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = bool(trust_remote_code)
        self.text_only = bool(text_only)
        self.supports_multimodal = not self.text_only
        self._pipeline_fn = pipeline_fn
        self._loaded_pipeline: Optional[Any] = None

    def _ensure_loaded(self) -> Any:
        if self._pipeline_fn is not None:
            return self._pipeline_fn
        if self._loaded_pipeline is not None:
            return self._loaded_pipeline

        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "TransformersDecisionBackend requires `transformers`. "
                "Install via: pip install 'laya-as-judge[torch]'"
            ) from exc

        # Prefer standardized HF DecisionPipeline / AutoModelForDecision if available
        if hasattr(transformers, "DecisionPipeline"):
            kwargs: Dict[str, Any] = {"model": self.model_id, "trust_remote_code": self.trust_remote_code}
            if self.device is not None:
                kwargs["device"] = self.device
            self._loaded_pipeline = transformers.DecisionPipeline(**kwargs)
            return self._loaded_pipeline

        if hasattr(transformers, "AutoModelForDecision"):
            model = transformers.AutoModelForDecision.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
            if self.device is not None and hasattr(model, "to"):
                model = model.to(self.device)
            self._loaded_pipeline = model
            return self._loaded_pipeline

        raise ImportError(
            "Your installed `transformers` version does not yet expose `DecisionPipeline` or "
            "`AutoModelForDecision`. Upgrade `transformers` or serve GGUF models locally via "
            "`SystemOneHTTPBackend` (`llama-server /v1/systemone`)."
        )

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self.text_only and _contains_media(state):
            raise ValueError("TransformersDecisionBackend is configured in text_only=True mode.")

        normalized_q = normalize_decision_questions(questions)
        pipe = self._ensure_loaded()

        t0 = time.perf_counter()
        if callable(pipe):
            raw = pipe(state, normalized_q)
        elif hasattr(pipe, "predict"):
            raw = pipe.predict(state, normalized_q)
        else:
            raise RuntimeError(f"Loaded Hugging Face decision object {type(pipe).__name__} is not callable.")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_answers = raw.get("answers", raw) if isinstance(raw, Mapping) else {}
        answers: Dict[str, Any] = {}
        for qid, qspec in normalized_q.items():
            labels, probs = _extract_simplex_from_answer(raw_answers.get(qid, {}), qspec)
            answers[qid] = _pack_canonical_answer(labels, probs, qspec)

        usage = raw.get("usage", {}) if isinstance(raw, Mapping) else {}
        return {
            "model": str((raw.get("model") if isinstance(raw, Mapping) else None) or self.model_id),
            "answers": answers,
            "usage": {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": 0,
            },
            "cost_usd": 0.0,
            "latency_ms": round(elapsed_ms, 2),
        }
