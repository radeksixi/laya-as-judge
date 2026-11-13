"""Vega-U (`vegaml` / `laya.Agent(vegaml=...)`) decision backend with conformal sets, abstention, and TTT heads."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .base import BaseDecisionEngine, serialize_state, validate_question_spec

_MEDIA_KEYS = ("image", "audio", "video")


def split_media_from_state(
    state: Union[str, Dict[str, Any], List[Any]],
) -> tuple[Union[str, Dict[str, Any], List[Any]], Dict[str, Any]]:
    """Extract optional `image`, `audio`, or `video` payloads from a dict state for VegaAgent."""
    if not isinstance(state, dict):
        return state, {}
    media: Dict[str, Any] = {}
    clean_state: Dict[str, Any] = {}
    for k, v in state.items():
        kl = str(k).lower()
        if kl in _MEDIA_KEYS and v is not None:
            media[kl] = v
        elif kl in ("_image_bytes", "_audio_bytes", "_video_bytes", "_pil_image", "_frames"):
            # Internal multimodal transport keys used by Tier-2; omit from text serialization
            continue
        else:
            clean_state[k] = v
    return clean_state, media


class VegaBackend(BaseDecisionEngine):
    """Universal Vega-U (`nandakishorm/vega-U`) backend via `laya.Agent(vegaml=...)`.

    Capabilities:
      - 73k-token multilingual & multimodal universal reader (`0.8b` or `4b`) + 14.3M physics engine
      - Conformal prediction sets (`prediction_set`), `abstain`, `unbound`, and automatic `route`
      - Test-Time Training (`fit_ttt`) fitting regularized linear/projection heads + calibration
        on labeled training splits without modifying backbone weights.
    """

    supports_multimodal: bool = True

    def __init__(
        self,
        model_id: str = "0.8b",
        device: Optional[str] = None,
        alpha: float = 0.10,
        risk: Union[str, float] = "0.1",
        target_precision: float = 0.95,
        folds: int = 5,
        calibration_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 16,
        **kwargs: Any,
    ):
        try:
            import laya
            from laya.vega import is_available
        except ImportError as exc:
            raise ImportError(
                "VegaBackend requires `laya>=0.4.23` and `vegaml`. Install via `pip install -U laya vegaml`."
            ) from exc

        if not is_available():
            raise ImportError("`vegaml` package is not available in the active Python environment.")

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        vega_arg: Union[bool, str] = True if model_id in ("", "default", "vega", "vega-u", "0.8b") else model_id
        self._agent = laya.Agent(
            vegaml=vega_arg,
            device=device,
            alpha=alpha,
            risk=risk,
            target_precision=target_precision,
            folds=folds,
            **kwargs,
        )
        display_name = f"vega-U/{model_id}" if model_id not in ("", "default", "vega", "vega-u") else "vega-U/0.8b"
        super().__init__(model_id=display_name)
        self.device = str(getattr(self._agent, "device", device))
        self.batch_size = max(1, int(batch_size))

        if calibration_dir is not None and Path(calibration_dir).exists():
            self.load_calibration(calibration_dir)

    @property
    def fitted_heads(self) -> List[str]:
        """Return question IDs that currently have fitted TTT heads attached."""
        vega_obj = getattr(self._agent, "vega", None)
        heads = getattr(vega_obj, "heads", {}) if vega_obj is not None else {}
        return sorted(heads.keys())

    def fit_ttt(
        self,
        rows: Sequence[Mapping[str, Any]],
        questions: Dict[str, Dict[str, Any]],
        save_dir: Optional[Union[str, Path]] = None,
        **fit_kwargs: Any,
    ) -> Dict[str, Any]:
        """Fit TTT heads + conformal/temperature calibration on strictly disjoint training rows.

        Args:
            rows: Sequence of `{"state": ..., "expected": {qid: gold_label}}` dicts.
            questions: Question definitions matching `validate_question_spec`.
            save_dir: Optional directory path to persist fitted heads & calibration.
        """
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        prepared_rows = []
        for r in rows:
            st, _ = split_media_from_state(r["state"])
            prepared_rows.append({"state": st, "expected": dict(r["expected"])})

        report = self._agent.fit(prepared_rows, normalized_q, **fit_kwargs)
        if save_dir is not None:
            self.save_calibration(save_dir)
        return dict(report) if isinstance(report, Mapping) else {"status": "fitted"}

    def save_calibration(self, path: Union[str, Path]) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._agent.save_calibration(str(out))

    def load_calibration(self, path: Union[str, Path]) -> None:
        self._agent.load_calibration(str(path))

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run Vega-U typed decision prediction with conformal set and routing metadata."""
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        clean_state, media = split_media_from_state(state)
        t0 = time.perf_counter()

        call_kwargs = dict(kwargs)
        for mk, mv in media.items():
            call_kwargs.setdefault(mk, mv)

        # If media is passed, pass clean_state only when no media key overrides state
        if any(k in call_kwargs for k in _MEDIA_KEYS):
            # When image/audio/video is provided AND text context exists, prepend text to instructions if non-empty
            if clean_state not in ("", {}, [], None):
                ctx_str = serialize_state(clean_state)
                augmented_q: Dict[str, Dict[str, Any]] = {}
                for qid, qspec in normalized_q.items():
                    qcopy = dict(qspec)
                    qcopy["instructions"] = f"{qspec['instructions']}\nContext: {ctx_str[:1500]}"
                    augmented_q[qid] = qcopy
                normalized_q = augmented_q
            raw = self._agent.predict(None, normalized_q, **call_kwargs)
        else:
            raw = self._agent.predict(clean_state, normalized_q, **call_kwargs)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw["latency_ms"] = round(latency_ms, 2)
        raw.setdefault("model", self.model_id)
        return raw

    def predict_batch(
        self,
        states: List[Union[str, Dict[str, Any], List[Any]]],
        questions: Dict[str, Dict[str, Any]],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run Vega-U over a batch of states."""
        if not states:
            return []
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        bs = max(1, int(batch_size or self.batch_size))

        # Check if all states are pure text (no media keys) and VegaAgent supports predict_batch
        cleaned_pairs = [split_media_from_state(s) for s in states]
        has_any_media = any(bool(m) for _, m in cleaned_pairs)

        if not has_any_media and hasattr(self._agent, "predict_batch"):
            clean_states = [cs for cs, _ in cleaned_pairs]
            results: List[Dict[str, Any]] = []
            for start in range(0, len(clean_states), bs):
                chunk = clean_states[start : start + bs]
                t0 = time.perf_counter()
                try:
                    chunk_res = self._agent.predict_batch(chunk, normalized_q, batch_size=bs)
                except TypeError:
                    chunk_res = [self._agent.predict(s, normalized_q, **kwargs) for s in chunk]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                per_ms = round(elapsed_ms / max(1, len(chunk)), 2)
                for r in chunk_res:
                    r.setdefault("latency_ms", per_ms)
                    r.setdefault("model", self.model_id)
                results.extend(chunk_res)
            return results

        return [self.predict(s, normalized_q, **kwargs) for s in states]
