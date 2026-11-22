"""HTTP backend for local or remote System-One / Decisions API servers.

Connects to any Jev / `/v1/systemone` or OpenAI/Google/Azure `/v1/decisions`
compatible endpoint (for example local `llama-server`, `OneJev`, `vLLM`,
`Ollama`, or another `laya-judge serve` instance) using only the Python
standard library (`urllib.request`), with zero extra required dependencies.

Key features:
  - Text-only enforcement option (strips or rejects non-text media payloads).
  - Full wire-protocol compatibility (`/v1/systemone` and `/v1/decisions`,
    supporting both `state`/`input` and `noul`/`bool` aliases).
  - Optional cyclic permutation averaging (`symmetry_permutations`) and
    adaptive margin-triggered symmetry (`adaptive_symmetry_threshold`) to
    eliminate option-order (position) bias on `choice` questions.
  - Optional post-hoc probability calibration (`TemperatureScaler` or
    `MetaCalibrator`) to recover calibration after Q4/Q8 GGUF quantization.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import urllib.error
import urllib.request

import numpy as np

from .base import (
    BaseDecisionEngine,
    confidence_from_probs,
    serialize_state,
    validate_question_spec,
)

TransportFn = Callable[[str, Dict[str, Any], Dict[str, str], float], Dict[str, Any]]

_BINARY_MEDIA_KEYS = {
    "image",
    "images",
    "screenshot",
    "screenshots",
    "audio",
    "video",
    "frames",
    "pixel_values",
    "_pil_image",
    "_frames",
    "_image_bytes",
    "_audio_bytes",
    "_video_bytes",
}


def normalize_decision_questions(
    questions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Validate and normalize decision questions, accepting `bool` as an alias for `noul`."""
    normalized: Dict[str, Dict[str, Any]] = {}
    for qid, raw_spec in questions.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Question '{qid}' must be a dictionary")
        spec = dict(raw_spec)
        if spec.get("type") == "bool":
            spec["type"] = "noul"
        normalized[str(qid)] = validate_question_spec(str(qid), spec)
    return normalized


def _contains_media(state: Any) -> bool:
    if isinstance(state, dict):
        for k, v in state.items():
            if str(k).lower() in _BINARY_MEDIA_KEYS and v not in (None, "", [], {}):
                return True
    return False


def _strip_media(state: Any) -> Any:
    if isinstance(state, dict):
        return {k: v for k, v in state.items() if str(k).lower() not in _BINARY_MEDIA_KEYS}
    return state


def _default_urllib_transport(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_s: float,
) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} from {url}: {err_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error calling {url}: {exc.reason}") from exc


def _extract_simplex_from_answer(
    ans: Mapping[str, Any],
    qspec: Mapping[str, Any],
) -> Tuple[List[str], np.ndarray]:
    """Extract canonical label order and normalized probability vector from one answer."""
    qtype = qspec["type"]
    if qtype == "noul":
        labels = ["false", "true"]
        if "noul" in ans:
            p_true = float(ans["noul"])
        elif "bool" in ans:
            p_true = float(ans["bool"])
        elif "probability" in ans:
            p_true = float(ans["probability"])
        elif isinstance(ans.get("probabilities"), Mapping) and "true" in ans["probabilities"]:
            p_true = float(ans["probabilities"]["true"])
        else:
            p_true = 0.5
        p_true = float(np.clip(p_true, 1e-6, 1.0 - 1e-6))
        return labels, np.array([1.0 - p_true, p_true], dtype=np.float64)

    if qtype == "score":
        k = len(qspec["criteria"])
        labels = [str(i) for i in range(k)]
        pdict = ans.get("probabilities") or {}
        arr = np.array([float(pdict.get(lbl, 0.0)) for lbl in labels], dtype=np.float64)
    else:
        crit = qspec["criteria"]
        labels = [str(k) for k in (crit.keys() if isinstance(crit, Mapping) else crit)]
        pdict = ans.get("probabilities") or {}
        if pdict:
            arr = np.array([float(pdict.get(lbl, 0.0)) for lbl in labels], dtype=np.float64)
        else:
            chosen = str(ans.get("choice", labels[0] if labels else ""))
            arr = np.array([0.9 if lbl == chosen else 0.1 / max(1, len(labels) - 1) for lbl in labels], dtype=np.float64)

    arr = np.clip(arr, 1e-6, 1.0)
    s = float(np.sum(arr))
    if s <= 0.0 or not np.all(np.isfinite(arr)):
        arr = np.ones(len(labels), dtype=np.float64) / max(1, len(labels))
    else:
        arr /= s
    return labels, arr


def _pack_canonical_answer(
    labels: Sequence[str],
    probs: np.ndarray,
    qspec: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Pack a normalized probability simplex back into canonical Laya/SystemOne answer format."""
    qtype = qspec["type"]
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0)
    p /= float(np.sum(p))
    k = len(labels)
    conf = round(confidence_from_probs(p, k), 4)
    top1 = round(float(np.max(p)), 6)

    if qtype == "choice":
        out: Dict[str, Any] = {
            "type": "choice",
            "choice": str(labels[int(np.argmax(p))]),
            "probabilities": {str(lbl): round(float(val), 6) for lbl, val in zip(labels, p)},
            "confidence": conf,
            "answer_confidence": top1,
        }
    elif qtype == "score":
        exp_score = float(np.sum(np.arange(k, dtype=np.float64) * p))
        out = {
            "type": "score",
            "score": round(exp_score, 4),
            "legend": {str(i): str(c) for i, c in enumerate(qspec["criteria"])},
            "probabilities": {str(i): round(float(val), 6) for i, val in enumerate(p)},
            "confidence": conf,
            "answer_confidence": top1,
        }
    else:
        p_true = float(p[1]) if k >= 2 else 0.5
        out = {
            "type": "noul",
            "noul": round(p_true, 6),
            "probabilities": {"false": round(1.0 - p_true, 6), "true": round(p_true, 6)},
            "confidence": conf,
            "answer_confidence": round(float(max(p_true, 1.0 - p_true)), 6),
        }

    if extra:
        for k_meta in ("prediction_set", "abstain", "route", "symmetry_runs"):
            if k_meta in extra:
                out[k_meta] = extra[k_meta]
    return out


def _cyclic_shift_choice_question(qspec: Mapping[str, Any], shift: int) -> Dict[str, Any]:
    """Return a semantically identical copy of a `choice` question with criteria cyclically shifted."""
    crit = qspec["criteria"]
    if isinstance(crit, Mapping):
        items = list(crit.items())
        n = len(items)
        s = shift % n
        shifted = dict(items[s:] + items[:s])
    else:
        items_l = list(crit)
        n = len(items_l)
        s = shift % n
        shifted = items_l[s:] + items_l[:s]
    return {
        "type": "choice",
        "instructions": qspec["instructions"],
        "criteria": shifted,
    }


def _top2_margin(probs: np.ndarray) -> float:
    if len(probs) < 2:
        return 1.0
    ordered = np.sort(probs)[::-1]
    return float(ordered[0] - ordered[1])


class SystemOneHTTPBackend(BaseDecisionEngine):
    """HTTP client engine for local or remote System-One / Decisions API servers.

    Compatible with:
      - `llama-server` (`POST /v1/systemone`) running GGUF decision models
        such as `Kev-4B`, `Tev1-4B`, `Winnow-12B`, `Julia-1`, `Laya`, `GLiDE`.
      - `OneJev` / `vLLM` / `OpenJev` servers running `Mammoth-1` (27B),
        `Elephant-1` (9B), `GLiDE v1.2`, etc.
      - Another `laya-judge serve` instance (`/v1/systemone` or `/v1/decisions`).
    """

    supports_multimodal: bool = False

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model_id: str = "local-decision-model",
        endpoint: str = "/v1/systemone",
        api_key: Optional[str] = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        max_workers: int = 8,
        text_only: bool = True,
        reject_multimodal: bool = False,
        symmetry_permutations: int = 1,
        adaptive_symmetry_threshold: Optional[float] = None,
        calibrator: Optional[Any] = None,
        calibrator_tier_tag: str = "local_http_systemone",
        transport_fn: Optional[TransportFn] = None,
    ):
        super().__init__(model_id=model_id)
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.max_workers = max(1, int(max_workers))
        self.text_only = bool(text_only)
        self.reject_multimodal = bool(reject_multimodal)
        self.supports_multimodal = not self.text_only
        self.symmetry_permutations = max(1, int(symmetry_permutations))
        self.adaptive_symmetry_threshold = (
            float(adaptive_symmetry_threshold) if adaptive_symmetry_threshold is not None else None
        )
        self.calibrator = calibrator
        self.calibrator_tier_tag = calibrator_tier_tag
        self._transport: TransportFn = transport_fn or _default_urllib_transport

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        model_name: str,
    ) -> Dict[str, Any]:
        clean_state = _strip_media(state) if self.text_only else state
        if self.endpoint.endswith("/decisions"):
            return {
                "model": model_name,
                "input": clean_state,
                "state": clean_state,
                "questions": questions,
            }
        return {
            "model": model_name,
            "state": clean_state,
            "questions": questions,
        }

    def _post_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        headers = self._headers()
        for attempt in range(self.max_retries):
            try:
                return self._transport(self.url, payload, headers, self.timeout_s)
            except Exception as exc:
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(0.25 * (2**attempt))
        raise RuntimeError(
            f"SystemOneHTTPBackend request to {self.url} failed after {self.max_retries} attempt(s): {last_err}"
        ) from last_err

    def _apply_calibrator(self, probs: np.ndarray, qspec: Mapping[str, Any]) -> np.ndarray:
        if self.calibrator is None:
            return probs
        if hasattr(self.calibrator, "calibrate_probs"):
            try:
                return np.asarray(
                    self.calibrator.calibrate_probs(probs, tier_tag=self.calibrator_tier_tag),
                    dtype=np.float64,
                )
            except TypeError:
                return np.asarray(self.calibrator.calibrate_probs(probs), dtype=np.float64)
        if hasattr(self.calibrator, "temperature_for"):
            from ..calibration import rescale_probabilities

            temp = self.calibrator.temperature_for(qspec["type"], len(probs))
            return rescale_probabilities(probs, temp)
        return probs

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        model_override: Optional[str] = None,
        symmetry_permutations: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run typed decision questions over a local/remote System-One HTTP server."""
        if self.text_only and self.reject_multimodal and _contains_media(state):
            raise ValueError(
                "SystemOneHTTPBackend is configured in text-only mode and received non-text media fields."
            )

        normalized_q = normalize_decision_questions(questions)
        model_name = model_override or self.model_id
        req_perms = max(
            1,
            int(symmetry_permutations if symmetry_permutations is not None else self.symmetry_permutations),
        )

        t0 = time.perf_counter()
        raw_primary = self._post_with_retries(self._build_payload(state, normalized_q, model_name))
        total_in_tokens = int((raw_primary.get("usage") or {}).get("input_tokens", 0))
        if total_in_tokens <= 0:
            total_in_tokens = max(1, len(serialize_state(_strip_media(state))) // 4)

        primary_answers = raw_primary.get("answers") or {}
        simplices: Dict[str, Tuple[List[str], List[np.ndarray]]] = {}
        extra_meta: Dict[str, Dict[str, Any]] = {}

        for qid, qspec in normalized_q.items():
            raw_ans = primary_answers.get(qid, {})
            labels, probs = _extract_simplex_from_answer(raw_ans, qspec)
            simplices[qid] = (labels, [probs])
            extra_meta[qid] = dict(raw_ans) if isinstance(raw_ans, Mapping) else {}

        # Determine which `choice` questions should run cyclic permutation averaging
        # to cancel out option-order (position) bias.
        perm_targets: List[str] = []
        if req_perms > 1 or self.adaptive_symmetry_threshold is not None:
            eff_perms = max(req_perms, 2 if self.adaptive_symmetry_threshold is not None else 1)
            for qid, qspec in normalized_q.items():
                if qspec["type"] != "choice":
                    continue
                labels, p_list = simplices[qid]
                if len(labels) < 2:
                    continue
                if req_perms > 1:
                    perm_targets.append(qid)
                elif self.adaptive_symmetry_threshold is not None:
                    if _top2_margin(p_list[0]) < self.adaptive_symmetry_threshold:
                        perm_targets.append(qid)
        else:
            eff_perms = 1

        if perm_targets and eff_perms > 1:
            max_shifts = min(
                eff_perms,
                max(len(simplices[qid][0]) for qid in perm_targets),
            )
            for shift in range(1, max_shifts):
                shifted_qs: Dict[str, Dict[str, Any]] = {}
                for qid in perm_targets:
                    if shift < len(simplices[qid][0]):
                        shifted_qs[qid] = _cyclic_shift_choice_question(normalized_q[qid], shift)
                if not shifted_qs:
                    continue
                raw_shift = self._post_with_retries(self._build_payload(state, shifted_qs, model_name))
                total_in_tokens += int((raw_shift.get("usage") or {}).get("input_tokens", 0))
                shift_answers = raw_shift.get("answers") or {}
                for qid in shifted_qs:
                    # Extract probabilities back in CANONICAL label order using normalized_q[qid]
                    _, p_canonical = _extract_simplex_from_answer(shift_answers.get(qid, {}), normalized_q[qid])
                    simplices[qid][1].append(p_canonical)

        final_answers: Dict[str, Any] = {}
        for qid, qspec in normalized_q.items():
            labels, p_runs = simplices[qid]
            if len(p_runs) > 1:
                # Log-linear geometric pooling across option permutations
                log_avg = np.mean([np.log(np.clip(p, 1e-6, 1.0)) for p in p_runs], axis=0)
                log_avg -= float(np.max(log_avg))
                fused = np.exp(log_avg)
                fused /= float(np.sum(fused))
            else:
                fused = p_runs[0]

            calibrated = self._apply_calibrator(fused, qspec)
            meta = dict(extra_meta.get(qid, {}))
            if len(p_runs) > 1:
                meta["symmetry_runs"] = len(p_runs)
            final_answers[qid] = _pack_canonical_answer(labels, calibrated, qspec, extra=meta)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "model": str(raw_primary.get("model") or model_name),
            "answers": final_answers,
            "usage": {
                "input_tokens": total_in_tokens,
                "output_tokens": 0,
            },
            "cost_usd": 0.0,
            "latency_ms": round(latency_ms, 2),
        }

    def predict_batch(
        self,
        states: List[Union[str, Dict[str, Any], List[Any]]],
        questions: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Execute a batch of states concurrently against the local HTTP decision server."""
        if not states:
            return []
        workers = min(len(states), self.max_workers)
        if workers <= 1:
            return [self.predict(s, questions, **kwargs) for s in states]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(self.predict, s, questions, **kwargs) for s in states]
            return [f.result() for f in futs]
