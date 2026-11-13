"""Gemini Interactions API Tier-2 Decision Engine (`google-genai >= 2.30.1`) with strict budget tracking."""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .engine.base import (
    BaseDecisionEngine,
    confidence_from_probs,
    render_options,
    serialize_state,
    validate_question_spec,
)

# Official Gemini Developer API Paid-Tier rates (USD per 1M tokens, Nov 2026)
GEMINI_MODEL_PRICING_USD_PER_1M: Dict[str, Tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-4-argon": (2.00, 10.00),
}


class BudgetExceededError(RuntimeError):
    """Raised when cumulative Gemini API spend reaches `CostTracker.max_budget_usd`."""


class CostTracker:
    """Thread-safe micro-dollar cost tracker with optional persistent ledger file."""

    def __init__(
        self,
        max_budget_usd: float = 1.80,
        ledger_path: Optional[Union[str, Path]] = None,
    ):
        self.max_budget_usd = float(max_budget_usd)
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self._lock = threading.Lock()
        self.session_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_thought_tokens = 0
        self.session_cost_usd = 0.0
        self.by_model: Dict[str, Dict[str, float]] = {}

        self._prior_ledger_cost_usd = 0.0
        if self.ledger_path is not None and self.ledger_path.is_file():
            try:
                data = json.loads(self.ledger_path.read_text())
                self._prior_ledger_cost_usd = float(data.get("cumulative_cost_usd", 0.0))
            except Exception:
                self._prior_ledger_cost_usd = 0.0

    @property
    def cumulative_cost_usd(self) -> float:
        return self._prior_ledger_cost_usd + self.session_cost_usd

    def check_can_spend(self, estimated_next_usd: float = 0.001) -> None:
        with self._lock:
            if self.cumulative_cost_usd + estimated_next_usd > self.max_budget_usd:
                raise BudgetExceededError(
                    f"Budget guard triggered: cumulative spend ${self.cumulative_cost_usd:.4f} + "
                    f"${estimated_next_usd:.4f} exceeds max_budget_usd=${self.max_budget_usd:.2f}"
                )

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        thought_tokens: int = 0,
    ) -> float:
        in_rate, out_rate = GEMINI_MODEL_PRICING_USD_PER_1M.get(model, (0.75, 3.75))
        billed_out = max(0, int(output_tokens) + int(thought_tokens))
        cost = (max(0, int(input_tokens)) / 1_000_000.0) * in_rate + (billed_out / 1_000_000.0) * out_rate

        with self._lock:
            self.session_calls += 1
            self.session_input_tokens += int(input_tokens)
            self.session_output_tokens += int(output_tokens)
            self.session_thought_tokens += int(thought_tokens)
            self.session_cost_usd += cost

            mstat = self.by_model.setdefault(
                model,
                {"calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0, "thought_tokens": 0.0, "cost_usd": 0.0},
            )
            mstat["calls"] += 1
            mstat["input_tokens"] += int(input_tokens)
            mstat["output_tokens"] += int(output_tokens)
            mstat["thought_tokens"] += int(thought_tokens)
            mstat["cost_usd"] += cost

            if self.ledger_path is not None:
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "cumulative_cost_usd": round(self._prior_ledger_cost_usd + self.session_cost_usd, 6),
                    "last_session_calls": self.session_calls,
                    "last_session_cost_usd": round(self.session_cost_usd, 6),
                    "max_budget_usd": self.max_budget_usd,
                    "by_model": self.by_model,
                }
                self.ledger_path.write_text(json.dumps(payload, indent=2))

        return cost

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_calls": self.session_calls,
                "session_input_tokens": self.session_input_tokens,
                "session_output_tokens": self.session_output_tokens,
                "session_thought_tokens": self.session_thought_tokens,
                "session_cost_usd": round(self.session_cost_usd, 6),
                "cumulative_cost_usd": round(self._prior_ledger_cost_usd + self.session_cost_usd, 6),
                "max_budget_usd": self.max_budget_usd,
                "by_model": dict(self.by_model),
            }


def _encode_image_part(
    img_obj: Any,
    max_side: int = 896,
    jpeg_quality: int = 82,
    image_resolution: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Encode PIL Image, bytes, file path, or HF image dict into an inline base64 Gemini image part."""
    if img_obj is None:
        return None
    valid_res = image_resolution if image_resolution in ("low", "medium", "high", "ultra_high") else None
    try:
        from PIL import Image

        pil_img: Optional[Image.Image] = None
        if isinstance(img_obj, Image.Image):
            pil_img = img_obj
        elif isinstance(img_obj, dict) and "bytes" in img_obj and img_obj["bytes"]:
            pil_img = Image.open(io.BytesIO(img_obj["bytes"]))
        elif isinstance(img_obj, dict) and "path" in img_obj and img_obj["path"]:
            pil_img = Image.open(img_obj["path"])
        elif isinstance(img_obj, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(img_obj))
        elif isinstance(img_obj, (str, Path)) and Path(img_obj).is_file():
            pil_img = Image.open(img_obj)
        elif isinstance(img_obj, np.ndarray):
            pil_img = Image.fromarray(img_obj.astype(np.uint8))

        if pil_img is None:
            return None

        if pil_img.mode not in ("RGB", "L"):
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            if "A" in pil_img.getbands():
                bg.paste(pil_img, mask=pil_img.split()[-1])
                pil_img = bg
            else:
                pil_img = pil_img.convert("RGB")
        elif pil_img.mode == "L":
            pil_img = pil_img.convert("RGB")

        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        part: Dict[str, str] = {"type": "image", "data": b64, "mime_type": "image/jpeg"}
        if valid_res is not None:
            part["resolution"] = valid_res
        return part
    except Exception:
        if isinstance(img_obj, (bytes, bytearray)):
            part = {
                "type": "image",
                "data": base64.b64encode(bytes(img_obj)).decode("utf-8"),
                "mime_type": "image/jpeg",
            }
            if valid_res is not None:
                part["resolution"] = valid_res
            return part
        return None


def _encode_audio_part(audio_obj: Any, max_seconds: float = 35.0) -> Optional[Dict[str, str]]:
    """Encode HF audio dict `{'array': ..., 'sampling_rate': ...}`, WAV/MP3 bytes, or path into inline audio."""
    if audio_obj is None:
        return None
    try:
        if isinstance(audio_obj, dict):
            if "bytes" in audio_obj and audio_obj["bytes"]:
                raw_b = bytes(audio_obj["bytes"])
                mime = "audio/mp3" if raw_b[:3] == b"ID3" or raw_b[:2] == b"\xff\xfb" else "audio/wav"
                if len(raw_b) <= 8_000_000:
                    return {"type": "audio", "data": base64.b64encode(raw_b).decode("utf-8"), "mime_type": mime}
            if "array" in audio_obj and audio_obj["array"] is not None:
                import soundfile as sf

                arr = np.asarray(audio_obj["array"], dtype=np.float32)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=-1)
                sr = int(audio_obj.get("sampling_rate", 16000))
                # Downsample >16kHz audio cleanly via integer decimation or linear interp to save payload bytes
                target_sr = 16000
                if sr > target_sr:
                    new_len = max(1, int(round(len(arr) * target_sr / float(sr))))
                    x_old = np.linspace(0.0, 1.0, len(arr), endpoint=False)
                    x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
                    arr = np.interp(x_new, x_old, arr).astype(np.float32)
                    sr = target_sr
                max_samples = int(sr * max_seconds)
                if len(arr) > max_samples:
                    arr = arr[:max_samples]
                buf = io.BytesIO()
                sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
                return {
                    "type": "audio",
                    "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
                    "mime_type": "audio/wav",
                }
            if "path" in audio_obj and audio_obj["path"] and Path(audio_obj["path"]).is_file():
                audio_obj = Path(audio_obj["path"])

        if isinstance(audio_obj, (str, Path)) and Path(audio_obj).is_file():
            p = Path(audio_obj)
            raw_b = p.read_bytes()
            ext = p.suffix.lower()
            mime = "audio/mp3" if ext == ".mp3" else "audio/flac" if ext == ".flac" else "audio/wav"
            return {"type": "audio", "data": base64.b64encode(raw_b).decode("utf-8"), "mime_type": mime}

        if isinstance(audio_obj, (bytes, bytearray)):
            raw_b = bytes(audio_obj)
            mime = "audio/mp3" if raw_b[:3] == b"ID3" else "audio/wav"
            return {"type": "audio", "data": base64.b64encode(raw_b).decode("utf-8"), "mime_type": mime}
    except Exception:
        return None
    return None


def _encode_video_part(video_obj: Any) -> Optional[Dict[str, str]]:
    """Encode inline MP4 bytes, local video file (<14MB), or YouTube URL into a Gemini video part."""
    if video_obj is None:
        return None
    if isinstance(video_obj, str) and (
        video_obj.startswith("https://www.youtube.com/") or video_obj.startswith("https://youtu.be/")
    ):
        return {"type": "video", "uri": video_obj, "mime_type": "video/mp4"}
    if isinstance(video_obj, (str, Path)) and Path(video_obj).is_file():
        raw_b = Path(video_obj).read_bytes()
        if len(raw_b) <= 14_000_000:
            return {"type": "video", "data": base64.b64encode(raw_b).decode("utf-8"), "mime_type": "video/mp4"}
    if isinstance(video_obj, (bytes, bytearray)) and len(video_obj) <= 14_000_000:
        return {"type": "video", "data": base64.b64encode(bytes(video_obj)).decode("utf-8"), "mime_type": "video/mp4"}
    return None


def extract_media_parts(
    state: Union[str, Dict[str, Any], List[Any]],
    image_max_side: int = 896,
    image_resolution: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract encoded inline Gemini media parts (images, storyboards, audio, video) from `state`."""
    if not isinstance(state, dict):
        return []

    media_parts: List[Dict[str, Any]] = []
    for k in ("image", "_pil_image", "screenshot"):
        if k in state and state[k] is not None:
            part = _encode_image_part(state[k], max_side=image_max_side, image_resolution=image_resolution)
            if part is not None:
                media_parts.append(part)
    for k in ("images", "screenshots", "_frames"):
        if k in state and isinstance(state[k], (list, tuple)):
            for item in state[k]:
                part = _encode_image_part(item, max_side=image_max_side, image_resolution=image_resolution)
                if part is not None:
                    media_parts.append(part)
    for k in ("audio", "_audio_bytes"):
        if k in state and state[k] is not None:
            part = _encode_audio_part(state[k])
            if part is not None:
                media_parts.append(part)
    for k in ("video", "_video_bytes"):
        if k in state and state[k] is not None:
            part = _encode_video_part(state[k])
            if part is not None:
                media_parts.append(part)
    return media_parts


def prepare_multimodal_input(
    state: Union[str, Dict[str, Any], List[Any]],
    prompt_text: str,
    image_max_side: int = 896,
    image_resolution: Optional[str] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """Build `interactions.create(input=...)` combining optional inline media parts + compact text prompt."""
    media_parts = extract_media_parts(state, image_max_side=image_max_side, image_resolution=image_resolution)
    if not media_parts:
        return prompt_text

    media_parts.append({"type": "text", "text": prompt_text})
    return media_parts


def strip_binary_state_fields(state: Union[str, Dict[str, Any], List[Any]]) -> Union[str, Dict[str, Any], List[Any]]:
    """Strip non-JSON binary/PIL media payloads from `state` before text serialization."""
    if not isinstance(state, dict):
        return state
    skip = {
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
    return {k: v for k, v in state.items() if str(k).lower() not in skip}


def _canonical_labels(qspec: Mapping[str, Any]) -> List[str]:
    qtype = qspec["type"]
    if qtype == "noul":
        return ["false", "true"]
    if qtype == "score":
        return [str(i) for i in range(len(qspec["criteria"]))]
    crit = qspec["criteria"]
    return [str(k) for k in (crit.keys() if isinstance(crit, Mapping) else crit)]


def build_questions_schema(questions: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Construct compact JSON Schema for multi-question typed probabilistic decision output."""
    props: Dict[str, Any] = {}
    req: List[str] = []
    for qid, qspec in questions.items():
        labels = _canonical_labels(qspec)
        props[qid] = {
            "type": "object",
            "properties": {
                "c": {"type": "string", "enum": labels},
                "p": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["c", "p"],
        }
        req.append(qid)
    return {
        "type": "object",
        "properties": props,
        "required": req,
    }


def build_decision_prompt(
    clean_state: Union[str, Dict[str, Any], List[Any]],
    questions: Mapping[str, Mapping[str, Any]],
    system_hint: Optional[str] = None,
    packed_mode: bool = False,
) -> str:
    """Build concise, calibrated decision prompt requesting exact label `c` and calibrated integer percentages `p`."""
    state_txt = serialize_state(clean_state)
    lines: List[str] = [
        "You are a calibrated typed-decision evaluator. Return compact single-line JSON (no indentation or newlines).",
        "Evaluate the STATE (and any attached media) and answer every question key.",
        "For each question key, output:",
        '  "c": best option ID from the allowed enum,',
        '  "p": calibrated integer percentages [p_0, ..., p_{K-1}] in exact listed option order summing to 100 (e.g. [5, 15, 80]; reflect genuine uncertainty, never output 100 unless 100% certain).',
    ]
    if packed_mode:
        lines.append(
            "PACKED MULTI-ITEM RULE: Each item I<j> lists its assigned question key(s) q<k>. "
            "Evaluate each item STRICTLY and INDEPENDENTLY using ONLY its own state/media."
        )
    if system_hint:
        lines.append(f"Domain guidance: {system_hint}")
    if state_txt and state_txt != "{}":
        lines.append(f"\nSTATE:\n{state_txt}")
    lines.append("\nQUESTIONS:")

    # Deduplicate identical rubrics when multiple packed items share the exact same question spec & options
    grouped_specs: Dict[Tuple[str, str, str], List[str]] = {}
    spec_meta: Dict[Tuple[str, str, str], int] = {}
    for qid, qspec in questions.items():
        labels = _canonical_labels(qspec)
        opts_rendered = render_options(dict(qspec))
        opt_lines = " | ".join(f"[{idx}:{lbl}] {txt}" for idx, (lbl, txt) in enumerate(zip(labels, opts_rendered)))
        sig = (str(qspec["type"]), str(qspec["instructions"]), opt_lines)
        grouped_specs.setdefault(sig, []).append(qid)
        spec_meta[sig] = len(labels)

    for (qtype, ins, opt_lines), qid_list in grouped_specs.items():
        n_lbl = spec_meta[(qtype, ins, opt_lines)]
        if len(qid_list) == 1:
            lines.append(f"- {qid_list[0]} ({qtype}): {ins}\n  Options ({n_lbl}): {opt_lines}")
        else:
            joined_keys = ", ".join(qid_list)
            lines.append(f"- [{joined_keys}] ({qtype}): {ins}\n  Options ({n_lbl}): {opt_lines}")
    return "\n".join(lines)


def parse_structured_answers(
    raw_json_text: str,
    questions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Parse Gemini JSON output into Laya-compatible typed `answers` with normalized probability simplexes."""
    try:
        parsed = json.loads(raw_json_text) if raw_json_text else {}
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    answers: Dict[str, Dict[str, Any]] = {}
    for qid, qspec in questions.items():
        qtype = qspec["type"]
        labels = _canonical_labels(qspec)
        k = len(labels)
        item = parsed.get(qid, {}) if isinstance(parsed.get(qid), dict) else {}
        chosen = str(item.get("c", labels[0]))
        if chosen not in labels:
            # Case-insensitive fallback match
            low_map = {lbl.lower(): lbl for lbl in labels}
            chosen = low_map.get(chosen.lower().strip(), labels[0])
        chosen_idx = labels.index(chosen)

        raw_p = item.get("p")
        probs: Optional[np.ndarray] = None
        if isinstance(raw_p, list) and len(raw_p) == k:
            try:
                arr = np.array([max(0.0, float(x)) for x in raw_p], dtype=np.float64)
                if np.all(np.isfinite(arr)) and float(np.sum(arr)) > 0.0:
                    probs = arr / float(np.sum(arr))
            except Exception:
                probs = None

        if probs is None:
            # Smooth fallback distribution centered on `chosen`
            probs = np.full(k, 0.15 / max(1, k - 1), dtype=np.float64) if k > 1 else np.ones(1, dtype=np.float64)
            if k > 1:
                probs[chosen_idx] = 0.85
        else:
            # Ensure `chosen` and `argmax(probs)` agree if model wrote slightly inconsistent `p` vs `c`
            if int(np.argmax(probs)) != chosen_idx:
                # Trust explicit enum choice `c` by swapping mass with argmax slot or boosting `c`
                max_idx = int(np.argmax(probs))
                probs[chosen_idx], probs[max_idx] = probs[max_idx], probs[chosen_idx]

        # Laplace floor 1e-4 to avoid degenerate log(0) in NLL calibration
        probs = np.clip(probs, 1e-4, 1.0)
        probs /= float(np.sum(probs))
        conf = round(confidence_from_probs(probs, k), 4)

        if qtype == "choice":
            answers[qid] = {
                "type": "choice",
                "choice": labels[int(np.argmax(probs))],
                "probabilities": {lbl: round(float(p), 6) for lbl, p in zip(labels, probs)},
                "confidence": conf,
                "answer_confidence": round(float(np.max(probs)), 6),
            }
        elif qtype == "score":
            exp_score = float(np.sum(np.arange(k, dtype=np.float64) * probs))
            answers[qid] = {
                "type": "score",
                "score": round(exp_score, 4),
                "legend": {str(i): str(c) for i, c in enumerate(qspec["criteria"])},
                "probabilities": {str(i): round(float(p), 6) for i, p in enumerate(probs)},
                "confidence": conf,
                "answer_confidence": round(float(np.max(probs)), 6),
            }
        else:
            p_true = float(probs[1]) if k >= 2 else 0.5
            answers[qid] = {
                "type": "noul",
                "noul": round(p_true, 6),
                "probabilities": {"false": round(1.0 - p_true, 6), "true": round(p_true, 6)},
                "confidence": conf,
                "answer_confidence": round(float(max(p_true, 1.0 - p_true)), 6),
            }
    return answers


class GeminiDecisionEngine(BaseDecisionEngine):
    """Cloud Tier-2 Decision Engine built on `google-genai` Interactions API (`client.interactions.create`).

    Supports text, images, audio, and video with strict JSON Schema output, concurrency, retries,
    and micro-dollar `CostTracker` enforcement.
    """

    supports_multimodal: bool = True

    def __init__(
        self,
        model_id: str = "gemini-3.5-flash-lite",
        thinking_level: Optional[str] = "low",
        cost_tracker: Optional[CostTracker] = None,
        max_budget_usd: float = 1.80,
        max_workers: int = 8,
        max_retries: int = 3,
        image_max_side: int = 896,
        image_resolution: Optional[str] = None,
        system_hint: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        super().__init__(model_id=model_id)
        self.thinking_level = thinking_level
        self.cost_tracker = cost_tracker or CostTracker(max_budget_usd=max_budget_usd)
        self.max_workers = max(1, int(max_workers))
        self.max_retries = max(1, int(max_retries))
        self.image_max_side = int(image_max_side)
        self.image_resolution = image_resolution
        self.system_hint = system_hint
        self._api_key = api_key
        self._client = client
        self._client_lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                from google import genai

                key = self._api_key or os.environ.get("GEMINI_API_KEY")
                if not key:
                    raise RuntimeError(
                        "GEMINI_API_KEY environment variable is not set. "
                        "Export GEMINI_API_KEY before calling GeminiDecisionEngine."
                    )
                self._client = genai.Client(api_key=key)
        return self._client

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        model_override: Optional[str] = None,
        thinking_level_override: Optional[str] = None,
        system_hint_override: Optional[str] = None,
        image_max_side_override: Optional[int] = None,
        image_resolution_override: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute structured typed decisions via `client.interactions.create(...)`."""
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        target_model = model_override or self.model_id
        target_thinking = thinking_level_override if thinking_level_override is not None else self.thinking_level
        hint = system_hint_override if system_hint_override is not None else self.system_hint
        eff_img_side = int(image_max_side_override) if image_max_side_override is not None else self.image_max_side
        eff_img_res = image_resolution_override if image_resolution_override is not None else self.image_resolution

        self.cost_tracker.check_can_spend(0.0005)

        clean_state = strip_binary_state_fields(state)
        prompt = build_decision_prompt(clean_state, normalized_q, system_hint=hint)
        interaction_input = prepare_multimodal_input(
            state, prompt, image_max_side=eff_img_side, image_resolution=eff_img_res
        )
        schema = build_questions_schema(normalized_q)

        req_kwargs: Dict[str, Any] = {
            "model": target_model,
            "input": interaction_input,
            "store": False,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
        }
        if target_thinking:
            req_kwargs["generation_config"] = {"thinking_level": target_thinking}

        t0 = time.perf_counter()
        last_err: Optional[Exception] = None
        interaction: Any = None
        client = self._get_client()

        for attempt in range(self.max_retries):
            try:
                interaction = client.interactions.create(**req_kwargs)
                break
            except Exception as exc:
                last_err = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(0.8 * (2**attempt))
        if interaction is None:
            raise RuntimeError(f"Gemini Interactions API failed after {self.max_retries} attempts: {last_err}")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage_obj = getattr(interaction, "usage", None)
        in_toks = int(getattr(usage_obj, "total_input_tokens", 0) or 0)
        out_toks = int(getattr(usage_obj, "total_output_tokens", 0) or 0)
        thought_toks = int(getattr(usage_obj, "total_thought_tokens", 0) or 0)

        call_cost = self.cost_tracker.record_usage(
            model=target_model,
            input_tokens=in_toks,
            output_tokens=out_toks,
            thought_tokens=thought_toks,
        )
        raw_text = getattr(interaction, "output_text", "") or ""
        answers = parse_structured_answers(raw_text, normalized_q)

        return {
            "model": target_model,
            "answers": answers,
            "usage": {
                "input_tokens": in_toks,
                "output_tokens": out_toks + thought_toks,
                "thought_tokens": thought_toks,
            },
            "cost_usd": round(call_cost, 8),
            "latency_ms": round(latency_ms, 2),
        }

    def _predict_packed_chunk(
        self,
        chunk: Sequence[Tuple[Union[str, Dict[str, Any], List[Any]], Dict[str, Dict[str, Any]]]],
        model_override: Optional[str] = None,
        thinking_level_override: Optional[str] = None,
        system_hint_override: Optional[str] = None,
        image_max_side_override: Optional[int] = None,
        image_resolution_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute 1 multi-item `interactions.create` call for `m` independent `(state, questions)` pairs."""
        if not chunk:
            return []
        if len(chunk) == 1:
            st0, qs0 = chunk[0]
            return [
                self.predict(
                    st0,
                    qs0,
                    model_override=model_override,
                    thinking_level_override=thinking_level_override,
                    system_hint_override=system_hint_override,
                    image_max_side_override=image_max_side_override,
                    image_resolution_override=image_resolution_override,
                )
            ]

        norm_chunk = [
            (st, {qid: validate_question_spec(qid, q) for qid, q in qs.items()})
            for st, qs in chunk
        ]
        target_model = model_override or self.model_id
        target_thinking = thinking_level_override if thinking_level_override is not None else self.thinking_level
        hint = system_hint_override if system_hint_override is not None else self.system_hint
        eff_img_side = int(image_max_side_override) if image_max_side_override is not None else self.image_max_side
        eff_img_res = image_resolution_override if image_resolution_override is not None else self.image_resolution

        self.cost_tracker.check_can_spend(0.0005 * len(norm_chunk))

        packed_q: Dict[str, Dict[str, Any]] = {}
        key_map: List[Dict[str, str]] = []
        clean_states: List[Union[str, Dict[str, Any], List[Any]]] = []
        media_lists: List[List[Dict[str, Any]]] = []
        flat_q_counter = 0
        for j, (st_j, qs_j) in enumerate(norm_chunk):
            clean_states.append(strip_binary_state_fields(st_j))
            media_lists.append(extract_media_parts(st_j, image_max_side=eff_img_side, image_resolution=eff_img_res))
            j_map: Dict[str, str] = {}
            for qid, qspec in qs_j.items():
                short_k = f"q{flat_q_counter}"
                flat_q_counter += 1
                packed_q[short_k] = qspec
                j_map[qid] = short_k
            key_map.append(j_map)

        any_media = any(bool(ml) for ml in media_lists)
        if any_media:
            parts: List[Dict[str, Any]] = []
            for j, (clean_j, media_j, j_map) in enumerate(zip(clean_states, media_lists, key_map)):
                st_str = serialize_state(clean_j)
                q_keys_str = ", ".join(j_map.values())
                parts.append({"type": "text", "text": f"=== BEGIN I{j} (keys: {q_keys_str}) ===\nSTATE I{j}:\n{st_str}"})
                parts.extend(media_j)
                parts.append({"type": "text", "text": f"=== END I{j} ==="})
            footer = build_decision_prompt({}, packed_q, system_hint=hint, packed_mode=True)
            parts.append({"type": "text", "text": footer})
            interaction_input: Union[str, List[Dict[str, Any]]] = parts
        else:
            packed_state = {
                f"I{j} (keys: {', '.join(key_map[j].values())})": clean_j
                for j, clean_j in enumerate(clean_states)
            }
            interaction_input = build_decision_prompt(packed_state, packed_q, system_hint=hint, packed_mode=True)

        schema = build_questions_schema(packed_q)
        req_kwargs: Dict[str, Any] = {
            "model": target_model,
            "input": interaction_input,
            "store": False,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
        }
        if target_thinking:
            req_kwargs["generation_config"] = {"thinking_level": target_thinking}

        t0 = time.perf_counter()
        interaction: Any = None
        client = self._get_client()
        for attempt in range(self.max_retries):
            try:
                interaction = client.interactions.create(**req_kwargs)
                break
            except Exception:
                if attempt + 1 < self.max_retries:
                    time.sleep(0.8 * (2**attempt))

        if interaction is None:
            # Automatic fault-tolerant fallback to individual item calls
            return [
                self.predict(
                    st_j,
                    qs_j,
                    model_override=model_override,
                    thinking_level_override=thinking_level_override,
                    system_hint_override=system_hint_override,
                    image_max_side_override=image_max_side_override,
                    image_resolution_override=image_resolution_override,
                )
                for st_j, qs_j in norm_chunk
            ]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage_obj = getattr(interaction, "usage", None)
        in_toks = int(getattr(usage_obj, "total_input_tokens", 0) or 0)
        out_toks = int(getattr(usage_obj, "total_output_tokens", 0) or 0)
        thought_toks = int(getattr(usage_obj, "total_thought_tokens", 0) or 0)
        call_cost = self.cost_tracker.record_usage(
            model=target_model,
            input_tokens=in_toks,
            output_tokens=out_toks,
            thought_tokens=thought_toks,
        )

        raw_text = getattr(interaction, "output_text", "") or ""
        try:
            raw_obj = json.loads(raw_text) if raw_text else {}
        except Exception:
            raw_obj = {}
        hit_keys = sum(1 for k in packed_q if isinstance(raw_obj, dict) and k in raw_obj)
        if hit_keys < max(1, int(0.8 * len(packed_q))):
            return [
                self.predict(
                    st_j,
                    qs_j,
                    model_override=model_override,
                    thinking_level_override=thinking_level_override,
                    system_hint_override=system_hint_override,
                    image_max_side_override=image_max_side_override,
                    image_resolution_override=image_resolution_override,
                )
                for st_j, qs_j in norm_chunk
            ]

        packed_answers = parse_structured_answers(raw_text, packed_q)
        m = len(norm_chunk)
        total_qs = max(1, len(packed_q))
        billed_out = out_toks + thought_toks

        # Conserve exact token & USD totals across the `m` unpacked item outputs
        q_counts = [max(1, len(qs_j)) for _, qs_j in norm_chunk]
        weights = [qc / float(sum(q_counts)) for qc in q_counts]
        in_alloc = [int(round(in_toks * w)) for w in weights]
        in_alloc[0] += in_toks - sum(in_alloc)
        out_alloc = [int(round(billed_out * w)) for w in weights]
        out_alloc[0] += billed_out - sum(out_alloc)
        th_alloc = [int(round(thought_toks * w)) for w in weights]
        th_alloc[0] += thought_toks - sum(th_alloc)
        cost_alloc = [call_cost * w for w in weights]

        results: List[Dict[str, Any]] = []
        for j, (_, qs_j) in enumerate(norm_chunk):
            item_ans = {qid: packed_answers[key_map[j][qid]] for qid in qs_j}
            results.append(
                {
                    "model": target_model,
                    "answers": item_ans,
                    "usage": {
                        "input_tokens": max(0, in_alloc[j]),
                        "output_tokens": max(0, out_alloc[j]),
                        "thought_tokens": max(0, th_alloc[j]),
                    },
                    "cost_usd": round(cost_alloc[j], 8),
                    "latency_ms": round(latency_ms / float(m), 2),
                }
            )
        return results

    def predict_packed_pairs(
        self,
        pairs: Sequence[Tuple[Union[str, Dict[str, Any], List[Any]], Dict[str, Dict[str, Any]]]],
        pack_size: int = 1,
        max_workers: Optional[int] = None,
        model_override: Optional[str] = None,
        thinking_level_override: Optional[str] = None,
        system_hint_override: Optional[str] = None,
        image_max_side_override: Optional[int] = None,
        image_resolution_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute `(state, questions)` pairs packed in chunks of `pack_size` across concurrent workers."""
        if not pairs:
            return []
        eff_pack = max(1, int(pack_size))
        chunks = [list(pairs[i : i + eff_pack]) for i in range(0, len(pairs), eff_pack)]
        workers = min(len(chunks), max(1, int(max_workers or self.max_workers)))

        if workers <= 1:
            nested = [
                self._predict_packed_chunk(
                    ch,
                    model_override=model_override,
                    thinking_level_override=thinking_level_override,
                    system_hint_override=system_hint_override,
                    image_max_side_override=image_max_side_override,
                    image_resolution_override=image_resolution_override,
                )
                for ch in chunks
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [
                    pool.submit(
                        self._predict_packed_chunk,
                        ch,
                        model_override=model_override,
                        thinking_level_override=thinking_level_override,
                        system_hint_override=system_hint_override,
                        image_max_side_override=image_max_side_override,
                        image_resolution_override=image_resolution_override,
                    )
                    for ch in chunks
                ]
                nested = [f.result() for f in futs]

        flat: List[Dict[str, Any]] = []
        for sub in nested:
            flat.extend(sub)
        return flat

    def predict_batch(
        self,
        states: List[Union[str, Dict[str, Any], List[Any]]],
        questions: Dict[str, Dict[str, Any]],
        max_workers: Optional[int] = None,
        pack_size: int = 1,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Execute concurrent (and optionally multi-item packed) `interactions.create` calls across `states`."""
        if not states:
            return []
        if int(pack_size) > 1 and len(states) > 1:
            return self.predict_packed_pairs(
                [(s, questions) for s in states],
                pack_size=int(pack_size),
                max_workers=max_workers,
                model_override=kwargs.get("model_override"),
                thinking_level_override=kwargs.get("thinking_level_override"),
                system_hint_override=kwargs.get("system_hint_override"),
                image_max_side_override=kwargs.get("image_max_side_override"),
                image_resolution_override=kwargs.get("image_resolution_override"),
            )
        workers = min(len(states), max(1, int(max_workers or self.max_workers)))
        if workers <= 1:
            return [self.predict(s, questions, **kwargs) for s in states]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.predict, s, questions, **kwargs) for s in states]
            return [f.result() for f in futures]
