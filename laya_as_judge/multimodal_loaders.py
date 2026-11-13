"""Multimodal & Domain-Enriched Benchmark Loaders + Strictly Disjoint TTT Trainer."""

from __future__ import annotations

import io
import json
import re
import subprocess
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .calibration import ClassMultiplierOptimizer

_WHISPER_LOCK = threading.Lock()
_WHISPER_MODEL: Any = None


def transcribe_pcm16_fast(pcm_f32: np.ndarray, max_chars: int = 2400) -> str:
    """Transcribe 16kHz mono float32 waveform locally via cached `faster-whisper-tiny` ($0.00 cost)."""
    global _WHISPER_MODEL
    if pcm_f32 is None or len(pcm_f32) < 1600:
        return ""
    try:
        from faster_whisper import WhisperModel

        with _WHISPER_LOCK:
            if _WHISPER_MODEL is None:
                try:
                    import torch

                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    dev = "cpu"
                ctype = "float16" if dev == "cuda" else "int8"
                try:
                    _WHISPER_MODEL = WhisperModel("tiny", device=dev, compute_type=ctype)
                except Exception:
                    _WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")

        segments, _ = _WHISPER_MODEL.transcribe(
            pcm_f32,
            beam_size=1,
            vad_filter=True,
            without_timestamps=False,
        )
        lines: List[str] = []
        for seg in segments:
            txt = (seg.text or "").strip()
            if txt:
                mm = int(seg.start // 60)
                ss = int(seg.start % 60)
                lines.append(f"[{mm:02d}:{ss:02d}] {txt}")
        full = " ".join(lines)
        if len(full) <= max_chars:
            return full
        # Preserve beginning, middle, and conclusion of long video transcripts
        part = max_chars // 3 - 6
        mid = len(full) // 2
        return (
            full[:part]
            + " ... "
            + full[max(0, mid - part // 2) : mid + part // 2]
            + " ... "
            + full[-part:]
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 1. Enriched When2Call Schema & Candidate Response Analyzer
# ---------------------------------------------------------------------------


def analyze_when2call_row(row: Mapping[str, Any], compact_tool_fn: Any) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build enriched When2Call state & per-case MCQ criteria using ONLY public MCQ inputs (`question`, `tools`, `answers`)."""
    raw_tools = list(row.get("tools") or [])
    parsed_tools: List[Dict[str, Any]] = []
    compact_tools: List[Dict[str, Any]] = []
    declared_names: Dict[str, Dict[str, Any]] = {}

    for rt in raw_tools:
        compact_tools.append(compact_tool_fn(rt))
        try:
            td = json.loads(rt) if isinstance(rt, str) else dict(rt)
            if isinstance(td, dict) and td.get("name"):
                tname = str(td["name"])
                parsed_tools.append(td)
                declared_names[tname] = td
                # Also index short suffix after '.' if namespaced
                if "." in tname:
                    declared_names.setdefault(tname.split(".")[-1], td)
        except Exception:
            pass

    ans_dict = dict(row.get("answers") or {})
    tc_raw = str(ans_dict.get("tool_call") or "").strip()
    schema_audit: Dict[str, Any] = {
        "num_available_tools": len(parsed_tools),
        "available_tool_names": [str(t.get("name", "")) for t in parsed_tools],
    }

    if tc_raw:
        try:
            tc_obj = json.loads(tc_raw)
            if isinstance(tc_obj, dict):
                cand_name = str(tc_obj.get("name", ""))
                cand_args = tc_obj.get("arguments", {}) or {}
                in_exact = cand_name in [str(t.get("name", "")) for t in parsed_tools]
                schema_audit["candidate_tool_call_name"] = cand_name
                schema_audit["candidate_tool_exists_in_available_tools"] = bool(in_exact)
                if in_exact and isinstance(cand_args, dict):
                    matched_td = next(t for t in parsed_tools if str(t.get("name", "")) == cand_name)
                    params = matched_td.get("parameters", {}) or {}
                    req_list = list(params.get("required") or []) if isinstance(params, dict) else []
                    missing_req = [
                        rk
                        for rk in req_list
                        if rk not in cand_args or cand_args[rk] in (None, "", [], {})
                    ]
                    schema_audit["declared_required_params"] = req_list
                    schema_audit["missing_required_params_in_call"] = missing_req
        except Exception:
            schema_audit["candidate_tool_call_parseable_json"] = False

    candidate_responses: Dict[str, str] = {}
    for k_opt in ("direct", "tool_call", "request_for_info", "cannot_answer"):
        if ans_dict.get(k_opt):
            candidate_responses[k_opt] = str(ans_dict[k_opt])[:140]

    from .eval_suite import WHEN2CALL_CRITERIA

    state = {
        "user_request": str(row.get("question", "")),
        "available_tools": compact_tools,
        "schema_audit": schema_audit,
        "candidate_responses": candidate_responses,
    }
    return state, dict(WHEN2CALL_CRITERIA)


# ---------------------------------------------------------------------------
# 2. Enriched Amazon ESCI State Builder
# ---------------------------------------------------------------------------

_ACCESSORY_PATTERNS = re.compile(
    r"\b(for|compatible with|fits|replacement|case|cover|protector|charger|cable|adapter|mount|stand|strap|holder|refill|kit for)\b",
    re.IGNORECASE,
)


def build_enriched_esci_state(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Construct compact, high-signal Amazon ESCI state from public query & product fields."""
    q_str = str(row.get("query") or "").strip()
    title = str(row.get("product_title") or "").strip()
    brand = str(row.get("product_brand") or "").strip()
    color = str(row.get("product_color") or "").strip()
    bullets = str(row.get("product_bullet_point") or "").strip()
    desc = str(row.get("product_description") or "").strip()

    if brand.lower() == "none":
        brand = ""
    if color.lower() == "none":
        color = ""
    if bullets.lower() == "none":
        bullets = ""
    if desc.lower() == "none":
        desc = ""

    q_toks = [w for w in re.findall(r"[a-z0-9]+", q_str.lower()) if len(w) > 1]
    t_toks = set(re.findall(r"[a-z0-9]+", f"{title} {brand} {color}".lower()))
    overlap = round(sum(1 for w in q_toks if w in t_toks) / max(1, len(q_toks)), 2) if q_toks else 0.0

    state: Dict[str, Any] = {
        "query": q_str,
        "product_title": title,
        "brand": brand,
        "color": color,
        "query_term_overlap": overlap,
    }
    if bullets:
        state["bullets"] = bullets[:420]
    if desc:
        state["description"] = desc[:280]
    return state


# ---------------------------------------------------------------------------
# 3. Multimodal-Mind2Web Loader (Set-of-Mark Viewport Grounding + DOM)
# ---------------------------------------------------------------------------


def _parse_cand_json(raw_cand: str) -> Dict[str, Any]:
    try:
        d = json.loads(raw_cand) if isinstance(raw_cand, str) else dict(raw_cand)
    except Exception:
        return {"tag": "unknown", "text": str(raw_cand)[:120], "bbox": None}

    attrs_raw = d.get("attributes", "{}")
    try:
        attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) else dict(attrs_raw or {})
    except Exception:
        attrs = {}

    bbox: Optional[Tuple[float, float, float, float]] = None
    rect_str = attrs.get("bounding_box_rect")
    if isinstance(rect_str, str):
        parts = [p.strip() for p in rect_str.split(",")]
        if len(parts) == 4:
            try:
                x, y, w, h = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
                if w > 1 and h > 1:
                    bbox = (x, y, w, h)
            except Exception:
                bbox = None

    # Extract clean human-readable attributes
    keep_attrs: Dict[str, str] = {}
    for ak in ("id", "name", "type", "role", "aria_label", "aria-label", "placeholder", "title", "value", "href", "alt", "class"):
        if ak in attrs and attrs[ak]:
            lim_c = 38 if ak == "class" else 72
            keep_attrs[ak] = str(attrs[ak])[:lim_c]
    text_val = str(d.get("text") or attrs.get("inner_text") or "").strip()[:110]
    return {
        "tag": str(d.get("tag", "elem")),
        "text": text_val,
        "attrs": keep_attrs,
        "bbox": bbox,
    }


def _render_som_mind2web_crop(
    screenshot: Any,
    labeled_candidates: Sequence[Tuple[str, Dict[str, Any]]],
) -> Any:
    """Crop tall Mind2Web screenshot around candidate bounding boxes and draw Set-of-Mark `[A]..[E]` overlays."""
    from PIL import Image, ImageDraw

    if not isinstance(screenshot, Image.Image):
        return screenshot
    img = screenshot.convert("RGB")
    W, H = img.size

    valid_boxes = [(lbl, c["bbox"]) for lbl, c in labeled_candidates if c.get("bbox") is not None]
    if not valid_boxes:
        crop_y0, crop_y1 = 0, min(H, 1050)
    else:
        ys_top = [max(0.0, b[1]) for _, b in valid_boxes]
        ys_bot = [min(float(H), b[1] + b[3]) for _, b in valid_boxes]
        min_y, max_y = min(ys_top), max(ys_bot)
        if max_y - min_y <= 1250:
            mid_y = 0.5 * (min_y + max_y)
            half = max(480.0, 0.5 * (max_y - min_y) + 180.0)
            crop_y0 = int(max(0.0, mid_y - half))
            crop_y1 = int(min(float(H), crop_y0 + 2 * half))
            crop_y0 = int(max(0, crop_y1 - int(2 * half)))
        else:
            # Anchor around median y while keeping up to 1400px vertical window
            med_y = float(np.median(ys_top))
            crop_y0 = int(max(0.0, med_y - 650.0))
            crop_y1 = int(min(float(H), crop_y0 + 1350.0))

    cropped = img.crop((0, crop_y0, W, max(crop_y0 + 64, crop_y1)))
    draw = ImageDraw.Draw(cropped)
    palette = [
        (220, 38, 38),
        (37, 99, 235),
        (22, 163, 74),
        (217, 119, 6),
        (147, 51, 234),
        (8, 145, 178),
    ]
    for idx, (lbl, cand) in enumerate(labeled_candidates):
        bbox = cand.get("bbox")
        if bbox is None:
            continue
        x, y, w, h = bbox
        ry = y - crop_y0
        if ry + h < 0 or ry > cropped.size[1]:
            continue
        color = palette[idx % len(palette)]
        x0 = max(0, int(x))
        y0 = max(0, int(ry))
        x1 = min(cropped.size[0] - 1, int(x + w))
        y1 = min(cropped.size[1] - 1, int(ry + h))
        if x1 > x0 and y1 > y0:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            tag_y0 = max(0, y0 - 18)
            draw.rectangle([x0, tag_y0, min(cropped.size[0] - 1, x0 + 28), tag_y0 + 18], fill=color)
            draw.text((x0 + 4, tag_y0 + 2), f"[{lbl}]", fill=(255, 255, 255))
    return cropped


def load_mind2web_multimodal_cases(limit: int, seed: int = 42) -> List[Any]:
    """Load `osunlp/Multimodal-Mind2Web` (`test_task`) as 5-way visual Set-of-Mark + DOM candidate grounding."""
    from datasets import load_dataset
    from .eval_suite import PreparedCase

    rng = np.random.default_rng(seed)
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="test_task", streaming=True).shuffle(
        seed=seed, buffer_size=400
    )
    letters = ["A", "B", "C", "D", "E"]
    interactive_tags = {"button", "a", "input", "select", "textarea", "option", "label", "span", "li", "div", "img"}
    cases: List[PreparedCase] = []

    for idx, row in enumerate(ds):
        pos_list = list(row.get("pos_candidates") or [])
        neg_list = list(row.get("neg_candidates") or [])
        if not pos_list or len(neg_list) < 4:
            continue

        pos_parsed = _parse_cand_json(pos_list[0])
        # Parse a pool of negative candidates, preferring interactive visible UI elements
        neg_pool_raw = neg_list[:60]
        parsed_negs = [_parse_cand_json(nc) for nc in neg_pool_raw]
        good_negs = [
            nc
            for nc in parsed_negs
            if nc["tag"].lower() in interactive_tags
            and nc.get("bbox") is not None
            and nc["bbox"][3] < 600
        ]
        if len(good_negs) < 4:
            good_negs = [nc for nc in parsed_negs if nc["tag"].lower() not in ("html", "body")]
        if len(good_negs) < 4:
            good_negs = parsed_negs
        chosen_neg_idx = rng.choice(len(good_negs), size=4, replace=False)
        selected_cands = [pos_parsed] + [good_negs[int(i)] for i in chosen_neg_idx]

        order = rng.permutation(5)
        gold_slot = int(np.where(order == 0)[0][0])
        gold_letter = letters[gold_slot]

        labeled_cands: List[Tuple[str, Dict[str, Any]]] = []
        crit: Dict[str, str] = {}
        for slot_i, orig_i in enumerate(order):
            lbl = letters[slot_i]
            c = selected_cands[int(orig_i)]
            labeled_cands.append((lbl, c))
            bbox_str = (
                f"bbox=({int(c['bbox'][0])},{int(c['bbox'][1])},{int(c['bbox'][2])}x{int(c['bbox'][3])})"
                if c.get("bbox")
                else "bbox=unlisted"
            )
            txt_part = f" text={c['text']!r}" if c["text"] else ""
            crit[lbl] = f"<{c['tag']}>{txt_part} attrs={json.dumps(c['attrs'], ensure_ascii=False)} {bbox_str}"

        som_img = _render_som_mind2web_crop(row.get("screenshot"), labeled_cands)

        try:
            op_info = json.loads(row.get("operation", "{}"))
        except Exception:
            op_info = {"raw": str(row.get("operation", ""))}

        act_reprs = list(row.get("action_reprs") or [])
        try:
            t_idx = int(row.get("target_action_index", 0))
        except Exception:
            t_idx = 0
        prev_steps = act_reprs[max(0, t_idx - 6) : t_idx]

        state: Dict[str, Any] = {
            "screenshot": som_img,
            "website": str(row.get("website", "")),
            "domain": f"{row.get('domain', '')}/{row.get('subdomain', '')}",
            "task_goal": str(row.get("confirmed_task", "")),
            "previous_actions": prev_steps,
            "next_operation_type": op_info,
        }
        q = {
            "element": {
                "type": "choice",
                "instructions": (
                    "Inspect the webpage screenshot (with colored Set-of-Mark boxes [A]-[E]), `task_goal`, "
                    "`previous_actions`, and `next_operation_type`. Select the exact target UI element [A]-[E] "
                    "to interact with next."
                ),
                "criteria": crit,
            }
        }
        cases.append(
            PreparedCase(
                case_id=f"m2w-{row.get('action_uid', idx)}",
                state=state,
                questions=q,
                gold={"element": (gold_letter, gold_slot, letters)},
            )
        )
        if len(cases) >= limit:
            break

    return cases


# ---------------------------------------------------------------------------
# 4. MMAU Audio Understanding Loader (`gamma-lab-umd/MMAU-test-mini`)
# ---------------------------------------------------------------------------


def _clean_choice_prefix(text: str, idx: int) -> str:
    s = str(text).strip()
    return re.sub(r"^\([A-Za-z]\)\s*", "", s)


def load_mmau_cases(limit: int, seed: int = 42) -> List[Any]:
    """Load official `gamma-lab-umd/MMAU-test-mini` with raw audio + optional local CUDA Whisper transcript hint."""
    import soundfile as sf
    from datasets import Audio, load_dataset
    from .eval_suite import PreparedCase

    ds = load_dataset("gamma-lab-umd/MMAU-test-mini", split="test").cast_column("context", Audio(decode=False))
    n_total = len(ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)[: min(limit, n_total)]
    letter_pool = ["A", "B", "C", "D", "E", "F"]
    cases: List[PreparedCase] = []

    for row_idx in perm:
        r = ds[int(row_idx)]
        raw_choices = list(r.get("choices") or [])
        if len(raw_choices) < 2:
            continue
        k = len(raw_choices)
        labels = letter_pool[:k]
        ans_raw = str(r.get("answer") or "").strip()

        # Match gold answer either by exact string, stripped string, or leading `(A)` letter
        gold_idx: Optional[int] = None
        for i, ch in enumerate(raw_choices):
            if str(ch).strip() == ans_raw or _clean_choice_prefix(str(ch), i).lower() == _clean_choice_prefix(ans_raw, i).lower():
                gold_idx = i
                break
        if gold_idx is None:
            m = re.match(r"^\(?([A-Fa-f])\)?", ans_raw)
            if m and m.group(1).upper() in labels:
                gold_idx = labels.index(m.group(1).upper())
        if gold_idx is None:
            continue

        gold_letter = labels[gold_idx]
        crit = {labels[i]: _clean_choice_prefix(str(raw_choices[i]), i) for i in range(k)}

        ctx = r.get("context") or {}
        raw_bytes = ctx.get("bytes")
        audio_payload: Any = ctx
        asr_hint = ""
        duration_sec = 0.0
        if raw_bytes:
            try:
                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=-1)
                duration_sec = round(len(arr) / float(max(1, sr)), 2)
                if sr != 16000:
                    new_len = max(1, int(round(len(arr) * 16000.0 / float(sr))))
                    arr16 = np.interp(
                        np.linspace(0.0, 1.0, new_len, endpoint=False),
                        np.linspace(0.0, 1.0, len(arr), endpoint=False),
                        arr,
                    ).astype(np.float32)
                else:
                    arr16 = arr
                buf = io.BytesIO()
                sf.write(buf, arr16, 16000, format="WAV", subtype="PCM_16")
                audio_payload = {"bytes": buf.getvalue()}
                asr_hint = transcribe_pcm16_fast(arr16, max_chars=800)
            except Exception:
                audio_payload = {"bytes": raw_bytes}

        try:
            attrs = json.loads(r.get("other_attributes") or "{}")
        except Exception:
            attrs = {}

        state: Dict[str, Any] = {
            "audio": audio_payload,
            "question": str(r.get("instruction") or ""),
            "audio_duration_sec": duration_sec,
            "audio_category": str(attrs.get("task", attrs.get("category", "general"))),
        }
        if asr_hint:
            state["local_whisper_speech_transcript"] = asr_hint

        q = {
            "answer": {
                "type": "choice",
                "instructions": (
                    "Listen carefully to the attached audio clip (speech, environmental sounds, music, paralinguistics, "
                    "timing, and acoustics) and select the single correct option answering `question`."
                ),
                "criteria": crit,
            }
        }
        cases.append(
            PreparedCase(
                case_id=f"mmau-{attrs.get('id', int(row_idx))}",
                state=state,
                questions=q,
                gold={"answer": (gold_letter, gold_idx, labels)},
            )
        )

    return cases


# ---------------------------------------------------------------------------
# 5. Video-MME Loader (w/o Subtitles: HTTP Range MP4 -> 18-Frame Storyboards + Raw Audio ASR)
# ---------------------------------------------------------------------------


class _HttpRangeReader(io.RawIOBase):
    """Seekable HTTP Range reader allowing `zipfile.ZipFile` to fetch individual MP4s without downloading 5GB zips."""

    def __init__(self, url: str):
        import requests

        self.session = requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=20)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: Any) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.size - 1, self.pos + len(b) - 1)
        resp = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=45)
        data = resp.content
        n = len(data)
        b[:n] = data
        self.pos += n
        return n


_VIDEOMME_INDEX_CACHE = Path(tempfile.gettempdir()) / "laya_videomme_zip_index_v1.json"


def _get_videomme_zip_index() -> Dict[str, Tuple[int, str]]:
    """Map `videoID -> (chunk_idx_1_to_20, zip_entry_name)` across `lmms-eval/Video-MME`."""
    if _VIDEOMME_INDEX_CACHE.is_file():
        try:
            raw = json.loads(_VIDEOMME_INDEX_CACHE.read_text())
            return {k: (int(v[0]), str(v[1])) for k, v in raw.items()}
        except Exception:
            pass

    def _index_one(chunk_i: int) -> List[Tuple[str, int, str]]:
        url = f"https://huggingface.co/datasets/lmms-eval/Video-MME/resolve/main/videos_chunked_{chunk_i:02d}.zip"
        zf = zipfile.ZipFile(io.BufferedReader(_HttpRangeReader(url), buffer_size=131072))
        out: List[Tuple[str, int, str]] = []
        for name in zf.namelist():
            if name.endswith(".mp4"):
                vid = Path(name).stem
                out.append((vid, chunk_i, name))
        return out

    mapping: Dict[str, Tuple[int, str]] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        for items in pool.map(_index_one, range(1, 21)):
            for vid, ci, name in items:
                mapping[vid] = (ci, name)

    try:
        _VIDEOMME_INDEX_CACHE.write_text(json.dumps(mapping))
    except Exception:
        pass
    return mapping


def _extract_videomme_storyboards_and_asr(mp4_bytes: bytes) -> Tuple[List[Any], str, float]:
    """Decode 18 timestamped frames into two 3x3 high-res storyboard grids + transcribe raw MP4 audio wave."""
    import cv2
    from PIL import Image

    sheets: List[Any] = []
    transcript = ""
    duration_sec = 0.0

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        tf.write(mp4_bytes)
        tf.flush()

        cap = cv2.VideoCapture(tf.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0.1:
            fps = 25.0
        duration_sec = round(total_frames / fps, 1) if total_frames > 0 else 0.0

        n_target = 18
        if total_frames > 0:
            frame_indices = np.linspace(
                int(total_frames * 0.02),
                max(0, int(total_frames * 0.98) - 1),
                n_target,
                dtype=int,
            )
        else:
            frame_indices = np.arange(n_target, dtype=int)

        tile_w, tile_h = 384, 240
        tiles: List[np.ndarray] = []
        tot_mm = int(duration_sec // 60)
        tot_ss = int(duration_sec % 60)

        for rank, fidx in enumerate(frame_indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                if tiles:
                    tiles.append(tiles[-1].copy())
                continue
            resized = cv2.resize(frame_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cur_sec = fidx / fps
            mm = int(cur_sec // 60)
            ss = int(cur_sec % 60)
            badge = f"F{rank:02d} {mm:02d}:{ss:02d}/{tot_mm:02d}:{tot_ss:02d}"
            cv2.rectangle(resized, (0, 0), (195, 24), (0, 0, 0), -1)
            cv2.putText(resized, badge, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 120), 1, cv2.LINE_AA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tiles.append(rgb)
        cap.release()

        while len(tiles) < 18 and tiles:
            tiles.append(tiles[-1].copy())

        if len(tiles) >= 18:
            rows_img = [np.concatenate(tiles[r * 3 : (r + 1) * 3], axis=1) for r in range(6)]
            grid = np.concatenate(rows_img, axis=0)
            sheets.append(Image.fromarray(grid))

        # Extract raw audio wave from MP4 itself (strictly NO external subtitle files!)
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    tf.name,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "s16le",
                    "-",
                ],
                capture_output=True,
                timeout=20,
                check=False,
            )
            if proc.returncode == 0 and len(proc.stdout) >= 3200:
                pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                # Cap ASR to max 12 minutes evenly or first+last segments for ultra-long videos
                max_samples = 16000 * 600
                if len(pcm) > max_samples:
                    half_s = max_samples // 2
                    pcm = np.concatenate([pcm[:half_s], pcm[-half_s:]])
                transcript = transcribe_pcm16_fast(pcm, max_chars=2400)
        except Exception:
            transcript = ""

    return sheets, transcript, duration_sec


def load_video_mme_cases(limit: int, seed: int = 42) -> List[Any]:
    """Load `lmms-eval/Video-MME` (strictly w/o external subtitles), grouping 3 questions per video."""
    from datasets import load_dataset
    from .eval_suite import PreparedCase

    ds = load_dataset("lmms-eval/Video-MME", split="test")
    zip_index = _get_videomme_zip_index()

    # Group rows by videoID so all 3 questions of a video share 1 multimodal extraction & 1 API call!
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for r in ds:
        vid = str(r.get("videoID") or "")
        if vid in zip_index:
            by_video.setdefault(vid, []).append(dict(r))

    # Stratified permutation across short / medium / long durations
    rng = np.random.default_rng(seed)
    by_dur: Dict[str, List[str]] = {"short": [], "medium": [], "long": []}
    for vid, qrows in by_video.items():
        dur = str(qrows[0].get("duration", "short")).lower()
        by_dur.setdefault(dur, []).append(vid)
    for dkey in by_dur:
        rng.shuffle(by_dur[dkey])

    interleaved_vids: List[str] = []
    max_len = max((len(v) for v in by_dur.values()), default=0)
    for i in range(max_len):
        for dkey in ("short", "medium", "long"):
            if i < len(by_dur.get(dkey, [])):
                interleaved_vids.append(by_dur[dkey][i])

    # Determine how many videos we need to reach `limit` questions
    selected_vids: List[str] = []
    planned_q = 0
    for vid in interleaved_vids:
        selected_vids.append(vid)
        planned_q += len(by_video[vid])
        if planned_q >= limit:
            break

    def _prepare_one_video(vid: str) -> Optional[PreparedCase]:
        chunk_i, entry_name = zip_index[vid]
        url = f"https://huggingface.co/datasets/lmms-eval/Video-MME/resolve/main/videos_chunked_{chunk_i:02d}.zip"
        try:
            zf = zipfile.ZipFile(io.BufferedReader(_HttpRangeReader(url), buffer_size=262144))
            mp4_bytes = zf.read(entry_name)
        except Exception:
            return None

        sheets, transcript, dur_sec = _extract_videomme_storyboards_and_asr(mp4_bytes)
        if not sheets:
            return None

        qrows = by_video[vid]
        labels = ["A", "B", "C", "D"]
        questions_map: Dict[str, Dict[str, Any]] = {}
        gold_map: Dict[str, Tuple[Any, int, List[Any]]] = {}

        for qrow in qrows:
            qid_raw = str(qrow.get("question_id", len(questions_map)))
            qkey = f"q_{qid_raw.replace('-', '_')}"
            ans = str(qrow.get("answer", "")).strip().upper()
            if ans not in labels:
                continue
            opts_list = list(qrow.get("options") or [])
            if len(opts_list) < 4:
                continue
            crit = {}
            for li, opt_str in zip(labels, opts_list[:4]):
                cleaned = re.sub(r"^[A-Da-d][\.\)]\s*", "", str(opt_str).strip())
                crit[li] = cleaned
            questions_map[qkey] = {
                "type": "choice",
                "instructions": f"[{qrow.get('task_type', 'Video QA')}] {qrow.get('question', '')}",
                "criteria": crit,
            }
            gold_map[qkey] = (ans, labels.index(ans), labels)

        if not questions_map:
            return None

        meta0 = qrows[0]
        state: Dict[str, Any] = {
            "_frames": sheets,
            "video_duration_category": str(meta0.get("duration", "")),
            "video_length_seconds": dur_sec,
            "domain": f"{meta0.get('domain', '')} / {meta0.get('sub_category', '')}",
            "frame_layout": "The attached 6x3 storyboard image shows 18 chronological frames F01..F18 (left-to-right, top-to-bottom), each stamped with exact MM:SS timestamp.",
        }
        if transcript:
            state["raw_audio_speech_transcript"] = transcript
        else:
            state["raw_audio_speech_transcript"] = "(no intelligible speech detected; rely on visual frames)"

        return PreparedCase(
            case_id=f"vmme-{vid}",
            state=state,
            questions=questions_map,
            gold=gold_map,
        )

    cases: List[PreparedCase] = []
    total_q = 0
    # Fetch & decode MP4s with bounded concurrency
    with ThreadPoolExecutor(max_workers=4) as pool:
        for case in pool.map(_prepare_one_video, selected_vids):
            if case is None:
                continue
            # Trim final video's questions if exceeding `limit`
            if total_q + len(case.questions) > limit:
                keep_keys = list(case.questions.keys())[: max(1, limit - total_q)]
                case = PreparedCase(
                    case_id=case.case_id,
                    state=case.state,
                    questions={k: case.questions[k] for k in keep_keys},
                    gold={k: case.gold[k] for k in keep_keys},
                )
            cases.append(case)
            total_q += len(case.questions)
            if total_q >= limit:
                break

    return cases


# ---------------------------------------------------------------------------
# 6. Strictly Disjoint Train-Split TTTHead & ClassMultiplier Fitter
# ---------------------------------------------------------------------------


def fit_disjoint_ttt_for_omega(
    omega_engine: Any,
    benchmarks: Sequence[str],
    eval_limit: int = 250,
    seed: int = 42,
    esci_train_samples: int = 320,
    esci_val_samples: int = 80,
    when2call_train_samples: int = 260,
    pubmedqa_train_samples: int = 260,
) -> Dict[str, Any]:
    """Fit Vega-U `TTTHead`s and Macro-F1 `ClassMultiplierOptimizer` on STRICTLY DISJOINT training splits.

    Leakage guarantees:
      - `amazon_esci`: Uses `tasksource/esci` `split="train"` (`test` uses `split="test"`).
      - `when2call`: Splits `nvidia/When2Call` by `source_id` prompt-family group; train uses ONLY groups
        excluded from the `eval_limit` test cases (`train_groups ∩ test_groups = ∅`).
      - `pubmedqa`: Uses `perm[eval_limit:]` of `pqa_labeled` (`train_indices ∩ eval_indices = ∅`).
    """
    from datasets import load_dataset
    from .eval_suite import ESCI_CRITERIA, ESCI_LABEL_ORDER, PUBMEDQA_CRITERIA, _compact_tool_spec, _format_pubmed_abstract

    vega_backend = getattr(omega_engine, "local_vega", None)
    if vega_backend is None or not hasattr(vega_backend, "fit_ttt"):
        return {"status": "skipped_no_vega"}

    rng = np.random.default_rng(seed + 10007)
    report: Dict[str, Any] = {}

    if "amazon_esci" in benchmarks and "esci" not in vega_backend.fitted_heads:
        ds_tr = load_dataset("tasksource/esci", split="train", streaming=True).shuffle(
            seed=seed + 99, buffer_size=6000
        )
        by_cls: Dict[str, List[Dict[str, Any]]] = {lbl: [] for lbl in ESCI_LABEL_ORDER}
        t2_val_by_cls: Dict[str, List[Dict[str, Any]]] = {lbl: [] for lbl in ESCI_LABEL_ORDER}
        t2_val_targets = {"Irrelevant": 10, "Complement": 8, "Substitute": 14, "Exact": 18}
        natural_val: List[Dict[str, Any]] = []
        # Mildly minority-enriched train distribution (`Exact: 38%, Substitute: 31%, Irrelevant: 18%, Complement: 13%`)
        # so rare `Complement` (~3% natural) gets strong representation without extreme prior distortion.
        scale = max(0.25, float(esci_train_samples) / 900.0)
        cls_targets = {
            "Irrelevant": max(30, int(round(160 * scale))),
            "Complement": max(25, int(round(120 * scale))),
            "Substitute": max(50, int(round(280 * scale))),
            "Exact": max(60, int(round(340 * scale))),
        }
        for r in ds_tr:
            if r.get("product_locale", "us") != "us":
                continue
            lbl = str(r.get("esci_label", ""))
            if lbl not in ESCI_LABEL_ORDER:
                continue
            if len(t2_val_by_cls[lbl]) < t2_val_targets[lbl]:
                t2_val_by_cls[lbl].append(r)
                continue
            if len(natural_val) < esci_val_samples:
                natural_val.append(r)
                continue
            if len(by_cls[lbl]) < cls_targets[lbl]:
                by_cls[lbl].append(r)
            if (
                all(len(by_cls[k]) >= cls_targets[k] for k in ESCI_LABEL_ORDER)
                and all(len(t2_val_by_cls[k]) >= t2_val_targets[k] for k in ESCI_LABEL_ORDER)
                and len(natural_val) >= esci_val_samples
            ):
                break

        q_esci = {
            "esci": {
                "type": "choice",
                "instructions": "Classify Amazon ESCI shopping relevance between `query` and `product_title`.",
                "criteria": {lbl: ESCI_CRITERIA[lbl] for lbl in ESCI_LABEL_ORDER},
            }
        }
        train_rows = []
        for lbl in ESCI_LABEL_ORDER:
            for r in by_cls[lbl]:
                train_rows.append({"state": build_enriched_esci_state(r), "expected": {"esci": lbl}})
        rng.shuffle(train_rows)

        if len(train_rows) >= 40:
            fit_res = vega_backend.fit_ttt(train_rows, q_esci)
            report["amazon_esci_ttt"] = {"train_rows": len(train_rows), "fit": fit_res}

            if len(natural_val) >= 40:
                val_states = [build_enriched_esci_state(r) for r in natural_val]
                val_gold = [ESCI_LABEL_ORDER.index(str(r["esci_label"])) for r in natural_val]
                laya_b = getattr(omega_engine, "local_laya", None)
                laya_val = laya_b.predict_batch(val_states, q_esci) if laya_b is not None else [None] * len(val_states)
                vega_val = vega_backend.predict_batch(val_states, q_esci)
                fused_local_val = [omega_engine._fuse_local(lo, vo, q_esci) for lo, vo in zip(laya_val, vega_val)]
                local_probs = [
                    np.array(
                        [float(o["answers"]["esci"]["probabilities"].get(lbl, 0.0)) for lbl in ESCI_LABEL_ORDER],
                        dtype=np.float64,
                    )
                    for o in fused_local_val
                ]
                local_opt = ClassMultiplierOptimizer(num_classes=len(ESCI_LABEL_ORDER), l2_shrinkage=0.012).fit(
                    local_probs, val_gold
                )
                omega_engine.register_class_optimizer("esci", local_opt, scope="local")
                report["amazon_esci_local_multipliers"] = [round(float(x), 4) for x in local_opt.multipliers]

    if "pubmedqa" in benchmarks and "pubmed_decision" not in vega_backend.fitted_heads:
        ds_pm = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        n_total = len(ds_pm)
        # Reconstruct exact eval permutation `np.random.default_rng(seed).permutation(n_total)[:eval_limit]`
        eval_rng = np.random.default_rng(seed)
        perm = eval_rng.permutation(n_total)
        eval_set = set(int(x) for x in perm[: min(eval_limit, n_total)])
        disjoint_pool = [int(x) for x in perm[min(eval_limit, n_total) :] if int(x) not in eval_set]

        if len(disjoint_pool) >= 60:
            q_pm = {
                "pubmed_decision": {
                    "type": "choice",
                    "instructions": "Based strictly on the biomedical study in `abstract`, answer `question`.",
                    "criteria": PUBMEDQA_CRITERIA,
                }
            }
            pm_rows = []
            for idx in disjoint_pool[:pubmedqa_train_samples]:
                r = ds_pm[idx]
                lbl = str(r.get("final_decision", "")).lower().strip()
                if lbl not in PUBMEDQA_CRITERIA:
                    continue
                pm_rows.append(
                    {
                        "state": {
                            "question": str(r.get("question", "")),
                            "abstract": _format_pubmed_abstract(r.get("context", {}), max_chars=3400),
                        },
                        "expected": {"pubmed_decision": lbl},
                    }
                )
            if len(pm_rows) >= 40:
                fit_res = vega_backend.fit_ttt(pm_rows, q_pm)
                report["pubmedqa_ttt"] = {"disjoint_train_rows": len(pm_rows), "fit": fit_res}

    if "when2call" in benchmarks and "w2c_action" not in vega_backend.fitted_heads:
        ds_w2c = load_dataset("nvidia/When2Call", split="mcq")
        n_total = len(ds_w2c)
        eval_rng = np.random.default_rng(seed)
        perm = eval_rng.permutation(n_total)
        # Collect eval indices AND their `source_id` prompt-family groups to guarantee group-disjointness!
        eval_indices: List[int] = []
        eval_source_groups = set()
        for idx in perm:
            r = ds_w2c[int(idx)]
            if str(r.get("correct_answer", "")) not in ("direct", "tool_call", "request_for_info", "cannot_answer"):
                continue
            eval_indices.append(int(idx))
            sid = str(r.get("source_id") or "").split("-")[0]
            if sid:
                eval_source_groups.add((str(r.get("source") or ""), sid))
            if len(eval_indices) >= eval_limit:
                break
        eval_idx_set = set(eval_indices)

        from .eval_suite import WHEN2CALL_CRITERIA

        q_w2c = {
            "w2c_action": {
                "type": "choice",
                "instructions": (
                    "Given `user_request`, `available_tools`, and `schema_audit`, decide the right agent routing action."
                ),
                "criteria": WHEN2CALL_CRITERIA,
            }
        }
        w2c_rows = []
        for idx in perm:
            ii = int(idx)
            if ii in eval_idx_set:
                continue
            r = ds_w2c[ii]
            sid = str(r.get("source_id") or "").split("-")[0]
            if sid and (str(r.get("source") or ""), sid) in eval_source_groups:
                continue
            lbl = str(r.get("correct_answer", ""))
            if lbl not in WHEN2CALL_CRITERIA:
                continue
            st, _ = analyze_when2call_row(r, _compact_tool_spec)
            w2c_rows.append({"state": st, "expected": {"w2c_action": lbl}})
            if len(w2c_rows) >= when2call_train_samples:
                break

        if len(w2c_rows) >= 60:
            fit_res = vega_backend.fit_ttt(w2c_rows, q_w2c)
            report["when2call_ttt"] = {"group_disjoint_train_rows": len(w2c_rows), "fit": fit_res}

    return report
