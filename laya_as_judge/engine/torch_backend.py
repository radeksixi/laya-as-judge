"""PyTorch / Transformers / Upstream Laya backend for Linux and NVIDIA/AMD/CPU hardware."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import (
    QTYPES,
    QTYPE_NAMES,
    BaseDecisionEngine,
    confidence_from_probs,
    render_options,
    serialize_state,
    validate_question_spec,
)


def _temp_bucket(qtype: int, k: int) -> str:
    """Cardinality temperature bucket matching upstream Laya checkpoints."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPE_NAMES[int(qtype)]}:{size}"


class TorchBackend(BaseDecisionEngine):
    """PyTorch / Upstream Laya backend for Linux and NVIDIA/AMD/CPU hardware.

    Supports:
    1. Official upstream 'laya' runtime (`pip install laya`) including `Agent`,
       subfolders (`"typed-decisions"`, `"multilingual"`), `Router`, `VegaAgent`,
       `max_len` overrides (up to 8192 on multilingual), and native `predict_batch`.
    2. Direct PyTorch + Transformers + Safetensors execution of Laya's `DecisionModel`
       architecture when `laya` is not installed.
    """

    def __init__(
        self,
        model_id: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        torch_dtype: str = "float16",
        preload: bool = False,
        subfolder: Optional[str] = None,
        use_router: bool = False,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        batch_size: int = 32,
        vegaml: Optional[Union[bool, str]] = None,
    ):
        # Allow shorthand aliases such as "router", "english", "typed-decisions", "multilingual"
        if model_id == "router":
            use_router = True
            model_id = "convaiinnovations/laya"
        elif model_id in ("english", "default", "laya") and subfolder is None:
            model_id = "convaiinnovations/laya"
        elif ":" in model_id and subfolder is None and not Path(model_id).exists():
            model_id, subfolder = model_id.split(":", 1)
        elif model_id in ("typed-decisions", "multilingual") and subfolder is None:
            subfolder = model_id
            model_id = "convaiinnovations/laya"

        display_id = f"{model_id}/router" if use_router else (f"{model_id}/{subfolder}" if subfolder else model_id)
        super().__init__(model_id=display_id)
        self._repo_id = model_id
        self.subfolder = subfolder
        self.default_max_len = max_len
        self.default_head_max_len = head_max_len
        self.batch_size = batch_size
        self._is_router = False
        self._is_upstream_laya = False
        self._has_neural_model = False

        # 1. Prefer official upstream 'laya' package if installed
        try:
            import laya

            self._laya = laya
            if use_router and hasattr(laya, "Router"):
                router_kwargs: Dict[str, Any] = {"device": device, "preload": preload}
                if vegaml is None:
                    router_kwargs["vega"] = False
                else:
                    router_kwargs["vega"] = vegaml
                try:
                    self._agent = laya.Router(**router_kwargs)
                except TypeError:
                    self._agent = laya.Router(device=device, preload=preload)
                self._is_router = True
            elif hasattr(laya, "load"):
                load_kwargs: Dict[str, Any] = {"device": device}
                if subfolder is not None:
                    load_kwargs["subfolder"] = subfolder
                if vegaml is not None:
                    load_kwargs["vegaml"] = vegaml
                self._agent = laya.load(model_id, **load_kwargs)
            elif hasattr(laya, "Agent"):
                self._agent = laya.Agent(model_id, device=device, subfolder=subfolder)
            else:
                self._agent = laya.Router(device=device, preload=preload)
                self._is_router = True
            self._is_upstream_laya = True
            return
        except ImportError:
            self._is_upstream_laya = False

        # 2. Direct PyTorch + Transformers + Safetensors execution of Laya DecisionModel
        try:
            import torch
            import torch.nn as nn
            from huggingface_hub import snapshot_download
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "TorchBackend requires the official 'laya' runtime (pip install laya) "
                "or 'torch' and 'transformers'. Install with: pip install 'laya-as-judge[torch]'"
            ) from e

        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        local_path = Path(model_id).expanduser()
        if not local_path.exists():
            allow = [f"{subfolder}/*"] if subfolder else None
            local_path = Path(snapshot_download(model_id, allow_patterns=allow))
        if subfolder:
            local_path = local_path / subfolder
        self.model_dir = local_path

        cfg_path = self.model_dir / "rl_agent_config.json"
        if cfg_path.is_file():
            self.cfg = json.loads(cfg_path.read_text())
        else:
            self.cfg = {"temperature": [1.0, 1.0, 1.0], "max_len": 512, "head_max_len": 192, "head_layers": 2}

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir / "tokenizer"))

        weights_path = self.model_dir / "model.safetensors"
        encoder_dir = self.model_dir / "encoder"
        try:
            from safetensors.torch import load_file

            if weights_path.is_file() and encoder_dir.is_dir():
                ecfg = AutoConfig.from_pretrained(str(encoder_dir))
                enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
                head_layers = int(self.cfg.get("head_layers", 2))
                n_act = len(self.cfg.get("act_costs", {"escalate": 0.5})) + 1

                class _DecisionModel(nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.encoder = enc
                        d = enc.config.hidden_size
                        nhead = max(1, d // 64)
                        layer = nn.TransformerEncoderLayer(
                            d, nhead, 4 * d, 0.1, batch_first=True, norm_first=True
                        )
                        self.head = (
                            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
                            if head_layers > 0
                            else None
                        )
                        self.type_emb = nn.Embedding(3, d)
                        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
                        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
                        self.register_buffer("temperature", torch.ones(3))

                    def forward(self, input_ids: Any, attention_mask: Any, marker_pos: Any, marker_mask: Any, qtype: Any) -> Any:
                        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                        h = h + self.type_emb(qtype)[:, None, :]
                        if self.head is not None:
                            pad = ~attention_mask.bool()
                            for lyr in self.head.layers:
                                h = lyr(h, src_key_padding_mask=pad)
                        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
                        m = torch.gather(h, 1, idx)
                        logits = self.scorer(m).squeeze(-1).float()
                        logits = logits.masked_fill(~marker_mask, -1e4)
                        p = torch.softmax(logits.detach(), -1)
                        k = marker_mask.sum(-1).clamp(min=2).float()
                        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
                        top2 = p.topk(2, -1).values
                        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
                        pooled = h[:, 0].float()
                        act_logits = self.act_head(torch.cat([pooled, feats], -1))
                        return logits, act_logits

                self._model = _DecisionModel()
                self._model.load_state_dict(load_file(str(weights_path)), strict=True)
                self._model.to(self.device).eval()
                if hasattr(self._model.encoder, "config"):
                    self._model.encoder.config.reference_compile = False
                self._has_neural_model = True
        except Exception as exc:
            warnings.warn(
                f"TorchBackend could not load neural checkpoint ({exc}); falling back to uniform scaffold. "
                "Install 'laya' (pip install laya) or 'safetensors' for full neural inference.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _build_sequence(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        q: Dict[str, Any],
        max_len: int,
        head_max_len: int,
    ) -> tuple[List[int], List[int]]:
        tok = self.tokenizer
        mask_tok = tok.mask_token or "[MASK]"
        opts = render_options(q)
        ins = str(q["instructions"]).replace(mask_tok, " ")
        head_ids = tok(f"{q['type']} question: {ins}", add_special_tokens=False)["input_ids"]
        opt_ids = [
            [tok.mask_token_id] + tok(" " + o.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
            for o in opts
        ]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[: max(8, opt_budget)]
        ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
        markers: List[int] = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(tok.sep_token_id)
        room = max(0, max_len - len(ids) - 1)
        st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:room]
        ids = ids + st + [tok.sep_token_id]
        return ids[:max_len], [m for m in markers if m < max_len]

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute decision questions forward pass."""
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        eff_max_len = max_len if max_len is not None else self.default_max_len
        eff_head_len = head_max_len if head_max_len is not None else self.default_head_max_len
        t0 = time.perf_counter()

        if self._is_upstream_laya:
            call_kwargs: Dict[str, Any] = {}
            if eff_max_len is not None:
                call_kwargs["max_len"] = eff_max_len
            if eff_head_len is not None:
                call_kwargs["head_max_len"] = eff_head_len
            call_kwargs.update(kwargs)
            raw_result = self._agent.predict(state, normalized_q, **call_kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            raw_result["latency_ms"] = round(latency_ms, 2)
            return raw_result

        ml = eff_max_len or int(self.cfg.get("max_len", 512))
        hml = eff_head_len or int(self.cfg.get("head_max_len", 192))

        if self._has_neural_model:
            torch = self.torch
            qids = list(normalized_q.keys())
            seqs, markers_list, qtypes = [], [], []
            for qid in qids:
                q = normalized_q[qid]
                seq, markers = self._build_sequence(state, q, ml, hml)
                seqs.append(seq)
                markers_list.append(markers)
                qtypes.append(QTYPES[q["type"]])

            n = len(seqs)
            L = max(len(s) for s in seqs)
            kmax = max(len(m) for m in markers_list)
            pad_id = self.tokenizer.pad_token_id or 0
            ids_t = torch.full((n, L), pad_id, dtype=torch.long)
            att_t = torch.zeros((n, L), dtype=torch.long)
            mpos_t = torch.zeros((n, kmax), dtype=torch.long)
            mmask_t = torch.zeros((n, kmax), dtype=torch.bool)
            for i, (s, m) in enumerate(zip(seqs, markers_list)):
                ids_t[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                att_t[i, : len(s)] = 1
                mpos_t[i, : len(m)] = torch.tensor(m, dtype=torch.long)
                mmask_t[i, : len(m)] = True

            with torch.no_grad():
                logits_t, act_t = self._model(
                    ids_t.to(self.device),
                    att_t.to(self.device),
                    mpos_t.to(self.device),
                    mmask_t.to(self.device),
                    torch.tensor(qtypes, dtype=torch.long, device=self.device),
                )
                logits_np = logits_t.float().cpu().numpy()
                act_np = torch.softmax(act_t.float(), -1).cpu().numpy()

            temps = self.cfg.get("temperature", [1.0, 1.0, 1.0])
            temp_by_opts = self.cfg.get("temperature_by_options", {})
            answers: Dict[str, Any] = {}
            for r, qid in enumerate(qids):
                q = normalized_q[qid]
                k = len(markers_list[r])
                qt = qtypes[r]
                t_val = float(temp_by_opts.get(_temp_bucket(qt, k), temps[qt]))
                z = logits_np[r, :k] / max(0.05, t_val)
                probs = np.exp(z - np.max(z))
                probs /= np.sum(probs)
                conf = confidence_from_probs(probs, k)
                ext = {"act_probability": round(float(act_np[r, 0]), 4)}
                if q["type"] == "choice":
                    labels = list(q["criteria"].keys())
                    answers[qid] = {
                        "type": "choice",
                        "choice": labels[int(np.argmax(probs))],
                        "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)},
                        "confidence": round(conf, 4),
                        "rl_agent": ext,
                    }
                elif q["type"] == "score":
                    expected_score = float(np.sum(np.arange(k) * probs))
                    answers[qid] = {
                        "type": "score",
                        "score": round(expected_score, 4),
                        "legend": {str(i): str(c) for i, c in enumerate(q["criteria"])},
                        "probabilities": {str(i): round(float(p), 4) for i, p in enumerate(probs)},
                        "confidence": round(conf, 4),
                        "rl_agent": ext,
                    }
                else:
                    p_true = float(probs[1]) if k >= 2 else 0.5
                    answers[qid] = {
                        "type": "noul",
                        "noul": round(p_true, 4),
                        "confidence": round(conf, 4),
                        "rl_agent": ext,
                    }

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "model": f"laya-torch ({self.model_id})",
                "answers": answers,
                "usage": {"input_tokens": int(att_t.sum()), "output_tokens": 0},
                "latency_ms": round(latency_ms, 2),
            }

        # Fallback uniform scaffold when weights are missing
        state_str = serialize_state(state)
        answers = {}
        total_input_tokens = 0
        for qid, q in normalized_q.items():
            opts = render_options(q)
            prompt = f"{q['type']} question: {q['instructions']} Options: {'; '.join(opts)} State: {state_str}"
            tokens = self.tokenizer(prompt, truncation=True, max_length=ml)
            total_input_tokens += len(tokens["input_ids"])
            k = len(opts)
            probs = np.ones(k, dtype=np.float32) / max(1, k)
            conf = confidence_from_probs(probs, k)
            ans: Dict[str, Any] = {
                "type": q["type"],
                "confidence": round(conf, 4),
                "action": {"act_probability": 0.05},
            }
            if q["type"] == "choice":
                labels = list(q["criteria"].keys())
                ans["choice"] = labels[int(np.argmax(probs))]
                ans["probabilities"] = {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)}
            elif q["type"] == "score":
                expected_score = float(np.sum(np.arange(k) * probs))
                ans["score"] = round(expected_score, 4)
                ans["legend"] = {str(i): str(c) for i, c in enumerate(q["criteria"])}
                ans["probabilities"] = {str(i): round(float(p), 4) for i, p in enumerate(probs)}
            else:
                ans["noul"] = 0.5
            answers[qid] = ans

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "model": f"laya-torch-scaffold ({self.model_id})",
            "answers": answers,
            "usage": {"input_tokens": total_input_tokens, "output_tokens": 0},
            "latency_ms": round(latency_ms, 2),
        }

    def predict_batch(
        self,
        states: List[Union[str, Dict[str, Any], List[Any]]],
        questions: Dict[str, Dict[str, Any]],
        batch_size: Optional[int] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        sort_by_length: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Execute batched forward passes across multiple states."""
        if not states:
            return []
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        bs = batch_size or self.batch_size
        eff_max_len = max_len if max_len is not None else self.default_max_len
        eff_head_len = head_max_len if head_max_len is not None else self.default_head_max_len

        if self._is_upstream_laya and hasattr(self._agent, "predict_batch"):
            results: List[Dict[str, Any]] = []
            for start in range(0, len(states), bs):
                chunk = states[start : start + bs]
                t0 = time.perf_counter()
                if self._is_router:
                    reqs: List[Dict[str, Any]] = []
                    for s in chunk:
                        req: Dict[str, Any] = {"state": s, "questions": normalized_q}
                        if eff_max_len is not None:
                            req["max_len"] = eff_max_len
                        if eff_head_len is not None:
                            req["head_max_len"] = eff_head_len
                        reqs.append(req)
                    chunk_res = self._agent.predict_batch(
                        reqs, batch_size=bs, sort_by_length=sort_by_length
                    )
                else:
                    call_kwargs: Dict[str, Any] = {
                        "batch_size": bs,
                        "sort_by_length": sort_by_length,
                    }
                    if eff_max_len is not None:
                        call_kwargs["max_len"] = eff_max_len
                    if eff_head_len is not None:
                        call_kwargs["head_max_len"] = eff_head_len
                    call_kwargs.update(kwargs)
                    chunk_res = self._agent.predict_batch(chunk, normalized_q, **call_kwargs)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                per_item_ms = round(elapsed_ms / max(1, len(chunk)), 2)
                for r in chunk_res:
                    r.setdefault("latency_ms", per_item_ms)
                results.extend(chunk_res)
            return results

        return [
            self.predict(s, normalized_q, max_len=eff_max_len, head_max_len=eff_head_len, **kwargs)
            for s in states
        ]
