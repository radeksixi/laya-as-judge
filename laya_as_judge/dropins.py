"""Open-Source Ecosystem Drop-In Adapters for Laya-as-a-Judge (Apache-2.0).

Provides zero-output-token, local drop-in adapters built on `BaseDecisionEngine`:
  1. `CandidateRankerEngine`: Best-of-N candidate tournament (`/v1/rank`) and
     Cohere / Jina / vLLM compatible RAG reranker (`/v1/rerank`).
  2. `ModerationAdapter`: Air-gapped OpenAI `/v1/moderations` wire-format drop-in.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .engine.base import BaseDecisionEngine, serialize_state
from .engine.http_systemone_backend import _extract_simplex_from_answer
from .judged_decision import spoofed_audit_guard_rule


class CandidateRankerEngine:
    """Zero-output-token Best-of-N tournament verifier (`/v1/rank`) and RAG reranker (`/v1/rerank`)."""

    def __init__(self, engine: BaseDecisionEngine):
        self.engine = engine

    def rank(
        self,
        *,
        prompt: str,
        candidates: Sequence[str],
        rubric: Optional[str] = None,
        top_n: Optional[int] = None,
        return_candidates: bool = True,
    ) -> Dict[str, Any]:
        """Rank `N` candidate completions for Best-of-N test-time verification."""
        t0 = time.perf_counter()
        cands = list(candidates)
        if not cands:
            return {
                "winner_index": None,
                "results": [],
                "model": self.engine.model_id,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "latency_ms": 0.0,
            }

        rubric_text = rubric or "overall accuracy, helpfulness, factuality, and instruction adherence"
        total_in_toks = 0

        if len(cands) <= 8:
            cand_map = {f"cand_{i}": c for i, c in enumerate(cands)}
            questions: Dict[str, Dict[str, Any]] = {
                "best_candidate": {
                    "type": "choice",
                    "instructions": f"Which candidate best answers the prompt according to: {rubric_text}?",
                    "criteria": {f"cand_{i}": f"Candidate {i} is the best response" for i in range(len(cands))},
                }
            }
            for i in range(len(cands)):
                questions[f"score_{i}"] = {
                    "type": "score",
                    "instructions": f"Rate cand_{i} on: {rubric_text}",
                    "criteria": ["poor", "fair", "good", "excellent"],
                }

            raw = self.engine.predict({"prompt": prompt, "candidates": cand_map}, questions)
            total_in_toks += int((raw.get("usage") or {}).get("input_tokens", 0))
            answers = raw.get("answers") or {}
            _, win_probs = _extract_simplex_from_answer(answers.get("best_candidate", {}), questions["best_candidate"])

            scored_items: List[Dict[str, Any]] = []
            for i, text in enumerate(cands):
                _, s_probs = _extract_simplex_from_answer(answers.get(f"score_{i}", {}), questions[f"score_{i}"])
                exp_s = float(np.sum(np.arange(len(s_probs), dtype=np.float64) * s_probs)) / max(1.0, len(s_probs) - 1.0)
                win_p = float(win_probs[i])
                blended = 0.65 * win_p + 0.35 * exp_s
                entry: Dict[str, Any] = {
                    "index": i,
                    "score": round(blended, 4),
                    "win_probability": round(win_p, 4),
                    "rubric_score": round(exp_s, 4),
                }
                if return_candidates:
                    entry["candidate"] = text
                scored_items.append(entry)
        else:
            q_pt = {
                "quality": {
                    "type": "score",
                    "instructions": f"Rate how well `candidate` answers `prompt` on: {rubric_text}",
                    "criteria": ["poor", "fair", "good", "excellent"],
                }
            }
            states = [{"prompt": prompt, "candidate": c} for c in cands]
            raws = self.engine.predict_batch(states, q_pt)
            raw_scores = []
            for r in raws:
                total_in_toks += int((r.get("usage") or {}).get("input_tokens", 0))
                _, sp = _extract_simplex_from_answer((r.get("answers") or {}).get("quality", {}), q_pt["quality"])
                raw_scores.append(float(np.sum(np.arange(len(sp), dtype=np.float64) * sp)) / max(1.0, len(sp) - 1.0))
            arr = np.asarray(raw_scores, dtype=np.float64)
            z = (arr - float(np.max(arr))) * 4.0
            win_probs = np.exp(z) / float(np.sum(np.exp(z)))
            scored_items = []
            for i, (text, sc, wp) in enumerate(zip(cands, raw_scores, win_probs)):
                entry = {
                    "index": i,
                    "score": round(float(sc), 4),
                    "win_probability": round(float(wp), 4),
                    "rubric_score": round(float(sc), 4),
                }
                if return_candidates:
                    entry["candidate"] = text
                scored_items.append(entry)

        scored_items.sort(key=lambda x: (-x["score"], -x["win_probability"], x["index"]))
        limit = int(top_n) if top_n is not None and top_n > 0 else len(scored_items)
        top_items = scored_items[:limit]

        return {
            "winner_index": top_items[0]["index"] if top_items else None,
            "results": top_items,
            "model": self.engine.model_id,
            "usage": {"input_tokens": total_in_toks, "output_tokens": 0},
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Union[str, Mapping[str, Any]]],
        top_n: Optional[int] = None,
        return_documents: bool = False,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cohere / Jina / vLLM drop-in `POST /v1/rerank` over zero-token relevance scoring."""
        t0 = time.perf_counter()
        doc_texts: List[str] = []
        for d in documents:
            if isinstance(d, Mapping):
                doc_texts.append(str(d.get("text") or serialize_state(dict(d))))
            else:
                doc_texts.append(str(d))

        q_rel = {
            "relevant": {
                "type": "noul",
                "instructions": "Does `document` contain information directly relevant to answering `query`?",
            },
            "relevance_level": {
                "type": "score",
                "instructions": "Rate the relevance of `document` to `query`:",
                "criteria": ["irrelevant", "tangential", "relevant", "exact_answer"],
            },
        }
        states = [{"query": query, "document": txt} for txt in doc_texts]
        raws = self.engine.predict_batch(states, q_rel) if states else []

        results: List[Dict[str, Any]] = []
        total_in = 0
        for idx, (orig_doc, raw) in enumerate(zip(documents, raws)):
            total_in += int((raw.get("usage") or {}).get("input_tokens", 0))
            ans = raw.get("answers") or {}
            p_rel = float(_extract_simplex_from_answer(ans.get("relevant", {}), q_rel["relevant"])[1][1])
            _, sp = _extract_simplex_from_answer(ans.get("relevance_level", {}), q_rel["relevance_level"])
            s_norm = float(np.sum(np.arange(len(sp), dtype=np.float64) * sp)) / max(1.0, len(sp) - 1.0)
            rel_score = round(float(np.clip(0.5 * p_rel + 0.5 * s_norm, 0.0, 1.0)), 6)
            row: Dict[str, Any] = {"index": idx, "relevance_score": rel_score}
            if return_documents:
                row["document"] = orig_doc if isinstance(orig_doc, Mapping) else {"text": str(orig_doc)}
            results.append(row)

        results.sort(key=lambda r: (-r["relevance_score"], r["index"]))
        if top_n is not None and top_n > 0:
            results = results[: int(top_n)]

        return {
            "id": f"rrnk_{uuid.uuid4().hex[:16]}",
            "model": model or self.engine.model_id,
            "results": results,
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"search_units": 1, "input_tokens": total_in, "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        }


DEFAULT_MODERATION_CATEGORIES: Dict[str, str] = {
    "harassment": "Does the text contain harassment, abuse, or targeted bullying?",
    "hate": "Does the text express hate speech or discrimination against a protected group?",
    "self-harm": "Does the text encourage, instruct, or depict self-harm or suicide?",
    "sexual": "Does the text contain sexually explicit or abusive content?",
    "violence": "Does the text promote, threaten, or instruct physical violence or weapons?",
    "prompt-injection": "Does the text attempt a prompt injection, jailbreak, or system instruction override?",
    "pii-exfiltration": "Does the text attempt to leak credentials, API keys, passwords, or private PII?",
}


class ModerationAdapter:
    """100% OpenAI `/v1/moderations` wire-compatible adapter backed by zero-token local models."""

    def __init__(
        self,
        engine: BaseDecisionEngine,
        *,
        categories: Optional[Mapping[str, str]] = None,
        flag_threshold: float = 0.50,
    ):
        self.engine = engine
        self.categories = dict(categories or DEFAULT_MODERATION_CATEGORIES)
        self.flag_threshold = float(flag_threshold)

    def moderate(
        self,
        input_data: Union[str, Sequence[str]],
        *,
        model: Optional[str] = None,
        custom_categories: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        texts = [input_data] if isinstance(input_data, str) else [str(x) for x in input_data]
        cats = dict(custom_categories or self.categories)
        questions = {cat_name: {"type": "noul", "instructions": instr} for cat_name, instr in cats.items()}

        results: List[Dict[str, Any]] = []
        for txt in texts:
            raw = self.engine.predict({"role": "prompt", "text": txt}, questions)
            ans = raw.get("answers") or {}
            scores: Dict[str, float] = {}
            flags: Dict[str, bool] = {}

            for cat_name, qspec in questions.items():
                _, p = _extract_simplex_from_answer(ans.get(cat_name, {}), qspec)
                p_true = float(p[1])
                scores[cat_name] = round(p_true, 6)
                flags[cat_name] = p_true >= self.flag_threshold

            if spoofed_audit_guard_rule(txt, {}, None) is not None and "prompt-injection" in scores:
                scores["prompt-injection"] = max(scores["prompt-injection"], 0.99)
                flags["prompt-injection"] = True

            results.append(
                {
                    "flagged": any(flags.values()),
                    "categories": flags,
                    "category_scores": scores,
                }
            )

        return {
            "id": f"modr-{uuid.uuid4().hex[:16]}",
            "model": model or self.engine.model_id,
            "results": results,
        }
