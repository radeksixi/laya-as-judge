"""OmegaDecisionEngine: Conformal Budget-Routed Hybrid Local (Laya + Vega-U + TTT) + Cloud (Gemini) Engine."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .calibration import ClassMultiplierOptimizer
from .cascade import has_unsupported_modality
from .engine.base import BaseDecisionEngine, confidence_from_probs, validate_question_spec
from .tier2_gemini import CostTracker, GeminiDecisionEngine


def _extract_simplex(ans: Mapping[str, Any], qspec: Mapping[str, Any]) -> Tuple[List[str], np.ndarray]:
    """Return canonical label list and normalized probability simplex for one answer."""
    qtype = qspec["type"]
    if qtype == "noul":
        labels = ["false", "true"]
        p_true = float(ans.get("noul", 0.5))
        arr = np.array([1.0 - p_true, p_true], dtype=np.float64)
    elif qtype == "score":
        k = len(qspec["criteria"])
        labels = [str(i) for i in range(k)]
        pdict = ans.get("probabilities", {}) or {}
        arr = np.array([float(pdict.get(lbl, 0.0)) for lbl in labels], dtype=np.float64)
    else:
        crit = qspec["criteria"]
        labels = [str(k) for k in (crit.keys() if isinstance(crit, Mapping) else crit)]
        pdict = ans.get("probabilities", {}) or {}
        arr = np.array([float(pdict.get(lbl, 0.0)) for lbl in labels], dtype=np.float64)

    arr = np.clip(arr, 1e-6, 1.0)
    s = float(np.sum(arr))
    if s <= 0.0 or not np.all(np.isfinite(arr)):
        arr = np.ones(len(labels), dtype=np.float64) / max(1, len(labels))
    else:
        arr /= s
    return labels, arr


def _pack_simplex_to_answer(
    labels: Sequence[str],
    probs: np.ndarray,
    qspec: Mapping[str, Any],
    tier_tag: str,
) -> Dict[str, Any]:
    """Pack a fused probability simplex back into Laya's canonical answer dict."""
    qtype = qspec["type"]
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0)
    p /= float(np.sum(p))
    k = len(labels)
    conf = round(confidence_from_probs(p, k), 4)
    top1 = round(float(np.max(p)), 6)

    if qtype == "choice":
        return {
            "type": "choice",
            "choice": str(labels[int(np.argmax(p))]),
            "probabilities": {str(lbl): round(float(val), 6) for lbl, val in zip(labels, p)},
            "confidence": conf,
            "answer_confidence": top1,
            "omega_tier": tier_tag,
        }
    if qtype == "score":
        exp_score = float(np.sum(np.arange(k, dtype=np.float64) * p))
        return {
            "type": "score",
            "score": round(exp_score, 4),
            "legend": {str(i): str(c) for i, c in enumerate(qspec["criteria"])},
            "probabilities": {str(i): round(float(val), 6) for i, val in enumerate(p)},
            "confidence": conf,
            "answer_confidence": top1,
            "omega_tier": tier_tag,
        }
    p_true = float(p[1]) if k >= 2 else 0.5
    return {
        "type": "noul",
        "noul": round(p_true, 6),
        "probabilities": {"false": round(1.0 - p_true, 6), "true": round(p_true, 6)},
        "confidence": conf,
        "answer_confidence": round(float(max(p_true, 1.0 - p_true)), 6),
        "omega_tier": tier_tag,
    }


def _uncertainty_score_from_raw(raw_out: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]) -> float:
    """Compute epistemic + aleatoric uncertainty score for Knapsack budget escalation."""
    ans_map = raw_out.get("answers", {}) or {}
    if not ans_map:
        return 1.0
    scores: List[float] = []
    for qid, qspec in questions.items():
        ans = ans_map.get(qid, {})
        _, probs = _extract_simplex(ans, qspec)
        sorted_p = np.sort(probs)[::-1]
        p1 = float(sorted_p[0])
        p2 = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
        margin = p1 - p2
        pset = ans.get("prediction_set")
        pset_pen = 0.12 * max(0, len(pset) - 1) if isinstance(pset, list) else 0.0
        abstain_pen = 0.18 if ans.get("abstain") is True else 0.0
        route_pen = 0.10 if ans.get("route") == "review" else 0.0
        scores.append((1.0 - margin) + pset_pen + abstain_pen + route_pen)
    return float(np.mean(scores))


class OmegaDecisionEngine(BaseDecisionEngine):
    """Hybrid Conformal Cascade Engine (`Laya` + `Vega-U` + `TTTHead` locally -> `Gemini` Tier-2).

    Achieves frontier accuracy across text & multimodal benchmarks while keeping `Total price`
    a small fraction of cloud-only models via conformal Top-K uncertainty routing.
    """

    supports_multimodal: bool = True

    def __init__(
        self,
        local_laya: Optional[BaseDecisionEngine] = None,
        local_vega: Optional[BaseDecisionEngine] = None,
        tier2_engine: Optional[GeminiDecisionEngine] = None,
        cost_tracker: Optional[CostTracker] = None,
        escalation_threshold: float = 0.78,
        max_escalation_fraction: float = 0.35,
        vega_weight: float = 0.60,
        ttt_vega_weight: float = 0.82,
        tier2_blend_weight: float = 0.88,
        ttt_tier2_blend_weight: float = 0.74,
        default_tier2_model: str = "gemini-3.5-flash-lite",
        max_budget_usd: float = 1.80,
    ):
        self.local_laya = local_laya
        self.local_vega = local_vega
        self.cost_tracker = cost_tracker or (
            tier2_engine.cost_tracker if tier2_engine is not None else CostTracker(max_budget_usd=max_budget_usd)
        )
        self.tier2_engine = tier2_engine or GeminiDecisionEngine(
            model_id=default_tier2_model,
            cost_tracker=self.cost_tracker,
        )
        self.escalation_threshold = float(escalation_threshold)
        self.max_escalation_fraction = float(np.clip(max_escalation_fraction, 0.0, 1.0))
        self.vega_weight = float(vega_weight)
        self.ttt_vega_weight = float(ttt_vega_weight)
        self.tier2_blend_weight = float(tier2_blend_weight)
        self.ttt_tier2_blend_weight = float(ttt_tier2_blend_weight)
        self.class_optimizers: Dict[str, ClassMultiplierOptimizer] = {}
        self.tier2_class_optimizers: Dict[str, ClassMultiplierOptimizer] = {}

        parts = []
        if self.local_laya is not None:
            parts.append(self.local_laya.model_id)
        if self.local_vega is not None:
            parts.append(self.local_vega.model_id)
        parts.append(self.tier2_engine.model_id)
        super().__init__(model_id="omega(" + " + ".join(parts) + ")")

        self.stats: Dict[str, float] = {
            "total_cases": 0.0,
            "local_resolved": 0.0,
            "tier2_escalated": 0.0,
            "tier2_heavy_escalated": 0.0,
            "multimodal_routed": 0.0,
            "total_cost_usd": 0.0,
        }

    def register_class_optimizer(
        self,
        question_id: str,
        optimizer: ClassMultiplierOptimizer,
        scope: str = "local",
    ) -> None:
        """Attach a disjoint-train-fitted Macro-F1 `ClassMultiplierOptimizer` (`scope='local'|'tier2'|'both'`)."""
        if scope in ("local", "both"):
            self.class_optimizers[question_id] = optimizer
        if scope in ("tier2", "both"):
            self.tier2_class_optimizers[question_id] = optimizer

    def _has_ttt_head(self, qid: str) -> bool:
        fitted = getattr(self.local_vega, "fitted_heads", []) if self.local_vega is not None else []
        return qid in fitted

    def _fuse_local(
        self,
        laya_out: Optional[Mapping[str, Any]],
        vega_out: Optional[Mapping[str, Any]],
        questions: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Log-linear ensemble of local Laya encoder and local Vega-U (+ optional TTT head)."""
        if laya_out is None and vega_out is None:
            raise RuntimeError("At least one local engine is required for `_fuse_local`.")
        if laya_out is None:
            return dict(vega_out)  # type: ignore[arg-type]
        if vega_out is None:
            return dict(laya_out)

        fused_answers: Dict[str, Any] = {}
        for qid, qspec in questions.items():
            a_laya = (laya_out.get("answers") or {}).get(qid, {})
            a_vega = (vega_out.get("answers") or {}).get(qid, {})
            labels, p_laya = _extract_simplex(a_laya, qspec)
            _, p_vega = _extract_simplex(a_vega, qspec)
            w_v = self.ttt_vega_weight if self._has_ttt_head(qid) else self.vega_weight
            log_p = (1.0 - w_v) * np.log(np.clip(p_laya, 1e-6, 1.0)) + w_v * np.log(np.clip(p_vega, 1e-6, 1.0))
            log_p -= np.max(log_p)
            p_ens = np.exp(log_p)
            p_ens /= float(np.sum(p_ens))

            packed = _pack_simplex_to_answer(labels, p_ens, qspec, tier_tag="local_laya_vega")
            # Preserve Vega conformal metadata for downstream routing
            for meta_key in ("prediction_set", "abstain", "unbound", "route"):
                if meta_key in a_vega:
                    packed[meta_key] = a_vega[meta_key]
            fused_answers[qid] = packed

        in_toks = max(
            int((laya_out.get("usage") or {}).get("input_tokens", 0)),
            int((vega_out.get("usage") or {}).get("input_tokens", 0)),
        )
        lat = float(laya_out.get("latency_ms", 0.0)) + float(vega_out.get("latency_ms", 0.0))
        return {
            "model": "local_omega",
            "answers": fused_answers,
            "usage": {"input_tokens": in_toks, "output_tokens": 0},
            "cost_usd": 0.0,
            "latency_ms": round(lat, 2),
        }

    @staticmethod
    def _blend_two_tier2(
        stage_a_out: Mapping[str, Any],
        stage_b_out: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        heavy_weight: float = 0.78,
    ) -> Dict[str, Any]:
        """Blend Stage-A (`flash-lite`) and Stage-B (`3.8-flash`) posteriors while summing exact costs."""
        answers: Dict[str, Any] = {}
        for qid, qspec in questions.items():
            labels, pa = _extract_simplex((stage_a_out.get("answers") or {}).get(qid, {}), qspec)
            _, pb = _extract_simplex((stage_b_out.get("answers") or {}).get(qid, {}), qspec)
            pf = (1.0 - heavy_weight) * pa + heavy_weight * pb
            answers[qid] = _pack_simplex_to_answer(labels, pf, qspec, tier_tag="omega_heavy_tier2")

        ua = stage_a_out.get("usage") or {}
        ub = stage_b_out.get("usage") or {}
        return {
            "model": f"{stage_a_out.get('model', 't2a')}+{stage_b_out.get('model', 't2b')}",
            "answers": answers,
            "usage": {
                "input_tokens": int(ua.get("input_tokens", 0)) + int(ub.get("input_tokens", 0)),
                "output_tokens": int(ua.get("output_tokens", 0)) + int(ub.get("output_tokens", 0)),
            },
            "cost_usd": float(stage_a_out.get("cost_usd", 0.0)) + float(stage_b_out.get("cost_usd", 0.0)),
            "latency_ms": round(float(stage_a_out.get("latency_ms", 0.0)) + float(stage_b_out.get("latency_ms", 0.0)), 2),
        }

    def _merge_tier2(
        self,
        local_out: Optional[Mapping[str, Any]],
        tier2_out: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        blend_weight_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Blend local posterior (carrying domain TTT priors) with Tier-2 Gemini posterior."""
        answers: Dict[str, Any] = {}
        for qid, qspec in questions.items():
            a_t2 = (tier2_out.get("answers") or {}).get(qid, {})
            labels, p_t2 = _extract_simplex(a_t2, qspec)
            if local_out is not None and qid in (local_out.get("answers") or {}):
                _, p_loc = _extract_simplex(local_out["answers"][qid], qspec)
                if blend_weight_override is not None:
                    w_t2 = float(blend_weight_override)
                else:
                    w_t2 = self.ttt_tier2_blend_weight if self._has_ttt_head(qid) else self.tier2_blend_weight
                p_final = (1.0 - w_t2) * p_loc + w_t2 * p_t2
                tag = "omega_blended_tier2"
            else:
                p_final = p_t2
                tag = "omega_direct_tier2"

            if qid in self.tier2_class_optimizers:
                p_final = self.tier2_class_optimizers[qid].apply(p_final)

            answers[qid] = _pack_simplex_to_answer(labels, p_final, qspec, tier_tag=tag)

        t2_usage = tier2_out.get("usage") or {}
        loc_lat = float(local_out.get("latency_ms", 0.0)) if local_out is not None else 0.0
        return {
            "model": f"{self.model_id}->{tier2_out.get('model', 'tier2')}",
            "answers": answers,
            "usage": {
                "input_tokens": int(t2_usage.get("input_tokens", 0)),
                "output_tokens": int(t2_usage.get("output_tokens", 0)),
            },
            "cost_usd": float(tier2_out.get("cost_usd", 0.0)),
            "latency_ms": round(loc_lat + float(tier2_out.get("latency_ms", 0.0)), 2),
        }

    def _finalize_local_only(
        self,
        local_out: Dict[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Apply any registered local `ClassMultiplierOptimizer` to purely local predictions."""
        if not self.class_optimizers:
            return local_out
        ans_map = dict(local_out.get("answers") or {})
        for qid, qspec in questions.items():
            if qid in self.class_optimizers and qid in ans_map:
                labels, probs = _extract_simplex(ans_map[qid], qspec)
                adj = self.class_optimizers[qid].apply(probs)
                ans_map[qid] = _pack_simplex_to_answer(labels, adj, qspec, tier_tag="local_laya_vega")
        out = dict(local_out)
        out["answers"] = ans_map
        return out

    def predict(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Dict[str, Any]],
        force_tier2: bool = False,
        tier2_model: Optional[str] = None,
        thinking_level: Optional[str] = None,
        system_hint: Optional[str] = None,
        escalation_threshold: Optional[float] = None,
        image_max_side: Optional[int] = None,
        image_resolution: Optional[str] = None,
        tier2_blend_weight: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        self.stats["total_cases"] += 1

        # 1. Direct Tier-2 route for multimodal inputs or forced Tier-2 tasks
        if force_tier2 or has_unsupported_modality(state) or (self.local_laya is None and self.local_vega is None):
            if has_unsupported_modality(state):
                self.stats["multimodal_routed"] += 1
            self.stats["tier2_escalated"] += 1
            t2_out = self.tier2_engine.predict(
                state,
                normalized_q,
                model_override=tier2_model,
                thinking_level_override=thinking_level,
                system_hint_override=system_hint,
                image_max_side_override=image_max_side,
                image_resolution_override=image_resolution,
            )
            self.stats["total_cost_usd"] += float(t2_out.get("cost_usd", 0.0))
            return self._merge_tier2(None, t2_out, normalized_q, blend_weight_override=tier2_blend_weight)

        # 2. Local GPU pass (Laya + Vega-U; strip Tier-2-only precedent context so 512/1024-token encoders stay clean)
        loc_state = self._local_clean_state(state)
        laya_out = self.local_laya.predict(loc_state, normalized_q) if self.local_laya is not None else None
        vega_out = self.local_vega.predict(loc_state, normalized_q) if self.local_vega is not None else None
        local_out = self._fuse_local(laya_out, vega_out, normalized_q)

        thresh = float(escalation_threshold if escalation_threshold is not None else self.escalation_threshold)
        top1_vals = [
            float(_extract_simplex(local_out["answers"][qid], qspec)[1].max())
            for qid, qspec in normalized_q.items()
        ]
        min_top1 = min(top1_vals) if top1_vals else 0.0
        any_review = any(
            local_out["answers"][qid].get("route") == "review"
            or len(local_out["answers"][qid].get("prediction_set") or []) > 1
            for qid in normalized_q
        )

        if self.max_escalation_fraction <= 0.0 or (min_top1 >= thresh and not any_review):
            self.stats["local_resolved"] += 1
            return self._finalize_local_only(local_out, normalized_q)

        # 3. Conformal-ambiguous -> Escalate to Tier-2 and blend
        self.stats["tier2_escalated"] += 1
        t2_out = self.tier2_engine.predict(
            state,
            normalized_q,
            model_override=tier2_model,
            thinking_level_override=thinking_level,
            system_hint_override=system_hint,
            image_max_side_override=image_max_side,
            image_resolution_override=image_resolution,
        )
        self.stats["total_cost_usd"] += float(t2_out.get("cost_usd", 0.0))
        return self._merge_tier2(local_out, t2_out, normalized_q, blend_weight_override=tier2_blend_weight)

    @staticmethod
    def _local_clean_state(state: Union[str, Dict[str, Any], List[Any]]) -> Union[str, Dict[str, Any], List[Any]]:
        """Remove Tier-2-only precedent context before running 512/1024-token local encoders."""
        if not isinstance(state, dict) or "disjoint_domain_policy_precedents" not in state:
            return state
        return {k: v for k, v in state.items() if k != "disjoint_domain_policy_precedents"}

    def _run_tier2_pairs_with_optional_heavy(
        self,
        pairs: Sequence[Tuple[Union[str, Dict[str, Any], List[Any]], Dict[str, Dict[str, Any]]]],
        total_population_size: int,
        tier2_model: Optional[str] = None,
        thinking_level: Optional[str] = None,
        system_hint: Optional[str] = None,
        image_max_side: Optional[int] = None,
        image_resolution: Optional[str] = None,
        pack_size: int = 1,
        tier2_heavy_model: Optional[str] = None,
        heavy_thinking_level: Optional[str] = None,
        heavy_escalation_fraction: float = 0.0,
        heavy_threshold: float = 0.85,
        heavy_pack_size: Optional[int] = None,
        heavy_image_resolution: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute Stage-A Tier-2 calls concurrently (with optional multi-item `pack_size`), then Stage-B heavy."""
        from concurrent.futures import ThreadPoolExecutor

        if not pairs:
            return []
        workers = min(len(pairs), max(1, int(getattr(self.tier2_engine, "max_workers", 8))))
        eff_pack = max(1, int(pack_size))

        if eff_pack > 1 and hasattr(self.tier2_engine, "predict_packed_pairs") and len(pairs) > 1:
            stage_a = self.tier2_engine.predict_packed_pairs(
                pairs,
                pack_size=eff_pack,
                max_workers=workers,
                model_override=tier2_model,
                thinking_level_override=thinking_level,
                system_hint_override=system_hint,
                image_max_side_override=image_max_side,
                image_resolution_override=image_resolution,
            )
        elif workers <= 1:
            stage_a = [
                self.tier2_engine.predict(
                    st,
                    qs,
                    model_override=tier2_model,
                    thinking_level_override=thinking_level,
                    system_hint_override=system_hint,
                    image_max_side_override=image_max_side,
                    image_resolution_override=image_resolution,
                )
                for st, qs in pairs
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [
                    pool.submit(
                        self.tier2_engine.predict,
                        st,
                        qs,
                        model_override=tier2_model,
                        thinking_level_override=thinking_level,
                        system_hint_override=system_hint,
                        image_max_side_override=image_max_side,
                        image_resolution_override=image_resolution,
                    )
                    for st, qs in pairs
                ]
                stage_a = [f.result() for f in futs]

        if not tier2_heavy_model or heavy_escalation_fraction <= 0.0:
            return stage_a

        max_heavy_k = int(round(total_population_size * float(np.clip(heavy_escalation_fraction, 0.0, 1.0))))
        if max_heavy_k <= 0:
            return stage_a

        heavy_cands: List[Tuple[float, int]] = []
        for idx, (out_a, (_, qs)) in enumerate(zip(stage_a, pairs)):
            top1_vals = [
                float(_extract_simplex((out_a.get("answers") or {}).get(qid, {}), qspec)[1].max())
                for qid, qspec in qs.items()
            ]
            min_t1 = min(top1_vals) if top1_vals else 0.0
            if min_t1 < heavy_threshold:
                heavy_cands.append((_uncertainty_score_from_raw(out_a, qs), idx))

        heavy_cands.sort(key=lambda x: -x[0])
        heavy_idx = [idx for _, idx in heavy_cands[:max_heavy_k]]
        if not heavy_idx:
            return stage_a

        self.stats["tier2_heavy_escalated"] += len(heavy_idx)
        h_workers = min(len(heavy_idx), workers)
        eff_h_think = heavy_thinking_level if heavy_thinking_level is not None else thinking_level
        eff_h_pack = max(1, int(heavy_pack_size if heavy_pack_size is not None else eff_pack))
        eff_h_res = heavy_image_resolution if heavy_image_resolution is not None else image_resolution
        heavy_pairs = [pairs[i] for i in heavy_idx]

        if eff_h_pack > 1 and hasattr(self.tier2_engine, "predict_packed_pairs") and len(heavy_pairs) > 1:
            stage_b = self.tier2_engine.predict_packed_pairs(
                heavy_pairs,
                pack_size=eff_h_pack,
                max_workers=h_workers,
                model_override=tier2_heavy_model,
                thinking_level_override=eff_h_think,
                system_hint_override=system_hint,
                image_max_side_override=image_max_side,
                image_resolution_override=eff_h_res,
            )
        elif h_workers <= 1:
            stage_b = [
                self.tier2_engine.predict(
                    pairs[i][0],
                    pairs[i][1],
                    model_override=tier2_heavy_model,
                    thinking_level_override=eff_h_think,
                    system_hint_override=system_hint,
                    image_max_side_override=image_max_side,
                    image_resolution_override=eff_h_res,
                )
                for i in heavy_idx
            ]
        else:
            with ThreadPoolExecutor(max_workers=h_workers) as pool:
                futs = [
                    pool.submit(
                        self.tier2_engine.predict,
                        pairs[i][0],
                        pairs[i][1],
                        model_override=tier2_heavy_model,
                        thinking_level_override=eff_h_think,
                        system_hint_override=system_hint,
                        image_max_side_override=image_max_side,
                        image_resolution_override=eff_h_res,
                    )
                    for i in heavy_idx
                ]
                stage_b = [f.result() for f in futs]

        final_t2 = list(stage_a)
        for i_sub, out_b in zip(heavy_idx, stage_b):
            final_t2[i_sub] = self._blend_two_tier2(stage_a[i_sub], out_b, pairs[i_sub][1])
        return final_t2

    def predict_cases(
        self,
        case_pairs: Sequence[Tuple[Union[str, Dict[str, Any], List[Any]], Dict[str, Dict[str, Any]]]],
        force_tier2: bool = False,
        tier2_model: Optional[str] = None,
        thinking_level: Optional[str] = None,
        system_hint: Optional[str] = None,
        escalation_threshold: Optional[float] = None,
        max_escalation_fraction: Optional[float] = None,
        image_max_side: Optional[int] = None,
        image_resolution: Optional[str] = None,
        pack_size: int = 1,
        tier2_blend_weight: Optional[float] = None,
        tier2_heavy_model: Optional[str] = None,
        heavy_thinking_level: Optional[str] = None,
        heavy_escalation_fraction: float = 0.0,
        heavy_threshold: float = 0.85,
        heavy_pack_size: Optional[int] = None,
        heavy_image_resolution: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Execute heterogeneous `(state, questions)` cases with Knapsack Top-K budget & parallel Tier-2."""
        if not case_pairs:
            return []
        norm_pairs = [
            (st, {qid: validate_question_spec(qid, q) for qid, q in qs.items()})
            for st, qs in case_pairs
        ]
        n = len(norm_pairs)
        self.stats["total_cases"] += n

        if (
            force_tier2
            or any(has_unsupported_modality(st) for st, _ in norm_pairs)
            or (self.local_laya is None and self.local_vega is None)
        ):
            mm_count = sum(1 for st, _ in norm_pairs if has_unsupported_modality(st))
            self.stats["multimodal_routed"] += mm_count
            self.stats["tier2_escalated"] += n
            t2_outs = self._run_tier2_pairs_with_optional_heavy(
                norm_pairs,
                total_population_size=n,
                tier2_model=tier2_model,
                thinking_level=thinking_level,
                system_hint=system_hint,
                image_max_side=image_max_side,
                image_resolution=image_resolution,
                pack_size=pack_size,
                tier2_heavy_model=tier2_heavy_model,
                heavy_thinking_level=heavy_thinking_level,
                heavy_escalation_fraction=heavy_escalation_fraction,
                heavy_threshold=heavy_threshold,
                heavy_pack_size=heavy_pack_size,
                heavy_image_resolution=heavy_image_resolution,
            )
            merged: List[Dict[str, Any]] = []
            for (_, qs), t2 in zip(norm_pairs, t2_outs):
                self.stats["total_cost_usd"] += float(t2.get("cost_usd", 0.0))
                merged.append(self._merge_tier2(None, t2, qs, blend_weight_override=tier2_blend_weight))
            return merged

        local_outs: List[Dict[str, Any]] = []
        for st, qs in norm_pairs:
            loc_st = self._local_clean_state(st)
            lo = self.local_laya.predict(loc_st, qs) if self.local_laya is not None else None
            vo = self.local_vega.predict(loc_st, qs) if self.local_vega is not None else None
            local_outs.append(self._fuse_local(lo, vo, qs))

        eff_frac = float(
            max_escalation_fraction if max_escalation_fraction is not None else self.max_escalation_fraction
        )
        max_k = int(round(n * np.clip(eff_frac, 0.0, 1.0)))
        thresh = float(escalation_threshold if escalation_threshold is not None else self.escalation_threshold)

        candidates: List[Tuple[float, int]] = []
        for idx, (loc, (_, qs)) in enumerate(zip(local_outs, norm_pairs)):
            top1_vals = [
                float(_extract_simplex(loc["answers"][qid], qspec)[1].max())
                for qid, qspec in qs.items()
            ]
            min_top1 = min(top1_vals) if top1_vals else 0.0
            any_review = any(
                loc["answers"][qid].get("route") == "review"
                or len(loc["answers"][qid].get("prediction_set") or []) > 1
                for qid in qs
            )
            if min_top1 < thresh or any_review:
                candidates.append((_uncertainty_score_from_raw(loc, qs), idx))

        candidates.sort(key=lambda x: -x[0])
        escalate_indices = [idx for _, idx in candidates[:max_k]]
        escalate_set = set(escalate_indices)

        results: List[Optional[Dict[str, Any]]] = [None] * n
        for idx in range(n):
            if idx not in escalate_set:
                self.stats["local_resolved"] += 1
                results[idx] = self._finalize_local_only(local_outs[idx], norm_pairs[idx][1])

        if escalate_indices:
            self.stats["tier2_escalated"] += len(escalate_indices)
            esc_pairs = [norm_pairs[i] for i in escalate_indices]
            t2_outs = self._run_tier2_pairs_with_optional_heavy(
                esc_pairs,
                total_population_size=n,
                tier2_model=tier2_model,
                thinking_level=thinking_level,
                system_hint=system_hint,
                image_max_side=image_max_side,
                image_resolution=image_resolution,
                pack_size=pack_size,
                tier2_heavy_model=tier2_heavy_model,
                heavy_thinking_level=heavy_thinking_level,
                heavy_escalation_fraction=heavy_escalation_fraction,
                heavy_threshold=heavy_threshold,
                heavy_pack_size=heavy_pack_size,
                heavy_image_resolution=heavy_image_resolution,
            )
            for orig_idx, t2 in zip(escalate_indices, t2_outs):
                self.stats["total_cost_usd"] += float(t2.get("cost_usd", 0.0))
                results[orig_idx] = self._merge_tier2(
                    local_outs[orig_idx],
                    t2,
                    norm_pairs[orig_idx][1],
                    blend_weight_override=tier2_blend_weight,
                )

        return [r for r in results if r is not None]

    def predict_batch(
        self,
        states: List[Union[str, Dict[str, Any], List[Any]]],
        questions: Dict[str, Dict[str, Any]],
        force_tier2: bool = False,
        tier2_model: Optional[str] = None,
        thinking_level: Optional[str] = None,
        system_hint: Optional[str] = None,
        escalation_threshold: Optional[float] = None,
        max_escalation_fraction: Optional[float] = None,
        image_max_side: Optional[int] = None,
        image_resolution: Optional[str] = None,
        pack_size: int = 1,
        tier2_blend_weight: Optional[float] = None,
        tier2_heavy_model: Optional[str] = None,
        heavy_thinking_level: Optional[str] = None,
        heavy_escalation_fraction: float = 0.0,
        heavy_threshold: float = 0.85,
        heavy_pack_size: Optional[int] = None,
        heavy_image_resolution: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Batched local GPU inference + Knapsack Top-K uncertainty escalation to Gemini Tier-2."""
        if not states:
            return []
        normalized_q = {qid: validate_question_spec(qid, q) for qid, q in questions.items()}
        n = len(states)
        self.stats["total_cases"] += n

        # Direct Tier-2 batch if forced or multimodal
        if (
            force_tier2
            or any(has_unsupported_modality(s) for s in states)
            or (self.local_laya is None and self.local_vega is None)
        ):
            mm_count = sum(1 for s in states if has_unsupported_modality(s))
            self.stats["multimodal_routed"] += mm_count
            self.stats["tier2_escalated"] += n
            t2_outs = self._run_tier2_pairs_with_optional_heavy(
                [(s, normalized_q) for s in states],
                total_population_size=n,
                tier2_model=tier2_model,
                thinking_level=thinking_level,
                system_hint=system_hint,
                image_max_side=image_max_side,
                image_resolution=image_resolution,
                pack_size=pack_size,
                tier2_heavy_model=tier2_heavy_model,
                heavy_thinking_level=heavy_thinking_level,
                heavy_escalation_fraction=heavy_escalation_fraction,
                heavy_threshold=heavy_threshold,
                heavy_pack_size=heavy_pack_size,
                heavy_image_resolution=heavy_image_resolution,
            )
            merged: List[Dict[str, Any]] = []
            for t2 in t2_outs:
                self.stats["total_cost_usd"] += float(t2.get("cost_usd", 0.0))
                merged.append(self._merge_tier2(None, t2, normalized_q, blend_weight_override=tier2_blend_weight))
            return merged

        # Run fast local GPU batch on ALL states ($0.00 cost!)
        loc_states = [self._local_clean_state(s) for s in states]
        laya_outs = (
            self.local_laya.predict_batch(loc_states, normalized_q) if self.local_laya is not None else [None] * n
        )
        vega_outs = (
            self.local_vega.predict_batch(loc_states, normalized_q) if self.local_vega is not None else [None] * n
        )
        local_outs = [self._fuse_local(lo, vo, normalized_q) for lo, vo in zip(laya_outs, vega_outs)]

        eff_frac = float(
            max_escalation_fraction if max_escalation_fraction is not None else self.max_escalation_fraction
        )
        max_k = int(round(n * np.clip(eff_frac, 0.0, 1.0)))
        thresh = float(escalation_threshold if escalation_threshold is not None else self.escalation_threshold)

        # Rank candidates that fail local confidence / conformal check by uncertainty score
        candidates: List[Tuple[float, int]] = []
        for idx, loc in enumerate(local_outs):
            top1_vals = [
                float(_extract_simplex(loc["answers"][qid], qspec)[1].max())
                for qid, qspec in normalized_q.items()
            ]
            min_top1 = min(top1_vals) if top1_vals else 0.0
            any_review = any(
                loc["answers"][qid].get("route") == "review"
                or len(loc["answers"][qid].get("prediction_set") or []) > 1
                for qid in normalized_q
            )
            if min_top1 < thresh or any_review:
                unc = _uncertainty_score_from_raw(loc, normalized_q)
                candidates.append((unc, idx))

        candidates.sort(key=lambda x: -x[0])
        escalate_indices = [idx for _, idx in candidates[:max_k]]
        escalate_set = set(escalate_indices)

        results: List[Optional[Dict[str, Any]]] = [None] * n
        for idx in range(n):
            if idx not in escalate_set:
                self.stats["local_resolved"] += 1
                results[idx] = self._finalize_local_only(local_outs[idx], normalized_q)

        if escalate_indices:
            self.stats["tier2_escalated"] += len(escalate_indices)
            esc_pairs = [(states[i], normalized_q) for i in escalate_indices]
            t2_outs = self._run_tier2_pairs_with_optional_heavy(
                esc_pairs,
                total_population_size=n,
                tier2_model=tier2_model,
                thinking_level=thinking_level,
                system_hint=system_hint,
                image_max_side=image_max_side,
                image_resolution=image_resolution,
                pack_size=pack_size,
                tier2_heavy_model=tier2_heavy_model,
                heavy_thinking_level=heavy_thinking_level,
                heavy_escalation_fraction=heavy_escalation_fraction,
                heavy_threshold=heavy_threshold,
                heavy_pack_size=heavy_pack_size,
                heavy_image_resolution=heavy_image_resolution,
            )
            for orig_idx, t2 in zip(escalate_indices, t2_outs):
                self.stats["total_cost_usd"] += float(t2.get("cost_usd", 0.0))
                results[orig_idx] = self._merge_tier2(
                    local_outs[orig_idx],
                    t2,
                    normalized_q,
                    blend_weight_override=tier2_blend_weight,
                )

        return [r for r in results if r is not None]
