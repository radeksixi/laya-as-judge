"""Speculative / Cascaded judging combining fast local Laya with Tier 2 Decision/LLM fallback."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np

from .calibration import PRICING_PROFILES, PricingProfile
from .engine.base import confidence_from_probs
from .judges.base import BaseJudge
from .types import ChoiceResult, DecisionType, EvaluationReport, NoulResult, ScoreResult

logger = logging.getLogger("laya_as_judge")

MULTIMODAL_STATE_KEYS: Set[str] = {
    "image",
    "images",
    "screenshot",
    "screenshots",
    "audio",
    "video",
    "frames",
    "pixel_values",
}


def has_unsupported_modality(state: Any) -> bool:
    """Detect whether `state` carries non-text media (images, audio, video) unsupported by Laya."""
    if isinstance(state, dict):
        for k, v in state.items():
            if str(k).lower() in MULTIMODAL_STATE_KEYS and v not in (None, "", [], {}):
                return True
    return False


class CascadedJudge:
    """Speculative / Cascaded Judge architecture.

    Tier 1 (Laya): Evaluates locally in ~7-35ms at $0.00 cost.
    Tier 2 (Cloud Decision API e.g. Gemini Decision 1 / Jev 1.14, Vega, or Frontier LLM):
    Invoked only when:
      1. `state` contains non-text modalities (`image`/`audio`/`video`), OR
      2. Laya's calibrated routing confidence falls below `confidence_threshold`.

    Supported `routing_metric` modes:
      - `"top1_prob"` (default): Routes on `average_top1_probability` (`max_i p_i`).
        Directly aligned with `Top-1 ECE` and avoids the binary K=2 Shannon entropy trap
        (where p_max=0.80 yields entropy confidence of only 0.278).
      - `"entropy"`: Routes on normalized Shannon entropy `1 - H(p)/ln(K)`.
      - `"action_head"`: Routes on `1 - max(act_probability)` from Laya's RL escalation head.
    """

    def __init__(
        self,
        tier1_judge: BaseJudge,
        tier2_llm_judge_fn: Optional[
            Callable[[Union[str, Dict[str, Any], List[Any]], Dict[str, Any]], Dict[str, Any]]
        ] = None,
        confidence_threshold: float = 0.75,
        estimated_llm_cost_per_eval: float = 0.03,
        estimated_llm_latency_ms: float = 2000.0,
        routing_metric: str = "top1_prob",
        escalate_multimodal: bool = True,
        tier2_pricing: Optional[Union[str, PricingProfile]] = None,
    ):
        if routing_metric not in ("top1_prob", "entropy", "action_head"):
            raise ValueError(
                f"Unknown routing_metric '{routing_metric}'; expected 'top1_prob', 'entropy', or 'action_head'"
            )
        self.tier1_judge = tier1_judge
        self.tier2_llm_judge_fn = tier2_llm_judge_fn
        self.confidence_threshold = confidence_threshold
        self.estimated_llm_cost_per_eval = estimated_llm_cost_per_eval
        self.estimated_llm_latency_ms = estimated_llm_latency_ms
        self.routing_metric = routing_metric
        self.escalate_multimodal = escalate_multimodal

        if isinstance(tier2_pricing, str):
            if tier2_pricing not in PRICING_PROFILES:
                raise ValueError(f"Unknown tier2_pricing profile '{tier2_pricing}'")
            self.tier2_pricing: Optional[PricingProfile] = PRICING_PROFILES[tier2_pricing]
        else:
            self.tier2_pricing = tier2_pricing

        self.stats = {
            "total_evals": 0,
            "tier1_resolved": 0,
            "tier2_escalated": 0,
            "multimodal_escalated": 0,
            "total_input_tokens": 0,
            "tier2_input_tokens": 0,
            "total_cost_usd": 0.0,
            "total_cost_saved_usd": 0.0,
            "total_latency_ms": 0.0,
        }

    def _compute_routing_score(self, report: EvaluationReport) -> float:
        """Compute scalar confidence score in [0, 1] according to `self.routing_metric`."""
        if not report.judgements:
            return 0.0
        if self.routing_metric == "entropy":
            return report.average_confidence
        if self.routing_metric == "action_head":
            act_probs = [
                j.action_probability for j in report.judgements.values() if j.action_probability is not None
            ]
            if act_probs:
                return float(np.clip(1.0 - max(act_probs), 0.0, 1.0))
            return report.average_top1_probability
        return report.average_top1_probability

    def _estimate_baseline_cost(self, input_tokens: int) -> float:
        if self.tier2_pricing is not None:
            return self.tier2_pricing.compute_cost(input_tokens, eval_count=1)
        return self.estimated_llm_cost_per_eval

    def _apply_tier2_answers(self, report: EvaluationReport, llm_result: Dict[str, Any]) -> None:
        """If Tier 2 returns typed Jev/Decision-API `answers`, merge them into `report.judgements`."""
        answers = llm_result.get("answers")
        if not isinstance(answers, dict):
            return
        for qid, qdef in self.tier1_judge.questions.items():
            ans = answers.get(qid)
            if not isinstance(ans, dict):
                continue
            qtype = qdef["type"]
            if qtype == "choice":
                probs = ans.get("probabilities", {})
                p_arr = np.array(list(probs.values()), dtype=np.float32) if probs else np.array([1.0], dtype=np.float32)
                conf = float(ans.get("confidence", confidence_from_probs(p_arr, len(p_arr))))
                top1 = float(max(probs.values())) if probs else conf
                report.judgements[qid] = ChoiceResult(
                    name=qid,
                    decision_type=DecisionType.CHOICE,
                    confidence=conf,
                    top1_probability=top1,
                    choice=str(ans.get("choice", "")),
                    probabilities=probs,
                    raw_answer=ans,
                )
            elif qtype == "score":
                probs = ans.get("probabilities", {})
                p_arr = np.array(list(probs.values()), dtype=np.float32) if probs else np.array([1.0], dtype=np.float32)
                conf = float(ans.get("confidence", confidence_from_probs(p_arr, len(p_arr))))
                top1 = float(max(probs.values())) if probs else conf
                report.judgements[qid] = ScoreResult(
                    name=qid,
                    decision_type=DecisionType.SCORE,
                    confidence=conf,
                    top1_probability=top1,
                    score=float(ans.get("score", 0.0)),
                    legend=ans.get("legend", {}),
                    level_probabilities=probs,
                    raw_answer=ans,
                )
            elif qtype == "noul":
                p_true = float(ans.get("noul", 0.5))
                p_arr = np.array([1.0 - p_true, p_true], dtype=np.float32)
                conf = float(ans.get("confidence", round(confidence_from_probs(p_arr, 2), 4)))
                report.judgements[qid] = NoulResult(
                    name=qid,
                    decision_type=DecisionType.NOUL,
                    confidence=conf,
                    top1_probability=float(max(p_true, 1.0 - p_true)),
                    holds=(p_true >= 0.5),
                    prob_true=p_true,
                    raw_answer=ans,
                )

    def evaluate(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        metadata: Optional[Dict[str, Any]] = None,
        **predict_kwargs: Any,
    ) -> EvaluationReport:
        """Run speculative cascaded evaluation."""
        self.stats["total_evals"] += 1

        # Step 0: Direct modality gate for image/audio/video states
        if self.escalate_multimodal and self.tier2_llm_judge_fn is not None and has_unsupported_modality(state):
            self.stats["tier2_escalated"] += 1
            self.stats["multimodal_escalated"] += 1
            t_llm0 = time.perf_counter()
            llm_result = self.tier2_llm_judge_fn(state, self.tier1_judge.questions)
            llm_latency_ms = (time.perf_counter() - t_llm0) * 1000.0
            in_toks = int(llm_result.get("usage", {}).get("input_tokens", 512))
            cost = float(llm_result.get("cost_usd", self._estimate_baseline_cost(in_toks)))
            self.stats["total_input_tokens"] += in_toks
            self.stats["tier2_input_tokens"] += in_toks
            self.stats["total_latency_ms"] += llm_latency_ms
            self.stats["total_cost_usd"] += cost
            meta = dict(metadata or {})
            meta.update(
                {
                    "speculative_tier": "tier2_llm",
                    "escalated_to_llm": True,
                    "escalation_reason": "unsupported_modality",
                    "routing_metric": self.routing_metric,
                    "confidence_threshold": self.confidence_threshold,
                    "tier2_llm_result": llm_result,
                }
            )
            rep = EvaluationReport(
                state=state,
                judgements={},
                latency_ms=round(llm_latency_ms, 2),
                input_tokens=in_toks,
                output_tokens=int(llm_result.get("usage", {}).get("output_tokens", 0)),
                model="cascaded (tier2=multimodal-judge)",
                backend="Tier2Escalation",
                cost_usd=cost,
                metadata=meta,
            )
            self._apply_tier2_answers(rep, llm_result)
            return rep

        # Step 1: Run fast local Laya judge
        report = self.tier1_judge.evaluate(state, metadata=metadata, **predict_kwargs)
        routing_score = self._compute_routing_score(report)
        baseline_cost = self._estimate_baseline_cost(report.input_tokens)
        self.stats["total_input_tokens"] += report.input_tokens

        report.metadata["speculative_tier"] = "tier1_laya"
        report.metadata["average_confidence"] = report.average_confidence
        report.metadata["average_top1_probability"] = report.average_top1_probability
        report.metadata["routing_metric"] = self.routing_metric
        report.metadata["routing_score"] = routing_score
        report.metadata["confidence_threshold"] = self.confidence_threshold

        # Step 2: Check calibrated routing threshold
        if routing_score >= self.confidence_threshold or self.tier2_llm_judge_fn is None:
            self.stats["tier1_resolved"] += 1
            self.stats["total_cost_saved_usd"] += baseline_cost
            self.stats["total_latency_ms"] += report.latency_ms
            report.metadata["escalated_to_llm"] = False
            return report

        # Step 3: Low confidence -> Escalate to Tier 2 judge
        logger.info(
            "Low %s score (%.2f < %.2f). Escalating state to Tier 2 judge...",
            self.routing_metric,
            routing_score,
            self.confidence_threshold,
        )
        self.stats["tier2_escalated"] += 1
        self.stats["tier2_input_tokens"] += report.input_tokens

        t_llm0 = time.perf_counter()
        llm_result = self.tier2_llm_judge_fn(state, self.tier1_judge.questions)
        llm_latency_ms = (time.perf_counter() - t_llm0) * 1000.0

        cost = float(llm_result.get("cost_usd", baseline_cost))
        total_latency = report.latency_ms + llm_latency_ms
        self.stats["total_latency_ms"] += total_latency
        self.stats["total_cost_usd"] += cost

        self._apply_tier2_answers(report, llm_result)
        report.latency_ms = round(total_latency, 2)
        report.cost_usd = cost
        report.model = f"cascaded (tier1={report.model}, tier2=llm-judge)"
        report.metadata["speculative_tier"] = "tier2_llm"
        report.metadata["escalated_to_llm"] = True
        report.metadata["escalation_reason"] = "low_confidence"
        report.metadata["tier2_llm_result"] = llm_result

        return report

    def evaluate_batch(
        self,
        states: List[Union[str, Dict[str, Any], List[Any]]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **predict_kwargs: Any,
    ) -> List[EvaluationReport]:
        """Batched speculative cascade: batch Tier 1 first, escalate only low-confidence items."""
        if not states:
            return []

        results: List[Optional[EvaluationReport]] = [None] * len(states)
        text_indices: List[int] = []
        text_states: List[Union[str, Dict[str, Any], List[Any]]] = []
        text_metas: List[Dict[str, Any]] = []

        for idx, st in enumerate(states):
            meta = dict(metadatas[idx]) if metadatas and idx < len(metadatas) else {}
            if self.escalate_multimodal and self.tier2_llm_judge_fn is not None and has_unsupported_modality(st):
                results[idx] = self.evaluate(st, metadata=meta, **predict_kwargs)
            else:
                text_indices.append(idx)
                text_states.append(st)
                text_metas.append(meta)

        if text_states:
            t1_reports = self.tier1_judge.evaluate_batch(text_states, metadatas=text_metas, **predict_kwargs)
            for orig_idx, st, report in zip(text_indices, text_states, t1_reports):
                self.stats["total_evals"] += 1
                self.stats["total_input_tokens"] += report.input_tokens
                routing_score = self._compute_routing_score(report)
                baseline_cost = self._estimate_baseline_cost(report.input_tokens)

                report.metadata["speculative_tier"] = "tier1_laya"
                report.metadata["average_confidence"] = report.average_confidence
                report.metadata["average_top1_probability"] = report.average_top1_probability
                report.metadata["routing_metric"] = self.routing_metric
                report.metadata["routing_score"] = routing_score
                report.metadata["confidence_threshold"] = self.confidence_threshold

                if routing_score >= self.confidence_threshold or self.tier2_llm_judge_fn is None:
                    self.stats["tier1_resolved"] += 1
                    self.stats["total_cost_saved_usd"] += baseline_cost
                    self.stats["total_latency_ms"] += report.latency_ms
                    report.metadata["escalated_to_llm"] = False
                    results[orig_idx] = report
                else:
                    self.stats["tier2_escalated"] += 1
                    self.stats["tier2_input_tokens"] += report.input_tokens
                    t_llm0 = time.perf_counter()
                    llm_result = self.tier2_llm_judge_fn(st, self.tier1_judge.questions)
                    llm_latency_ms = (time.perf_counter() - t_llm0) * 1000.0
                    cost = float(llm_result.get("cost_usd", baseline_cost))
                    total_latency = report.latency_ms + llm_latency_ms
                    self.stats["total_latency_ms"] += total_latency
                    self.stats["total_cost_usd"] += cost
                    self._apply_tier2_answers(report, llm_result)
                    report.latency_ms = round(total_latency, 2)
                    report.cost_usd = cost
                    report.model = f"cascaded (tier1={report.model}, tier2=llm-judge)"
                    report.metadata["speculative_tier"] = "tier2_llm"
                    report.metadata["escalated_to_llm"] = True
                    report.metadata["escalation_reason"] = "low_confidence"
                    report.metadata["tier2_llm_result"] = llm_result
                    results[orig_idx] = report

        return [r for r in results if r is not None]

    def summary(self) -> Dict[str, Any]:
        """Summary of cascade performance and cost savings."""
        total = max(1, self.stats["total_evals"])
        tier1_pct = (self.stats["tier1_resolved"] / total) * 100.0
        escalation_pct = (self.stats["tier2_escalated"] / total) * 100.0

        return {
            "total_evaluations": self.stats["total_evals"],
            "routing_metric": self.routing_metric,
            "confidence_threshold": self.confidence_threshold,
            "tier1_resolved_count": self.stats["tier1_resolved"],
            "tier1_resolved_percent": round(tier1_pct, 1),
            "tier2_escalated_count": self.stats["tier2_escalated"],
            "tier2_escalation_percent": round(escalation_pct, 1),
            "multimodal_escalated_count": self.stats["multimodal_escalated"],
            "total_input_tokens": self.stats["total_input_tokens"],
            "tier2_input_tokens": self.stats["tier2_input_tokens"],
            "total_cost_usd": round(self.stats["total_cost_usd"], 6),
            "total_cost_saved_usd": round(self.stats["total_cost_saved_usd"], 6),
            "average_latency_ms": round(self.stats["total_latency_ms"] / total, 2),
        }
