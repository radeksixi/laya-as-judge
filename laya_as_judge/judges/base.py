"""Base judge class and result transformation logic."""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..engine import BaseDecisionEngine, confidence_from_probs, get_engine
from ..types import (
    ChoiceResult,
    DecisionType,
    EvaluationReport,
    JudgementResult,
    NoulResult,
    ScoreResult,
)


class BaseJudge(abc.ABC):
    """Abstract base class for all Laya-as-a-Judge evaluators."""

    def __init__(
        self,
        engine: Optional[BaseDecisionEngine] = None,
        backend: str = "auto",
        model_id: Optional[str] = None,
        **engine_kwargs: Any,
    ):
        if engine is not None:
            self.engine = engine
        else:
            self.engine = get_engine(backend=backend, model_id=model_id, **engine_kwargs)

    @property
    @abc.abstractmethod
    def questions(self) -> Dict[str, Dict[str, Any]]:
        """Dictionary of question specifications evaluated by this judge."""
        pass

    def evaluate(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        metadata: Optional[Dict[str, Any]] = None,
        **predict_kwargs: Any,
    ) -> EvaluationReport:
        """Run all typed evaluation questions on the provided state.

        Args:
            state: The state / context / prompt to evaluate.
            metadata: Optional additional metadata to store in report.
            **predict_kwargs: Optional engine overrides (e.g. max_len, head_max_len).

        Returns:
            EvaluationReport containing typed results for every question.
        """
        if predict_kwargs:
            raw_result = self.engine.predict(state, self.questions, **predict_kwargs)
        else:
            raw_result = self.engine.predict(state, self.questions)
        return self._format_report(state, raw_result, metadata or {})

    def evaluate_batch(
        self,
        states: List[Union[str, Dict[str, Any], List[Any]]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **predict_kwargs: Any,
    ) -> List[EvaluationReport]:
        """Run all typed evaluation questions over a batch of states."""
        if not states:
            return []
        raw_results = self.engine.predict_batch(states, self.questions, **predict_kwargs)
        reports: List[EvaluationReport] = []
        for idx, (state, raw_res) in enumerate(zip(states, raw_results)):
            meta = dict(metadatas[idx]) if metadatas and idx < len(metadatas) else {}
            reports.append(self._format_report(state, raw_res, meta))
        return reports

    def _format_report(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        raw_result: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> EvaluationReport:
        """Convert raw Laya answers into strictly typed JudgementResults."""
        answers = raw_result.get("answers", {})
        judgements: Dict[str, JudgementResult] = {}

        for qid, qdef in self.questions.items():
            ans = answers.get(qid)
            if ans is None:
                continue

            qtype = qdef["type"]
            act_prob = ans.get("action", {}).get("act_probability")
            if act_prob is None:
                act_prob = ans.get("rl_agent", {}).get("act_probability")
            if act_prob is not None:
                act_prob = float(act_prob)

            if qtype == "choice":
                probs_dict = ans.get("probabilities", {})
                if "confidence" in ans:
                    conf = float(ans["confidence"])
                elif probs_dict:
                    p_arr = np.array(list(probs_dict.values()), dtype=np.float32)
                    conf = confidence_from_probs(p_arr, len(p_arr))
                else:
                    conf = 0.5
                top1 = float(
                    ans["answer_confidence"]
                    if "answer_confidence" in ans
                    else (max(probs_dict.values()) if probs_dict else conf)
                )
                judgements[qid] = ChoiceResult(
                    name=qid,
                    decision_type=DecisionType.CHOICE,
                    confidence=conf,
                    top1_probability=top1,
                    choice=ans.get("choice", ""),
                    probabilities=probs_dict,
                    action_probability=act_prob,
                    raw_answer=ans,
                )
            elif qtype == "score":
                probs_dict = ans.get("probabilities", {})
                if "confidence" in ans:
                    conf = float(ans["confidence"])
                elif probs_dict:
                    p_arr = np.array(list(probs_dict.values()), dtype=np.float32)
                    conf = confidence_from_probs(p_arr, len(p_arr))
                else:
                    conf = 0.5
                top1 = float(
                    ans["answer_confidence"]
                    if "answer_confidence" in ans
                    else (max(probs_dict.values()) if probs_dict else conf)
                )
                judgements[qid] = ScoreResult(
                    name=qid,
                    decision_type=DecisionType.SCORE,
                    confidence=conf,
                    top1_probability=top1,
                    score=float(ans.get("score", 0.0)),
                    legend=ans.get("legend", {}),
                    level_probabilities=probs_dict,
                    action_probability=act_prob,
                    raw_answer=ans,
                )
            elif qtype == "noul":
                prob_true = float(ans.get("noul", 0.5))
                if "confidence" in ans:
                    conf = float(ans["confidence"])
                else:
                    # Upstream Laya returns {"type": "noul", "noul": p_true} without "confidence"
                    conf = round(
                        confidence_from_probs(np.array([1.0 - prob_true, prob_true], dtype=np.float32), 2),
                        4,
                    )
                top1 = float(max(prob_true, 1.0 - prob_true))
                judgements[qid] = NoulResult(
                    name=qid,
                    decision_type=DecisionType.NOUL,
                    confidence=conf,
                    top1_probability=top1,
                    holds=(prob_true >= 0.5),
                    prob_true=prob_true,
                    threshold=0.5,
                    action_probability=act_prob,
                    raw_answer=ans,
                )

        usage = raw_result.get("usage", {})
        latency_ms = raw_result.get("latency_ms", 0.0)

        return EvaluationReport(
            state=state,
            judgements=judgements,
            latency_ms=latency_ms,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),  # Laya guarantee: 0
            model=raw_result.get("model", self.engine.model_id),
            backend=type(self.engine).__name__,
            cost_usd=0.0,  # Zero cloud token cost!
            metadata=metadata,
        )
