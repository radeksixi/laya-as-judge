"""Benchmark comparison engine between Laya-as-a-Judge and LLM-as-a-Judge / Decision APIs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .calibration import (
    PRICING_PROFILES,
    accuracy_score,
    estimate_suite_prices,
    macro_f1_score,
    top1_ece,
)
from .judges.base import BaseJudge
from .types import ChoiceResult, EvaluationReport, NoulResult, ScoreResult


@dataclass
class BenchmarkComparison:
    """Detailed side-by-side benchmark comparison including Top-1 ECE and Total Price."""

    sample_count: int
    laya_p50_latency_ms: float
    laya_p95_latency_ms: float
    laya_throughput_qps: float
    laya_output_tokens: int
    laya_cost_usd: float
    laya_schema_failures: int

    llm_p50_latency_ms: float
    llm_p95_latency_ms: float
    llm_throughput_qps: float
    llm_output_tokens: int
    llm_cost_usd: float
    llm_schema_failures: int

    speedup_factor: float
    cost_reduction_factor: str
    laya_backend: str = "unknown"
    llm_baseline_simulated: bool = False
    laya_input_tokens: int = 0
    laya_accuracy: Optional[float] = None
    laya_macro_f1: Optional[float] = None
    laya_top1_ece: Optional[float] = None
    total_price_comparison_usd: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        laya_dict: Dict[str, Any] = {
            "p50_latency_ms": self.laya_p50_latency_ms,
            "p95_latency_ms": self.laya_p95_latency_ms,
            "throughput_qps": self.laya_throughput_qps,
            "input_tokens": self.laya_input_tokens,
            "output_tokens": self.laya_output_tokens,
            "cost_usd": self.laya_cost_usd,
            "schema_failure_rate": f"{(self.laya_schema_failures / self.sample_count) * 100:.1f}%",
        }
        if self.laya_accuracy is not None:
            laya_dict["accuracy"] = round(self.laya_accuracy, 4)
        if self.laya_macro_f1 is not None:
            laya_dict["macro_f1"] = round(self.laya_macro_f1, 4)
        if self.laya_top1_ece is not None:
            laya_dict["top1_ece"] = round(self.laya_top1_ece, 4)

        out: Dict[str, Any] = {
            "sample_count": self.sample_count,
            "speedup_factor": f"{self.speedup_factor:.1f}x",
            "cost_reduction": self.cost_reduction_factor,
            "laya_backend": self.laya_backend,
            "llm_baseline_simulated": self.llm_baseline_simulated,
            "laya": laya_dict,
            "llm_judge": {
                "p50_latency_ms": self.llm_p50_latency_ms,
                "p95_latency_ms": self.llm_p95_latency_ms,
                "throughput_qps": self.llm_throughput_qps,
                "output_tokens": self.llm_output_tokens,
                "cost_usd": self.llm_cost_usd,
                "schema_failure_rate": f"{(self.llm_schema_failures / self.sample_count) * 100:.1f}%",
            },
        }
        if self.total_price_comparison_usd:
            out["total_price_usd"] = self.total_price_comparison_usd
        return out


def _extract_primary_prediction(report: EvaluationReport, target_question: Optional[str] = None) -> tuple[Any, float]:
    """Extract primary predicted label and Top-1 probability from an EvaluationReport."""
    if not report.judgements:
        return None, 0.0
    qid = target_question or next(iter(report.judgements.keys()))
    j = report.judgements[qid]
    top1 = float(j.top1_probability if j.top1_probability is not None else j.confidence)
    if isinstance(j, ChoiceResult):
        return j.choice, top1
    if isinstance(j, NoulResult):
        return j.holds, top1
    if isinstance(j, ScoreResult):
        return int(round(j.score)), top1
    return None, top1


class BenchmarkRunner:
    """Runs head-to-head comparison between Laya and LLM-as-a-Judge / cloud Decision APIs."""

    def __init__(
        self,
        judge: BaseJudge,
        llm_judge_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ):
        self.judge = judge
        self.llm_judge_fn = llm_judge_fn

    def run(
        self,
        samples: List[Any],
        simulate_llm_if_missing: bool = True,
        gold_labels: Optional[Sequence[Any]] = None,
        target_question: Optional[str] = None,
        batch_size: int = 1,
    ) -> BenchmarkComparison:
        """Run benchmark on test samples with optional ground-truth accuracy & Top-1 ECE."""
        n = len(samples)
        if n == 0:
            raise ValueError("Benchmark samples list cannot be empty")
        if gold_labels is not None and len(gold_labels) != n:
            raise ValueError("gold_labels length must match samples length")

        laya_latencies: List[float] = []
        laya_input_tokens = 0
        laya_output_tokens = 0
        laya_schema_failures = 0
        reports: List[EvaluationReport] = []

        t0_laya = time.perf_counter()
        if batch_size > 1:
            for start in range(0, n, batch_size):
                chunk = samples[start : start + batch_size]
                chunk_reps = self.judge.evaluate_batch(chunk)
                reports.extend(chunk_reps)
        else:
            for sample in samples:
                reports.append(self.judge.evaluate(sample))
        total_laya_time = time.perf_counter() - t0_laya

        for rep in reports:
            laya_latencies.append(rep.latency_ms)
            laya_input_tokens += rep.input_tokens
            laya_output_tokens += rep.output_tokens
            if len(rep.judgements) != len(self.judge.questions):
                laya_schema_failures += 1

        laya_p50 = float(np.percentile(laya_latencies, 50))
        laya_p95 = float(np.percentile(laya_latencies, 95))
        laya_qps = n / max(0.001, total_laya_time)
        laya_cost = 0.0

        acc_val: Optional[float] = None
        f1_val: Optional[float] = None
        ece_val: Optional[float] = None
        if gold_labels is not None:
            preds: List[Any] = []
            top1_probs: List[float] = []
            for rep in reports:
                pred, p_max = _extract_primary_prediction(rep, target_question=target_question)
                preds.append(pred)
                top1_probs.append(p_max)
            correct_arr = [1 if yp == yt else 0 for yp, yt in zip(preds, gold_labels)]
            acc_val = accuracy_score(gold_labels, preds)
            f1_val = macro_f1_score(gold_labels, preds)
            ece_val = top1_ece(top1_probs, correct_arr)

        # 2. Benchmark LLM Judge
        llm_latencies: List[float] = []
        llm_output_tokens = 0
        llm_schema_failures = 0
        llm_cost = 0.0

        if self.llm_judge_fn:
            t0_llm = time.perf_counter()
            for sample in samples:
                t_item = time.perf_counter()
                try:
                    res = self.llm_judge_fn(sample)
                    llm_latencies.append((time.perf_counter() - t_item) * 1000.0)
                    llm_output_tokens += res.get("output_tokens", 250)
                    llm_cost += res.get("cost_usd", 0.03)
                except Exception:
                    llm_schema_failures += 1
                    llm_latencies.append((time.perf_counter() - t_item) * 1000.0)
            total_llm_time = time.perf_counter() - t0_llm
        else:
            # SIMULATED baseline (no LLM is called): synthetic numbers drawn from assumed
            # typical figures for a hosted autoregressive LLM judge.
            rng = np.random.default_rng(42)
            llm_latencies = (rng.normal(loc=1850.0, scale=350.0, size=n)).clip(min=900.0).tolist()
            llm_output_tokens = int(n * 240)
            llm_cost = round(
                PRICING_PROFILES["frontier_llm_judge"].compute_cost(
                    laya_input_tokens, output_tokens=llm_output_tokens, eval_count=n
                ),
                4,
            )
            if llm_cost == 0.0:
                llm_cost = round(n * 0.003, 4)
            llm_schema_failures = int(n * 0.065)
            total_llm_time = sum(llm_latencies) / 1000.0

        llm_p50 = float(np.percentile(llm_latencies, 50))
        llm_p95 = float(np.percentile(llm_latencies, 95))
        llm_qps = n / max(0.001, total_llm_time)
        speedup = llm_p50 / max(0.1, laya_p50)

        price_table = estimate_suite_prices(
            total_input_tokens=laya_input_tokens,
            eval_count=n,
            cascade_escalation_rate=0.20,
        )

        return BenchmarkComparison(
            sample_count=n,
            laya_p50_latency_ms=round(laya_p50, 2),
            laya_p95_latency_ms=round(laya_p95, 2),
            laya_throughput_qps=round(laya_qps, 1),
            laya_output_tokens=laya_output_tokens,
            laya_cost_usd=laya_cost,
            laya_schema_failures=laya_schema_failures,
            llm_p50_latency_ms=round(llm_p50, 2),
            llm_p95_latency_ms=round(llm_p95, 2),
            llm_throughput_qps=round(llm_qps, 1),
            llm_output_tokens=llm_output_tokens,
            llm_cost_usd=llm_cost,
            llm_schema_failures=llm_schema_failures,
            speedup_factor=round(speedup, 1),
            cost_reduction_factor="100% ($0.00 local vs cloud APIs)",
            laya_backend=type(self.judge.engine).__name__,
            llm_baseline_simulated=self.llm_judge_fn is None,
            laya_input_tokens=laya_input_tokens,
            laya_accuracy=acc_val,
            laya_macro_f1=f1_val,
            laya_top1_ece=ece_val,
            total_price_comparison_usd=price_table,
        )
