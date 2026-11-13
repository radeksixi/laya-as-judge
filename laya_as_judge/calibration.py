"""Calibration metrics (Top-1 ECE, Brier, NLL, AURC), post-hoc Temperature Scaling, and Total Price modeling."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .engine.base import QTYPES, QTYPE_NAMES, confidence_from_probs
from .judges.base import BaseJudge
from .types import (
    ChoiceResult,
    DecisionType,
    EvaluationReport,
    JudgementResult,
    NoulResult,
    ScoreResult,
)


def top1_ece(
    top1_probs: Union[Sequence[float], np.ndarray],
    correct: Union[Sequence[bool], Sequence[int], Sequence[float], np.ndarray],
    n_bins: int = 15,
) -> float:
    """Compute Top-1 Expected Calibration Error (ECE), where lower is better.

    Matches the 15-bin Top-1 ECE protocol used in Gemini Decision 1 and JevBench:
        ECE = sum_{b=1}^B (|I_b| / N) * |mean_{i in I_b}(p_max_i) - mean_{i in I_b}(correct_i)|

    Args:
        top1_probs: Array of predicted Top-1 probabilities (max_k p_k) in [0, 1].
        correct: Binary array (1/True if Top-1 prediction equals gold label, else 0/False).
        n_bins: Number of equal-width probability bins in [0, 1] (default: 15).

    Returns:
        Top-1 ECE in [0.0, 1.0]. Returns 0.0 if input is empty.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 1:
        raise ValueError(f"n_bins must be an integer >= 1, got {n_bins!r}")
    conf = np.asarray(top1_probs, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.shape != corr.shape:
        raise ValueError(f"Shape mismatch: top1_probs {conf.shape} vs correct {corr.shape}")
    if len(conf) == 0:
        return 0.0

    conf = np.clip(conf, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if idx == 0 else conf > lo) & (conf <= hi)
        if np.any(sel):
            ece += float(np.mean(sel)) * abs(float(np.mean(conf[sel])) - float(np.mean(corr[sel])))
    return float(ece)


def brier_score(
    probs_list: Sequence[Union[Sequence[float], np.ndarray]],
    gold_indices: Sequence[int],
) -> float:
    """Compute multi-class Brier score across fixed- or variable-cardinality distributions.

    For each item n with distribution p_n over K_n options and gold index y_n:
        BS_n = sum_{k=0}^{K_n - 1} (p_{n,k} - 1[k == y_n])^2
    """
    if len(probs_list) != len(gold_indices):
        raise ValueError("probs_list and gold_indices must have equal length")
    if not probs_list:
        return 0.0

    total = 0.0
    for probs, y in zip(probs_list, gold_indices):
        p = np.asarray(probs, dtype=np.float64)
        target = np.zeros_like(p)
        if 0 <= int(y) < len(p):
            target[int(y)] = 1.0
        total += float(np.sum((p - target) ** 2))
    return total / len(probs_list)


def negative_log_likelihood(
    probs_list: Sequence[Union[Sequence[float], np.ndarray]],
    gold_indices: Sequence[int],
    eps: float = 1e-12,
) -> float:
    """Compute mean Negative Log-Likelihood (cross-entropy / log-loss) over distributions."""
    if len(probs_list) != len(gold_indices):
        raise ValueError("probs_list and gold_indices must have equal length")
    if not probs_list:
        return 0.0

    total = 0.0
    for probs, y in zip(probs_list, gold_indices):
        p = np.asarray(probs, dtype=np.float64)
        p = p / max(eps, float(np.sum(p)))
        p_gold = float(p[int(y)]) if 0 <= int(y) < len(p) else eps
        total -= float(np.log(max(eps, min(1.0, p_gold))))
    return total / len(probs_list)


def aurc(
    confidences: Union[Sequence[float], np.ndarray],
    correct: Union[Sequence[bool], Sequence[int], Sequence[float], np.ndarray],
) -> float:
    """Area Under the Risk-Coverage Curve (AURC, lower is better).

    Measures selective prediction quality for CascadedJudge escalation thresholds.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return 0.0
    # Break ties deterministically via stable sort on negative confidence
    order = np.argsort(-conf, kind="mergesort")
    errors = 1.0 - corr[order]
    cum_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1, dtype=np.float64)
    return float(np.mean(cum_risk))


def selective_accuracy_at_coverage(
    confidences: Union[Sequence[float], np.ndarray],
    correct: Union[Sequence[bool], Sequence[int], Sequence[float], np.ndarray],
    coverage: float = 0.80,
) -> float:
    """Compute accuracy on the top `coverage` fraction of most-confident predictions."""
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return 0.0
    keep = max(1, int(round(len(conf) * coverage)))
    order = np.argsort(-conf, kind="mergesort")[:keep]
    return float(np.mean(corr[order]))


def accuracy_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Compute exact Top-1 accuracy."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return 0.0
    matches = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return float(matches / len(y_true))


def macro_f1_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None,
) -> float:
    """Compute unweighted Macro-F1 across classes (used e.g. by Amazon ESCI)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return 0.0

    classes = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred), key=str)
    if not classes:
        return 0.0

    f1s: List[float] = []
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def bucket_key_for_question(decision_type: Union[str, DecisionType], num_options: int) -> str:
    """Return cardinality calibration bucket matching Laya's temperature_by_options."""
    dtype_str = decision_type.value if isinstance(decision_type, DecisionType) else str(decision_type)
    size = "2" if num_options <= 2 else "3-5" if num_options <= 5 else "6-10" if num_options <= 10 else "11+"
    return f"{dtype_str}:{size}"


def rescale_probabilities(probs: Union[Sequence[float], np.ndarray], temperature: float) -> np.ndarray:
    """Apply post-hoc temperature scaling p'_i = softmax(log(p_i) / T) in probability space."""
    p = np.asarray(probs, dtype=np.float64)
    if len(p) <= 1 or abs(temperature - 1.0) < 1e-6:
        s = float(np.sum(p))
        return (p / s) if s > 0 else np.ones_like(p) / max(1, len(p))
    t_safe = max(0.05, min(20.0, float(temperature)))
    log_p = np.log(np.clip(p, 1e-12, 1.0)) / t_safe
    log_p -= np.max(log_p)
    scaled = np.exp(log_p)
    return scaled / np.sum(scaled)


def _fit_single_temperature(
    probs_list: Sequence[np.ndarray],
    gold_indices: Sequence[int],
    temp_min: float = 0.25,
    temp_max: float = 5.0,
    l2_prior_strength: float = 1.5,
) -> float:
    """Fit scalar temperature T minimizing regularized NLL via golden-section search on log(T).

    Includes empirical Bayes L2 shrinkage `(l2_prior_strength / N) * (log T)^2` toward
    the neutral prior T = 1.0 (`log T = 0`), preventing small-sample overfitting while
    vanishing asymptotically as validation size N grows.
    """
    if not probs_list:
        return 1.0
    log_ps = [np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)) for p in probs_list]
    ys = [int(y) for y in gold_indices]
    n_items = float(len(log_ps))
    reg_weight = float(max(0.0, l2_prior_strength)) / n_items

    def nll_at_log_t(log_t: float) -> float:
        inv_t = float(np.exp(-log_t))
        loss = 0.0
        for lp, y in zip(log_ps, ys):
            z = lp * inv_t
            z_max = float(np.max(z))
            lse = z_max + float(np.log(np.sum(np.exp(z - z_max))))
            loss += lse - float(z[y])
        return (loss / n_items) + reg_weight * (log_t * log_t)

    a = float(np.log(temp_min))
    b = float(np.log(temp_max))
    # Coarse grid bracket first to guard against boundary artifacts, then golden-section to 1e-5
    grid = np.linspace(a, b, 41)
    vals = [nll_at_log_t(float(x)) for x in grid]
    best_idx = int(np.argmin(vals))
    lo = float(grid[max(0, best_idx - 1)])
    hi = float(grid[min(len(grid) - 1, best_idx + 1)])

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = nll_at_log_t(c), nll_at_log_t(d)
    for _ in range(40):
        if abs(hi - lo) < 1e-5:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = nll_at_log_t(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = nll_at_log_t(d)

    opt_t = float(np.exp((lo + hi) / 2.0))
    return float(np.clip(opt_t, temp_min, temp_max))


class TemperatureScaler:
    """Post-hoc temperature calibrator reducing Top-1 ECE and NLL without weight updates.

    Fits a global scalar temperature T and optional per-bucket temperatures
    (e.g. 'noul:2', 'choice:3-5', 'score:3-5') on a held-out validation split,
    with empirical Bayes shrinkage toward T=1.0 for small validation sets.
    """

    def __init__(
        self,
        global_temperature: float = 1.0,
        bucket_temperatures: Optional[Dict[str, float]] = None,
        min_bucket_samples: int = 25,
        l2_prior_strength: float = 1.5,
    ):
        self.global_temperature = float(global_temperature)
        self.bucket_temperatures: Dict[str, float] = dict(bucket_temperatures or {})
        self.min_bucket_samples = int(min_bucket_samples)
        self.l2_prior_strength = float(l2_prior_strength)

    def fit(
        self,
        probs_list: Sequence[Union[Sequence[float], np.ndarray]],
        gold_indices: Sequence[int],
        buckets: Optional[Sequence[str]] = None,
    ) -> TemperatureScaler:
        """Fit global and optional bucket-specific temperatures minimizing regularized NLL."""
        if len(probs_list) != len(gold_indices):
            raise ValueError("probs_list and gold_indices must have the same length")
        arrays = [np.asarray(p, dtype=np.float64) for p in probs_list]
        if len(arrays) < 5:
            self.global_temperature = 1.0
            self.bucket_temperatures = {}
            return self

        self.global_temperature = _fit_single_temperature(
            arrays, gold_indices, l2_prior_strength=self.l2_prior_strength
        )
        self.bucket_temperatures = {}

        if buckets is not None:
            if len(buckets) != len(arrays):
                raise ValueError("buckets must match probs_list length")
            grouped_p: Dict[str, List[np.ndarray]] = {}
            grouped_y: Dict[str, List[int]] = {}
            for p, y, b in zip(arrays, gold_indices, buckets):
                grouped_p.setdefault(b, []).append(p)
                grouped_y.setdefault(b, []).append(int(y))
            for b, plist in grouped_p.items():
                if len(plist) >= self.min_bucket_samples:
                    self.bucket_temperatures[b] = _fit_single_temperature(
                        plist, grouped_y[b], l2_prior_strength=self.l2_prior_strength
                    )

        return self

    def temperature_for(self, decision_type: Union[str, DecisionType], num_options: int) -> float:
        bkey = bucket_key_for_question(decision_type, num_options)
        return self.bucket_temperatures.get(bkey, self.global_temperature)

    def calibrate_judgement(self, j: JudgementResult) -> JudgementResult:
        """Return a temperature-calibrated copy of a JudgementResult."""
        if isinstance(j, ChoiceResult) and j.probabilities:
            labels = list(j.probabilities.keys())
            raw_p = np.array([j.probabilities[k] for k in labels], dtype=np.float64)
            t = self.temperature_for(DecisionType.CHOICE, len(labels))
            cal_p = rescale_probabilities(raw_p, t)
            conf = round(confidence_from_probs(cal_p, len(labels)), 4)
            top1 = round(float(np.max(cal_p)), 4)
            best_lbl = labels[int(np.argmax(cal_p))]
            return ChoiceResult(
                name=j.name,
                decision_type=DecisionType.CHOICE,
                confidence=conf,
                top1_probability=top1,
                choice=best_lbl,
                probabilities={lbl: round(float(p), 4) for lbl, p in zip(labels, cal_p)},
                action_probability=j.action_probability,
                raw_answer=j.raw_answer,
            )

        if isinstance(j, ScoreResult) and j.level_probabilities:
            keys = sorted(j.level_probabilities.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
            raw_p = np.array([j.level_probabilities[k] for k in keys], dtype=np.float64)
            t = self.temperature_for(DecisionType.SCORE, len(keys))
            cal_p = rescale_probabilities(raw_p, t)
            conf = round(confidence_from_probs(cal_p, len(keys)), 4)
            top1 = round(float(np.max(cal_p)), 4)
            exp_score = round(float(np.sum(np.arange(len(keys)) * cal_p)), 4)
            return ScoreResult(
                name=j.name,
                decision_type=DecisionType.SCORE,
                confidence=conf,
                top1_probability=top1,
                score=exp_score,
                legend=j.legend,
                level_probabilities={k: round(float(p), 4) for k, p in zip(keys, cal_p)},
                action_probability=j.action_probability,
                raw_answer=j.raw_answer,
            )

        if isinstance(j, NoulResult):
            raw_p = np.array([1.0 - j.prob_true, j.prob_true], dtype=np.float64)
            t = self.temperature_for(DecisionType.NOUL, 2)
            cal_p = rescale_probabilities(raw_p, t)
            p_true = round(float(cal_p[1]), 4)
            conf = round(confidence_from_probs(cal_p, 2), 4)
            top1 = round(float(max(p_true, 1.0 - p_true)), 4)
            return NoulResult(
                name=j.name,
                decision_type=DecisionType.NOUL,
                confidence=conf,
                top1_probability=top1,
                holds=(p_true >= j.threshold),
                prob_true=p_true,
                threshold=j.threshold,
                action_probability=j.action_probability,
                raw_answer=j.raw_answer,
            )

        return j

    def calibrate_report(self, report: EvaluationReport) -> EvaluationReport:
        """Apply temperature scaling to all judgements in an EvaluationReport."""
        new_judgements = {qid: self.calibrate_judgement(j) for qid, j in report.judgements.items()}
        meta = dict(report.metadata)
        meta["temperature_calibrated"] = True
        meta["global_temperature"] = round(self.global_temperature, 4)
        return EvaluationReport(
            state=report.state,
            judgements=new_judgements,
            latency_ms=report.latency_ms,
            input_tokens=report.input_tokens,
            output_tokens=report.output_tokens,
            model=report.model,
            backend=report.backend,
            cost_usd=report.cost_usd,
            metadata=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_temperature": round(self.global_temperature, 4),
            "bucket_temperatures": {k: round(v, 4) for k, v in self.bucket_temperatures.items()},
            "min_bucket_samples": self.min_bucket_samples,
        }


class CalibratedJudge(BaseJudge):
    """Wraps any BaseJudge with a fitted post-hoc TemperatureScaler."""

    def __init__(self, base_judge: BaseJudge, scaler: TemperatureScaler):
        self.base_judge = base_judge
        self.scaler = scaler
        super().__init__(engine=base_judge.engine)

    @property
    def questions(self) -> Dict[str, Dict[str, Any]]:
        return self.base_judge.questions

    def evaluate(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        metadata: Optional[Dict[str, Any]] = None,
        **predict_kwargs: Any,
    ) -> EvaluationReport:
        rep = self.base_judge.evaluate(state, metadata=metadata, **predict_kwargs)
        return self.scaler.calibrate_report(rep)

    def evaluate_batch(
        self,
        states: List[Union[str, Dict[str, Any], List[Any]]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **predict_kwargs: Any,
    ) -> List[EvaluationReport]:
        reps = self.base_judge.evaluate_batch(states, metadatas=metadatas, **predict_kwargs)
        return [self.scaler.calibrate_report(r) for r in reps]


@dataclass(frozen=True)
class PricingProfile:
    """Token-based pricing model for computing realistic `Total price` across eval suites.

    Reflects both zero-output-token Decision APIs (Laya local, Gemini Decision 1,
    TypeSafe Jev 1.14, GPT-6 Luna) and generative LLM-as-a-Judge baselines.
    """

    name: str
    input_usd_per_1m_tokens: float
    output_usd_per_1m_tokens: float = 0.0
    assumed_output_tokens_per_eval: int = 0

    def compute_cost(
        self,
        input_tokens: int,
        output_tokens: Optional[int] = None,
        eval_count: int = 1,
    ) -> float:
        out_toks = (
            output_tokens
            if output_tokens is not None
            else (self.assumed_output_tokens_per_eval * max(0, eval_count))
        )
        in_cost = (max(0, input_tokens) / 1_000_000.0) * self.input_usd_per_1m_tokens
        out_cost = (max(0, out_toks) / 1_000_000.0) * self.output_usd_per_1m_tokens
        return float(in_cost + out_cost)


PRICING_PROFILES: Dict[str, PricingProfile] = {
    "laya_local": PricingProfile(
        name="Laya Local (MLX / CUDA / CPU)",
        input_usd_per_1m_tokens=0.0,
        output_usd_per_1m_tokens=0.0,
        assumed_output_tokens_per_eval=0,
    ),
    "typesafe_jev_1_14": PricingProfile(
        name="TypeSafe Jev 1.14",
        input_usd_per_1m_tokens=0.08,
        output_usd_per_1m_tokens=0.0,
        assumed_output_tokens_per_eval=0,
    ),
    "gemini_decision_1": PricingProfile(
        name="Gemini Decision 1",
        input_usd_per_1m_tokens=0.10,
        output_usd_per_1m_tokens=0.0,
        assumed_output_tokens_per_eval=0,
    ),
    "gpt_6_luna_decisions": PricingProfile(
        name="GPT-6 Luna (OpenAI Decisions)",
        input_usd_per_1m_tokens=0.11,
        output_usd_per_1m_tokens=0.0,
        assumed_output_tokens_per_eval=0,
    ),
    "gemini_3_5_flash_lite": PricingProfile(
        name="Gemini 3.5 Flash-Lite",
        input_usd_per_1m_tokens=0.30,
        output_usd_per_1m_tokens=2.50,
        assumed_output_tokens_per_eval=75,
    ),
    "gemini_3_8_flash": PricingProfile(
        name="Gemini 3.8 Flash",
        input_usd_per_1m_tokens=0.75,
        output_usd_per_1m_tokens=3.75,
        assumed_output_tokens_per_eval=90,
    ),
    "frontier_llm_judge": PricingProfile(
        name="Frontier Autoregressive LLM Judge",
        input_usd_per_1m_tokens=2.50,
        output_usd_per_1m_tokens=10.00,
        assumed_output_tokens_per_eval=240,
    ),
}


def estimate_suite_prices(
    total_input_tokens: int,
    eval_count: int,
    cascade_escalation_rate: float = 0.20,
    tier2_profile_key: str = "gemini_decision_1",
) -> Dict[str, float]:
    """Compare `Total price` (USD) for a given workload across all reference profiles + Cascade."""
    prices: Dict[str, float] = {}
    for key, prof in PRICING_PROFILES.items():
        prices[key] = round(prof.compute_cost(total_input_tokens, eval_count=eval_count), 6)

    t2 = PRICING_PROFILES[tier2_profile_key]
    esc = float(np.clip(cascade_escalation_rate, 0.0, 1.0))
    escalated_tokens = int(round(total_input_tokens * esc))
    escalated_evals = int(round(eval_count * esc))
    prices[f"cascade_laya_to_{tier2_profile_key}"] = round(
        t2.compute_cost(escalated_tokens, eval_count=escalated_evals), 6
    )
    return prices


# ---------------------------------------------------------------------------
# Argmax-Preserving MetaCalibrator (Logistic + PAVA Isotonic) & F1 Optimizer
# ---------------------------------------------------------------------------


def reconstruct_top1_calibrated_probs(
    probs: Union[Sequence[float], np.ndarray],
    calibrated_top1: float,
    eps: float = 1e-5,
) -> np.ndarray:
    """Adjust Top-1 probability to `calibrated_top1` while strictly preserving `argmax(probs)`.

    Preserves exact runner-up ratios whenever `q >= r_max / (1 + r_max) + eps`, and smoothly
    blends runner-up shares toward uniform `1 / (K - 1)` when `1/K < q < r_max / (1 + r_max)`,
    eliminating upward clamping bias on multi-class (`K >= 3`) uncertain predictions while
    guaranteeing 100% invariant Top-1 accuracy and Macro-F1 across the entire `[1/K + eps, 1 - eps]` simplex!
    """
    p = np.asarray(probs, dtype=np.float64)
    k = len(p)
    if k <= 1:
        return np.ones_like(p)
    p = np.clip(p, 1e-12, 1.0)
    p = p / float(np.sum(p))
    top_idx = int(np.argmax(p))

    rem_mask = np.ones(k, dtype=bool)
    rem_mask[top_idx] = False
    rem_mass = float(np.sum(p[rem_mask]))
    if rem_mass <= 1e-12:
        shares = np.full(k - 1, 1.0 / float(k - 1), dtype=np.float64)
    else:
        shares = p[rem_mask] / rem_mass

    min_argmax_safe = (1.0 + float(k - 1) * eps) / float(k) + 1e-9
    q = float(np.clip(calibrated_top1, min_argmax_safe, 1.0 - eps))

    if k > 2:
        r_max = float(np.max(shares))
        c_max = (q - eps) / max(1e-12, 1.0 - q)
        if r_max > c_max:
            u = 1.0 / float(k - 1)
            lam = float(np.clip((r_max - c_max) / max(1e-12, r_max - u), 0.0, 1.0))
            shares = (1.0 - lam) * shares + lam * u
            shares /= float(np.sum(shares))

    out = np.empty(k, dtype=np.float64)
    out[top_idx] = q
    out[rem_mask] = (1.0 - q) * shares
    return out / float(np.sum(out))


def _top1_meta_features(probs: np.ndarray, tier_tag: str = "default") -> np.ndarray:
    """Extract invariant Top-1 uncertainty features `[1, logit(p1), p1-p2, norm_ent, 1/K, tier_indicators, slope_inter]`."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p = p / float(np.sum(p))
    k = max(2, len(p))
    sorted_p = np.sort(p)[::-1]
    p1 = float(np.clip(sorted_p[0], 1e-5, 1.0 - 1e-5))
    p2 = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
    logit_p1 = float(np.log(p1 / (1.0 - p1)))
    margin = p1 - p2
    ent_conf = confidence_from_probs(p, k)
    inv_k = 1.0 / float(k)

    tl = str(tier_tag).lower()
    is_vega = 1.0 if "vega" in tl else 0.0
    is_gemini = 1.0 if ("gemini" in tl or "tier2" in tl) else 0.0
    return np.array([1.0, logit_p1, margin, ent_conf, inv_k, is_vega, is_gemini, logit_p1 * is_gemini], dtype=np.float64)


def _fit_pava_isotonic(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact linear-time weighted Pool Adjacent Violators Algorithm (PAVA) with tie-pooling in pure NumPy."""
    # Round to 1e-5 first so tied / near-tied probabilities are pooled into a single weighted point
    xr = np.round(np.asarray(x, dtype=np.float64), 5)
    yr = np.asarray(y, dtype=np.float64)
    uniq_x, inv = np.unique(xr, return_inverse=True)
    sum_y = np.bincount(inv, weights=yr)
    counts = np.bincount(inv).astype(np.float64)

    # Stack of blocks: [sum_y, weight, sum_wx]
    blocks: List[List[float]] = []
    for xi, syi, wi in zip(uniq_x, sum_y, counts):
        blocks.append([float(syi), float(wi), float(wi * xi)])
        while len(blocks) >= 2:
            prev_mean = blocks[-2][0] / blocks[-2][1]
            curr_mean = blocks[-1][0] / blocks[-1][1]
            if prev_mean >= curr_mean:
                b_last = blocks.pop()
                blocks[-1][0] += b_last[0]
                blocks[-1][1] += b_last[1]
                blocks[-1][2] += b_last[2]
            else:
                break

    knots_x: List[float] = []
    knots_y: List[float] = []
    for sy, cnt, swx in blocks:
        val = float(np.clip(sy / cnt, 0.01, 0.995))
        knots_x.append(swx / cnt)
        knots_y.append(val)
    return np.array(knots_x, dtype=np.float64), np.array(knots_y, dtype=np.float64)


class MetaCalibrator:
    """Smooth multi-feature Platt/Logistic meta-calibrator minimizing Top-1 ECE.

    Uses an unpenalized intercept (`sum(pred) == sum(y)` exact global calibration),
    L2 shrinkage toward identity Platt scaling `w0 = [0, 1, 0, ...]`, and Armijo
    backtracking line-search Newton optimization for guaranteed monotonic convergence.
    Guarantees 0.0000 change to Top-1 accuracy and Macro-F1 via `reconstruct_top1_calibrated_probs`.
    """

    def __init__(self, l2_reg: float = 3.0, use_isotonic: bool = False, isotonic_min_samples: int = 1200):
        self.l2_reg = float(l2_reg)
        self.use_isotonic = bool(use_isotonic)
        self.isotonic_min_samples = int(isotonic_min_samples)
        # Default identity weights: w = [0, 1, 0, ...] maps sigmoid(1 * logit_p1) -> p1
        self.weights = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._iso_x: Optional[np.ndarray] = None
        self._iso_y: Optional[np.ndarray] = None
        self._iso_blend: float = 0.0

    def fit(
        self,
        probs_list: Sequence[Union[Sequence[float], np.ndarray]],
        gold_indices: Sequence[int],
        tier_tags: Optional[Sequence[str]] = None,
    ) -> MetaCalibrator:
        """Fit damped-Newton L2-regularized logistic calibration with exact mean preservation."""
        n = len(probs_list)
        if n != len(gold_indices):
            raise ValueError("probs_list and gold_indices must have equal length")
        if n < 10:
            return self

        tags = list(tier_tags) if tier_tags is not None else ["default"] * n
        X = np.vstack([_top1_meta_features(np.asarray(p, dtype=np.float64), t) for p, t in zip(probs_list, tags)])
        y = np.array(
            [1.0 if int(np.argmax(p)) == int(g) else 0.0 for p, g in zip(probs_list, gold_indices)],
            dtype=np.float64,
        )

        # Prior vector w0 = [0, 1, 0, ...] so small samples shrink smoothly toward raw p1
        w0 = np.zeros(X.shape[1], dtype=np.float64)
        w0[1] = 1.0
        w = w0.copy()
        reg_diag = np.full(X.shape[1], self.l2_reg, dtype=np.float64)
        reg_diag[0] = 1e-6  # unpenalized intercept guarantees mean(pred) == mean(y)
        reg_diag[1] = min(1.0, self.l2_reg * 0.4)  # gentle prior on Platt logit_p1 slope

        def _loss(wv: np.ndarray) -> float:
            zv = np.clip(X @ wv, -20.0, 20.0)
            nll = float(np.sum(np.logaddexp(0.0, zv) - y * zv))
            pen = 0.5 * float(np.sum(reg_diag * ((wv - w0) ** 2)))
            return nll + pen

        cur_loss = _loss(w)
        for _ in range(30):
            z = np.clip(X @ w, -20.0, 20.0)
            pred = 1.0 / (1.0 + np.exp(-z))
            grad = X.T @ (pred - y) + reg_diag * (w - w0)
            s = np.clip(pred * (1.0 - pred), 1e-5, 0.25)
            H = (X.T * s) @ X + np.diag(reg_diag)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break

            slope = float(np.dot(grad, step))
            if slope <= 1e-12:
                break

            alpha = 1.0
            accepted = False
            while alpha >= 1e-4:
                w_try = w - alpha * step
                next_loss = _loss(w_try)
                if next_loss <= cur_loss - 1e-4 * alpha * slope:
                    w = w_try
                    cur_loss = next_loss
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted or float(np.max(np.abs(alpha * step))) < 1e-7:
                break

        self.weights = w
        if self.use_isotonic and n >= self.isotonic_min_samples:
            z_cal = np.clip(X @ self.weights, -20.0, 20.0)
            p_log = 1.0 / (1.0 + np.exp(-z_cal))
            self._iso_x, self._iso_y = _fit_pava_isotonic(p_log, y)
            self._iso_blend = float(min(0.25, n / 4000.0))
        else:
            self._iso_x, self._iso_y, self._iso_blend = None, None, 0.0

        return self

    def predict_top1_confidence(self, probs: Union[Sequence[float], np.ndarray], tier_tag: str = "default") -> float:
        feat = _top1_meta_features(np.asarray(probs, dtype=np.float64), tier_tag)
        z = float(np.clip(np.dot(feat, self.weights), -20.0, 20.0))
        p_log = float(1.0 / (1.0 + np.exp(-z)))
        if self._iso_x is not None and self._iso_y is not None and len(self._iso_x) >= 1 and self._iso_blend > 0.0:
            p_iso = float(np.interp(p_log, self._iso_x, self._iso_y))
            return float((1.0 - self._iso_blend) * p_log + self._iso_blend * p_iso)
        return p_log

    def calibrate_probs(
        self,
        probs: Union[Sequence[float], np.ndarray],
        tier_tag: str = "default",
    ) -> np.ndarray:
        """Return calibrated simplex preserving `argmax(probs)`."""
        q = self.predict_top1_confidence(probs, tier_tag=tier_tag)
        return reconstruct_top1_calibrated_probs(probs, q)


def oof_calibrate_pooled(
    probs_list: Sequence[Union[Sequence[float], np.ndarray]],
    gold_indices: Sequence[int],
    tier_tags: Optional[Sequence[str]] = None,
    n_folds: int = 5,
    seed: int = 42,
    l2_reg: float = 2.5,
) -> List[np.ndarray]:
    """Strictly out-of-fold K-fold cross-calibration with zero test-sample leakage.

    Each item `i` in fold `f` is calibrated exclusively by a `MetaCalibrator` fitted on the
    other `K - 1` folds (`train_idx ∩ val_idx = ∅`). Preserves every Top-1 prediction 100%.
    """
    n = len(probs_list)
    if n == 0:
        return []
    if n < max(12, n_folds * 2):
        return [np.asarray(p, dtype=np.float64) for p in probs_list]

    tags = list(tier_tags) if tier_tags is not None else ["default"] * n
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, int(max(2, n_folds)))

    out: List[Optional[np.ndarray]] = [None] * n
    for f_idx in range(len(folds)):
        val_idx = folds[f_idx]
        val_set = set(int(x) for x in val_idx)
        trn_idx = [int(i) for i in perm if int(i) not in val_set]

        cal = MetaCalibrator(l2_reg=l2_reg).fit(
            [probs_list[i] for i in trn_idx],
            [gold_indices[i] for i in trn_idx],
            tier_tags=[tags[i] for i in trn_idx],
        )
        for vi in val_idx:
            idx_int = int(vi)
            out[idx_int] = cal.calibrate_probs(probs_list[idx_int], tier_tag=tags[idx_int])

    return [p if p is not None else np.asarray(probs_list[i], dtype=np.float64) for i, p in enumerate(out)]


class ClassMultiplierOptimizer:
    """Coordinate-ascent post-hoc class multiplier tuning `argmax_c (m_c * p_c)` for Macro-F1.

    Must be fitted strictly on disjoint `train` / `validation` predictions (never on test!).
    """

    def __init__(self, num_classes: int, l2_shrinkage: float = 0.01):
        self.num_classes = int(num_classes)
        self.l2_shrinkage = float(l2_shrinkage)
        self.multipliers = np.ones(self.num_classes, dtype=np.float64)

    def fit(
        self,
        probs_list: Sequence[Union[Sequence[float], np.ndarray]],
        gold_indices: Sequence[int],
        n_rounds: int = 3,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> ClassMultiplierOptimizer:
        if not probs_list:
            return self
        P = np.vstack([np.asarray(p, dtype=np.float64) for p in probs_list])
        y_arr = np.array([int(g) for g in gold_indices], dtype=np.int64)
        w_arr = (
            np.asarray(sample_weights, dtype=np.float64)
            if sample_weights is not None
            else np.ones(len(y_arr), dtype=np.float64)
        )

        m = np.ones(self.num_classes, dtype=np.float64)
        grid = np.geomspace(0.68, 1.85, 35)

        def _weighted_macro_f1(preds_arr: np.ndarray) -> float:
            f1s: List[float] = []
            for c in range(self.num_classes):
                tp = float(np.sum(w_arr[(y_arr == c) & (preds_arr == c)]))
                fp = float(np.sum(w_arr[(y_arr != c) & (preds_arr == c)]))
                fn = float(np.sum(w_arr[(y_arr == c) & (preds_arr != c)]))
                denom = 2.0 * tp + fp + fn
                f1s.append((2.0 * tp / denom) if denom > 1e-12 else 0.0)
            return float(np.mean(f1s))

        def obj(mult: np.ndarray) -> float:
            preds_arr = np.argmax(P * mult[None, :], axis=1)
            f1 = _weighted_macro_f1(preds_arr)
            pen = self.l2_shrinkage * float(np.mean(np.log(mult) ** 2))
            return f1 - pen

        best_val = obj(m)
        for _ in range(max(1, n_rounds)):
            for c in range(1, self.num_classes):  # fix m[0]=1.0 for scale identifiability
                cur_best_m = m[c]
                for cand in grid:
                    m[c] = float(cand)
                    v = obj(m)
                    if v > best_val + 1e-6:
                        best_val = v
                        cur_best_m = float(cand)
                m[c] = cur_best_m

        self.multipliers = m / float(np.exp(np.mean(np.log(m))))
        return self

    def apply(self, probs: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float64)
        if len(p) != self.num_classes:
            return p
        scaled = p * self.multipliers
        s = float(np.sum(scaled))
        return (scaled / s) if s > 0 else p

