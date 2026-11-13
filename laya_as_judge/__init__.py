"""Laya-as-a-Judge: Fast, calibrated, token-free LLM & AI agent evaluation.

Uses Laya non-autoregressive typed decision models instead of slow, expensive,
and fragile LLM-as-a-Judge.
"""

from .batch import BatchEvaluator
from .benchmark import BenchmarkComparison, BenchmarkRunner
from .calibration import (
    PRICING_PROFILES,
    CalibratedJudge,
    ClassMultiplierOptimizer,
    MetaCalibrator,
    PricingProfile,
    TemperatureScaler,
    accuracy_score,
    aurc,
    brier_score,
    estimate_suite_prices,
    macro_f1_score,
    negative_log_likelihood,
    oof_calibrate_pooled,
    reconstruct_top1_calibrated_probs,
    selective_accuracy_at_coverage,
    top1_ece,
)
from .cascade import CascadedJudge, has_unsupported_modality
from .engine import BaseDecisionEngine, EmulatorBackend, VegaBackend, get_engine
from .omega import OmegaDecisionEngine
from .tier2_gemini import BudgetExceededError, CostTracker, GeminiDecisionEngine
from .eval_suite import (
    BENCHMARK_SPECS,
    BenchmarkSpec,
    BenchmarkTaskResult,
    DecisionBenchmarkSuite,
    SuiteEvaluationSummary,
    load_benchmark_cases,
)
from .judges import (
    AgentTrajectoryJudge,
    AnswerRelevanceJudge,
    BaseJudge,
    CustomJudge,
    CustomJudgeBuilder,
    FaithfulnessJudge,
    InstructionAdherenceJudge,
    PairwiseComparisonJudge,
    SafetyGuardJudge,
)
from .types import (
    BatchEvaluationReport,
    ChoiceResult,
    DecisionType,
    EvaluationReport,
    JudgementResult,
    NoulResult,
    ScoreResult,
)

__version__ = "0.4.0"

__all__ = [
    # Judges
    "BaseJudge",
    "FaithfulnessJudge",
    "AnswerRelevanceJudge",
    "InstructionAdherenceJudge",
    "SafetyGuardJudge",
    "PairwiseComparisonJudge",
    "AgentTrajectoryJudge",
    "CustomJudge",
    "CustomJudgeBuilder",
    "CalibratedJudge",
    # Architecture & Workflows
    "CascadedJudge",
    "OmegaDecisionEngine",
    "GeminiDecisionEngine",
    "CostTracker",
    "BudgetExceededError",
    "has_unsupported_modality",
    "BatchEvaluator",
    "BenchmarkRunner",
    "BenchmarkComparison",
    "DecisionBenchmarkSuite",
    "SuiteEvaluationSummary",
    "BenchmarkTaskResult",
    "BenchmarkSpec",
    "BENCHMARK_SPECS",
    "load_benchmark_cases",
    # Calibration & Cost Metrics
    "top1_ece",
    "brier_score",
    "negative_log_likelihood",
    "aurc",
    "selective_accuracy_at_coverage",
    "accuracy_score",
    "macro_f1_score",
    "TemperatureScaler",
    "MetaCalibrator",
    "ClassMultiplierOptimizer",
    "oof_calibrate_pooled",
    "reconstruct_top1_calibrated_probs",
    "PricingProfile",
    "PRICING_PROFILES",
    "estimate_suite_prices",
    # Engine & Factory
    "get_engine",
    "BaseDecisionEngine",
    "EmulatorBackend",
    "VegaBackend",
    # Types & Reports
    "DecisionType",
    "JudgementResult",
    "ScoreResult",
    "ChoiceResult",
    "NoulResult",
    "EvaluationReport",
    "BatchEvaluationReport",
]
