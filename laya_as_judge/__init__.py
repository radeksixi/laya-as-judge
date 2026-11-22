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
from .engine import (
    BaseDecisionEngine,
    EmulatorBackend,
    SystemOneHTTPBackend,
    VegaBackend,
    get_engine,
    normalize_decision_questions,
)
from .judged_decision import (
    DEFAULT_DECISION_AUDIT_QUESTIONS,
    DEFAULT_SECURITY_POLICY_RULES,
    ActionGate,
    AuditPolicy,
    JudgedDecisionEngine,
    JudgedDecisionReport,
    spoofed_audit_guard_rule,
)
from .omega import OmegaDecisionEngine
from .profiles import (
    HARDWARE_PROFILES,
    HardwareProfile,
    build_local_profile_engines,
    get_hardware_profile,
)
from .server import create_app
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

__version__ = "0.7.0"

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
    "JudgedDecisionEngine",
    "JudgedDecisionReport",
    "ActionGate",
    "AuditPolicy",
    "DEFAULT_DECISION_AUDIT_QUESTIONS",
    "DEFAULT_SECURITY_POLICY_RULES",
    "spoofed_audit_guard_rule",
    "HardwareProfile",
    "HARDWARE_PROFILES",
    "get_hardware_profile",
    "build_local_profile_engines",
    "create_app",
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
    "SystemOneHTTPBackend",
    "VegaBackend",
    "normalize_decision_questions",
    # Types & Reports
    "DecisionType",
    "JudgementResult",
    "ScoreResult",
    "ChoiceResult",
    "NoulResult",
    "EvaluationReport",
    "BatchEvaluationReport",
]
