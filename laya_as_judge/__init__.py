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
from .decision_bench import (
    build_decision_bench_command,
    build_decisionbench_request,
    evaluate_decisionbench_rows,
    format_evaluation_request_issue,
    parse_decisionbench_answer,
    run_decision_bench_subprocess,
    sample_contract_rows,
    verify_systemone_contract,
)
from .dropins import DEFAULT_MODERATION_CATEGORIES, CandidateRankerEngine, ModerationAdapter
from .integrations import (
    LayaJudgedGateNode,
    LayaToolAbstainError,
    LayaToolBlockedError,
    LayaToolGateMiddleware,
    SixiGuardRule,
    SixiRegressionCase,
    SixiScannerGuardPack,
    laya_gate_router,
    laya_tool_guard,
    load_sixi_scanner_rules,
    make_decision_router,
)
from .rail_guard import RailGuardEvaluator, load_rail_guard_dataset
from .engine import (
    BaseDecisionEngine,
    EmulatorBackend,
    SystemOneHTTPBackend,
    TransformersDecisionBackend,
    VegaBackend,
    get_engine,
    normalize_decision_questions,
)
from .eval_suite import (
    BENCHMARK_SPECS,
    BenchmarkSpec,
    BenchmarkTaskResult,
    DecisionBenchmarkSuite,
    SuiteEvaluationSummary,
    load_benchmark_cases,
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
from .judges import (
    AgentTrajectoryJudge,
    AnswerRelevanceJudge,
    BaseJudge,
    CustomJudge,
    CustomJudgeBuilder,
    FaithfulnessJudge,
    InstructionAdherenceJudge,
    PairwiseComparisonJudge,
    RedTeamVerdictJudge,
    SafetyGuardJudge,
)
from .mcp_server import LayaMCPServer, mcp_tool_definitions
from .omega import OmegaDecisionEngine
from .profiles import (
    HARDWARE_PROFILES,
    HardwareProfile,
    build_local_profile_engines,
    get_hardware_profile,
)
from .server import MetricsCollector, create_app
from .tier2_gemini import BudgetExceededError, CostTracker, GeminiDecisionEngine
from .types import (
    BatchEvaluationReport,
    ChoiceResult,
    DecisionType,
    EvaluationReport,
    JudgementResult,
    NoulResult,
    ScoreResult,
)

__version__ = "1.2.0"

__all__ = [
    # Judges
    "BaseJudge",
    "FaithfulnessJudge",
    "AnswerRelevanceJudge",
    "InstructionAdherenceJudge",
    "SafetyGuardJudge",
    "PairwiseComparisonJudge",
    "AgentTrajectoryJudge",
    "RedTeamVerdictJudge",
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
    "MetricsCollector",
    "create_app",
    # Ecosystem Drop-Ins
    "CandidateRankerEngine",
    "ModerationAdapter",
    "DEFAULT_MODERATION_CATEGORIES",
    # Integrations (FastMCP & LangGraph)
    "LayaToolBlockedError",
    "LayaToolAbstainError",
    "LayaToolGateMiddleware",
    "laya_tool_guard",
    "make_decision_router",
    "LayaJudgedGateNode",
    "laya_gate_router",
    # MCP Stdio Server
    "LayaMCPServer",
    "mcp_tool_definitions",
    # Closed-Loop sixi-scanner Integration
    "SixiGuardRule",
    "SixiRegressionCase",
    "SixiScannerGuardPack",
    "load_sixi_scanner_rules",
    # External Benchmark Harnesses (DecisionBench & RAIL Guard)
    "build_decisionbench_request",
    "parse_decisionbench_answer",
    "sample_contract_rows",
    "verify_systemone_contract",
    "evaluate_decisionbench_rows",
    "build_decision_bench_command",
    "run_decision_bench_subprocess",
    "format_evaluation_request_issue",
    "RailGuardEvaluator",
    "load_rail_guard_dataset",
    # Hybrid Cloud & Benchmarks
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
    "TransformersDecisionBackend",
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
