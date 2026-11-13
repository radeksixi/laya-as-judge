"""Example 06: Decision Benchmark Suite — Laya-Omega v0.4.0 (9 Benchmarks, Top-1 ECE & Total Price).

Demonstrates evaluating Laya / Laya-Omega (used alongside https://sixi.ai) across all 9 benchmarks
from the Gemini Decision 1 comparative evaluation table:
  - JevBench, Amazon ESCI (4-class Macro-F1), New Yorker Caption, Mind2Web Multimodal,
    PubMedQA, Global-MMLU-Lite, When2Call, MMAU Audio, and Video-MME (w/o subtitles)
  - Plus pooled 10-fold out-of-fold calibrated Top-1 ECE and Total Price (USD).

Usage:
  - Offline deterministic CI smoke test:
      python examples/06_decision_benchmark_suite.py
  - Live Hugging Face local Laya run:
      python examples/06_decision_benchmark_suite.py --hf
  - Full Hybrid Laya-Omega 3-Tier Cascade (requires GEMINI_API_KEY + local CUDA/CPU weights):
      python examples/06_decision_benchmark_suite.py --hf --omega --limit 250
"""

from __future__ import annotations

import sys

from laya_as_judge import (
    CostTracker,
    DecisionBenchmarkSuite,
    GeminiDecisionEngine,
    OmegaDecisionEngine,
    VegaBackend,
    get_engine,
)


def main() -> None:
    use_hf = "--hf" in sys.argv
    use_omega = "--omega" in sys.argv
    source = "hf" if use_hf else "fixture"
    limit = 50 if use_hf else 10
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])

    if use_omega:
        tracker = CostTracker(max_budget_usd=2.00)
        local_laya = get_engine(backend="torch", model_id="convaiinnovations/laya")
        local_vega = VegaBackend(model_id="0.8b", batch_size=32)
        tier2 = GeminiDecisionEngine(
            model_id="gemini-3.5-flash-lite",
            thinking_level="low",
            cost_tracker=tracker,
            max_workers=12,
        )
        engine = OmegaDecisionEngine(
            local_laya=local_laya,
            local_vega=local_vega,
            tier2_engine=tier2,
            cost_tracker=tracker,
        )
        suite = DecisionBenchmarkSuite(engine=engine, batch_size=32, fit_disjoint_ttt=use_hf)
    else:
        suite = DecisionBenchmarkSuite(backend="auto")

    print("=" * 90)
    print(f"Example 06: Decision Benchmark Suite (backend={type(suite.engine).__name__}, source={source}, limit={limit})")
    print("=" * 90)

    summary = suite.run_suite(
        benchmarks=DecisionBenchmarkSuite.ALL_TABLE_BENCHMARKS,
        limit_per_benchmark=limit,
        source=source,
        calibration_split=0.25,
        seed=42,
    )

    print(f"\n{'Benchmark':<22} | {'Metric':<10} | {'Score':>8} | {'Acc@80%':>8} | {'Cal ECE':>9} | {'Cost USD':>9}")
    print("-" * 90)
    for r in summary.results:
        if r.status != "completed":
            print(f"{r.display_name:<22} | {r.primary_metric_name:<10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>9} | {r.status}")
        else:
            sc = f"{(r.primary_score or 0.0) * 100:.2f}%"
            c80 = f"{(r.accuracy_at_80pct_coverage or 0.0) * 100:.1f}%"
            ece = f"{(r.calibrated_top1_ece if r.calibrated_top1_ece is not None else (r.top1_ece or 0.0)):.4f}"
            cost_s = f"${r.actual_cost_usd:.4f}"
            print(f"{r.display_name:<22} | {r.primary_metric_name:<10} | {sc:>8} | {c80:>8} | {ece:>9} | {cost_s:>9}")

    print("-" * 90)
    print(f"Pooled Top-1 ECE (raw):            {summary.pooled_top1_ece:.4f}")
    if summary.pooled_calibrated_top1_ece is not None:
        print(f"Pooled Top-1 ECE (10-fold OOF):    {summary.pooled_calibrated_top1_ece:.4f}")
    print(f"Pooled Brier / AURC:               {summary.pooled_brier:.4f} / {summary.pooled_aurc:.4f}")
    print(f"Total Questions / Tokens (in/out): {summary.total_questions_evaluated} / {summary.total_input_tokens:,} / {summary.total_output_tokens:,}")

    print("\nTotal Price Comparison (USD):")
    for profile_key, usd in summary.total_price_usd.items():
        print(f"  - {profile_key:<36}: ${usd:.6f}")


if __name__ == "__main__":
    main()
