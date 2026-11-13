"""Unit tests for MetaCalibrator, ClassMultiplierOptimizer, CostTracker, GeminiDecisionEngine, and OmegaDecisionEngine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from laya_as_judge import (
    BudgetExceededError,
    ClassMultiplierOptimizer,
    CostTracker,
    DecisionBenchmarkSuite,
    GeminiDecisionEngine,
    MetaCalibrator,
    OmegaDecisionEngine,
    accuracy_score,
    macro_f1_score,
    oof_calibrate_pooled,
    reconstruct_top1_calibrated_probs,
    top1_ece,
)
from laya_as_judge.engine import BaseDecisionEngine


def test_reconstruct_top1_preserves_argmax_across_all_cardinalities():
    """Verify `reconstruct_top1_calibrated_probs` never flips argmax even at aggressive down-calibration."""
    rng = np.random.default_rng(7)
    for k in (2, 3, 4, 5, 8):
        for _ in range(40):
            raw = rng.dirichlet(np.ones(k))
            orig_best = int(np.argmax(raw))
            for target_q in (0.05, 0.20, 0.35, 0.51, 0.80, 0.99):
                cal = reconstruct_top1_calibrated_probs(raw, target_q)
                assert np.isclose(np.sum(cal), 1.0)
                assert np.all(cal >= 0.0)
                assert int(np.argmax(cal)) == orig_best


def test_meta_calibrator_and_oof_calibration_slash_ece_without_changing_accuracy():
    """Simulate a heterogeneous 2-tier system with overconfident predictions (ECE > 0.16 -> < 0.025)."""
    rng = np.random.default_rng(2026)
    n = 600
    probs_list: List[np.ndarray] = []
    gold_indices: List[int] = []
    tags: List[str] = []

    for i in range(n):
        k = 4 if i % 2 == 0 else 3
        is_tier2 = (i % 3) == 0
        # Raw top-1 confidence ~0.95, whereas true accuracy is ~0.78 on tier2 and ~0.68 on local
        true_acc = 0.78 if is_tier2 else 0.68
        p = np.full(k, 0.05 / (k - 1), dtype=np.float64)
        p[0] = 0.95
        hit = rng.random() < true_acc
        probs_list.append(p)
        gold_indices.append(0 if hit else 1)
        tags.append("omega_blended_tier2" if is_tier2 else "local_laya_vega")

    orig_preds = [int(np.argmax(p)) for p in probs_list]
    orig_acc = accuracy_score(gold_indices, orig_preds)
    raw_top1 = np.array([float(np.max(p)) for p in probs_list])
    corr = np.array([1.0 if yp == yt else 0.0 for yp, yt in zip(orig_preds, gold_indices)])
    raw_ece = top1_ece(raw_top1, corr)
    assert raw_ece > 0.15

    oof_probs = oof_calibrate_pooled(probs_list, gold_indices, tier_tags=tags, n_folds=5, seed=42)
    cal_preds = [int(np.argmax(p)) for p in oof_probs]
    assert accuracy_score(gold_indices, cal_preds) == pytest.approx(orig_acc, abs=1e-12)

    cal_top1 = np.array([float(np.max(p)) for p in oof_probs])
    cal_ece = top1_ece(cal_top1, corr)
    assert cal_ece < 0.025


def test_class_multiplier_optimizer_improves_imbalanced_macro_f1():
    """Verify Macro-F1 coordinate search recovers minority class recall on ESCI-like skewed priors."""
    rng = np.random.default_rng(99)
    # Class 1 ('Complement') has low prior so raw posterior peaks at ~0.35 while Class 3 ('Exact') sits at ~0.42
    probs_tr, gold_tr = [], []
    probs_te, gold_te = [], []
    for split_probs, split_gold in ((probs_tr, gold_tr), (probs_te, gold_te)):
        for _ in range(160):
            split_probs.append(np.array([0.03, 0.05, 0.12, 0.80]))
            split_gold.append(3)
        for _ in range(50):
            split_probs.append(np.array([0.05, 0.05, 0.65, 0.25]))
            split_gold.append(2)
        for _ in range(30):
            split_probs.append(np.array([0.75, 0.05, 0.10, 0.10]))
            split_gold.append(0)
        for _ in range(20):
            # True minority class 1 where posterior is 0.36 vs 0.44 for class 3
            noise = rng.uniform(-0.02, 0.02)
            split_probs.append(np.array([0.08, 0.36 + noise, 0.12, 0.44 - noise]))
            split_gold.append(1)

    raw_preds_te = [int(np.argmax(p)) for p in probs_te]
    raw_f1 = macro_f1_score(gold_te, raw_preds_te, labels=[0, 1, 2, 3])

    opt = ClassMultiplierOptimizer(num_classes=4, l2_shrinkage=0.005).fit(probs_tr, gold_tr)
    adj_preds_te = [int(np.argmax(opt.apply(p))) for p in probs_te]
    adj_f1 = macro_f1_score(gold_te, adj_preds_te, labels=[0, 1, 2, 3])

    assert adj_f1 > raw_f1 + 0.15


def test_cost_tracker_and_mocked_gemini_decision_engine(tmp_path: Path):
    """Test `GeminiDecisionEngine` structured parsing, multimodal payload formatting, and `BudgetExceededError`."""
    ledger = tmp_path / "ledger.json"
    tracker = CostTracker(max_budget_usd=0.01, ledger_path=ledger)

    captured_calls: List[Dict[str, Any]] = []

    class FakeInteractions:
        def create(self, **kwargs: Any) -> Any:
            captured_calls.append(kwargs)
            payload = {"q1": {"c": "B", "p": [0.1, 0.8, 0.1]}}
            usage = SimpleNamespace(
                total_input_tokens=500,
                total_output_tokens=20,
                total_thought_tokens=30,
            )
            return SimpleNamespace(output_text=json.dumps(payload), usage=usage)

    fake_client = SimpleNamespace(interactions=FakeInteractions())
    eng = GeminiDecisionEngine(
        model_id="gemini-3.5-flash-lite",
        thinking_level="low",
        cost_tracker=tracker,
        client=fake_client,
    )
    q = {
        "q1": {
            "type": "choice",
            "instructions": "Pick option.",
            "criteria": {"A": "first", "B": "second", "C": "third"},
        }
    }
    res = eng.predict(
        {"screenshot": b"raw-img-bytes", "task": "click B"},
        q,
        image_resolution_override="medium",
    )
    assert res["answers"]["q1"]["choice"] == "B"
    assert res["usage"]["input_tokens"] == 500
    assert res["usage"]["output_tokens"] == 50
    assert res["cost_usd"] > 0.0
    assert ledger.is_file()

    # Verify multimodal inline part was constructed, resolution set, and binary key stripped from text prompt
    sent_input = captured_calls[0]["input"]
    assert isinstance(sent_input, list)
    assert sent_input[0]["type"] == "image"
    assert sent_input[0]["resolution"] == "medium"
    assert "raw-img-bytes" not in sent_input[-1]["text"]

    # Trigger budget cap
    tiny_tracker = CostTracker(max_budget_usd=0.0001)
    with pytest.raises(BudgetExceededError):
        tiny_tracker.check_can_spend(0.001)


def test_omega_engine_knapsack_budget_and_all_9_suite_fixtures():
    """Verify `OmegaDecisionEngine` respects `max_escalation_fraction` and unlocks all 9 suite benchmarks."""

    class UncertainLocalEngine(BaseDecisionEngine):
        def predict(self, state: Any, questions: Dict[str, Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
            ans: Dict[str, Any] = {}
            for qid, qspec in questions.items():
                if qspec["type"] == "noul":
                    ans[qid] = {"type": "noul", "noul": 0.55, "route": "review"}
                elif qspec["type"] == "score":
                    k = len(qspec["criteria"])
                    p = {str(i): 1.0 / k for i in range(k)}
                    ans[qid] = {"type": "score", "score": 1.0, "probabilities": p, "route": "review"}
                else:
                    crit = qspec["criteria"]
                    keys = list(crit.keys() if isinstance(crit, dict) else crit)
                    p = {str(x): 1.0 / len(keys) for x in keys}
                    ans[qid] = {"type": "choice", "choice": str(keys[0]), "probabilities": p, "route": "review"}
            return {"model": "mock-local", "answers": ans, "usage": {"input_tokens": 50, "output_tokens": 0}, "latency_ms": 2.0}

    t2_calls = 0

    class FakeInteractions:
        def create(self, **kwargs: Any) -> Any:
            nonlocal t2_calls
            t2_calls += 1
            schema_props = kwargs["response_format"]["schema"]["properties"]
            out = {}
            for qid, qschema in schema_props.items():
                enum_vals = qschema["properties"]["c"]["enum"]
                chosen = enum_vals[0]
                probs = [0.9] + [0.1 / max(1, len(enum_vals) - 1)] * (len(enum_vals) - 1)
                out[qid] = {"c": chosen, "p": probs}
            usage = SimpleNamespace(total_input_tokens=120, total_output_tokens=18, total_thought_tokens=10)
            return SimpleNamespace(output_text=json.dumps(out), usage=usage)

    t2 = GeminiDecisionEngine(
        model_id="gemini-3.5-flash-lite",
        client=SimpleNamespace(interactions=FakeInteractions()),
        max_workers=2,
    )
    omega = OmegaDecisionEngine(
        local_laya=UncertainLocalEngine(),
        local_vega=None,
        tier2_engine=t2,
        escalation_threshold=0.90,
        max_escalation_fraction=0.50,
    )

    # Batch of 4 uncertain text states with max_escalation_fraction=0.50 -> MUST escalate exactly 2!
    q_simple = {"ok": {"type": "noul", "instructions": "Is ok?"}}
    batch_out = omega.predict_batch(["s1", "s2", "s3", "s4"], q_simple)
    assert len(batch_out) == 4
    assert t2_calls == 2

    # Full 9-benchmark suite on offline fixtures with Omega -> all 9 completed (0 skipped)!
    suite = DecisionBenchmarkSuite(engine=omega, fit_disjoint_ttt=False)
    summary = suite.run_suite(
        benchmarks=DecisionBenchmarkSuite.ALL_TABLE_BENCHMARKS,
        limit_per_benchmark=4,
        source="fixture",
        calibration_split=0.25,
    )
    assert len(summary.results) == 9
    assert all(r.status == "completed" for r in summary.results)
    assert "actual_run_usd" in summary.total_price_usd


def test_gemini_predict_packed_pairs_reduces_calls_and_conserves_cost():
    """Verify `predict_packed_pairs` packs `N` items into `ceil(N / pack_size)` calls and conserves exact tokens/USD."""
    calls: List[Dict[str, Any]] = []

    class FakePackedInteractions:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            props = kwargs["response_format"]["schema"]["properties"]
            out = {}
            for k_idx, (qid, qschema) in enumerate(props.items()):
                enum_vals = qschema["properties"]["c"]["enum"]
                pick = enum_vals[k_idx % len(enum_vals)]
                probs = [0.05] * len(enum_vals)
                probs[enum_vals.index(pick)] = 1.0 - 0.05 * (len(enum_vals) - 1)
                out[qid] = {"c": pick, "p": probs}
            usage = SimpleNamespace(total_input_tokens=601, total_output_tokens=45, total_thought_tokens=110)
            return SimpleNamespace(output_text=json.dumps(out), usage=usage)

    tracker = CostTracker(max_budget_usd=1.0)
    eng = GeminiDecisionEngine(
        model_id="gemini-3.8-flash",
        client=SimpleNamespace(interactions=FakePackedInteractions()),
        cost_tracker=tracker,
        max_workers=2,
    )
    q_esci = {
        "esci": {
            "type": "choice",
            "instructions": "Classify ESCI relevance.",
            "criteria": {"Irrelevant": "bad", "Complement": "addon", "Substitute": "alt", "Exact": "match"},
        }
    }
    text_pairs = [({"query": f"q{i}", "product_title": f"p{i}"}, q_esci) for i in range(6)]
    res_text = eng.predict_packed_pairs(text_pairs, pack_size=3)
    assert len(res_text) == 6
    assert len(calls) == 2
    # Rubric deduplication: `Classify ESCI relevance.` should appear once in the packed prompt, not 3 times
    assert calls[0]["input"].count("Classify ESCI relevance.") == 1
    assert sum(r["usage"]["input_tokens"] for r in res_text) == 1202
    assert sum(r["usage"]["output_tokens"] for r in res_text) == 310
    assert sum(r["cost_usd"] for r in res_text) == pytest.approx(tracker.session_cost_usd, abs=1e-6)

    # Interleaved multimodal packing check (2 audio items in 1 call)
    mm_pairs = [
        ({"audio": b"wav0", "question": "first?"}, {"ans": {"type": "choice", "instructions": "Q0", "criteria": ["A", "B"]}}),
        ({"audio": b"wav1", "question": "second?"}, {"ans": {"type": "choice", "instructions": "Q1", "criteria": ["A", "B"]}}),
    ]
    res_mm = eng.predict_packed_pairs(mm_pairs, pack_size=2)
    assert len(res_mm) == 2
    assert len(calls) == 3
    mm_input = calls[2]["input"]
    assert isinstance(mm_input, list)
    audio_parts = [p for p in mm_input if p.get("type") == "audio"]
    assert len(audio_parts) == 2
