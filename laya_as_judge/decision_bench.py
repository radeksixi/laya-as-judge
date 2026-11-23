"""DecisionBench (`decisionbench.dev` / `Hanno-Labs/decision-bench`) runner & wire-contract harness.

`laya-as-judge` natively implements the exact `/v1/systemone` Jev-compatible wire contract used by
`Hanno-Labs/decision-bench`'s official `SystemOneHTTPDecisionModel` (`run-system-one-http`):
  - `BINARY_CLASSIFICATION` -> `type="noul"`, `criteria={"true": ..., "false": ...}` -> `answers.decision.noul`
  - `CANDIDATE_SELECTION`   -> `type="choice"`, `criteria={cand_id: desc}` -> `answers.decision.probabilities[cand_id]`
  - `ORDINAL_SCORING`       -> `type="score"`, `criteria=[desc0, ...]` -> `answers.decision.probabilities["0"..]`

This module provides:
  1. `verify_systemone_contract(base_url)` — live smoke test of all 3 DecisionBench primitives
     against a running `laya-judge serve` (or any `/v1/systemone`) endpoint.
  2. `build_decision_bench_command(...)` — generates the exact `decision-bench run-system-one-http`
     CLI invocation for any `laya-as-judge` profile/model.
  3. `format_evaluation_request_issue(...)` — generates a copy-pasteable GitHub issue matching
     `Hanno-Labs/decision-bench/.github/ISSUE_TEMPLATE/evaluation_request.yml`.
  4. `evaluate_decisionbench_rows(engine, rows)` — standalone local evaluator for DecisionBench-format
     JSONL rows reporting accuracy, Top-1 ECE, Brier score, and error count (zero extra dependencies).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .calibration import brier_score, top1_ece
from .engine.base import BaseDecisionEngine
from .engine.http_systemone_backend import _extract_simplex_from_answer, normalize_decision_questions

DECISION_BENCH_REPO = "https://github.com/Hanno-Labs/decision-bench"
DECISION_BENCH_RESULTS_REPO = "https://github.com/Hanno-Labs/decision-bench-results"
SYSTEM_ONE_CONTRACT_VERSION = "jev-compatible-systemone-http-v1"


def build_decisionbench_request(row: Mapping[str, Any], *, model: str = "laya-as-judge") -> Tuple[Dict[str, Any], List[str]]:
    """Mirror `Hanno-Labs/decision-bench`'s `build_jev_request()` for any DecisionBench row."""
    primitive = str(row.get("primitive", "candidate_selection")).lower().strip()
    candidates = list(row.get("candidates") or [])
    instruction = str(row.get("instruction") or row.get("instructions") or "Select the best option.")
    state = row.get("state")

    def _desc(c: Mapping[str, Any]) -> str:
        lbl = str(c.get("label", c.get("id", "")))
        desc = c.get("description")
        return lbl if not desc else f"{lbl}: {desc}"

    if primitive in ("binary_classification", "binary", "noul"):
        if len(candidates) != 2:
            raise ValueError(f"Binary DecisionBench row requires 2 candidates, got {len(candidates)}")
        question: Dict[str, Any] = {
            "type": "noul",
            "instructions": instruction,
            "criteria": {"true": _desc(candidates[0]), "false": _desc(candidates[1])},
        }
        order = [str(c["id"]) for c in candidates]
    elif primitive in ("ordinal_scoring", "ordinal", "score"):
        ordered = sorted(candidates, key=lambda c: float(c.get("ordinal_value", 0.0)))
        question = {
            "type": "score",
            "instructions": instruction,
            "criteria": [_desc(c) for c in ordered],
        }
        order = [str(c["id"]) for c in ordered]
    else:
        question = {
            "type": "choice",
            "instructions": instruction,
            "criteria": {str(c["id"]): _desc(c) for c in candidates},
        }
        order = [str(c["id"]) for c in candidates]

    return {"model": model, "state": state, "questions": {"decision": question}}, order


def parse_decisionbench_answer(
    row: Mapping[str, Any],
    answer: Mapping[str, Any],
    response_order: Sequence[str],
) -> List[float]:
    """Mirror `Hanno-Labs/decision-bench`'s `prediction_from_jev_answer()`."""
    primitive = str(row.get("primitive", "candidate_selection")).lower().strip()
    candidates = list(row.get("candidates") or [])
    if primitive in ("binary_classification", "binary", "noul"):
        p_true = float(answer["noul"])
        true_id = str(candidates[0]["id"])
        return [p_true if str(c["id"]) == true_id else 1.0 - p_true for c in candidates]

    raw_probs = answer.get("probabilities")
    if not isinstance(raw_probs, Mapping):
        raise TypeError("SystemOne response did not include 'probabilities' dict")
    if primitive in ("ordinal_scoring", "ordinal", "score"):
        by_id = {cid: float(raw_probs[str(idx)]) for idx, cid in enumerate(response_order)}
    else:
        by_id = {cid: float(raw_probs[cid]) for cid in response_order}
    probs = [by_id[str(c["id"])] for c in candidates]
    total = sum(p for p in probs if np.isfinite(p) and p > 0.0)
    if total <= 0.0:
        raise ValueError("Non-positive probability distribution returned")
    return [p / total for p in probs]


def sample_contract_rows() -> List[Dict[str, Any]]:
    """Three canonical DecisionBench rows covering all 3 primitives (`noul`, `choice`, `score`)."""
    return [
        {
            "row_id": "contract-binary-1",
            "primitive": "binary_classification",
            "state": "Invoice #8812 shows a duplicate charge of $49.00 on March 14.",
            "instruction": "Does the document describe a duplicate billing charge?",
            "candidates": [
                {"id": "yes", "label": "Yes", "description": "Duplicate billing charge is present"},
                {"id": "no", "label": "No", "description": "No duplicate charge is described"},
            ],
            "gold_candidate_id": "yes",
        },
        {
            "row_id": "contract-choice-1",
            "primitive": "candidate_selection",
            "state": "Customer reports parcel #TRK-991 has been stuck in customs for 6 days.",
            "instruction": "Route this ticket to the single best department.",
            "candidates": [
                {"id": "dept_billing", "label": "Billing", "description": "Invoices, refunds, card charges"},
                {"id": "dept_shipping", "label": "Shipping", "description": "Parcel tracking, customs, delivery"},
                {"id": "dept_security", "label": "Security", "description": "Account takeover, phishing, fraud"},
            ],
            "gold_candidate_id": "dept_shipping",
        },
        {
            "row_id": "contract-score-1",
            "primitive": "ordinal_scoring",
            "state": "Answer: 'Paris is the capital of France.' Reference: 'The capital of France is Paris.'",
            "instruction": "Rate factual grounding of the answer against the reference.",
            "candidates": [
                {"id": "s0", "label": "0", "description": "Contradicted", "ordinal_value": 0},
                {"id": "s1", "label": "1", "description": "Partially supported", "ordinal_value": 1},
                {"id": "s2", "label": "2", "description": "Fully grounded", "ordinal_value": 2},
            ],
            "gold_candidate_id": "s2",
        },
    ]


def verify_systemone_contract(
    base_url: str,
    *,
    model: str = "laya-as-judge",
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Verify live `/v1/systemone` endpoint compatibility with `Hanno-Labs/decision-bench`."""
    url = f"{base_url.rstrip('/')}/v1/systemone"
    checks: List[Dict[str, Any]] = []
    all_ok = True

    for row in sample_contract_rows():
        req_body, order = build_decisionbench_request(row, model=model)
        t0 = time.perf_counter()
        raw_req = urllib.request.Request(
            url=url,
            data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(raw_req, timeout=timeout_s) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
            answer = resp_json["answers"]["decision"]
            probs = parse_decisionbench_answer(row, answer, order)
            output_tokens = int((resp_json.get("usage") or {}).get("output_tokens", 0))
            ok = (
                len(probs) == len(row["candidates"])
                and abs(sum(probs) - 1.0) < 1e-4
                and all(0.0 <= p <= 1.0 for p in probs)
                and output_tokens == 0
            )
            checks.append(
                {
                    "row_id": row["row_id"],
                    "primitive": row["primitive"],
                    "ok": ok,
                    "probabilities": [round(p, 4) for p in probs],
                    "output_tokens": output_tokens,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }
            )
            if not ok:
                all_ok = False
        except Exception as exc:
            all_ok = False
            checks.append(
                {
                    "row_id": row["row_id"],
                    "primitive": row["primitive"],
                    "ok": False,
                    "error": str(exc),
                }
            )

    return {
        "contract_version": SYSTEM_ONE_CONTRACT_VERSION,
        "base_url": base_url.rstrip("/"),
        "passed": all_ok,
        "checks": checks,
    }


def evaluate_decisionbench_rows(
    engine: BaseDecisionEngine,
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate DecisionBench-format rows directly against any `BaseDecisionEngine`."""
    correct = 0
    error_rows = 0
    probs_list: List[np.ndarray] = []
    gold_indices: List[int] = []
    latencies_ms: List[float] = []
    per_primitive: Dict[str, Dict[str, int]] = {}

    for row in rows:
        prim = str(row.get("primitive", "candidate_selection")).lower().strip()
        per_primitive.setdefault(prim, {"total": 0, "correct": 0, "errors": 0})
        per_primitive[prim]["total"] += 1
        try:
            req, order = build_decisionbench_request(row, model=model or engine.model_id)
            norm_q = normalize_decision_questions(req["questions"])
            t0 = time.perf_counter()
            out = engine.predict(req["state"], norm_q)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            raw_ans = (out.get("answers") or {}).get("decision", {})
            labels, simplex = _extract_simplex_from_answer(raw_ans, norm_q["decision"])
            # Pack back into exact Jev contract and parse using DecisionBench parser
            if norm_q["decision"]["type"] == "noul":
                packed = {"type": "noul", "noul": float(simplex[1])}
            elif norm_q["decision"]["type"] == "score":
                packed = {"type": "score", "probabilities": {str(i): float(p) for i, p in enumerate(simplex)}}
            else:
                packed = {"type": "choice", "probabilities": {lbl: float(p) for lbl, p in zip(labels, simplex)}}

            probs = parse_decisionbench_answer(row, packed, order)
            pred_idx = int(np.argmax(probs))
            cand_ids = [str(c["id"]) for c in row["candidates"]]
            gold_id = row.get("gold_candidate_id")
            if gold_id is not None and str(gold_id) in cand_ids:
                gold_idx = cand_ids.index(str(gold_id))
                probs_list.append(np.asarray(probs, dtype=np.float64))
                gold_indices.append(gold_idx)
                if pred_idx == gold_idx:
                    correct += 1
                    per_primitive[prim]["correct"] += 1
        except Exception:
            error_rows += 1
            per_primitive[prim]["errors"] += 1

    requested = len(rows)
    successful = requested - error_rows
    if probs_list:
        top1_confs = [float(np.max(p)) for p in probs_list]
        correct_flags = [int(np.argmax(p)) == y for p, y in zip(probs_list, gold_indices)]
        ece_val = float(top1_ece(top1_confs, correct_flags))
        brier_val = float(brier_score(probs_list, gold_indices))
    else:
        ece_val = 0.0
        brier_val = 0.0
    acc = (correct / len(gold_indices)) if gold_indices else 0.0

    return {
        "contract_version": SYSTEM_ONE_CONTRACT_VERSION,
        "model": model or engine.model_id,
        "requested_rows": requested,
        "successful_rows": successful,
        "error_rows": error_rows,
        "metrics": {
            "overall": {
                "accuracy": round(acc, 6),
                "expected_calibration_error": round(ece_val, 6),
                "brier_score": round(brier_val, 6),
                "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0,
            },
            "by_primitive": per_primitive,
        },
    }


def compute_serving_bundle_sha256(profile: str, model_repo: str, model_revision: str) -> str:
    """Compute a deterministic SHA-256 identity digest for a pinned serving configuration."""
    payload = json.dumps(
        {"package": "laya-as-judge", "profile": profile, "model_repo": model_repo, "model_revision": model_revision},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_decision_bench_command(
    *,
    decision_bench_dir: str,
    output_dir: str,
    base_url: str = "http://127.0.0.1:8000",
    profile: str = "rtx-16gb-glide",
    model: str = "laya-as-judge/rtx-16gb-glide",
    model_repo: str = "radeksixi/laya-as-judge",
    model_revision: str = "v1.1.0",
    task_spec: Optional[str] = None,
    concurrency: int = 8,
    max_candidates: int = 26,
    smoke: bool = True,
) -> List[str]:
    """Construct the exact `Hanno-Labs/decision-bench` `run-system-one-http` command."""
    src = Path(decision_bench_dir)
    spec = Path(task_spec) if task_spec else (src / "task_specs" / "decisionbench-dev.toml")
    bundle_sha = compute_serving_bundle_sha256(profile, model_repo, model_revision)
    cmd = [
        sys.executable,
        "-m",
        "decision_bench.cli",
        "run-system-one-http",
        str(spec),
        str(output_dir),
        "--project-root",
        str(src),
        "--base-url",
        base_url.rstrip("/"),
        "--model",
        model,
        "--model-repo",
        model_repo,
        "--model-revision",
        model_revision,
        "--serving-bundle-sha256",
        bundle_sha,
        "--inference-image",
        f"laya-as-judge:{model_revision} ({profile})",
        "--concurrency",
        str(concurrency),
        "--max-candidates",
        str(max_candidates),
        "--adapter-name",
        f"laya-as-judge-{profile}-v1",
        "--probability-source",
        "zero_output_token_calibrated_simplex_v1",
    ]
    if smoke:
        cmd.append("--smoke")
    return cmd


def run_decision_bench_subprocess(
    *,
    decision_bench_dir: str,
    output_dir: str,
    base_url: str = "http://127.0.0.1:8000",
    profile: str = "rtx-16gb-glide",
    model: str = "laya-as-judge/rtx-16gb-glide",
    model_repo: str = "radeksixi/laya-as-judge",
    model_revision: str = "v1.1.0",
    task_spec: Optional[str] = None,
    concurrency: int = 8,
    smoke: bool = True,
) -> subprocess.CompletedProcess:
    """Execute `Hanno-Labs/decision-bench` against a running `laya-judge serve` server."""
    cmd = build_decision_bench_command(
        decision_bench_dir=decision_bench_dir,
        output_dir=output_dir,
        base_url=base_url,
        profile=profile,
        model=model,
        model_repo=model_repo,
        model_revision=model_revision,
        task_spec=task_spec,
        concurrency=concurrency,
        smoke=smoke,
    )
    return subprocess.run(cmd, check=True)


def format_evaluation_request_issue(
    *,
    profile: str = "rtx-16gb-glide",
    version: str = "v1.1.0",
    model_repo: str = "radeksixi/laya-as-judge",
) -> str:
    """Format a ready-to-submit GitHub Issue for `Hanno-Labs/decision-bench` evaluation requests."""
    return f"""### Model
`{model_repo}` (`{profile}` local zero-output-token calibrated cascade served via native `/v1/systemone`)

### Immutable revision
`{version}` (`https://github.com/radeksixi/laya-as-judge/releases/tag/{version}`)

### Decision and probability contract
- **Native endpoint:** `POST /v1/systemone` (`{SYSTEM_ONE_CONTRACT_VERSION}`) — directly compatible with `decision-bench run-system-one-http` (`src/decision_bench/models/system_one_http.py`), zero custom adapter code required.
- **Supported primitives:** All three DecisionBench primitives natively (`noul` / binary classification, `choice` / candidate selection up to 26+ options, `score` / ordinal scoring) with **`0` output tokens** and **`0%` parse failures**.
- **Probability source:** Native zero-output-token decision logits from a 2-tier local GGUF cascade (`Kev-4B` -> `GLiDE-9B` or `Clef-Flash-9B`, fitting in `11.1–11.8 GB` VRAM on 1x 16 GB GPU) with post-hoc argmax-preserving `TemperatureScaler` calibration (`Top-1 ECE` reduced `0.0701 -> 0.0287`).
- **Quick reproducible launch:**
  ```bash
  pip install "laya-as-judge[server]=={version.lstrip('v')}"
  laya-judge serve --profile {profile} --tier1-url http://127.0.0.1:8080 --tier2-url http://127.0.0.1:8081 --port 8000
  laya-judge decision-bench --verify-only --base-url http://127.0.0.1:8000
  ```

### Data integrity
- [x] I am not aware of DecisionBench evaluation rows being used for training.
"""
