# Changelog

All notable changes to **`laya-as-judge`** are documented in this file.

## [1.2.0] — 2026-11-23

**Closed Attack → Guard → Prove Loop (`RedTeamVerdictJudge` + Native `sixi-scanner` Guardrail Rules)**

### Added
- **Dedicated `RedTeamVerdictJudge` Preset (`laya_as_judge/judges/redteam.py`, `POST /v1/judge` with `judge="redteam"`):**
  - Zero-output-token 3-criterion red-team judge (`violation_confirmed: noul`, `is_benign_refusal: noul`, `severity: choice[none,low,medium,high,critical]`) with explicit refusal arbitration (`RedTeamVerdictJudge.summarize_verdict`) so polite refusals are never misclassified as breaks.
  - Updated `EvaluationReport.passed` and `EmulatorBackend` to handle `is_benign_refusal` cleanly in offline CI.
- **Native `sixi-scanner` Guardrail Loader (`laya_as_judge/integrations/sixi_scanner.py`):**
  - `load_sixi_scanner_rules(path)` loads `sixi-scanner --export-laya <path>` guard packs (and raw `sixi-scanner --format json` reports as fallback) into deterministic, sub-millisecond `PolicyRuleFn` gates covering `prompt_block`, `tool_block`, `mcp_surface_block`, and `output_block`.
  - Plugs directly into `JudgedDecisionEngine`, FastMCP `LayaToolGateMiddleware`, LangGraph `LayaJudgedGateNode`, and `create_app(rules_file=...)` / `laya-judge serve --rules-file <path>`.
  - `SixiScannerGuardPack.verify_regression(...)` and CLI command **`laya-judge verify-sixi <rules.json>`** replay exported attack vectors + benign probes in CI to prove closure with `0%` benign false blocks.
- **Runnable Example & Tests:** `examples/10_closed_redteam_guard_loop.py` and `tests/test_v12_redteam_and_sixi_loop.py`.

### Live End-to-End Verification (`$0.00` Local GPU)
- **Live `Qwen3.8-27B` (`Sixi-ASI 1.0.0`) × all 45 `sixi-scanner` techniques:** CUDA `RedTeamVerdictJudge` (`0` output tokens, `131.5 ms` avg) rejected 1 FP and confirmed 2 real findings (`probe.llm01.sleeper-memory`, `probe.llm01.ssti-echo`); exporting `--export-laya` closed **`2/2` findings (`still_open=0`, `0.29 ms` guard overhead, `0/10` benign false blocks)** on re-scan while `sixi-scanner diff` honestly caught 1 stochastic new finding (`probe.llm05.xss-reflection`).
- **Live DVMCP Challenge 4 (FastMCP Rug Pull):** `sixi-scanner surface --rug-pull --export-laya` exported `get_weather` (`surface.mcp.rug-pull`); `LayaToolGateMiddleware` blocked `get_weather` deterministically in **`1.00 ms`** (`ActionGate.BLOCK`) while committing clean `get_system_time` (`ActionGate.COMMIT`).

---

## [1.1.1] — 2026-11-23

**Official MCP Stdio Server (`server.json`), GitHub Marketplace Action (`action.yml`), Native CI Gate + SARIF 2.1.0 & A2A v1.0 Agent Card**

### Added
- **Zero-Dependency MCP Stdio Server (`laya_as_judge/mcp_server.py`, `uvx laya-as-judge mcp`):** Standard JSON-RPC 2.0 MCP server exposing `laya_decide`, `laya_judge`, `laya_judged_decision`, `laya_moderate`, and `laya_rerank` plus official MCP Registry manifest (`server.json`) and PyPI ownership tag (`<!-- mcp-name: io.github.radeksixi/laya-as-judge -->`).
- **GitHub Marketplace Composite Action (`action.yml`) & Native CI Gate (`laya-judge eval --gate-judgement --max-failing --format sarif`):** Fail-closed CI threshold gating (`0` pass / `1` fail / `2` error) and SARIF 2.1.0 output for GitHub Code Scanning.
- **A2A v1.0 Discovery & JSON-RPC Endpoint (`laya_as_judge/server.py`):** Serves `GET /.well-known/agent-card.json` (plus legacy `/.well-known/agent.json`) and `POST /a2a/v1` (`tasks/send` & `message/send`).
- Added `laya-as-judge` console script alias alongside `laya-judge` so `uvx laya-as-judge mcp` works without `--from`.

---

## [1.1.0] — 2026-11-23

**Ecosystem Drop-In APIs, FastMCP & LangGraph Integrations, DecisionBench Runner, RAIL Guard Benchmark & Hugging Face Space**

### Added
- **OSS Ecosystem Drop-In APIs (`laya_as_judge/dropins.py`, `laya_as_judge/server.py`):**
  - `POST /v1/decisions/batch` (`7.97×` faster than 32 sequential HTTP calls on loopback)
  - `POST /v1/moderations` (OpenAI-compatible moderation schema with deterministic `spoofed_audit_guard_rule`)
  - `POST /v1/rerank` (Cohere / Jina / vLLM-compatible RAG document reranker)
  - `POST /v1/rank` (`CandidateRankerEngine` Best-of-N tournament verifier)
  - `GET /metrics` (Prometheus text exposition via `MetricsCollector`)
  - `TransformersDecisionBackend` (`get_engine("hf-decision", ...)` adapter for Hugging Face `AutoModelForDecision` / `DecisionPipeline`)
- **Framework Integrations (`laya_as_judge/integrations/`):**
  - **FastMCP (`fastmcp.py`):** `LayaToolGateMiddleware` (FastMCP 2.x `on_call_tool` hook) and universal sync/async `@laya_tool_guard` decorator enforcing `COMMIT / FALLBACK / ABSTAIN / BLOCK`.
  - **LangGraph (`langgraph.py`):** `make_decision_router` (conditional-edge router), `LayaJudgedGateNode`, and `laya_gate_router`.
- **External Benchmark Harnesses:**
  - **DecisionBench (`laya_as_judge/decision_bench.py`, `laya-judge decision-bench`, `examples/decisionbench_laya_adapter.py`):** Native `jev-compatible-systemone-http-v1` contract verification across all 3 primitives (`noul`, `choice`, `score`), `Hanno-Labs/decision-bench` `run-system-one-http` runner bridge, and pre-filled evaluation issue generator.
  - **RAIL Guard Benchmark (`laya_as_judge/rail_guard.py`, `laya-judge rail-guard`):** Evaluates `responsible-ai-labs/rail-guard-benchmark` Pool A (content safety across 6 domains: **`95.0% F1`**, **`90.5%`** adversarial catch, **`0.0%` benign FPR** on local `Kev-4B-Q4_K_M` at `$0.00`) and Pool B (agent tool-call safety across 5 domains: **`95.7%`** unsafe tool-call intercept rate, **`91.4%`** gap-potential intercept rate via 4-way `ActionGate`).
- **Distribution & Interactive Demo:**
  - Lightweight CPU **Hugging Face Space** (`spaces/hf_demo/`, live at `https://huggingface.co/spaces/rbai86/laya-as-judge-demo`).
  - Official PyPI Trusted Publishing GitHub Actions workflow (`.github/workflows/publish-pypi.yml`).

---

## [1.0.0] — 2026-11-22

**Full Official Stable Release** (`Development Status :: 5 - Production/Stable`) — one Apache-2.0 open-source **Decision, Judge & Governance API** built for both solo developers (`$0` local GPUs / edge) and regulated air-gapped enterprises (EU AI Act / DORA / NIS2 / Sixi Assure).

### Highlights
- **Stable 3-Mode API Contract** across Python SDK, CLI (`laya-judge`), and FastAPI server (`laya-judge serve`):
  - **Mode 1 — Decision API (`POST /v1/decisions`, `POST /v1/systemone`):** drop-in wire compatibility with Jev / Google / Azure Decisions APIs (`state`/`input`, `noul`/`bool`, `choice`, `score`, `0` output tokens).
  - **Mode 2 — Judge API (`POST /v1/judge`):** 6 built-in typed presets + custom rubrics, batch JSONL evaluation, and calibration suite.
  - **Mode 3 — Hybrid "The Judged Decision" (`POST /v1/judged-decision`):** Decide -> Cross-Family Verify -> Orthogonal Judge -> Deterministic Policy Rules -> 4-way `ActionGate` (`COMMIT` / `FALLBACK` / `ABSTAIN` / `BLOCK`) + structured JSON `audit_trace`.
- **Fail-Closed Enterprise Governance Defaults:** `audit_policy="on_high_stakes"` defaults to fail-closed when `high_stakes_actions` is omitted; `DEFAULT_SECURITY_POLICY_RULES` (`spoofed_audit_guard_rule`) runs deterministically on every call.
- **Verified Production Scoreboard:**
  - Single 16 GB RTX GPU (`$0.00` cloud): `78.0%–78.7%` across 6 HF benchmarks (`99.1%` of Cloud Decision API), `12/12` red-team catch rate with `0.0%` false blocks, `0.0287` calibrated Q4 Top-1 ECE, `100%` Polish accuracy & `0.0%` option-order flips (`rtx-16gb`), and `3.3×` faster multi-question long-log throughput.
  - Hybrid multimodal `Laya-Omega`: `11/11` wins over Gemini Decision 1 across 2,250 questions (`0.0234` OOF Top-1 ECE, `$0.77` total).
- **Documentation & Hero Architecture:** Updated top-of-README v1.0.0 platform architecture graphic, API/Feature overview matrix, and full solo + enterprise deployment guides.

---

## [0.7.0] — 2026-11-22

**100% Local 3-Mode API, Hardware VRAM Profiles & 5 Live GPU Experiments**

### Added
- `SystemOneHTTPBackend` (`laya_as_judge/engine/http_systemone_backend.py`): stdlib HTTP client for local `llama-server /v1/systemone` & `/v1/decisions` with `bool`/`noul` aliasing, text-only guard, cyclic `symmetry_permutations` + `adaptive_symmetry_threshold`, and `TemperatureScaler`/`MetaCalibrator` hook.
- Curated local VRAM profiles (`laya_as_judge/profiles.py`): `edge-6gb` (`5.88 GB`), `rtx-16gb` (`11.75 GB`, `Clef-Flash-9B`), `rtx-16gb-glide` (`11.11 GB`, `GLiDE-9B-IQ4_NL`), `workstation-24gb` (`16.8–20.2 GB`), and `emulator` (`0 GB`).
- `JudgedDecisionEngine`, `ActionGate`, `AuditPolicy`, `JudgedDecisionReport`, and deterministic `spoofed_audit_guard_rule` (`laya_as_judge/judged_decision.py`).
- FastAPI 3-mode local server (`laya_as_judge/server.py`) and CLI commands (`laya-judge profiles`, `laya-judge serve`, `laya-judge judged-decision`).
- Automatic `joint_schema_tier2` routing (`>=4` questions on `>=1200`-char states) and Tier-1 overflow escalation in `OmegaDecisionEngine`.
- Comprehensive unit test suite (`tests/test_local_api_and_judged_decision.py`, 60/60 tests passing) and runnable `examples/07_local_judged_decision_api.py`.

---

## [0.4.0] — 2026-11-13

**Laya-Omega Hybrid Multimodal Cascade — 11 / 11 Wins vs. Gemini Decision 1**

### Added
- `OmegaDecisionEngine` combining local CUDA encoders (`Laya` + `Vega-U 0.8B` with disjoint Test-Time Training `TTTHead`s) with selective packed Gemini escalation (`gemini-3.5-flash-lite` -> `gemini-3.8-flash`).
- `MetaCalibrator` (10-fold out-of-fold argmax-preserving simplex meta-calibration reaching `0.0234` pooled Top-1 ECE) and `ClassMultiplierOptimizer` for rare-class Macro-F1.
- Full 9-benchmark multimodal `DecisionBenchmarkSuite` (`JevBench`, `Amazon ESCI`, `New Yorker Caption`, `Mind2Web`, `PubMedQA`, `Global-MMLU-Lite`, `When2Call`, `MMAU`, `Video-MME`).

---

## [0.2.0] — 2026-10-15

- Initial release of Laya-as-a-Judge Python SDK, 6 built-in judges, `CustomJudgeBuilder`, `CascadedJudge`, `BatchEvaluator`, `TemperatureScaler`, `MLXBackend`, `TorchBackend`, `EmulatorBackend`, and `laya-judge` CLI.
