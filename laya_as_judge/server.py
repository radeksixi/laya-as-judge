"""Local Text-Only HTTP API Server exposing Decision, Judge, Judged Decision & Ecosystem Drop-Ins.

Endpoints (Apache-2.0 Open Source):
  - Mode 1 (Decision — Cloud Drop-in):
      * `POST /v1/systemone`        TypeSafe Jev / llama.cpp / OpenRouter wire format
      * `POST /v1/decisions`        OpenAI / Google / Azure Decisions API (incl. `input` & `bool`)
      * `POST /v1/decisions/batch`  High-throughput batched decision evaluation
  - Mode 2 (Judge & Ecosystem Drop-ins):
      * `POST /v1/judge`            Built-in presets or custom typed rubrics
      * `POST /v1/moderations`      Air-gapped OpenAI Moderation API drop-in
      * `POST /v1/rerank`           Cohere / Jina / vLLM RAG reranker drop-in
      * `POST /v1/rank`             Best-of-N candidate tournament verifier
  - Mode 3 (Hybrid — The Judged Decision):
      * `POST /v1/judged-decision`  Decide -> Orthogonal Audit/Judge -> ActionGate (`commit/fallback/abstain/block`)
  - Operational & Observability:
      * `GET /health`, `GET /v1/models`, `GET /v1/profiles`, `GET /metrics` (Prometheus)

Requires optional server dependencies: `pip install 'laya-as-judge[server]'`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Union

from .dropins import CandidateRankerEngine, ModerationAdapter
from .engine.base import BaseDecisionEngine
from .engine.http_systemone_backend import (
    _contains_media,
    _extract_simplex_from_answer,
    _pack_canonical_answer,
    normalize_decision_questions,
)
from .judged_decision import ActionGate, AuditPolicy, JudgedDecisionEngine
from .judges import (
    AgentTrajectoryJudge,
    AnswerRelevanceJudge,
    BaseJudge,
    CustomJudge,
    FaithfulnessJudge,
    InstructionAdherenceJudge,
    PairwiseComparisonJudge,
    SafetyGuardJudge,
)
from .profiles import HARDWARE_PROFILES, build_local_profile_engines

PRESET_JUDGES = {
    "faithfulness": FaithfulnessJudge,
    "relevance": AnswerRelevanceJudge,
    "instruction": InstructionAdherenceJudge,
    "safety": SafetyGuardJudge,
    "pairwise": PairwiseComparisonJudge,
    "agent": AgentTrajectoryJudge,
}


class MetricsCollector:
    """Zero-dependency thread-safe Prometheus exposition metrics tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_by_endpoint: Dict[str, int] = {}
        self.gates_by_verdict: Dict[str, int] = {}
        self.latency_ms_sum: Dict[str, float] = {}
        self.extra_counters: Dict[str, int] = {}

    def record_request(self, endpoint: str, latency_ms: float) -> None:
        with self._lock:
            self.requests_by_endpoint[endpoint] = self.requests_by_endpoint.get(endpoint, 0) + 1
            self.latency_ms_sum[endpoint] = self.latency_ms_sum.get(endpoint, 0.0) + float(latency_ms)

    def record_gate(self, gate: str) -> None:
        with self._lock:
            self.gates_by_verdict[gate] = self.gates_by_verdict.get(gate, 0) + 1

    def inc_counter(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.extra_counters[name] = self.extra_counters.get(name, 0) + int(amount)

    def render_prometheus(self) -> str:
        with self._lock:
            lines: List[str] = [
                "# HELP laya_requests_total Total zero-output-token API requests by endpoint.",
                "# TYPE laya_requests_total counter",
            ]
            for ep, cnt in sorted(self.requests_by_endpoint.items()):
                lines.append(f'laya_requests_total{{endpoint="{ep}"}} {cnt}')
            lines.extend(
                [
                    "# HELP laya_request_latency_ms_sum Cumulative request latency in milliseconds.",
                    "# TYPE laya_request_latency_ms_sum counter",
                ]
            )
            for ep, tot in sorted(self.latency_ms_sum.items()):
                lines.append(f'laya_request_latency_ms_sum{{endpoint="{ep}"}} {tot:.2f}')
            lines.extend(
                [
                    "# HELP laya_action_gates_total Total ActionGate verdicts emitted.",
                    "# TYPE laya_action_gates_total counter",
                ]
            )
            for g, cnt in sorted(self.gates_by_verdict.items()):
                lines.append(f'laya_action_gates_total{{gate="{g}"}} {cnt}')
            for name, val in sorted(self.extra_counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {val}")
            return "\n".join(lines) + "\n"


try:
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import Field as _PydanticField

    class SystemOneRequest(_PydanticBaseModel):
        model: Optional[str] = None
        state: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        input: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        questions: Dict[str, Dict[str, Any]]
        symmetry_permutations: Optional[int] = _PydanticField(default=None, ge=1, le=8)

    class BatchDecisionRequest(_PydanticBaseModel):
        model: Optional[str] = None
        items: List[Union[str, Dict[str, Any], List[Any]]]
        questions: Dict[str, Dict[str, Any]]

    class JudgeRequest(_PydanticBaseModel):
        state: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        input: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        judge: Optional[str] = _PydanticField(
            default=None,
            description=f"Built-in preset name ({', '.join(PRESET_JUDGES.keys())})",
        )
        questions: Optional[Dict[str, Dict[str, Any]]] = None
        name: str = "CustomJudge"
        metadata: Optional[Dict[str, Any]] = None

    class JudgedDecisionRequest(_PydanticBaseModel):
        model: Optional[str] = None
        state: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        input: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        questions: Dict[str, Dict[str, Any]]
        judge: Optional[str] = _PydanticField(
            default=None,
            description="Optional built-in judge preset instead of default orthogonal decision audit.",
        )
        judge_questions: Optional[Dict[str, Dict[str, Any]]] = None
        audit_policy: Optional[str] = _PydanticField(
            default=None,
            description="One of 'always', 'on_high_stakes', or 'on_uncertain'.",
        )
        high_stakes_actions: Optional[List[str]] = None
        fallback_map: Optional[Dict[str, str]] = None
        min_top1_prob: Optional[float] = _PydanticField(default=None, ge=0.0, le=1.0)
        min_margin: Optional[float] = _PydanticField(default=None, ge=0.0, le=1.0)
        cross_verify: Optional[bool] = None
        metadata: Optional[Dict[str, Any]] = None

    class ModerationRequest(_PydanticBaseModel):
        input: Union[str, List[str]]
        model: Optional[str] = None
        categories: Optional[Dict[str, str]] = None

    class RerankRequest(_PydanticBaseModel):
        query: str
        documents: List[Union[str, Dict[str, Any]]]
        top_n: Optional[int] = _PydanticField(default=None, ge=1)
        return_documents: bool = False
        model: Optional[str] = None

    class RankRequest(_PydanticBaseModel):
        prompt: str
        candidates: List[str]
        rubric: Optional[str] = None
        top_n: Optional[int] = _PydanticField(default=None, ge=1)
        return_candidates: bool = True

except ImportError:  # pragma: no cover - optional server dependency
    SystemOneRequest = Any  # type: ignore[misc,assignment]
    BatchDecisionRequest = Any  # type: ignore[misc,assignment]
    JudgeRequest = Any  # type: ignore[misc,assignment]
    JudgedDecisionRequest = Any  # type: ignore[misc,assignment]
    ModerationRequest = Any  # type: ignore[misc,assignment]
    RerankRequest = Any  # type: ignore[misc,assignment]
    RankRequest = Any  # type: ignore[misc,assignment]


def create_app(
    *,
    profile: str = "emulator",
    decider_engine: Optional[BaseDecisionEngine] = None,
    judge_engine: Optional[BaseDecisionEngine] = None,
    guard_engine: Optional[BaseDecisionEngine] = None,
    tier1_backend: Optional[str] = None,
    tier1_model: Optional[str] = None,
    tier1_url: Optional[str] = None,
    tier2_url: Optional[str] = None,
    tier2_model: Optional[str] = None,
    device: Optional[str] = None,
    fallback_to_emulator: bool = True,
    default_audit_policy: str = "on_high_stakes",
    text_only: bool = True,
    run_ingress_guard: bool = False,
    cross_verify: bool = False,
    enable_default_security_rules: bool = True,
    calibrator: Optional[Any] = None,
) -> Any:
    """Create a FastAPI application serving Decision, Judge, Judged Decision, and OSS Drop-In APIs."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import PlainTextResponse
        import pydantic  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The HTTP server requires optional server dependencies. "
            "Install them via: pip install 'laya-as-judge[server]'"
        ) from exc

    from . import __version__

    if decider_engine is None:
        built_decider, built_judge, built_guard, hw_profile = build_local_profile_engines(
            profile_name=profile,
            tier1_backend=tier1_backend,
            tier1_model=tier1_model,
            tier1_url=tier1_url,
            tier2_url=tier2_url,
            tier2_model=tier2_model,
            device=device,
            fallback_to_emulator=fallback_to_emulator,
            calibrator=calibrator,
        )
        decider_engine = built_decider
        judge_engine = judge_engine or built_judge
        guard_engine = guard_engine or built_guard
    else:
        hw_profile = HARDWARE_PROFILES.get(profile, HARDWARE_PROFILES["emulator"])
        judge_engine = judge_engine or decider_engine

    judged_engine = JudgedDecisionEngine(
        decider_engine=decider_engine,
        judge_engine=judge_engine,
        guard_engine=guard_engine,
        audit_policy=default_audit_policy,
        cross_verify=cross_verify,
        run_ingress_guard=run_ingress_guard,
        enable_default_security_rules=enable_default_security_rules,
        calibrator=calibrator,
    )
    ranker_engine = CandidateRankerEngine(engine=judge_engine)
    moderation_adapter = ModerationAdapter(engine=judge_engine)
    metrics = MetricsCollector()

    preset_judge_instances: Dict[str, BaseJudge] = {
        name: cls(engine=judge_engine) for name, cls in PRESET_JUDGES.items()
    }

    app = FastAPI(
        title="Laya-as-a-Judge Local Decision & Evaluation API",
        description=(
            "100% local, text-only, zero-output-token API providing Decision (/v1/decisions, /v1/systemone, "
            "/v1/decisions/batch), Judge (/v1/judge), Ecosystem Drop-Ins (/v1/moderations, /v1/rerank, /v1/rank), "
            "and Hybrid The Judged Decision (/v1/judged-decision)."
        ),
        version=__version__,
    )

    def _resolve_state(
        state_val: Optional[Union[str, Dict[str, Any], List[Any]]],
        input_val: Optional[Union[str, Dict[str, Any], List[Any]]],
    ) -> Union[str, Dict[str, Any], List[Any]]:
        resolved = state_val if state_val is not None else input_val
        if resolved is None:
            raise HTTPException(status_code=422, detail="Either 'state' or 'input' must be provided.")
        if text_only and _contains_media(resolved):
            raise HTTPException(
                status_code=422,
                detail=(
                    "This server runs in text-only local mode and does not accept raw image/audio/video "
                    "media fields. Pass textual representations or deploy a multimodal Tier-2 engine."
                ),
            )
        return resolved

    def _run_decision_endpoint(
        req: SystemOneRequest,
        *,
        include_bool_alias: bool,
        endpoint_path: str,
    ) -> Dict[str, Any]:
        state = _resolve_state(req.state, req.input)
        raw_types = {
            qid: (qspec.get("type") if isinstance(qspec, Mapping) else None)
            for qid, qspec in req.questions.items()
        }
        try:
            normalized_q = normalize_decision_questions(req.questions)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        t0 = time.perf_counter()
        predict_kwargs: Dict[str, Any] = {}
        if req.symmetry_permutations is not None and hasattr(decider_engine, "symmetry_permutations"):
            predict_kwargs["symmetry_permutations"] = req.symmetry_permutations
        raw_out = (
            decider_engine.predict(state, normalized_q, **predict_kwargs)
            if predict_kwargs
            else decider_engine.predict(state, normalized_q)
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        answers: Dict[str, Any] = {}
        for qid, qspec in normalized_q.items():
            labels, probs = _extract_simplex_from_answer((raw_out.get("answers") or {}).get(qid, {}), qspec)
            packed = _pack_canonical_answer(labels, probs, qspec)
            if include_bool_alias and raw_types.get(qid) == "bool" and packed["type"] == "noul":
                packed["bool"] = packed["noul"]
                packed["holds"] = packed["noul"] >= 0.5
            answers[qid] = packed

        usage = raw_out.get("usage") or {}
        metrics.record_request(endpoint_path, elapsed_ms)
        return {
            "model": req.model or raw_out.get("model") or decider_engine.model_id,
            "answers": answers,
            "usage": {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": 0,
            },
            "latency_ms": round(float(raw_out.get("latency_ms", elapsed_ms)), 2),
        }

    # --- Operational & Observability Endpoints ---

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "text_only": text_only,
            "profile": hw_profile.name,
            "max_vram_gb": hw_profile.max_vram_gb,
            "modes": ["decision", "judge", "judged_decision"],
            "dropins": ["decisions_batch", "moderations", "rerank", "rank", "prometheus_metrics"],
            "models": {
                "decider": decider_engine.model_id,
                "judge": judge_engine.model_id,
                "guard": guard_engine.model_id if guard_engine is not None else None,
            },
        }

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": decider_engine.model_id,
                    "object": "model",
                    "role": "decider",
                    "backend": type(decider_engine).__name__,
                    "text_only": text_only,
                },
                {
                    "id": judge_engine.model_id,
                    "object": "model",
                    "role": "judge",
                    "backend": type(judge_engine).__name__,
                    "text_only": text_only,
                },
            ],
            "profile": hw_profile.to_dict(),
        }

    @app.get("/v1/profiles")
    def list_profiles() -> Dict[str, Any]:
        return {
            "active_profile": hw_profile.name,
            "profiles": {k: v.to_dict() for k, v in HARDWARE_PROFILES.items()},
        }

    @app.get("/metrics")
    def prometheus_metrics() -> Any:
        return PlainTextResponse(metrics.render_prometheus(), media_type="text/plain; version=0.0.4")

    # --- Mode 1: Decision API ---

    @app.post("/v1/systemone")
    def systemone_decision(req: SystemOneRequest) -> Dict[str, Any]:
        """TypeSafe Jev / llama.cpp `/v1/systemone` compatible endpoint."""
        return _run_decision_endpoint(req, include_bool_alias=False, endpoint_path="/v1/systemone")

    @app.post("/v1/decisions")
    def decisions_api(req: SystemOneRequest) -> Dict[str, Any]:
        """OpenAI / Google / Azure `/v1/decisions` compatible endpoint."""
        return _run_decision_endpoint(req, include_bool_alias=True, endpoint_path="/v1/decisions")

    @app.post("/v1/decisions/batch")
    def decisions_batch_api(req: BatchDecisionRequest) -> Dict[str, Any]:
        """High-throughput batched decision evaluation with zero output tokens."""
        t0 = time.perf_counter()
        for item in req.items:
            if text_only and _contains_media(item):
                raise HTTPException(status_code=422, detail="Media payloads rejected in text-only local mode.")
        try:
            normalized_q = normalize_decision_questions(req.questions)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        raw_outs = decider_engine.predict_batch(list(req.items), normalized_q) if req.items else []
        results: List[Dict[str, Any]] = []
        total_in = 0
        for raw_out in raw_outs:
            total_in += int((raw_out.get("usage") or {}).get("input_tokens", 0))
            ans_map: Dict[str, Any] = {}
            for qid, qspec in normalized_q.items():
                labels, probs = _extract_simplex_from_answer((raw_out.get("answers") or {}).get(qid, {}), qspec)
                ans_map[qid] = _pack_canonical_answer(labels, probs, qspec)
            results.append({"answers": ans_map})

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics.record_request("/v1/decisions/batch", elapsed_ms)
        return {
            "model": req.model or decider_engine.model_id,
            "count": len(results),
            "results": results,
            "usage": {"input_tokens": total_in, "output_tokens": 0},
            "latency_ms": round(elapsed_ms, 2),
        }

    # --- Mode 2: Judge & Ecosystem Drop-Ins ---

    @app.post("/v1/judge")
    def run_judge(req: JudgeRequest) -> Dict[str, Any]:
        """Evaluate state with a preset or custom typed rubric judge."""
        t0 = time.perf_counter()
        state = _resolve_state(req.state, req.input)

        if req.questions:
            try:
                norm_q = normalize_decision_questions(req.questions)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            active_judge: BaseJudge = CustomJudge(name=req.name, questions_dict=norm_q, engine=judge_engine)
        elif req.judge:
            key = req.judge.strip().lower()
            if key not in preset_judge_instances:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown judge preset '{req.judge}'. Expected one of: {', '.join(PRESET_JUDGES.keys())}",
                )
            active_judge = preset_judge_instances[key]
        else:
            raise HTTPException(
                status_code=422,
                detail="Provide either 'judge' (preset name) or 'questions' (custom rubric).",
            )

        report = active_judge.evaluate(state, metadata=req.metadata)
        payload = report.to_dict()
        payload["passed"] = report.passed
        payload["average_confidence"] = round(report.average_confidence, 4)
        payload["average_top1_probability"] = round(report.average_top1_probability, 4)
        metrics.record_request("/v1/judge", (time.perf_counter() - t0) * 1000.0)
        return payload

    @app.post("/v1/moderations")
    def run_moderations(req: ModerationRequest) -> Dict[str, Any]:
        """Air-gapped OpenAI `/v1/moderations` drop-in endpoint ($0 local, 0 output tokens)."""
        t0 = time.perf_counter()
        out = moderation_adapter.moderate(req.input, model=req.model, custom_categories=req.categories)
        metrics.record_request("/v1/moderations", (time.perf_counter() - t0) * 1000.0)
        return out

    @app.post("/v1/rerank")
    def run_rerank(req: RerankRequest) -> Dict[str, Any]:
        """Cohere / Jina / vLLM drop-in `/v1/rerank` RAG relevance endpoint."""
        t0 = time.perf_counter()
        out = ranker_engine.rerank(
            query=req.query,
            documents=req.documents,
            top_n=req.top_n,
            return_documents=req.return_documents,
            model=req.model,
        )
        metrics.record_request("/v1/rerank", (time.perf_counter() - t0) * 1000.0)
        return out

    @app.post("/v1/rank")
    def run_rank(req: RankRequest) -> Dict[str, Any]:
        """Zero-output-token Best-of-N candidate tournament verifier."""
        t0 = time.perf_counter()
        out = ranker_engine.rank(
            prompt=req.prompt,
            candidates=req.candidates,
            rubric=req.rubric,
            top_n=req.top_n,
            return_candidates=req.return_candidates,
        )
        metrics.record_request("/v1/rank", (time.perf_counter() - t0) * 1000.0)
        return out

    # --- Mode 3: Hybrid — The Judged Decision ---

    @app.post("/v1/judged-decision")
    def run_judged_decision(req: JudgedDecisionRequest) -> Dict[str, Any]:
        """Hybrid mode: Decide -> Orthogonal Audit/Judge -> Deterministic Action Gate."""
        t0 = time.perf_counter()
        state = _resolve_state(req.state, req.input)

        try:
            norm_q = normalize_decision_questions(req.questions)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        judge_override: Optional[BaseJudge] = None
        if req.judge_questions:
            try:
                norm_jq = normalize_decision_questions(req.judge_questions)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            judge_override = CustomJudge(name="CustomAuditJudge", questions_dict=norm_jq, engine=judge_engine)
        elif req.judge:
            key = req.judge.strip().lower()
            if key not in preset_judge_instances:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown judge preset '{req.judge}'. Expected one of: {', '.join(PRESET_JUDGES.keys())}",
                )
            judge_override = preset_judge_instances[key]

        if req.audit_policy is not None:
            try:
                AuditPolicy(req.audit_policy)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid 'audit_policy'; expected 'always', 'on_high_stakes', or 'on_uncertain'.",
                ) from exc

        result = judged_engine.evaluate(
            state,
            norm_q,
            audit_policy=req.audit_policy,
            high_stakes_actions=req.high_stakes_actions,
            fallback_map=req.fallback_map,
            judge_override=judge_override,
            min_top1_prob=req.min_top1_prob,
            min_margin=req.min_margin,
            cross_verify=req.cross_verify,
            metadata=req.metadata,
        )
        metrics.record_gate(result.gate.value)
        metrics.record_request("/v1/judged-decision", (time.perf_counter() - t0) * 1000.0)
        return result.to_dict()

    # Attach engines & helpers on app.state for clean composition by commercial extensions (e.g. sixi-assure-runtime)
    app.state.decider_engine = decider_engine
    app.state.judge_engine = judge_engine
    app.state.guard_engine = guard_engine
    app.state.judged_engine = judged_engine
    app.state.ranker_engine = ranker_engine
    app.state.moderation_adapter = moderation_adapter
    app.state.preset_judges = preset_judge_instances
    app.state.metrics = metrics
    app.state.hardware_profile = hw_profile
    app.state.text_only = text_only

    return app
