"""Local Text-Only HTTP API Server exposing all 3 modes: Decision, Judge, and Judged Decision.

Endpoints:
  - Mode 1 (Decision — Cloud Drop-in):
      * `POST /v1/systemone`  (TypeSafe Jev / llama.cpp / OpenRouter wire format)
      * `POST /v1/decisions`  (OpenAI / Google / Azure Decisions API wire format, incl. `input` & `bool` aliases)
  - Mode 2 (Judge — Structured Evaluation & Guardrails):
      * `POST /v1/judge`      (Built-in presets or custom typed rubrics)
  - Mode 3 (Hybrid — The Judged Decision):
      * `POST /v1/judged-decision`  (Decide -> Orthogonal Audit/Judge -> ActionGate: commit/fallback/abstain/block)
  - Operational:
      * `GET /health`
      * `GET /v1/models`
      * `GET /v1/profiles`

Requires optional server dependencies: `pip install 'laya-as-judge[server]'`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Union

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
from .profiles import HARDWARE_PROFILES, HardwareProfile, build_local_profile_engines

PRESET_JUDGES = {
    "faithfulness": FaithfulnessJudge,
    "relevance": AnswerRelevanceJudge,
    "instruction": InstructionAdherenceJudge,
    "safety": SafetyGuardJudge,
    "pairwise": PairwiseComparisonJudge,
    "agent": AgentTrajectoryJudge,
}

try:
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import Field as _PydanticField

    class SystemOneRequest(_PydanticBaseModel):
        model: Optional[str] = None
        state: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        input: Optional[Union[str, Dict[str, Any], List[Any]]] = None
        questions: Dict[str, Dict[str, Any]]
        symmetry_permutations: Optional[int] = _PydanticField(default=None, ge=1, le=8)

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

except ImportError:  # pragma: no cover - optional server dependency
    SystemOneRequest = Any  # type: ignore[misc,assignment]
    JudgeRequest = Any  # type: ignore[misc,assignment]
    JudgedDecisionRequest = Any  # type: ignore[misc,assignment]


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
    """Create a FastAPI application serving Decision, Judge, and Hybrid Judged Decision APIs."""
    try:
        from fastapi import FastAPI, HTTPException
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

    preset_judge_instances: Dict[str, BaseJudge] = {
        name: cls(engine=judge_engine) for name, cls in PRESET_JUDGES.items()
    }

    app = FastAPI(
        title="Laya-as-a-Judge Local Decision & Governance API",
        description=(
            "100% local, text-only, zero-output-token API providing three unified modes: "
            "Decision (/v1/decisions, /v1/systemone), Judge (/v1/judge), and Hybrid "
            "The Judged Decision (/v1/judged-decision)."
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

    def _run_decision_endpoint(req: SystemOneRequest, *, include_bool_alias: bool) -> Dict[str, Any]:
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
        raw_out = decider_engine.predict(state, normalized_q, **predict_kwargs) if predict_kwargs else decider_engine.predict(state, normalized_q)
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
        return {
            "model": req.model or raw_out.get("model") or decider_engine.model_id,
            "answers": answers,
            "usage": {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": 0,
            },
            "latency_ms": round(float(raw_out.get("latency_ms", elapsed_ms)), 2),
        }

    # --- Operational Endpoints ---

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "text_only": text_only,
            "profile": hw_profile.name,
            "max_vram_gb": hw_profile.max_vram_gb,
            "modes": ["decision", "judge", "judged_decision"],
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

    # --- Mode 1: Decision API (Cloud Drop-In) ---

    @app.post("/v1/systemone")
    def systemone_decision(req: SystemOneRequest) -> Dict[str, Any]:
        """TypeSafe Jev / llama.cpp `/v1/systemone` compatible endpoint."""
        return _run_decision_endpoint(req, include_bool_alias=False)

    @app.post("/v1/decisions")
    def decisions_api(req: SystemOneRequest) -> Dict[str, Any]:
        """OpenAI / Google / Azure `/v1/decisions` compatible endpoint."""
        return _run_decision_endpoint(req, include_bool_alias=True)

    # --- Mode 2: Judge API ---

    @app.post("/v1/judge")
    def run_judge(req: JudgeRequest) -> Dict[str, Any]:
        """Evaluate state with a preset or custom typed rubric judge."""
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
        return payload

    # --- Mode 3: Hybrid — The Judged Decision ---

    @app.post("/v1/judged-decision")
    def run_judged_decision(req: JudgedDecisionRequest) -> Dict[str, Any]:
        """Hybrid mode: Decide -> Orthogonal Audit/Judge -> Deterministic Action Gate."""
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
        return result.to_dict()

    # Attach engines on app.state for programmatic inspection/testing
    app.state.decider_engine = decider_engine
    app.state.judge_engine = judge_engine
    app.state.guard_engine = guard_engine
    app.state.judged_engine = judged_engine
    app.state.hardware_profile = hw_profile

    return app
