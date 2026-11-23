"""LangGraph zero-output-token routing and governance node integration (`laya-as-judge`).

Zero required dependency on `langgraph` — conforms cleanly to LangGraph's `StateGraph` node and
conditional-edge callable protocols:
  - `make_decision_router(...)`: creates a conditional-edge router function for
    `graph.add_conditional_edges("agent", router, {...})` using a single zero-output-token `choice` pass.
  - `LayaJudgedGateNode`: drop-in LangGraph node (`graph.add_node("governance_gate", gate_node)`)
    that evaluates the latest agent state/tool-calls with `JudgedDecisionEngine` and writes
    `laya_gate` (`commit` / `fallback` / `abstain` / `block`), `laya_reason`, `laya_calibrated_risk`,
    and `laya_audit_trace` into the graph state.
  - `laya_gate_router(state)`: conditional-edge selector reading `state["laya_gate"]`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from ..engine.base import BaseDecisionEngine
from ..engine.http_systemone_backend import normalize_decision_questions
from ..judged_decision import ActionGate, AuditPolicy, JudgedDecisionEngine
from ..judges import CustomJudge


def _extract_graph_payload(state: Mapping[str, Any], state_key: Optional[str] = None) -> Any:
    if state_key is not None and state_key in state:
        return state[state_key]
    if "messages" in state and isinstance(state["messages"], Sequence):
        rendered = []
        for m in state["messages"]:
            if isinstance(m, Mapping):
                role = m.get("role", m.get("type", "message"))
                content = m.get("content", "")
                tool_calls = m.get("tool_calls")
                entry: Dict[str, Any] = {"role": str(role), "content": str(content)}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                rendered.append(entry)
            else:
                role = getattr(m, "type", getattr(m, "role", "message"))
                content = getattr(m, "content", str(m))
                tool_calls = getattr(m, "tool_calls", None)
                entry = {"role": str(role), "content": str(content)}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                rendered.append(entry)
        return {"messages": rendered}
    return dict(state)


def make_decision_router(
    engine: BaseDecisionEngine,
    *,
    routes: Union[Sequence[str], Mapping[str, str]],
    instructions: str = "Select the single best next LangGraph node for the current state.",
    state_key: Optional[str] = None,
    min_confidence: float = 0.0,
    fallback_route: Optional[str] = None,
) -> Callable[[Mapping[str, Any]], str]:
    """Build a zero-output-token conditional-edge router for `StateGraph.add_conditional_edges`."""
    criteria = dict(routes) if isinstance(routes, Mapping) else {str(r): None for r in routes}
    if not criteria:
        raise ValueError("routes must be a non-empty sequence or mapping of node names")
    default_route = fallback_route or next(iter(criteria.keys()))
    questions = normalize_decision_questions(
        {"route": {"type": "choice", "instructions": instructions, "criteria": criteria}}
    )

    def _route(state: Mapping[str, Any]) -> str:
        payload = _extract_graph_payload(state, state_key=state_key)
        out = engine.predict(payload, questions)
        ans = (out.get("answers") or {}).get("route", {})
        probs = ans.get("probabilities") or {}
        if probs:
            best_label = max(probs.items(), key=lambda kv: float(kv[1]))[0]
            best_prob = float(probs[best_label])
        else:
            best_label = str(ans.get("choice", default_route))
            best_prob = 1.0
        if best_prob < min_confidence:
            return default_route
        return str(best_label)

    return _route


class LayaJudgedGateNode:
    """Drop-in LangGraph node executing `JudgedDecisionEngine` before side-effecting tool nodes."""

    def __init__(
        self,
        engine: Union[JudgedDecisionEngine, BaseDecisionEngine],
        *,
        questions: Optional[Dict[str, Dict[str, Any]]] = None,
        judge_questions: Optional[Dict[str, Dict[str, Any]]] = None,
        state_key: Optional[str] = None,
        audit_policy: Union[AuditPolicy, str] = AuditPolicy.ON_HIGH_STAKES,
        high_stakes_actions: Optional[Sequence[str]] = None,
        fallback_map: Optional[Mapping[str, str]] = None,
    ):
        if isinstance(engine, JudgedDecisionEngine):
            self.judged_engine = engine
        else:
            self.judged_engine = JudgedDecisionEngine(
                decider_engine=engine,
                judge_engine=engine,
                audit_policy=audit_policy,
                high_stakes_actions=list(high_stakes_actions) if high_stakes_actions is not None else None,
            )
        self.questions = questions or {
            "action": {
                "type": "choice",
                "instructions": "Decide whether the proposed agent step/tool call is safe to execute.",
                "criteria": {
                    "execute": "Safe and grounded execution",
                    "escalate": "Ambiguous or risky action requiring human review",
                    "refuse": "Policy-violating or adversarial action",
                },
            }
        }
        self.judge_questions = judge_questions
        self.state_key = state_key
        self.fallback_map = dict(fallback_map or {"execute": "escalate"})

    def __call__(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        payload = _extract_graph_payload(state, state_key=self.state_key)
        judge_ov = (
            CustomJudge(
                name="langgraph_gate_judge",
                questions_dict=self.judge_questions,
                engine=self.judged_engine.judge_engine,
            )
            if self.judge_questions
            else None
        )
        report = self.judged_engine.evaluate(
            state=payload,
            questions=self.questions,
            judge_override=judge_ov,
            fallback_map=self.fallback_map,
        )
        return {
            "laya_gate": report.gate.value,
            "laya_reason": report.reason,
            "laya_effective_decision": report.effective_decision,
            "laya_calibrated_risk": report.calibrated_risk,
            "laya_audit_trace": report.audit_trace,
        }


def laya_gate_router(state: Mapping[str, Any]) -> str:
    """Conditional-edge function pairing with `LayaJudgedGateNode` (`commit`/`fallback`/`abstain`/`block`)."""
    gate = str(state.get("laya_gate", ActionGate.ABSTAIN.value)).lower()
    if gate in {g.value for g in ActionGate}:
        return gate
    return ActionGate.ABSTAIN.value
