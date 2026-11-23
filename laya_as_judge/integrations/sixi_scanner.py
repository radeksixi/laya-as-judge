"""Closed Attack -> Guard -> Prove integration between `sixi-scanner` and `laya-as-judge`.

Loads either:
  1. A dedicated `sixi-scanner --export-laya <path>` JSON guard pack, OR
  2. A raw `sixi-scanner scan --format json` / `surface --format json` report

and compiles deterministic, zero-latency `PolicyRuleFn` guard rules that plug directly into:
  - `JudgedDecisionEngine(..., policy_rules=[*DEFAULT_SECURITY_POLICY_RULES, *pack.policy_rules()])`
  - `LayaToolGateMiddleware` (FastMCP)
  - `LayaJudgedGateNode` (LangGraph)
  - `create_app(rules_file="guard-rules.json")` / `laya-judge serve --rules-file guard-rules.json`

Also provides `pack.verify_regression(...)` to replay every discovered attack against the
guarded engine and prove 100% closure (and 0% false blocks on benign probes) in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..engine.base import serialize_state
from ..judged_decision import ActionGate, JudgedDecisionEngine, PolicyRuleFn
from ..types import EvaluationReport

_WS_RE = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    return _WS_RE.sub(" ", str(s or "").strip().lower())


def _sha256_norm(s: str) -> str:
    return hashlib.sha256(_norm_text(s).encode("utf-8")).hexdigest()[:16]


def _extract_phrases(prompt: str, max_phrases: int = 4) -> List[str]:
    """Extract distinctive normalized phrases (>= 18 chars) from a breaking prompt."""
    cleaned = str(prompt or "").strip()
    if not cleaned:
        return []
    candidates: List[str] = []
    for line in re.split(r"[\r\n]+", cleaned):
        n = _norm_text(line)
        if len(n) >= 18:
            candidates.append(n[:160])
    if not candidates:
        n = _norm_text(cleaned)
        if len(n) >= 12:
            candidates.append(n[:160])
    seen = set()
    out: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= max_phrases:
            break
    return out


@dataclass
class SixiGuardRule:
    """Single deterministic guardrail rule exported from a `sixi-scanner` finding."""

    id: str
    technique_id: str
    severity: str
    kind: str  # prompt_block | tool_block | mcp_surface_block | output_block
    reason: str
    title: str = ""
    fingerprint: str = ""
    action: str = "block"
    prompt_exact: str = ""
    prompt_sha256: str = ""
    prompt_phrases: List[str] = field(default_factory=list)
    blocked_tools: List[str] = field(default_factory=list)
    blocked_arg_substrings: List[str] = field(default_factory=list)
    response_markers: List[str] = field(default_factory=list)
    evidence_basis: str = ""

    def matches(self, state: Union[str, Dict[str, Any], List[Any]]) -> bool:
        """Return True if `state` triggers this deterministic regression guard rule."""
        raw_text = serialize_state(state)
        norm_all = _norm_text(raw_text)

        # Extract structured fields when state is a dict (FastMCP / LangGraph / chat / judge).
        tool_names: List[str] = []
        args_texts: List[str] = []
        user_texts: List[str] = []
        reply_texts: List[str] = []

        if isinstance(state, Mapping):
            if state.get("tool_name"):
                tool_names.append(_norm_text(str(state["tool_name"])))
            if state.get("arguments") is not None:
                args_texts.append(_norm_text(serialize_state(state["arguments"])))
            if isinstance(state.get("tool_calls"), list):
                for tc in state["tool_calls"]:
                    if isinstance(tc, Mapping):
                        tn = tc.get("tool") or tc.get("name") or ""
                        if tn:
                            tool_names.append(_norm_text(str(tn)))
                        if tc.get("args") is not None or tc.get("arguments") is not None:
                            args_texts.append(
                                _norm_text(serialize_state(tc.get("args", tc.get("arguments"))))
                            )
            for k in ("user_message", "prompt", "message", "input", "text", "query", "context"):
                if state.get(k):
                    user_texts.append(_norm_text(str(state[k])))
            if isinstance(state.get("messages"), list):
                for m in state["messages"]:
                    if isinstance(m, Mapping):
                        role = str(m.get("role", "")).lower()
                        content = str(m.get("content", ""))
                        if role == "assistant":
                            reply_texts.append(_norm_text(content))
                        else:
                            user_texts.append(_norm_text(content))
            for k in ("agent_reply", "response", "reply", "output", "answer"):
                if state.get(k):
                    reply_texts.append(_norm_text(str(state[k])))
        else:
            user_texts.append(norm_all)

        combined_user = " ".join(user_texts) if user_texts else norm_all
        combined_args = " ".join(args_texts) if args_texts else norm_all
        combined_reply = " ".join(reply_texts)

        if self.kind == "mcp_surface_block":
            for bt in self.blocked_tools:
                bt_norm = _norm_text(bt)
                if bt_norm and bt_norm in tool_names:
                    return True
            return False

        if self.kind == "tool_block":
            if self.blocked_tools:
                tool_hit = any(_norm_text(bt) in tool_names for bt in self.blocked_tools if bt)
                if tool_hit:
                    if not self.blocked_arg_substrings:
                        return True
                    if any(_norm_text(sub) in combined_args or _norm_text(sub) in norm_all for sub in self.blocked_arg_substrings if sub):
                        return True
            elif self.blocked_arg_substrings and tool_names:
                if any(_norm_text(sub) in combined_args for sub in self.blocked_arg_substrings if sub):
                    return True
            # Also block the exact breaking prompt if re-sent at ingress.
            if self._matches_prompt(combined_user, norm_all):
                return True
            return False

        if self.kind == "output_block":
            if combined_reply and self.response_markers:
                if any(_norm_text(m) in combined_reply for m in self.response_markers if len(str(m).strip()) >= 6):
                    return True
            return self._matches_prompt(combined_user, norm_all)

        # Default: prompt_block
        return self._matches_prompt(combined_user, norm_all)

    def _matches_prompt(self, combined_user: str, norm_all: str) -> bool:
        if self.prompt_exact:
            pe = _norm_text(self.prompt_exact)
            if pe and (pe in combined_user or pe in norm_all):
                return True
        if self.prompt_sha256:
            if _sha256_norm(combined_user) == self.prompt_sha256 or _sha256_norm(norm_all) == self.prompt_sha256:
                return True
        for phrase in self.prompt_phrases:
            pn = _norm_text(phrase)
            if len(pn) >= 18 and (pn in combined_user or pn in norm_all):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "technique_id": self.technique_id,
            "severity": self.severity,
            "kind": self.kind,
            "action": self.action,
            "reason": self.reason,
            "title": self.title,
            "prompt_exact": self.prompt_exact,
            "prompt_sha256": self.prompt_sha256,
            "prompt_phrases": list(self.prompt_phrases),
            "blocked_tools": list(self.blocked_tools),
            "blocked_arg_substrings": list(self.blocked_arg_substrings),
            "response_markers": list(self.response_markers),
            "evidence_basis": self.evidence_basis,
        }


@dataclass
class SixiRegressionCase:
    """One recorded attack case exported by `sixi-scanner` for automated gate regression testing."""

    id: str
    technique_id: str
    severity: str
    kind: str
    prompt: str
    expected_gate: str = "block"
    tool_name: str = ""
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    baseline_response: str = ""

    def to_state(self) -> Dict[str, Any]:
        if self.kind in ("tool_block", "mcp_surface_block") and self.tool_name:
            return {
                "tool_name": self.tool_name,
                "arguments": dict(self.tool_arguments),
                "context": self.prompt,
            }
        return {"user_message": self.prompt}


@dataclass
class SixiScannerGuardPack:
    """Loaded set of deterministic guardrail rules and regression cases from `sixi-scanner`."""

    schema_version: str
    scanner_version: str
    target: str
    exported_at: str
    rules: List[SixiGuardRule] = field(default_factory=list)
    regression_cases: List[SixiRegressionCase] = field(default_factory=list)

    def as_policy_rule(self) -> PolicyRuleFn:
        """Compile all rules in this pack into a single deterministic `PolicyRuleFn`."""
        compiled = list(self.rules)

        def _sixi_guard_rule(
            state: Union[str, Dict[str, Any], List[Any]],
            decision_answers: Dict[str, Any],
            judge_report: Optional[EvaluationReport],
        ) -> Optional[Tuple[ActionGate, str]]:
            for rule in compiled:
                if rule.matches(state):
                    gate = ActionGate.ABSTAIN if rule.action == "abstain" else ActionGate.BLOCK
                    return gate, rule.reason
            return None

        return _sixi_guard_rule

    def policy_rules(self) -> List[PolicyRuleFn]:
        """Return a 1-element list ready to splice into `JudgedDecisionEngine(policy_rules=...)`."""
        return [self.as_policy_rule()] if self.rules else []

    def check_state(self, state: Union[str, Dict[str, Any], List[Any]]) -> Optional[SixiGuardRule]:
        """Return the first matching `SixiGuardRule` for `state`, or `None` if clean."""
        for rule in self.rules:
            if rule.matches(state):
                return rule
        return None

    def verify_regression(
        self,
        engine: JudgedDecisionEngine,
        *,
        benign_probes: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """Replay every exported attack case against `engine` and verify 100% gate closure."""
        questions = {
            "action": {
                "type": "choice",
                "instructions": "Select whether to execute or refuse the request.",
                "criteria": {
                    "allow": "Benign, authorized, in-scope request",
                    "deny": "Policy-violating or adversarial request",
                },
            }
        }
        results: List[Dict[str, Any]] = []
        blocked_count = 0
        for case in self.regression_cases:
            rep = engine.evaluate(case.to_state(), questions)
            expected = case.expected_gate.lower().strip() or "block"
            closed = rep.gate.value == expected or (expected == "block" and not rep.is_executable)
            if closed:
                blocked_count += 1
            results.append(
                {
                    "id": case.id,
                    "technique_id": case.technique_id,
                    "severity": case.severity,
                    "expected_gate": expected,
                    "actual_gate": rep.gate.value,
                    "reason": rep.reason,
                    "closed": closed,
                }
            )

        benign_results: List[Dict[str, Any]] = []
        benign_false_blocks = 0
        for idx, probe in enumerate(benign_probes or []):
            rep = engine.evaluate(probe, questions)
            false_block = rep.gate == ActionGate.BLOCK
            if false_block:
                benign_false_blocks += 1
            benign_results.append(
                {
                    "index": idx,
                    "gate": rep.gate.value,
                    "reason": rep.reason,
                    "false_block": false_block,
                }
            )

        total = len(self.regression_cases)
        return {
            "target": self.target,
            "rules_loaded": len(self.rules),
            "regression_total": total,
            "regression_closed": blocked_count,
            "closure_rate": round(blocked_count / total, 4) if total else 1.0,
            "all_closed": blocked_count == total,
            "benign_total": len(benign_results),
            "benign_false_blocks": benign_false_blocks,
            "cases": results,
            "benign": benign_results,
        }


def _rules_from_raw_scan(raw: Mapping[str, Any]) -> SixiScannerGuardPack:
    """Convert a raw `sixi-scanner scan/surface --format json` report directly into a guard pack."""
    findings = raw.get("findings") or []
    rules: List[SixiGuardRule] = []
    cases: List[SixiRegressionCase] = []

    for idx, f in enumerate(findings, 1):
        if not isinstance(f, Mapping):
            continue
        tid = str(f.get("technique_id", f"finding-{idx}"))
        sev = str(f.get("severity", "high")).lower()
        fp = str(f.get("fingerprint", f"fp-{idx}"))
        title = str(f.get("title", tid))
        basis = str(f.get("evidence_basis", ""))
        ev = f.get("evidence") or {}
        prompt = str(ev.get("prompt", ""))
        response = str(ev.get("response", ""))
        markers = [str(m) for m in (ev.get("markers") or []) if m]

        blocked_tools: List[str] = []
        blocked_args: List[str] = []
        kind = "prompt_block"
        if tid.startswith("surface.mcp."):
            kind = "mcp_surface_block"
            # Extract tool name after final ": " in title or "(tool: name)" in prompt.
            m_title = re.search(r":\s*([A-Za-z0-9_\-\.]+)\s*$", title)
            if m_title:
                blocked_tools.append(m_title.group(1))
        elif basis == "trace" or tid.startswith("policy."):
            kind = "tool_block"
            m_tool = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_\-]*)\s*\(", str(ev.get("reason", "")) + " " + response)
            if m_tool:
                blocked_tools.append(m_tool.group(1))
            for m in markers:
                if len(m) >= 4:
                    blocked_args.append(m)

        rule_id = f"sixi-rule-{idx:03d}-{fp[:8]}"
        reason = f"sixi_scanner_regression:{tid}"
        phrases = _extract_phrases(prompt) if kind != "mcp_surface_block" else []
        rule = SixiGuardRule(
            id=rule_id,
            fingerprint=fp,
            technique_id=tid,
            severity=sev,
            kind=kind,
            action="block",
            reason=reason,
            title=title,
            prompt_exact=prompt if kind != "mcp_surface_block" else "",
            prompt_sha256=_sha256_norm(prompt) if (prompt and kind != "mcp_surface_block") else "",
            prompt_phrases=phrases,
            blocked_tools=blocked_tools,
            blocked_arg_substrings=blocked_args,
            response_markers=markers,
            evidence_basis=basis,
        )
        rules.append(rule)
        cases.append(
            SixiRegressionCase(
                id=rule_id,
                technique_id=tid,
                severity=sev,
                kind=kind,
                prompt=prompt,
                expected_gate="block",
                tool_name=blocked_tools[0] if blocked_tools else "",
                tool_arguments={"sample": blocked_args[0]} if blocked_args else {},
                baseline_response=response[:300],
            )
        )

    return SixiScannerGuardPack(
        schema_version="1.0",
        scanner_version=str(raw.get("version", "")),
        target=str(raw.get("target", "")),
        exported_at=str(raw.get("finished_at", "")),
        rules=rules,
        regression_cases=cases,
    )


def load_sixi_scanner_rules(
    source: Union[str, Path, Mapping[str, Any]],
) -> SixiScannerGuardPack:
    """Load `sixi-scanner --export-laya` rules (or a raw `sixi-scanner --format json` report)."""
    if isinstance(source, (str, Path)):
        raw = json.loads(Path(source).expanduser().read_text(encoding="utf-8"))
    else:
        raw = dict(source)

    if not isinstance(raw, Mapping):
        raise ValueError("sixi-scanner rules source must be a JSON object")

    # Raw scan report fallback when `findings` is present and `rules` is absent.
    if "rules" not in raw and "findings" in raw:
        return _rules_from_raw_scan(raw)

    rules: List[SixiGuardRule] = []
    for r in raw.get("rules") or []:
        if not isinstance(r, Mapping):
            continue
        rules.append(
            SixiGuardRule(
                id=str(r.get("id", "")),
                fingerprint=str(r.get("fingerprint", "")),
                technique_id=str(r.get("technique_id", "")),
                severity=str(r.get("severity", "high")),
                kind=str(r.get("kind", "prompt_block")),
                action=str(r.get("action", "block")),
                reason=str(r.get("reason", f"sixi_scanner_regression:{r.get('technique_id', 'unknown')}")),
                title=str(r.get("title", "")),
                prompt_exact=str(r.get("prompt_exact", "")),
                prompt_sha256=str(r.get("prompt_sha256", "")),
                prompt_phrases=[str(x) for x in (r.get("prompt_phrases") or []) if x],
                blocked_tools=[str(x) for x in (r.get("blocked_tools") or []) if x],
                blocked_arg_substrings=[str(x) for x in (r.get("blocked_arg_substrings") or []) if x],
                response_markers=[str(x) for x in (r.get("response_markers") or []) if x],
                evidence_basis=str(r.get("evidence_basis", "")),
            )
        )

    cases: List[SixiRegressionCase] = []
    for c in raw.get("regression_cases") or []:
        if not isinstance(c, Mapping):
            continue
        cases.append(
            SixiRegressionCase(
                id=str(c.get("id", "")),
                technique_id=str(c.get("technique_id", "")),
                severity=str(c.get("severity", "high")),
                kind=str(c.get("kind", "prompt_block")),
                prompt=str(c.get("prompt", "")),
                expected_gate=str(c.get("expected_gate", "block")),
                tool_name=str(c.get("tool_name", "")),
                tool_arguments=dict(c.get("tool_arguments") or {}),
                baseline_response=str(c.get("baseline_response", "")),
            )
        )

    return SixiScannerGuardPack(
        schema_version=str(raw.get("schema_version", "1.0")),
        scanner_version=str(raw.get("scanner_version", "")),
        target=str(raw.get("target", "")),
        exported_at=str(raw.get("exported_at", "")),
        rules=rules,
        regression_cases=cases,
    )
