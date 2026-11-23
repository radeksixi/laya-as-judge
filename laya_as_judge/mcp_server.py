"""Zero-dependency Model Context Protocol (MCP) stdio JSON-RPC 2.0 server for `laya-as-judge`.

Exposes `laya-as-judge`'s zero-output-token Decision, Judge, Hybrid Judged-Decision (`ActionGate`),
Moderation, and Rerank engines directly as standard MCP tools so any MCP client (Claude Desktop,
Cursor, Windsurf, VS Code, LangGraph, FastMCP) can invoke them via:

    uvx laya-as-judge mcp
    # or with local neural servers ($0 cloud):
    laya-judge mcp --profile rtx-16gb --tier1-url http://127.0.0.1:8080 --tier2-url http://127.0.0.1:8081
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Mapping, Optional, TextIO

from .dropins import CandidateRankerEngine, ModerationAdapter
from .engine.base import BaseDecisionEngine
from .engine.http_systemone_backend import _extract_simplex_from_answer, _pack_canonical_answer, normalize_decision_questions
from .judged_decision import AuditPolicy, JudgedDecisionEngine
from .judges import (
    AgentTrajectoryJudge,
    AnswerRelevanceJudge,
    FaithfulnessJudge,
    InstructionAdherenceJudge,
    PairwiseComparisonJudge,
    SafetyGuardJudge,
)
from .profiles import build_local_profile_engines

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

_PRESET_JUDGES = {
    "faithfulness": FaithfulnessJudge,
    "relevance": AnswerRelevanceJudge,
    "instruction": InstructionAdherenceJudge,
    "safety": SafetyGuardJudge,
    "pairwise": PairwiseComparisonJudge,
    "agent": AgentTrajectoryJudge,
}


def mcp_tool_definitions() -> List[Dict[str, Any]]:
    """Return MCP `tools/list` schema descriptors for all 5 zero-output-token Laya tools."""
    return [
        {
            "name": "laya_decide",
            "description": (
                "Evaluate typed decision questions (noul/bool, choice, score) over a state with zero output tokens "
                "and calibrated probability simplices."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": {"description": "Text or JSON object representing the context to evaluate."},
                    "questions": {
                        "type": "object",
                        "description": "Mapping of question_id -> {type: 'noul'|'choice'|'score', instructions, criteria}.",
                    },
                },
                "required": ["state", "questions"],
            },
        },
        {
            "name": "laya_judge",
            "description": (
                "Run a built-in zero-output-token evaluation preset (faithfulness, safety, relevance, "
                "instruction, pairwise, agent) over an input state."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "judge": {
                        "type": "string",
                        "enum": list(_PRESET_JUDGES.keys()),
                        "default": "safety",
                        "description": "Built-in judge preset name.",
                    },
                    "state": {"description": "Input text or structured dict for the judge preset."},
                },
                "required": ["state"],
            },
        },
        {
            "name": "laya_judged_decision",
            "description": (
                "Hybrid governance gate (Decide -> Orthogonal Judge -> Deterministic Policy Rules) returning "
                "an explicit 4-way ActionGate (commit, fallback, abstain, block) and JSON audit_trace."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": {"description": "Agent state or user request to decide and audit."},
                    "questions": {"type": "object", "description": "Typed operational decision questions."},
                    "high_stakes_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Action labels that force orthogonal judge auditing (fail-closed if omitted).",
                    },
                    "fallback_map": {
                        "type": "object",
                        "description": "Optional safe substitute mapping (e.g. {'issue_refund': 'request_more_info'}).",
                    },
                    "audit_policy": {
                        "type": "string",
                        "enum": ["on_high_stakes", "always", "on_uncertain"],
                        "default": "on_high_stakes",
                    },
                },
                "required": ["state", "questions"],
            },
        },
        {
            "name": "laya_moderate",
            "description": (
                "OpenAI-compatible multi-category content moderation with deterministic prompt-injection "
                "and spoofed-audit detection (zero output tokens)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": {"description": "Single string or array of strings to moderate."},
                },
                "required": ["input"],
            },
        },
        {
            "name": "laya_rerank",
            "description": "Zero-output-token RAG document reranker returning documents sorted by relevance_score.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Candidate documents to rank.",
                    },
                    "top_n": {"type": "integer", "minimum": 1, "description": "Optional top-N cutoff."},
                },
                "required": ["query", "documents"],
            },
        },
    ]


class LayaMCPServer:
    """In-process MCP JSON-RPC 2.0 dispatcher backed by any `laya-as-judge` engine profile."""

    def __init__(
        self,
        *,
        profile: str = "emulator",
        decider_engine: Optional[BaseDecisionEngine] = None,
        judge_engine: Optional[BaseDecisionEngine] = None,
        tier1_url: Optional[str] = None,
        tier2_url: Optional[str] = None,
        fallback_to_emulator: bool = True,
    ):
        if decider_engine is None:
            decider, judge_eng, guard_eng, hw = build_local_profile_engines(
                profile_name=profile,
                tier1_url=tier1_url,
                tier2_url=tier2_url,
                fallback_to_emulator=fallback_to_emulator,
            )
            self.decider_engine = decider
            self.judge_engine = judge_engine or judge_eng
            self.guard_engine = guard_eng
            self.profile_name = hw.name
        else:
            self.decider_engine = decider_engine
            self.judge_engine = judge_engine or decider_engine
            self.guard_engine = None
            self.profile_name = profile

        self.judged_engine = JudgedDecisionEngine(
            decider_engine=self.decider_engine,
            judge_engine=self.judge_engine,
            guard_engine=self.guard_engine,
            audit_policy=AuditPolicy.ON_HIGH_STAKES,
        )
        self.moderation = ModerationAdapter(engine=self.judge_engine)
        self.ranker = CandidateRankerEngine(engine=self.judge_engine)
        self.preset_judges = {k: cls(engine=self.judge_engine) for k, cls in _PRESET_JUDGES.items()}

    def _dispatch_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        if name == "laya_decide":
            norm_q = normalize_decision_questions(arguments["questions"])
            raw = self.decider_engine.predict(arguments["state"], norm_q)
            answers: Dict[str, Any] = {}
            for qid, qspec in norm_q.items():
                labels, probs = _extract_simplex_from_answer((raw.get("answers") or {}).get(qid, {}), qspec)
                answers[qid] = _pack_canonical_answer(labels, probs, qspec)
            return {
                "model": raw.get("model", self.decider_engine.model_id),
                "answers": answers,
                "usage": {"input_tokens": int((raw.get("usage") or {}).get("input_tokens", 0)), "output_tokens": 0},
            }

        if name == "laya_judge":
            key = str(arguments.get("judge", "safety")).strip().lower()
            if key not in self.preset_judges:
                raise ValueError(f"Unknown judge preset '{key}'. Expected one of: {', '.join(self.preset_judges)}")
            state = arguments["state"]
            if not isinstance(state, dict):
                state = {"text": str(state)}
            return self.preset_judges[key].evaluate(state).to_dict()

        if name == "laya_judged_decision":
            norm_q = normalize_decision_questions(arguments["questions"])
            rep = self.judged_engine.evaluate(
                arguments["state"],
                norm_q,
                audit_policy=arguments.get("audit_policy"),
                high_stakes_actions=arguments.get("high_stakes_actions"),
                fallback_map=arguments.get("fallback_map"),
            )
            return rep.to_dict()

        if name == "laya_moderate":
            return self.moderation.moderate(arguments["input"])

        if name == "laya_rerank":
            return self.ranker.rerank(
                query=str(arguments["query"]),
                documents=list(arguments["documents"]),
                top_n=arguments.get("top_n"),
                return_documents=True,
            )

        raise KeyError(f"Unknown MCP tool '{name}'")

    def handle_message(self, msg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Process one JSON-RPC 2.0 request/notification dict; returns response dict or None for notifications."""
        from . import __version__

        req_id = msg.get("id")
        method = str(msg.get("method", ""))
        params = msg.get("params") or {}

        # Notifications carry no id and expect no response
        if req_id is None and method.startswith("notifications/"):
            return None

        if method == "initialize":
            client_proto = str(params.get("protocolVersion", SUPPORTED_PROTOCOL_VERSIONS[0]))
            proto = client_proto if client_proto in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "laya-as-judge", "version": __version__},
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": mcp_tool_definitions()}}

        if method == "tools/call":
            tool_name = str(params.get("name", ""))
            tool_args = params.get("arguments") or {}
            try:
                out = self._dispatch_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def serve_stdio(self, stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None) -> None:
        """Run the blocking newline-delimited JSON-RPC 2.0 MCP stdio loop."""
        in_stream = stdin or sys.stdin
        out_stream = stdout or sys.stdout
        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {exc}"}}
                out_stream.write(json.dumps(err, ensure_ascii=False) + "\n")
                out_stream.flush()
                continue
            if not isinstance(msg, dict):
                continue
            resp = self.handle_message(msg)
            if resp is not None:
                out_stream.write(json.dumps(resp, ensure_ascii=False) + "\n")
                out_stream.flush()
