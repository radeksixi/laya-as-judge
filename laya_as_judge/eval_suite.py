"""Multi-benchmark evaluation suite for typed decision models (Accuracy, Macro-F1, Top-1 ECE, Total Price).

Covers the Gemini Decision 1 / TypeSafe Jev 1.14 comparative benchmark table:
  - Text & Typed-Decision Core:
      1. `jevbench` (JevBench public split: noul, choice, score)
      2. `amazon_esci` (Amazon Shopping Queries ESCI: 4-class Macro-F1)
      3. `when2call` (NVIDIA When2Call: 4-way tool invocation routing)
      4. `pubmedqa` (PubMedQA labeled: yes / no / maybe biomedical grounding)
      5. `global_mmlu_lite` (Cohere Global-MMLU-Lite: multilingual 4-way QA)
      6. `new_yorker_caption` (New Yorker Caption Contest text matching: 5-way choice)
  - Raw Multimodal Gate (documented N/A on text-only Laya; auto-escalated in CascadedJudge):
      7. `mind2web_multimodal` (UI screenshot + DOM action selection)
      8. `mmau` (Massive Multi-Task Audio Understanding)
      9. `video_mme` (Video-MME without subtitles)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .calibration import (
    PRICING_PROFILES,
    MetaCalibrator,
    TemperatureScaler,
    accuracy_score,
    aurc,
    brier_score,
    bucket_key_for_question,
    estimate_suite_prices,
    macro_f1_score,
    negative_log_likelihood,
    oof_calibrate_pooled,
    rescale_probabilities,
    selective_accuracy_at_coverage,
    top1_ece,
)
from .engine import BaseDecisionEngine, get_engine


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata and capability specification for a benchmark in the evaluation table."""

    key: str
    display_name: str
    primary_metric: str  # "accuracy" or "macro_f1"
    modality: str  # "text", "image+text", "audio", "video"
    laya_text_supported: bool
    recommended_checkpoint: str  # "typed-decisions", "english", "multilingual", or "tier2_multimodal"
    recommended_max_len: int = 512
    description: str = ""


BENCHMARK_SPECS: Dict[str, BenchmarkSpec] = {
    "jevbench": BenchmarkSpec(
        key="jevbench",
        display_name="JevBench (public)",
        primary_metric="accuracy",
        modality="text",
        laya_text_supported=True,
        recommended_checkpoint="english",
        recommended_max_len=1024,
        description="Typed probabilistic decisions (noul, choice, score) across 12 business domains.",
    ),
    "amazon_esci": BenchmarkSpec(
        key="amazon_esci",
        display_name="Amazon ESCI",
        primary_metric="macro_f1",
        modality="text",
        laya_text_supported=True,
        recommended_checkpoint="multilingual",
        recommended_max_len=512,
        description="4-level ordinal e-commerce search relevance (Irrelevant, Complement, Substitute, Exact).",
    ),
    "when2call": BenchmarkSpec(
        key="when2call",
        display_name="When2Call",
        primary_metric="accuracy",
        modality="text",
        laya_text_supported=True,
        recommended_checkpoint="typed-decisions",
        recommended_max_len=1024,
        description="Agent tool-calling & abstention judgment (direct, tool_call, request_for_info, cannot_answer).",
    ),
    "pubmedqa": BenchmarkSpec(
        key="pubmedqa",
        display_name="PubMedQA",
        primary_metric="accuracy",
        modality="text",
        laya_text_supported=True,
        recommended_checkpoint="typed-decisions",
        recommended_max_len=1024,
        description="Biomedical research QA grounded in PubMed abstracts (yes, no, maybe).",
    ),
    "global_mmlu_lite": BenchmarkSpec(
        key="global_mmlu_lite",
        display_name="Global-MMLU-Lite",
        primary_metric="accuracy",
        modality="text",
        laya_text_supported=True,
        recommended_checkpoint="router",
        recommended_max_len=512,
        description="Multilingual knowledge & reasoning across languages; stress-tests OOD calibration & cascade.",
    ),
    "new_yorker_caption": BenchmarkSpec(
        key="new_yorker_caption",
        display_name="New Yorker Caption",
        primary_metric="accuracy",
        modality="text",
        laya_text_supported=True,
        recommended_checkpoint="english",
        recommended_max_len=512,
        description="Text-only cartoon scene description to punchline caption matching (5 candidates).",
    ),
    "mind2web_multimodal": BenchmarkSpec(
        key="mind2web_multimodal",
        display_name="Mind2Web Multimodal",
        primary_metric="accuracy",
        modality="image+text",
        laya_text_supported=False,
        recommended_checkpoint="tier2_multimodal",
        recommended_max_len=2048,
        description="Requires webpage screenshots; unsupported by text-only Laya (escalate to Tier 2 or use DOM text).",
    ),
    "mmau": BenchmarkSpec(
        key="mmau",
        display_name="MMAU",
        primary_metric="accuracy",
        modality="audio",
        laya_text_supported=False,
        recommended_checkpoint="tier2_multimodal",
        recommended_max_len=1024,
        description="Raw audio perception & reasoning; requires native audio decision model (Tier 2).",
    ),
    "video_mme": BenchmarkSpec(
        key="video_mme",
        display_name="Video-MME",
        primary_metric="accuracy",
        modality="video",
        laya_text_supported=False,
        recommended_checkpoint="tier2_multimodal",
        recommended_max_len=2048,
        description="Raw video understanding without subtitles; requires native video decision model (Tier 2).",
    ),
}


@dataclass
class PreparedCase:
    """One state with one or more typed questions and aligned ground-truth labels."""

    case_id: str
    state: Union[str, Dict[str, Any], List[Any]]
    questions: Dict[str, Dict[str, Any]]
    # Mapping question_id -> (gold_label_str_or_val, gold_option_index, option_labels_list)
    gold: Dict[str, Tuple[Any, int, List[Any]]]


@dataclass
class BenchmarkTaskResult:
    """Aggregated result for a single benchmark in the suite."""

    benchmark: str
    display_name: str
    status: str  # "completed" or "skipped_unsupported_modality"
    primary_metric_name: str
    primary_score: Optional[float] = None
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    top1_ece: Optional[float] = None
    calibrated_top1_ece: Optional[float] = None
    brier: Optional[float] = None
    nll: Optional[float] = None
    aurc_score: Optional[float] = None
    accuracy_at_80pct_coverage: Optional[float] = None
    sample_count: int = 0
    case_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    actual_cost_usd: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    fitted_temperature: Optional[float] = None
    notes: str = ""

    # Raw per-question arrays kept in-memory for cross-suite pooled ECE & calibration
    _probs_list: List[np.ndarray] = field(default_factory=list, repr=False)
    _gold_indices: List[int] = field(default_factory=list, repr=False)
    _buckets: List[str] = field(default_factory=list, repr=False)
    _tier_tags: List[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "display_name": self.display_name,
            "status": self.status,
            "primary_metric": self.primary_metric_name,
            "primary_score": round(self.primary_score, 4) if self.primary_score is not None else None,
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "macro_f1": round(self.macro_f1, 4) if self.macro_f1 is not None else None,
            "top1_ece": round(self.top1_ece, 4) if self.top1_ece is not None else None,
            "calibrated_top1_ece": (
                round(self.calibrated_top1_ece, 4) if self.calibrated_top1_ece is not None else None
            ),
            "brier": round(self.brier, 4) if self.brier is not None else None,
            "nll": round(self.nll, 4) if self.nll is not None else None,
            "aurc": round(self.aurc_score, 4) if self.aurc_score is not None else None,
            "accuracy_at_80pct_coverage": (
                round(self.accuracy_at_80pct_coverage, 4)
                if self.accuracy_at_80pct_coverage is not None
                else None
            ),
            "sample_count": self.sample_count,
            "case_count": self.case_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "actual_cost_usd": round(self.actual_cost_usd, 6),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "fitted_temperature": round(self.fitted_temperature, 4) if self.fitted_temperature is not None else None,
            "notes": self.notes,
        }


@dataclass
class SuiteEvaluationSummary:
    """Full table summary across benchmarks + pooled bottom rows (`Top-1 ECE` & `Total price`)."""

    backend: str
    model: str
    results: List[BenchmarkTaskResult]
    pooled_top1_ece: float
    pooled_calibrated_top1_ece: Optional[float]
    pooled_brier: float
    pooled_aurc: float
    total_questions_evaluated: int
    total_cases_evaluated: int
    total_input_tokens: int
    total_output_tokens: int
    total_price_usd: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "total_questions_evaluated": self.total_questions_evaluated,
            "total_cases_evaluated": self.total_cases_evaluated,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "pooled_top1_ece": round(self.pooled_top1_ece, 4),
            "pooled_calibrated_top1_ece": (
                round(self.pooled_calibrated_top1_ece, 4)
                if self.pooled_calibrated_top1_ece is not None
                else None
            ),
            "pooled_brier": round(self.pooled_brier, 4),
            "pooled_aurc": round(self.pooled_aurc, 4),
            "total_price_usd": self.total_price_usd,
            "benchmarks": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Dataset Loaders (Hugging Face + deterministic offline CI fixtures)
# ---------------------------------------------------------------------------

ESCI_CRITERIA: Dict[str, str] = {
    "Exact": "exact match: product IS the requested item type and violates NO explicit brand, model, size, color, or constraint in the query (default positive match for broad queries)",
    "Substitute": "substitute: functional replacement of the SAME product type ONLY when it misses or changes an explicit brand, model, size, color, count, or constraint stated in the query",
    "Complement": "complement: separate companion item, accessory, replacement part, consumable, attachment, or related series installment meant to be used WITH/ALONGSIDE the queried item rather than replacing it",
    "Irrelevant": "irrelevant: wrong product category, superficial keyword overlap, or item a shopper searching this query would neither use as a replacement nor buy alongside it",
}

ESCI_LABEL_ORDER: List[str] = ["Irrelevant", "Complement", "Substitute", "Exact"]

ESCI_SCORE_LEVELS: List[str] = [
    "Irrelevant: unrelated product or wrong category for the query",
    "Complement: accessory, add-on, or companion item rather than the main product",
    "Substitute: functional alternative missing a specific brand, model, or attribute",
    "Exact: exact match satisfying all requirements of the search query",
]

WHEN2CALL_CRITERIA: Dict[str, str] = {
    "direct": "answer directly in natural language without calling any tool (no tool needed)",
    "tool_call": "invoke one of the provided tools because a matching tool and required parameters are available",
    "request_for_info": "ask the user a clarifying question because a matching tool exists but required details are missing",
    "cannot_answer": "politely decline or state inability because none of the provided tools can perform the requested action",
}

PUBMEDQA_CRITERIA: Dict[str, str] = {
    "yes": "the study results and abstract evidence support an affirmative answer",
    "no": "the study results and abstract evidence refute the hypothesis or show no effect",
    "maybe": "the results are inconclusive, mixed, conditional, or insufficient to answer definitively",
}


def _compact_tool_spec(raw_tool: str) -> Dict[str, Any]:
    """Compress verbose JSON Schema tool definitions so multiple tools fit in 512-1024 tokens."""
    try:
        t = json.loads(raw_tool) if isinstance(raw_tool, str) else dict(raw_tool)
    except Exception:
        return {"raw": str(raw_tool)[:240]}
    params = t.get("parameters", {})
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    req = set(params.get("required", [])) if isinstance(params, dict) else set()
    compact_props: Dict[str, str] = {}
    for pname, pinfo in props.items():
        if isinstance(pinfo, dict):
            ptype = pinfo.get("type", "any")
            flag = "required" if pname in req else "optional"
            desc = str(pinfo.get("description", ""))[:50]
            compact_props[pname] = f"{ptype} ({flag}): {desc}"
        else:
            compact_props[pname] = str(pinfo)[:45]
    return {
        "name": t.get("name", ""),
        "description": str(t.get("description", ""))[:120],
        "parameters": compact_props,
    }


def _format_pubmed_abstract(ctx_dict: Dict[str, Any], max_chars: int = 1900) -> str:
    """Format PubMedQA abstract preserving RESULTS/CONCLUSIONS tail when truncating."""
    paragraphs = list(ctx_dict.get("contexts", []))
    sec_labels = list(ctx_dict.get("labels", []))
    parts: List[str] = []
    for idx, para in enumerate(paragraphs):
        lbl = sec_labels[idx] if idx < len(sec_labels) else f"SEC_{idx}"
        parts.append(f"[{lbl}] {para}")
    full_text = " ".join(parts)
    if len(full_text) <= max_chars:
        return full_text
    # Keep beginning (background/objective) + full tail (methods/results)
    head_budget = int(max_chars * 0.30)
    tail_budget = max_chars - head_budget - 5
    return full_text[:head_budget] + " ... " + full_text[-tail_budget:]


def load_benchmark_cases(
    benchmark: str,
    limit: int = 100,
    source: str = "hf",
    seed: int = 42,
    mmlu_langs: Sequence[str] = ("en", "de", "es", "fr"),
    include_multimodal: bool = False,
) -> List[PreparedCase]:
    """Load prepared evaluation cases from Hugging Face ('hf') or deterministic fixtures ('fixture')."""
    if source == "fixture":
        return _load_fixture_cases(benchmark, limit=limit)

    from datasets import load_dataset
    from .multimodal_loaders import (
        analyze_when2call_row,
        build_enriched_esci_state,
        load_mind2web_multimodal_cases,
        load_mmau_cases,
        load_video_mme_cases,
    )

    rng = np.random.default_rng(seed)

    if benchmark == "jevbench":
        ds = load_dataset("JevBench/jevbench", "questions", split="test")
        all_grouped: Dict[str, Dict[str, Any]] = {}
        for row in ds:
            gold_str = str(row.get("gold", "")).strip()
            if gold_str == "" or gold_str.lower() == "unknown":
                continue
            cid = str(row["case_id"])
            if cid not in all_grouped:
                dom = str(row.get("domain") or cid.split("-")[0])
                all_grouped[cid] = {
                    "state": row["state"],
                    "domain": dom,
                    "questions": {},
                    "gold": {},
                    "raw_gold_tuple": {},
                }
            qkey = str(row["question_key"])
            qtype = str(row["type"])
            ins = str(row["instructions"])
            if qtype == "noul":
                qdef: Dict[str, Any] = {"type": "noul", "instructions": ins}
                g_bool = gold_str.lower() == "true"
                all_grouped[cid]["questions"][qkey] = qdef
                all_grouped[cid]["gold"][qkey] = (g_bool, 1 if g_bool else 0, [False, True])
                all_grouped[cid]["raw_gold_tuple"][qkey] = "true" if g_bool else "false"
            elif qtype == "choice":
                opts = row.get("options") or []
                crit = {str(o["id"]): str(o.get("description") or "") for o in opts}
                labels = list(crit.keys())
                if gold_str not in crit:
                    continue
                all_grouped[cid]["questions"][qkey] = {
                    "type": "choice",
                    "instructions": ins,
                    "criteria": crit,
                }
                all_grouped[cid]["gold"][qkey] = (gold_str, labels.index(gold_str), labels)
                all_grouped[cid]["raw_gold_tuple"][qkey] = gold_str
            elif qtype == "score":
                levels = [str(x) for x in (row.get("levels") or [])]
                g_idx = int(gold_str)
                if g_idx < 0 or g_idx >= len(levels):
                    continue
                all_grouped[cid]["questions"][qkey] = {
                    "type": "score",
                    "instructions": ins,
                    "criteria": levels,
                }
                all_grouped[cid]["gold"][qkey] = (g_idx, g_idx, list(range(len(levels))))
                all_grouped[cid]["raw_gold_tuple"][qkey] = str(g_idx)

        cids = [cid for cid, d in all_grouped.items() if d["questions"]]
        perm_cids = [cids[int(i)] for i in rng.permutation(len(cids))]
        raw_selected: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        total_q = 0
        for cid in perm_cids:
            data = all_grouped[cid]
            if total_q + len(data["questions"]) > limit and raw_selected:
                keep = list(data["questions"].keys())[: max(1, limit - total_q)]
                qs = {k: data["questions"][k] for k in keep}
                gs = {k: data["gold"][k] for k in keep}
            else:
                qs = data["questions"]
                gs = data["gold"]
            raw_selected.append((cid, qs, gs))
            total_q += len(qs)
            if total_q >= limit:
                break

        eval_cid_set = {cid for cid, _, _ in raw_selected}
        precedents_by_domain: Dict[str, List[Dict[str, str]]] = {}
        if include_multimodal:
            precedent_cid_set: set[str] = set()
            for other_cid in cids:
                if other_cid in eval_cid_set:
                    continue
                precedent_cid_set.add(other_cid)
                od = all_grouped[other_cid]
                dom_list = precedents_by_domain.setdefault(od["domain"], [])
                if len(dom_list) < 14 and od["raw_gold_tuple"]:
                    dom_list.append(dict(od["raw_gold_tuple"]))
            assert eval_cid_set.isdisjoint(precedent_cid_set)

        selected_cases: List[PreparedCase] = []
        for cid, qs, gs in raw_selected:
            data = all_grouped[cid]
            if include_multimodal:
                st_payload: Union[str, Dict[str, Any]] = {
                    "domain": data["domain"],
                    "report": data["state"],
                    "disjoint_domain_policy_precedents": precedents_by_domain.get(data["domain"], []),
                }
            else:
                st_payload = data["state"]
            selected_cases.append(PreparedCase(case_id=cid, state=st_payload, questions=qs, gold=gs))
        return selected_cases

    if benchmark == "amazon_esci":
        ds = load_dataset("tasksource/esci", split="test", streaming=True).shuffle(
            seed=seed, buffer_size=2000
        )
        pool: List[Dict[str, Any]] = []
        pool_target = max(limit * 3, 150)
        for row in ds:
            if row.get("product_locale", "us") != "us":
                continue
            lbl = str(row.get("esci_label", ""))
            if lbl not in ESCI_LABEL_ORDER:
                continue
            pool.append(row)
            if len(pool) >= pool_target:
                break
        indices = rng.permutation(len(pool))[:limit]
        esci_crit = {lbl: ESCI_CRITERIA[lbl] for lbl in ESCI_LABEL_ORDER}
        cases: List[PreparedCase] = []
        for idx in indices:
            r = pool[int(idx)]
            lbl = str(r["esci_label"])
            g_idx = ESCI_LABEL_ORDER.index(lbl)
            state = build_enriched_esci_state(r)
            q = {
                "esci": {
                    "type": "choice",
                    "instructions": "Classify Amazon ESCI shopping relevance between `query` and `product_title`.",
                    "criteria": esci_crit,
                }
            }
            cases.append(
                PreparedCase(
                    case_id=f"esci-{r.get('example_id', idx)}",
                    state=state,
                    questions=q,
                    gold={"esci": (lbl, g_idx, list(ESCI_LABEL_ORDER))},
                )
            )
        return cases

    if benchmark == "when2call":
        ds = load_dataset("nvidia/When2Call", split="mcq")
        n_total = len(ds)
        perm = rng.permutation(n_total)
        labels = list(WHEN2CALL_CRITERIA.keys())
        cases = []
        for idx in perm:
            r = ds[int(idx)]
            lbl = str(r.get("correct_answer", ""))
            if lbl not in WHEN2CALL_CRITERIA:
                continue
            state, crit = analyze_when2call_row(r, _compact_tool_spec)
            q = {
                "w2c_action": {
                    "type": "choice",
                    "instructions": (
                        "Given `user_request`, `available_tools`, `schema_audit`, and `candidate_responses`, "
                        "decide the right agent routing action."
                    ),
                    "criteria": crit,
                }
            }
            cases.append(
                PreparedCase(
                    case_id=f"w2c-{r.get('uuid', idx)}",
                    state=state,
                    questions=q,
                    gold={"w2c_action": (lbl, labels.index(lbl), labels)},
                )
            )
            if len(cases) >= limit:
                break
        return cases

    if benchmark == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        n_total = len(ds)
        indices = rng.permutation(n_total)[: min(limit, n_total)]
        labels = list(PUBMEDQA_CRITERIA.keys())
        cases = []
        for idx in indices:
            r = ds[int(idx)]
            lbl = str(r.get("final_decision", "")).lower().strip()
            if lbl not in PUBMEDQA_CRITERIA:
                continue
            abstract_str = _format_pubmed_abstract(r.get("context", {}), max_chars=1850)
            meshes = list(r.get("meshes") or [])[:8]
            state = {
                "question": str(r.get("question", "")),
                "abstract": abstract_str,
            }
            if meshes:
                state["meshes"] = meshes
            q = {
                "pubmed_decision": {
                    "type": "choice",
                    "instructions": "Based strictly on the biomedical study in `abstract`, answer `question`.",
                    "criteria": PUBMEDQA_CRITERIA,
                }
            }
            cases.append(
                PreparedCase(
                    case_id=f"pubmed-{r.get('pubid', idx)}",
                    state=state,
                    questions=q,
                    gold={"pubmed_decision": (lbl, labels.index(lbl), labels)},
                )
            )
        return cases

    if benchmark == "global_mmlu_lite":
        per_lang = max(1, (limit + len(mmlu_langs) - 1) // max(1, len(mmlu_langs)))
        labels = ["A", "B", "C", "D"]
        cases = []
        for lang_i, lang in enumerate(mmlu_langs):
            try:
                ds = load_dataset("CohereLabs/Global-MMLU-Lite", lang, split="test", streaming=True).shuffle(
                    seed=seed + lang_i, buffer_size=300
                )
            except Exception:
                continue
            count_lang = 0
            for idx, r in enumerate(ds):
                ans = str(r.get("answer", "")).strip().upper()
                if ans not in labels:
                    continue
                crit = {
                    "A": str(r.get("option_a", "")),
                    "B": str(r.get("option_b", "")),
                    "C": str(r.get("option_c", "")),
                    "D": str(r.get("option_d", "")),
                }
                state = {
                    "language": lang,
                    "subject": str(r.get("subject", "")),
                    "question": str(r.get("question", "")),
                }
                q = {
                    "answer": {
                        "type": "choice",
                        "instructions": "Select the single factually correct option for `question`.",
                        "criteria": crit,
                    }
                }
                cases.append(
                    PreparedCase(
                        case_id=f"mmlu-{lang}-{r.get('sample_id', idx)}",
                        state=state,
                        questions=q,
                        gold={"answer": (ans, labels.index(ans), labels)},
                    )
                )
                count_lang += 1
                if count_lang >= per_lang or len(cases) >= limit:
                    break
            if len(cases) >= limit:
                break
        return cases[:limit]

    if benchmark == "new_yorker_caption":
        ds = load_dataset("jmhessel/newyorker_caption_contest", "matching", split="test", streaming=True).shuffle(
            seed=seed, buffer_size=250
        )
        cols = ["instance_id", "image_location", "image_description", "image_uncanny_description", "caption_choices", "label"]
        ds = ds.select_columns(cols)
        cases = []
        letter_pool = ["A", "B", "C", "D", "E", "F", "G"]
        for idx, r in enumerate(ds):
            choices_list = list(r.get("caption_choices") or [])
            gold_letter = str(r.get("label", "")).strip().upper()
            k = len(choices_list)
            if k < 2:
                continue
            labels = letter_pool[:k]
            if gold_letter not in labels:
                continue
            crit = {labels[i]: str(choices_list[i]) for i in range(k)}
            state: Dict[str, Any] = {
                "location": str(r.get("image_location") or ""),
                "scene_description": str(r.get("image_description") or ""),
                "comic_element": str(r.get("image_uncanny_description") or ""),
            }
            q = {
                "caption": {
                    "type": "choice",
                    "instructions": (
                        "Which candidate caption best fits the New Yorker cartoon (combining visual irony, "
                        "`scene_description`, and `comic_element`)?"
                    ),
                    "criteria": crit,
                }
            }
            cases.append(
                PreparedCase(
                    case_id=f"nyc-{r.get('instance_id', idx)}",
                    state=state,
                    questions=q,
                    gold={"caption": (gold_letter, labels.index(gold_letter), labels)},
                )
            )
            if len(cases) >= limit:
                break
        return cases

    if benchmark == "mind2web_multimodal":
        return load_mind2web_multimodal_cases(limit=limit, seed=seed)

    if benchmark == "mmau":
        return load_mmau_cases(limit=limit, seed=seed)

    if benchmark == "video_mme":
        return load_video_mme_cases(limit=limit, seed=seed)

    raise ValueError(f"Unsupported benchmark loader: '{benchmark}'")


def _load_fixture_cases(benchmark: str, limit: int = 10) -> List[PreparedCase]:
    """Deterministic built-in mini fixtures for fast offline CI testing."""
    if benchmark == "jevbench":
        cases = [
            PreparedCase(
                case_id="fix-jev-1",
                state="Subject: Urgent: Data export completely broken since v2.4 upgrade. Need engineering fix or we cancel.",
                questions={
                    "feature_failing": {
                        "type": "noul",
                        "instructions": "Is a product feature currently failing for this customer?",
                    },
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this ticket?",
                        "criteria": {
                            "engineering": "broken product feature, bug, crash",
                            "billing": "invoice, refund, duplicate charge",
                            "sales": "pricing quote or enterprise upgrade",
                        },
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "Rate ticket urgency:",
                        "criteria": ["low", "medium", "high", "critical_churn_risk"],
                    },
                },
                gold={
                    "feature_failing": (True, 1, [False, True]),
                    "team": ("engineering", 0, ["engineering", "billing", "sales"]),
                    "urgency": (3, 3, [0, 1, 2, 3]),
                },
            ),
            PreparedCase(
                case_id="fix-jev-2",
                state="Hi team, we were charged twice on invoice #88412. Please refund the duplicate payment.",
                questions={
                    "duplicate_charge": {
                        "type": "noul",
                        "instructions": "Does the customer report a duplicate billing charge?",
                    },
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this ticket?",
                        "criteria": {
                            "engineering": "broken product feature, bug, crash",
                            "billing": "invoice, refund, duplicate charge",
                            "sales": "pricing quote or enterprise upgrade",
                        },
                    },
                },
                gold={
                    "duplicate_charge": (True, 1, [False, True]),
                    "team": ("billing", 1, ["engineering", "billing", "sales"]),
                },
            ),
        ]
        return cases[:limit]

    if benchmark == "amazon_esci":
        level_ids = list(range(len(ESCI_LABEL_ORDER)))
        items = [
            ("wireless noise cancelling headphones", "Sony WH-1000XM5 Wireless Noise Cancelling Headphones", "Exact"),
            ("wireless noise cancelling headphones", "Hard EVA Carrying Case for Sony WH-1000XM5 Headphones", "Complement"),
            ("wireless noise cancelling headphones", "Wired Studio Monitor Over-Ear Headphones (3.5mm Corded)", "Substitute"),
            ("wireless noise cancelling headphones", "Stainless Steel Kitchen Garlic Press", "Irrelevant"),
        ]
        return [
            PreparedCase(
                case_id=f"fix-esci-{i}",
                state={"query": qstr, "product_title": title},
                questions={
                    "esci": {
                        "type": "score",
                        "instructions": "Rate how closely `product_title` matches the shopper's search `query`:",
                        "criteria": ESCI_SCORE_LEVELS,
                    }
                },
                gold={"esci": (ESCI_LABEL_ORDER.index(lbl), ESCI_LABEL_ORDER.index(lbl), level_ids)},
            )
            for i, (qstr, title, lbl) in enumerate(items[:limit])
        ]

    if benchmark == "when2call":
        labels = list(WHEN2CALL_CRITERIA.keys())
        items = [
            (
                "What is 12 multiplied by 7?",
                [{"name": "get_weather", "description": "Fetch city weather", "parameters": {"city": "string (required)"}}],
                "direct",
            ),
            (
                "Check the weather in Warsaw right now.",
                [{"name": "get_weather", "description": "Fetch city weather", "parameters": {"city": "string (required)"}}],
                "tool_call",
            ),
            (
                "Check the current weather for me please.",
                [{"name": "get_weather", "description": "Fetch city weather", "parameters": {"city": "string (required)"}}],
                "request_for_info",
            ),
            (
                "Book me a flight to Tokyo tomorrow.",
                [{"name": "get_weather", "description": "Fetch city weather", "parameters": {"city": "string (required)"}}],
                "cannot_answer",
            ),
        ]
        return [
            PreparedCase(
                case_id=f"fix-w2c-{i}",
                state={"user_request": req, "available_tools": tools},
                questions={
                    "action": {
                        "type": "choice",
                        "instructions": "Choose the right agent tool-calling decision.",
                        "criteria": WHEN2CALL_CRITERIA,
                    }
                },
                gold={"action": (lbl, labels.index(lbl), labels)},
            )
            for i, (req, tools, lbl) in enumerate(items[:limit])
        ]

    if benchmark == "pubmedqa":
        labels = list(PUBMEDQA_CRITERIA.keys())
        items = [
            (
                "Does low-dose intervention X reduce systolic blood pressure?",
                "[RESULTS] In 420 randomized patients, intervention X reduced systolic blood pressure by 11.4 mmHg (p<0.001).",
                "yes",
            ),
            (
                "Does supplement Y prevent seasonal influenza infection?",
                "[RESULTS] Incidence of influenza was identical between supplement Y (14.2%) and placebo (14.5%, p=0.89).",
                "no",
            ),
            (
                "Is biomarker Z predictive of long-term remission?",
                "[RESULTS] Biomarker Z showed a non-significant trend in small pilot cohorts (n=18); larger trials are required.",
                "maybe",
            ),
        ]
        return [
            PreparedCase(
                case_id=f"fix-pmqa-{i}",
                state={"question": qstr, "abstract": abst},
                questions={
                    "decision": {
                        "type": "choice",
                        "instructions": "Based strictly on `abstract`, answer `question`.",
                        "criteria": PUBMEDQA_CRITERIA,
                    }
                },
                gold={"decision": (lbl, labels.index(lbl), labels)},
            )
            for i, (qstr, abst, lbl) in enumerate(items[:limit])
        ]

    if benchmark == "global_mmlu_lite":
        labels = ["A", "B", "C", "D"]
        return [
            PreparedCase(
                case_id="fix-mmlu-pl-1",
                state={"language": "pl", "subject": "astronomy", "question": "Która planeta jest najbliżej Słońca?"},
                questions={
                    "answer": {
                        "type": "choice",
                        "instructions": "Select the correct option for `question`.",
                        "criteria": {"A": "Wenus", "B": "Merkury", "C": "Mars", "D": "Jowisz"},
                    }
                },
                gold={"answer": ("B", 1, labels)},
            ),
            PreparedCase(
                case_id="fix-mmlu-en-1",
                state={"language": "en", "subject": "biology", "question": "Which organelle produces ATP via cellular respiration?"},
                questions={
                    "answer": {
                        "type": "choice",
                        "instructions": "Select the correct option for `question`.",
                        "criteria": {"A": "Ribosome", "B": "Golgi apparatus", "C": "Mitochondrion", "D": "Lysosome"},
                    }
                },
                gold={"answer": ("C", 2, labels)},
            ),
        ][:limit]

    if benchmark == "new_yorker_caption":
        labels = ["A", "B", "C"]
        return [
            PreparedCase(
                case_id="fix-nyc-1",
                state={
                    "location": "corporate boardroom",
                    "scene_description": "Two dogs in business suits point at a downward chart.",
                    "comic_element": "Dogs acting as Wall Street executives blaming cats.",
                },
                questions={
                    "caption": {
                        "type": "choice",
                        "instructions": "Choose the funniest fitting cartoon caption.",
                        "criteria": {
                            "A": "The quarterly squirrel projections were overly optimistic.",
                            "B": "Please pass the salt.",
                            "C": "The train arrives at noon.",
                        },
                    }
                },
                gold={"caption": ("A", 0, labels)},
            )
        ][:limit]

    if benchmark == "mind2web_multimodal":
        labels = ["A", "B", "C", "D"]
        return [
            PreparedCase(
                case_id="fix-m2w-1",
                state={
                    "screenshot": b"mock-jpeg-bytes",
                    "website": "shop.example.com",
                    "task_goal": "Click the checkout button",
                    "next_operation_type": {"op": "CLICK"},
                },
                questions={
                    "element": {
                        "type": "choice",
                        "instructions": "Select the target UI element [A]-[D].",
                        "criteria": {
                            "A": "<button> text='Proceed to Checkout'",
                            "B": "<a> text='Privacy Policy'",
                            "C": "<input> name='newsletter'",
                            "D": "<div> text='Footer'",
                        },
                    }
                },
                gold={"element": ("A", 0, labels)},
            )
        ][:limit]

    if benchmark == "mmau":
        labels = ["A", "B", "C", "D"]
        return [
            PreparedCase(
                case_id="fix-mmau-1",
                state={
                    "audio": {"bytes": b"mock-wav"},
                    "question": "Which instrument is playing the solo?",
                    "audio_category": "music",
                },
                questions={
                    "answer": {
                        "type": "choice",
                        "instructions": "Listen to the audio and select the right option.",
                        "criteria": {"A": "Violin", "B": "Drums", "C": "Tuba", "D": "Silence"},
                    }
                },
                gold={"answer": ("A", 0, labels)},
            )
        ][:limit]

    if benchmark == "video_mme":
        labels = ["A", "B", "C", "D"]
        return [
            PreparedCase(
                case_id="fix-vmme-1",
                state={
                    "_frames": [b"mock-grid-1"],
                    "video_duration_category": "short",
                    "raw_audio_speech_transcript": "[00:05] Welcome to our chemistry demonstration.",
                },
                questions={
                    "q_001_1": {
                        "type": "choice",
                        "instructions": "What subject is demonstrated in the video?",
                        "criteria": {"A": "Chemistry", "B": "Soccer", "C": "Cooking", "D": "Gardening"},
                    }
                },
                gold={"q_001_1": ("A", 0, labels)},
            )
        ][:limit]

    raise ValueError(f"No fixture defined for benchmark '{benchmark}'")


# ---------------------------------------------------------------------------
# Suite Runner
# ---------------------------------------------------------------------------


def _extract_probs_and_pred(
    ans: Dict[str, Any],
    qdef: Dict[str, Any],
    option_labels: List[Any],
) -> Tuple[np.ndarray, Any]:
    """Extract normalized probability array over `option_labels` and Top-1 predicted label."""
    qtype = qdef["type"]
    k = len(option_labels)

    if qtype == "noul":
        p_true = float(ans.get("noul", 0.5))
        probs = np.array([1.0 - p_true, p_true], dtype=np.float64)
        probs = np.clip(probs, 0.0, 1.0)
        probs /= max(1e-12, float(np.sum(probs)))
        pred = bool(option_labels[int(np.argmax(probs))])
        return probs, pred

    if qtype == "choice":
        pdict = ans.get("probabilities", {})
        probs = np.array([float(pdict.get(str(lbl), 0.0)) for lbl in option_labels], dtype=np.float64)
        if float(np.sum(probs)) <= 0.0:
            probs = np.ones(k, dtype=np.float64) / max(1, k)
        else:
            probs /= float(np.sum(probs))
        pred = option_labels[int(np.argmax(probs))]
        return probs, pred

    if qtype == "score":
        pdict = ans.get("probabilities", {})
        probs = np.array([float(pdict.get(str(i), 0.0)) for i in range(k)], dtype=np.float64)
        if float(np.sum(probs)) <= 0.0:
            probs = np.ones(k, dtype=np.float64) / max(1, k)
        else:
            probs /= float(np.sum(probs))
        # Modal level aligns exact prediction with Top-1 probability max_i p_i for Top-1 ECE
        pred = option_labels[int(np.argmax(probs))]
        return probs, pred

    raise ValueError(f"Unknown question type: {qtype}")


OMEGA_BENCHMARK_POLICIES: Dict[str, Dict[str, Any]] = {
    "jevbench": {
        "force_tier2": True,
        "tier2_model": "gemini-3.8-flash",
        "thinking_level": "low",
        "pack_size": 1,
        "tier2_blend_weight": 1.0,
        "system_hint": (
            "First verify every boolean ('noul') fact strictly from numeric/status details in `report`. "
            "Then determine `choice` and 0-indexed ordinal `score` levels by aligning those boolean facts with "
            "the option descriptions and `disjoint_domain_policy_precedents` (precedent policy tuples from other cases in the same domain)."
        ),
    },
    "amazon_esci": {
        "tier2_model": "gemini-3.8-flash",
        "thinking_level": "low",
        "escalation_threshold": 0.99,
        "max_escalation_fraction": 0.96,
        "pack_size": 10,
        "tier2_heavy_model": "gemini-3.8-flash",
        "heavy_thinking_level": "medium",
        "heavy_escalation_fraction": 0.24,
        "heavy_threshold": 0.82,
        "heavy_pack_size": 8,
        "tier2_blend_weight": 0.96,
        "system_hint": (
            "Amazon KDD Cup ESCI 4-class Shopping Queries rules (do NOT over-predict 'Substitute'!): "
            "1) 'Exact' (~65%): If the product IS the requested item type and violates NO explicit brand/model/size/color/feature "
            "stated in `query`, choose 'Exact' (when in doubt between 'Exact' and 'Substitute' on a broad query, prefer 'Exact'). "
            "2) 'Substitute' (~22%): Reserve STRICTLY for functional replacements of the SAME product type that fail a specific "
            "stated constraint in `query` (wrong brand, model, size, color, count, or missing requested feature). "
            "3) 'Complement' (~3.5%): Choose 'Complement' when the product is a separate companion accessory, case, cable, "
            "replacement part, refill/consumable, or series add-on used WITH/ALONGSIDE the queried item rather than replacing it "
            "(never label the main product itself or a bundle containing the main product as 'Complement'). "
            "4) 'Irrelevant' (~10%): Wrong category, incompatible target entity, or superficial keyword overlap that neither "
            "replaces nor pairs with the queried item -> choose 'Irrelevant', NOT 'Substitute'."
        ),
    },
    "new_yorker_caption": {
        "force_tier2": True,
        "tier2_model": "gemini-3.8-flash",
        "thinking_level": "low",
        "pack_size": 12,
        "system_hint": (
            "New Yorker cartoon caption contest: identify the exact visual irony, double-meaning pun, or absurd "
            "social contrast linking `location`, `scene_description`, and `comic_element` to the sharpest editorial punchline."
        ),
    },
    "mind2web_multimodal": {
        "force_tier2": True,
        "tier2_model": "gemini-3.5-flash-lite",
        "thinking_level": "low",
        "image_max_side": 820,
        "image_resolution": "medium",
        "pack_size": 2,
        "tier2_heavy_model": "gemini-3.8-flash",
        "heavy_thinking_level": "low",
        "heavy_escalation_fraction": 0.19,
        "heavy_threshold": 0.86,
        "heavy_pack_size": 4,
        "heavy_image_resolution": "medium",
        "system_hint": (
            "Web UI Set-of-Mark grounding: match `task_goal`, `previous_actions`, and `next_operation_type` "
            "to the exact labeled bounding box [A]-[E] and DOM element attributes."
        ),
    },
    "pubmedqa": {
        "tier2_model": "gemini-3.8-flash",
        "thinking_level": "low",
        "escalation_threshold": 0.99,
        "max_escalation_fraction": 0.95,
        "pack_size": 8,
        "tier2_blend_weight": 0.98,
        "system_hint": (
            "PubMedQA labeled dataset (~55% yes, ~34% no, ~11% maybe): the conclusion paragraph of the abstract was "
            "removed, so infer what the original authors concluded from [RESULTS]! Statistical significance (p<0.05) "
            "or clear directional empirical findings -> 'yes' or 'no' (even if routine caveats apply). Choose 'maybe' "
            "(~11%) strictly when primary endpoints are non-significant/conflicting so authors give a conditional answer."
        ),
    },
    "global_mmlu_lite": {
        "tier2_model": "gemini-3.5-flash-lite",
        "thinking_level": "low",
        "escalation_threshold": 0.98,
        "max_escalation_fraction": 0.94,
        "pack_size": 2,
        "tier2_heavy_model": "gemini-3.8-flash",
        "heavy_thinking_level": "low",
        "heavy_escalation_fraction": 0.32,
        "heavy_threshold": 0.90,
        "heavy_pack_size": 8,
        "tier2_blend_weight": 0.96,
        "system_hint": "Verify multilingual STEM, humanities, and factual logic carefully before selecting A/B/C/D.",
    },
    "when2call": {
        "tier2_model": "gemini-3.5-flash-lite",
        "thinking_level": "low",
        "escalation_threshold": 0.95,
        "max_escalation_fraction": 0.88,
        "pack_size": 2,
        "tier2_heavy_model": "gemini-3.8-flash",
        "heavy_thinking_level": "low",
        "heavy_escalation_fraction": 0.14,
        "heavy_threshold": 0.86,
        "heavy_pack_size": 6,
        "tier2_blend_weight": 0.96,
        "system_hint": (
            "When2Call rubric: check `schema_audit` and `available_tools`! "
            "If `candidate_tool_exists_in_available_tools` is False and a tool is needed -> 'cannot_answer'. "
            "If no tool is needed for general knowledge/chitchat -> 'direct'. "
            "If a matching tool IS in `available_tools` and all required params are known -> 'tool_call'; "
            "if a matching tool IS in `available_tools` but user omitted required details -> 'request_for_info'."
        ),
    },
    "mmau": {
        "force_tier2": True,
        "tier2_model": "gemini-3.5-flash-lite",
        "thinking_level": "low",
        "pack_size": 10,
        "tier2_heavy_model": "gemini-3.8-flash",
        "heavy_thinking_level": "low",
        "heavy_escalation_fraction": 0.22,
        "heavy_threshold": 0.88,
        "heavy_pack_size": 12,
        "system_hint": (
            "Analyze acoustic events, speaker tone/identity, musical rhythm/instrumentation, temporal ordering, "
            "and spoken content directly from each item's audio clip."
        ),
    },
    "video_mme": {
        "force_tier2": True,
        "tier2_model": "gemini-3.8-flash",
        "thinking_level": "low",
        "image_max_side": 900,
        "image_resolution": "medium",
        "pack_size": 7,
        "system_hint": (
            "Examine all 18 chronological frames F01..F18 in the 6x3 storyboard image (with MM:SS timestamps) "
            "together with `raw_audio_speech_transcript` for each video item to answer its questions accurately."
        ),
    },
}


class DecisionBenchmarkSuite:
    """Evaluates Laya / Vega-U / Omega across the 9-benchmark decision table with Top-1 ECE & Total Price."""

    DEFAULT_BENCHMARKS: Tuple[str, ...] = (
        "jevbench",
        "amazon_esci",
        "when2call",
        "pubmedqa",
        "global_mmlu_lite",
        "new_yorker_caption",
    )

    ALL_TABLE_BENCHMARKS: Tuple[str, ...] = (
        "jevbench",
        "amazon_esci",
        "new_yorker_caption",
        "mind2web_multimodal",
        "pubmedqa",
        "global_mmlu_lite",
        "when2call",
        "mmau",
        "video_mme",
    )

    def __init__(
        self,
        engine: Optional[BaseDecisionEngine] = None,
        backend: str = "auto",
        model_id: Optional[str] = None,
        batch_size: int = 32,
        auto_route_checkpoints: bool = False,
        fit_disjoint_ttt: bool = True,
        **engine_kwargs: Any,
    ):
        self._backend_name = backend
        self._explicit_model_id = model_id
        self._engine_kwargs = dict(engine_kwargs)
        self.auto_route_checkpoints = bool(auto_route_checkpoints and engine is None and model_id is None)
        self.engine = engine or get_engine(backend=backend, model_id=model_id, **engine_kwargs)
        self.batch_size = max(1, int(batch_size))
        self.fit_disjoint_ttt = bool(fit_disjoint_ttt)
        self._ttt_fitted_for: set[str] = set()
        self._engine_cache: Dict[str, BaseDecisionEngine] = {}
        if not self.auto_route_checkpoints:
            self._engine_cache["default"] = self.engine

    def _get_task_engine(self, spec: BenchmarkSpec) -> BaseDecisionEngine:
        """Return either the fixed suite engine or the cached task-recommended Laya checkpoint."""
        if (
            not self.auto_route_checkpoints
            or type(self.engine).__name__ in ("EmulatorBackend", "OmegaDecisionEngine", "GeminiDecisionEngine", "VegaBackend")
        ):
            return self.engine
        ckpt = spec.recommended_checkpoint
        if ckpt not in self._engine_cache:
            self._engine_cache[ckpt] = get_engine(
                backend=self._backend_name,
                model_id=ckpt,
                **self._engine_kwargs,
            )
        return self._engine_cache[ckpt]

    def evaluate_benchmark(
        self,
        benchmark: str,
        limit: int = 100,
        source: str = "hf",
        calibration_split: float = 0.0,
        seed: int = 42,
        max_len: Optional[int] = None,
    ) -> BenchmarkTaskResult:
        """Run a single benchmark from `BENCHMARK_SPECS` and compute all quality, calibration & cost metrics."""
        if benchmark not in BENCHMARK_SPECS:
            raise ValueError(
                f"Unknown benchmark '{benchmark}'. Available: {list(BENCHMARK_SPECS.keys())}"
            )
        spec = BENCHMARK_SPECS[benchmark]
        task_engine = self._get_task_engine(spec)
        multimodal_ok = bool(getattr(task_engine, "supports_multimodal", False))

        if not spec.laya_text_supported and not multimodal_ok:
            return BenchmarkTaskResult(
                benchmark=spec.key,
                display_name=spec.display_name,
                status="skipped_unsupported_modality",
                primary_metric_name=spec.primary_metric,
                notes=(
                    f"Requires {spec.modality} input; text-only Laya encoder returns N/A "
                    f"(use OmegaDecisionEngine or CascadedJudge to route {spec.modality} items)."
                ),
            )

        is_omega = type(task_engine).__name__ == "OmegaDecisionEngine"
        if is_omega and source == "hf" and self.fit_disjoint_ttt and benchmark not in self._ttt_fitted_for:
            if benchmark in ("amazon_esci", "pubmedqa", "when2call"):
                from .multimodal_loaders import fit_disjoint_ttt_for_omega

                fit_disjoint_ttt_for_omega(task_engine, [benchmark], eval_limit=limit, seed=seed)
                self._ttt_fitted_for.add(benchmark)

        cases = load_benchmark_cases(
            benchmark,
            limit=limit,
            source=source,
            seed=seed,
            include_multimodal=multimodal_ok,
        )
        if not cases:
            return BenchmarkTaskResult(
                benchmark=spec.key,
                display_name=spec.display_name,
                status="empty",
                primary_metric_name=spec.primary_metric,
                notes="No cases loaded.",
            )

        eff_max_len = max_len or spec.recommended_max_len
        omega_kwargs: Dict[str, Any] = dict(OMEGA_BENCHMARK_POLICIES.get(benchmark, {})) if is_omega else {}

        # Check if all cases share an identical question schema (enables fast `predict_batch`!)
        first_q = cases[0].questions
        uniform_schema = all(c.questions == first_q for c in cases)

        raw_outputs: List[Dict[str, Any]] = []
        if uniform_schema and len(cases) > 1:
            states = [c.state for c in cases]
            try:
                raw_outputs = task_engine.predict_batch(
                    states, first_q, batch_size=self.batch_size, max_len=eff_max_len, **omega_kwargs
                )
            except TypeError:
                raw_outputs = task_engine.predict_batch(states, first_q)
        else:
            # Multi-question / per-case schemas (e.g. JevBench, Global-MMLU, New Yorker, Mind2Web, MMAU, Video-MME)
            if hasattr(task_engine, "predict_cases") and len(cases) > 1:
                raw_outputs = task_engine.predict_cases(
                    [(c.state, c.questions) for c in cases], max_len=eff_max_len, **omega_kwargs
                )
            elif type(task_engine).__name__ == "GeminiDecisionEngine" and len(cases) > 1:
                from concurrent.futures import ThreadPoolExecutor

                t2_workers = min(len(cases), getattr(task_engine, "max_workers", 8))
                with ThreadPoolExecutor(max_workers=max(1, t2_workers)) as pool:
                    futs = [pool.submit(task_engine.predict, c.state, c.questions) for c in cases]
                    raw_outputs = [f.result() for f in futs]
            else:
                for c in cases:
                    try:
                        raw_outputs.append(
                            task_engine.predict(c.state, c.questions, max_len=eff_max_len, **omega_kwargs)
                        )
                    except TypeError:
                        raw_outputs.append(task_engine.predict(c.state, c.questions))

        y_true: List[Any] = []
        y_pred: List[Any] = []
        probs_list: List[np.ndarray] = []
        gold_indices: List[int] = []
        buckets: List[str] = []
        tier_tags: List[str] = []
        latencies: List[float] = []
        total_in_toks = 0
        total_out_toks = 0
        total_actual_cost = 0.0

        for case, raw_res in zip(cases, raw_outputs):
            latencies.append(float(raw_res.get("latency_ms", 0.0)))
            usage = raw_res.get("usage", {})
            total_in_toks += int(usage.get("input_tokens", 0))
            total_out_toks += int(usage.get("output_tokens", 0))
            total_actual_cost += float(raw_res.get("cost_usd", 0.0))
            ans_map = raw_res.get("answers", {})
            for qid, qdef in case.questions.items():
                if qid not in case.gold or qid not in ans_map:
                    continue
                gold_val, gold_idx, opt_labels = case.gold[qid]
                probs, pred_val = _extract_probs_and_pred(ans_map[qid], qdef, opt_labels)
                y_true.append(gold_val)
                y_pred.append(pred_val)
                probs_list.append(probs)
                gold_indices.append(gold_idx)
                buckets.append(bucket_key_for_question(qdef["type"], len(opt_labels)))
                tier_tags.append(str(ans_map[qid].get("omega_tier", task_engine.model_id)))

        top1_probs = np.array([float(np.max(p)) for p in probs_list], dtype=np.float64)
        correct_arr = np.array([1.0 if yp == yt else 0.0 for yp, yt in zip(y_pred, y_true)], dtype=np.float64)

        acc = accuracy_score(y_true, y_pred)
        f1 = (
            macro_f1_score(y_true, y_pred, labels=ESCI_LABEL_ORDER)
            if benchmark == "amazon_esci" and y_true and isinstance(y_true[0], str)
            else macro_f1_score(y_true, y_pred)
        )
        ece_raw = top1_ece(top1_probs, correct_arr)
        bs = brier_score(probs_list, gold_indices)
        nll_val = negative_log_likelihood(probs_list, gold_indices)
        aurc_val = aurc(top1_probs, correct_arr)
        acc_80 = selective_accuracy_at_coverage(top1_probs, correct_arr, coverage=0.80)

        cal_ece: Optional[float] = None
        fitted_t: Optional[float] = None
        if 0.05 <= calibration_split <= 0.80 and len(probs_list) >= 20:
            oof_probs = oof_calibrate_pooled(
                probs_list, gold_indices, tier_tags=tier_tags, n_folds=5, seed=seed
            )
            oof_top1 = np.array([float(np.max(p)) for p in oof_probs], dtype=np.float64)
            cal_ece = top1_ece(oof_top1, correct_arr)
        elif 0.05 <= calibration_split <= 0.80 and len(probs_list) >= 8:
            n_val = max(3, int(round(len(probs_list) * calibration_split)))
            if len(probs_list) - n_val >= 3:
                scaler = TemperatureScaler(min_bucket_samples=15).fit(
                    probs_list[:n_val], gold_indices[:n_val], buckets=buckets[:n_val]
                )
                fitted_t = scaler.global_temperature
                eval_probs = [
                    rescale_probabilities(
                        p,
                        scaler.bucket_temperatures.get(b, scaler.global_temperature),
                    )
                    for p, b in zip(probs_list[n_val:], buckets[n_val:])
                ]
                eval_top1 = np.array([float(np.max(p)) for p in eval_probs], dtype=np.float64)
                cal_ece = top1_ece(eval_top1, correct_arr[n_val:])

        primary = f1 if spec.primary_metric == "macro_f1" else acc
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

        return BenchmarkTaskResult(
            benchmark=spec.key,
            display_name=spec.display_name,
            status="completed",
            primary_metric_name=spec.primary_metric,
            primary_score=primary,
            accuracy=acc,
            macro_f1=f1,
            top1_ece=ece_raw,
            calibrated_top1_ece=cal_ece,
            brier=bs,
            nll=nll_val,
            aurc_score=aurc_val,
            accuracy_at_80pct_coverage=acc_80,
            sample_count=len(y_true),
            case_count=len(cases),
            input_tokens=total_in_toks,
            output_tokens=total_out_toks,
            actual_cost_usd=total_actual_cost,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            fitted_temperature=fitted_t,
            notes=f"checkpoint={task_engine.model_id}",
            _probs_list=probs_list,
            _gold_indices=gold_indices,
            _buckets=buckets,
            _tier_tags=tier_tags,
        )

    def run_suite(
        self,
        benchmarks: Optional[Sequence[str]] = None,
        limit_per_benchmark: int = 100,
        source: str = "hf",
        calibration_split: float = 0.25,
        cascade_escalation_rate: float = 0.20,
        seed: int = 42,
        progress_callback: Optional[Callable[[BenchmarkTaskResult, int, int], None]] = None,
    ) -> SuiteEvaluationSummary:
        """Run multiple benchmarks and compute pooled `Top-1 ECE` and `Total price`."""
        default_list = (
            self.ALL_TABLE_BENCHMARKS
            if getattr(self.engine, "supports_multimodal", False)
            else self.DEFAULT_BENCHMARKS
        )
        target_benchmarks = list(benchmarks or default_list)
        results: List[BenchmarkTaskResult] = []

        pooled_probs: List[np.ndarray] = []
        pooled_gold: List[int] = []
        pooled_buckets: List[str] = []
        pooled_tags: List[str] = []
        total_questions = 0
        total_cases = 0
        total_in_tokens = 0
        total_out_tokens = 0
        total_actual_cost = 0.0

        total_b = len(target_benchmarks)
        for idx, bkey in enumerate(target_benchmarks, start=1):
            res = self.evaluate_benchmark(
                bkey,
                limit=limit_per_benchmark,
                source=source,
                calibration_split=calibration_split,
                seed=seed,
            )
            results.append(res)
            if progress_callback is not None:
                progress_callback(res, idx, total_b)
            if res.status == "completed":
                pooled_probs.extend(res._probs_list)
                pooled_gold.extend(res._gold_indices)
                pooled_buckets.extend(res._buckets)
                pooled_tags.extend(res._tier_tags)
                total_questions += res.sample_count
                total_cases += res.case_count
                total_in_tokens += res.input_tokens
                total_out_tokens += res.output_tokens
                total_actual_cost += res.actual_cost_usd

        if pooled_probs:
            p_top1 = np.array([float(np.max(p)) for p in pooled_probs], dtype=np.float64)
            p_corr = np.array(
                [1.0 if int(np.argmax(p)) == int(y) else 0.0 for p, y in zip(pooled_probs, pooled_gold)],
                dtype=np.float64,
            )
            p_ece = top1_ece(p_top1, p_corr)
            p_brier = brier_score(pooled_probs, pooled_gold)
            p_aurc = aurc(p_top1, p_corr)

            p_cal_ece: Optional[float] = None
            if 0.05 <= calibration_split <= 0.80 and len(pooled_probs) >= 20:
                oof_probs = oof_calibrate_pooled(
                    pooled_probs,
                    pooled_gold,
                    tier_tags=pooled_tags,
                    n_folds=10 if len(pooled_probs) >= 200 else 5,
                    seed=seed,
                )
                oof_top1 = np.array([float(np.max(p)) for p in oof_probs], dtype=np.float64)
                p_cal_ece = top1_ece(oof_top1, p_corr)
            elif 0.05 <= calibration_split <= 0.80 and len(pooled_probs) >= 12:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(pooled_probs))
                n_val = max(4, int(round(len(pooled_probs) * calibration_split)))
                if len(pooled_probs) - n_val >= 4:
                    val_idx = perm[:n_val]
                    tst_idx = perm[n_val:]
                    scaler = TemperatureScaler(min_bucket_samples=15).fit(
                        [pooled_probs[i] for i in val_idx],
                        [pooled_gold[i] for i in val_idx],
                        buckets=[pooled_buckets[i] for i in val_idx],
                    )
                    cal_probs = [
                        rescale_probabilities(
                            pooled_probs[i],
                            scaler.bucket_temperatures.get(pooled_buckets[i], scaler.global_temperature),
                        )
                        for i in tst_idx
                    ]
                    cal_top1 = np.array([float(np.max(p)) for p in cal_probs], dtype=np.float64)
                    cal_corr = p_corr[tst_idx]
                    p_cal_ece = top1_ece(cal_top1, cal_corr)
        else:
            p_ece, p_brier, p_aurc, p_cal_ece = 0.0, 0.0, 0.0, None

        prices = estimate_suite_prices(
            total_input_tokens=total_in_tokens,
            eval_count=max(1, total_cases),
            cascade_escalation_rate=cascade_escalation_rate,
        )
        prices["actual_run_usd"] = round(total_actual_cost, 6)

        return SuiteEvaluationSummary(
            backend=type(self.engine).__name__,
            model=(
                f"{self.engine.model_id} (auto-routed)"
                if self.auto_route_checkpoints
                else self.engine.model_id
            ),
            results=results,
            pooled_top1_ece=p_ece,
            pooled_calibrated_top1_ece=p_cal_ece,
            pooled_brier=p_brier,
            pooled_aurc=p_aurc,
            total_questions_evaluated=total_questions,
            total_cases_evaluated=total_cases,
            total_input_tokens=total_in_tokens,
            total_output_tokens=total_out_tokens,
            total_price_usd=prices,
        )
