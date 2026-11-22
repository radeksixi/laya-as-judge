"""Hardware VRAM profiles and local-only text decision/judge engine builders.

Provides curated, cross-family text-only configurations tailored to real local
hardware budgets:
  - `edge-6gb`: Max 4-6 GB VRAM (NVIDIA Jetson Orin Nano 8GB / RTX 3050 / RTX 4060 8GB)
  - `rtx-16gb`: Max 12-15 GB VRAM (NVIDIA RTX 16GB cards: 4060 Ti 16GB / 4080 / 5070 Ti / 5080)
  - `workstation-24gb`: 24-48+ GB VRAM (RTX 3090/4090/5090 / A6000 / L40S)
  - `emulator`: Zero-VRAM offline CI & development profile (no weights downloaded)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, Optional, Tuple

from .engine.base import BaseDecisionEngine
from .engine.emulator_backend import EmulatorBackend
from .engine.factory import get_engine
from .engine.http_systemone_backend import SystemOneHTTPBackend
from .omega import OmegaDecisionEngine

logger = logging.getLogger("laya_as_judge")


@dataclass(frozen=True)
class HardwareProfile:
    """Specification of a local text-only hardware deployment profile."""

    name: str
    display_name: str
    target_hardware: str
    max_vram_gb: float
    estimated_vram_gb: str
    tier1_backend: str
    tier1_model: str
    tier1_vram_gb: float
    tier2_backend: str
    tier2_model: str
    tier2_vram_gb: float
    tier2_default_url: Optional[str]
    guard_model: str
    guard_vram_gb: float
    escalation_threshold: float
    max_escalation_fraction: float
    adaptive_symmetry_threshold: Optional[float]
    symmetry_permutations: int
    recommended_tier2_cmd: str
    notes: str
    joint_schema_tier2: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "edge-6gb": HardwareProfile(
        name="edge-6gb",
        display_name="Edge / Small GPU (4-6 GB VRAM)",
        target_hardware="NVIDIA Jetson Orin Nano 8GB (unified) or RTX 3050/4060 8GB",
        max_vram_gb=6.0,
        estimated_vram_gb="3.8 - 5.8 GB (incl. 4k KV-cache)",
        tier1_backend="torch",
        tier1_model="convaiinnovations/laya",
        tier1_vram_gb=0.85,
        tier2_backend="systemone",
        tier2_model="ggml-org/Kev-4B-GGUF:Q4_K_M",
        tier2_vram_gb=3.0,
        tier2_default_url="http://127.0.0.1:8081",
        guard_model="ggml-org/Laya-GGUF:Q8_0 (0.45 GB) or convaiinnovations/laya",
        guard_vram_gb=0.45,
        escalation_threshold=0.78,
        max_escalation_fraction=0.35,
        adaptive_symmetry_threshold=0.22,
        symmetry_permutations=2,
        recommended_tier2_cmd=(
            "llama-server -hf ggml-org/Kev-4B-GGUF:Q4_K_M --port 8081 --ctx-size 4096 -b 2048 -ub 2048 --parallel 2"
        ),
        notes=(
            "Cross-family edge stack: ModernBERT Laya (ggml-org/Laya-GGUF:Q8_0 ~0.45 GB or PyTorch "
            "convaiinnovations/laya) as Tier-1 + Guard, paired with Qwen3.5-based Kev-4B Q4_K_M "
            "(~3.0 GB) as Tier-2 / Judge via local llama-server /v1/systemone. Verified live under 6 GB VRAM."
        ),
    ),
    "rtx-16gb": HardwareProfile(
        name="rtx-16gb",
        display_name="Single RTX 16GB Workstation (12-15 GB VRAM, Clef-Flash)",
        target_hardware="NVIDIA RTX 4060 Ti 16GB / 4080 / 5060 Ti 16GB / 5070 Ti / 5080 16GB",
        max_vram_gb=15.0,
        estimated_vram_gb="11.0 - 12.5 GB (measured 11.75 GB incl. 4k KV-cache)",
        tier1_backend="systemone",
        tier1_model="ggml-org/Kev-4B-GGUF:Q4_K_M",
        tier1_vram_gb=3.0,
        tier2_backend="systemone",
        tier2_model="ggml-org/Clef-Flash-GGUF:Q4_K_M",
        tier2_vram_gb=6.8,
        tier2_default_url="http://127.0.0.1:8081",
        guard_model="ggml-org/Laya-GGUF:Q8_0 (0.45 GB) or convaiinnovations/laya",
        guard_vram_gb=0.45,
        escalation_threshold=0.82,
        max_escalation_fraction=0.40,
        adaptive_symmetry_threshold=None,
        symmetry_permutations=1,
        recommended_tier2_cmd=(
            "llama-server -hf ggml-org/Clef-Flash-GGUF:Q4_K_M --port 8081 --ctx-size 8192 -b 2048 -ub 2048 --parallel 2"
        ),
        notes=(
            "Recommended all-round single-16GB-GPU production stack: Kev-4B Q4_K_M (~3.0 GB) as fast "
            "Tier-1 + Cloudflare Clef-Flash 9B Q4_K_M (~6.8 GB, joint-schema bidirectional head) as "
            "Tier-2 / Judge. Measured live: 78.03% 6-bench avg, 12/12 red-team (0% false blocks), "
            "0.0% symmetry flip rate, 12/12 Polish, and 3.3x faster multi-question throughput."
        ),
        joint_schema_tier2=True,
    ),
    "rtx-16gb-glide": HardwareProfile(
        name="rtx-16gb-glide",
        display_name="Single RTX 16GB Workstation (12-15 GB VRAM, GLiDE-9B English)",
        target_hardware="NVIDIA RTX 4060 Ti 16GB / 4080 / 5060 Ti 16GB / 5070 Ti / 5080 16GB",
        max_vram_gb=15.0,
        estimated_vram_gb="10.5 - 12.0 GB (measured 11.11 GB incl. 4k KV-cache)",
        tier1_backend="systemone",
        tier1_model="ggml-org/Kev-4B-GGUF:Q4_K_M",
        tier1_vram_gb=3.0,
        tier2_backend="systemone",
        tier2_model="bartowski/fastino_GLiDE-9B-GGUF:IQ4_NL",
        tier2_vram_gb=5.8,
        tier2_default_url="http://127.0.0.1:8081",
        guard_model="ggml-org/Laya-GGUF:Q8_0 (0.45 GB) or convaiinnovations/laya",
        guard_vram_gb=0.45,
        escalation_threshold=0.82,
        max_escalation_fraction=0.40,
        adaptive_symmetry_threshold=None,
        symmetry_permutations=1,
        recommended_tier2_cmd=(
            "llama-server -hf bartowski/fastino_GLiDE-9B-GGUF:IQ4_NL --port 8081 --ctx-size 8192 -b 2048 -ub 2048 --parallel 2"
        ),
        notes=(
            "Highest English reasoning & tool-calling single-GPU stack: Kev-4B Q4_K_M + Fastino "
            "GLiDE-9B IQ4_NL. Measured live: 78.73% 6-bench avg (When2Call 92.5%, New Yorker 77.5%, "
            "ESCI 69.85%), 0.0287 OOF Top-1 ECE after TemperatureScaler, 12/12 red-team under "
            "audit_policy='on_high_stakes'."
        ),
        joint_schema_tier2=False,
    ),
    "workstation-24gb": HardwareProfile(
        name="workstation-24gb",
        display_name="High-VRAM Workstation / Server (24-48 GB VRAM)",
        target_hardware="NVIDIA RTX 3090/4090/5090 24-32GB / 2x16GB / A6000 / L40S 48GB",
        max_vram_gb=48.0,
        estimated_vram_gb="20 - 26 GB (incl. 4k-8k KV-cache)",
        tier1_backend="systemone",
        tier1_model="ggml-org/Kev-4B-GGUF:Q4_K_M",
        tier1_vram_gb=3.0,
        tier2_backend="systemone",
        tier2_model="ggml-org/OpenJev-GGUF:Q4_K_M",
        tier2_vram_gb=19.4,
        tier2_default_url="http://127.0.0.1:8081",
        guard_model="ggml-org/Laya-GGUF:Q8_0 (or Julia-1-Q8_0)",
        guard_vram_gb=0.45,
        escalation_threshold=0.84,
        max_escalation_fraction=0.45,
        adaptive_symmetry_threshold=0.25,
        symmetry_permutations=3,
        recommended_tier2_cmd=(
            "llama-server -hf ggml-org/OpenJev-GGUF:Q4_K_M --port 8081 --ctx-size 4096 -b 2048 -ub 2048 --parallel 2"
        ),
        notes=(
            "Frontier local text stack: Kev-4B Q4_K_M (or Vega-U 4B with TTT) cascading to "
            "ggml-org/OpenJev-GGUF:Q4_K_M (~19.4 GB) or full-precision Clef-Flash-BF16 (~17 GB) "
            "via native llama-server /v1/systemone."
        ),
    ),
    "emulator": HardwareProfile(
        name="emulator",
        display_name="Offline Emulator / CI (0 GB VRAM)",
        target_hardware="Any CPU / CI runner (no GPU or model downloads required)",
        max_vram_gb=0.0,
        estimated_vram_gb="0.0 GB (heuristic backend)",
        tier1_backend="emulator",
        tier1_model="convaiinnovations/laya-emulator",
        tier1_vram_gb=0.0,
        tier2_backend="emulator",
        tier2_model="convaiinnovations/laya-emulator-judge",
        tier2_vram_gb=0.0,
        tier2_default_url=None,
        guard_model="convaiinnovations/laya-emulator",
        guard_vram_gb=0.0,
        escalation_threshold=0.75,
        max_escalation_fraction=0.35,
        adaptive_symmetry_threshold=None,
        symmetry_permutations=1,
        recommended_tier2_cmd="(none required — runs fully in-process)",
        notes=(
            "Deterministic zero-weight emulator profile for testing all 3 API modes "
            "(Decision, Judge, Hybrid Judged Decision) without downloading models."
        ),
    ),
}


def get_hardware_profile(name: str) -> HardwareProfile:
    """Look up a hardware deployment profile by name."""
    key = name.strip().lower()
    if key not in HARDWARE_PROFILES:
        valid = ", ".join(sorted(HARDWARE_PROFILES.keys()))
        raise ValueError(f"Unknown hardware profile '{name}'; expected one of: {valid}")
    return HARDWARE_PROFILES[key]


def build_local_profile_engines(
    profile_name: str = "emulator",
    *,
    tier1_backend: Optional[str] = None,
    tier1_model: Optional[str] = None,
    tier1_url: Optional[str] = None,
    tier2_url: Optional[str] = None,
    tier2_model: Optional[str] = None,
    tier2_endpoint: str = "/v1/systemone",
    device: Optional[str] = None,
    fallback_to_emulator: bool = True,
    calibrator: Optional[Any] = None,
    tier1_transport_fn: Optional[Any] = None,
    tier2_transport_fn: Optional[Any] = None,
) -> Tuple[BaseDecisionEngine, BaseDecisionEngine, Optional[BaseDecisionEngine], HardwareProfile]:
    """Instantiate `(decider_engine, judge_engine, guard_engine, profile)` for a local hardware profile.

    Constructs a 100% local, text-only stack (`$0.00` cloud cost):
      - `decider_engine`: Either a single local engine or a local-only `OmegaDecisionEngine`
        cascading Tier-1 -> local Tier-2 (`SystemOneHTTPBackend`) when `tier2_url` is active.
      - `judge_engine`: The stronger cross-family verifier (Tier-2 when configured, otherwise
        distinct local/emulator engine).
      - `guard_engine`: Lightweight fast ingress safety engine (`Laya` / `EmulatorBackend`).
    """
    profile = get_hardware_profile(profile_name)

    if profile.name == "emulator" and not tier1_url and not tier2_url:
        decider = EmulatorBackend(model_id=tier1_model or profile.tier1_model)
        judge = EmulatorBackend(model_id=tier2_model or profile.tier2_model)
        guard = EmulatorBackend(model_id=profile.guard_model)
        return decider, judge, guard, profile

    # 1. Build Tier-1 local engine
    t1_kind = (tier1_backend or profile.tier1_backend).lower().strip()
    t1_mid = tier1_model or profile.tier1_model
    t1_engine: Optional[BaseDecisionEngine] = None

    if tier1_url:
        t1_engine = SystemOneHTTPBackend(
            base_url=tier1_url,
            model_id=t1_mid,
            endpoint=tier2_endpoint,
            text_only=True,
            calibrator=calibrator,
            transport_fn=tier1_transport_fn,
        )
    else:
        try:
            ek: Dict[str, Any] = {}
            if device is not None and t1_kind in ("torch", "vega", "vegaml", "vega-u"):
                ek["device"] = device
            t1_engine = get_engine(backend=t1_kind, model_id=t1_mid, **ek)
        except Exception as exc:
            if not fallback_to_emulator:
                raise
            logger.warning(
                "Profile '%s': Tier-1 backend '%s' unavailable (%s); falling back to EmulatorBackend.",
                profile.name,
                t1_kind,
                exc,
            )
            t1_engine = EmulatorBackend(model_id=f"{profile.name}-tier1-emulator")

    # 2. Build Tier-2 local HTTP engine (if URL provided or profile specifies local server)
    t2_mid = tier2_model or profile.tier2_model
    t2_engine: Optional[BaseDecisionEngine] = None
    eff_t2_url = tier2_url

    if eff_t2_url:
        t2_engine = SystemOneHTTPBackend(
            base_url=eff_t2_url,
            model_id=t2_mid,
            endpoint=tier2_endpoint,
            text_only=True,
            symmetry_permutations=1,
            adaptive_symmetry_threshold=profile.adaptive_symmetry_threshold,
            calibrator=calibrator,
            calibrator_tier_tag=f"{profile.name}_tier2",
            transport_fn=tier2_transport_fn,
        )

    # 3. Fast ingress guard engine (reuses lightweight Laya if Tier-1 is Laya, else tries Laya/emulator)
    guard_engine: Optional[BaseDecisionEngine] = None
    if t1_kind == "torch" and isinstance(t1_engine, BaseDecisionEngine):
        guard_engine = t1_engine
    else:
        try:
            gk: Dict[str, Any] = {}
            if device is not None:
                gk["device"] = device
            guard_engine = get_engine(backend="torch", model_id="convaiinnovations/laya", **gk)
        except Exception:
            guard_engine = EmulatorBackend(model_id=f"{profile.name}-guard-emulator")

    # 4. Compose Decider & Judge
    if t2_engine is not None and t1_engine is not None:
        is_vega = t1_kind in ("vega", "vegaml", "vega-u")
        decider_engine: BaseDecisionEngine = OmegaDecisionEngine(
            local_laya=None if is_vega else t1_engine,
            local_vega=t1_engine if is_vega else None,
            tier2_engine=t2_engine,
            escalation_threshold=profile.escalation_threshold,
            max_escalation_fraction=profile.max_escalation_fraction,
            joint_schema_tier2=profile.joint_schema_tier2,
        )
        judge_engine: BaseDecisionEngine = t2_engine
    else:
        assert t1_engine is not None
        decider_engine = t1_engine
        judge_engine = t1_engine

    return decider_engine, judge_engine, guard_engine, profile
