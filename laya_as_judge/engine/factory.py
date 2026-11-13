"""Factory and autodetection for Laya decision backends."""

from __future__ import annotations

import logging
import platform
from typing import Any, Optional

from .base import BaseDecisionEngine
from .emulator_backend import EmulatorBackend

logger = logging.getLogger("laya_as_judge")

_EMULATOR_NOTICE_SHOWN = False


def get_engine(
    backend: str = "auto",
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> BaseDecisionEngine:
    """Create or autodetect the best available Laya decision engine.

    Args:
        backend: One of 'auto', 'mlx', 'torch', or 'emulator'.
        model_id: Model checkpoint identifier or local directory path.
        **kwargs: Additional parameters passed to backend constructor.

    Returns:
        An instance of BaseDecisionEngine.
    """
    backend = backend.lower().strip()

    if backend == "mlx":
        from .mlx_backend import MLXBackend
        mid = model_id or "aac6fef/laya-mlx"
        return MLXBackend(model_id=mid, **kwargs)

    if backend in ("vega", "vegaml", "vega-u"):
        from .vega_backend import VegaBackend
        mid = model_id or "0.8b"
        return VegaBackend(model_id=mid, **kwargs)

    if backend == "gemini":
        from ..tier2_gemini import GeminiDecisionEngine
        mid = model_id or "gemini-3.5-flash-lite"
        return GeminiDecisionEngine(model_id=mid, **kwargs)

    if backend == "omega":
        from ..omega import OmegaDecisionEngine
        from ..tier2_gemini import GeminiDecisionEngine
        local_laya = None
        local_vega = None
        try:
            from .torch_backend import TorchBackend
            local_laya = TorchBackend(model_id="convaiinnovations/laya", device=kwargs.get("device"))
        except Exception as e:
            logger.debug("Omega local Laya unavailable: %s", e)
        try:
            from .vega_backend import VegaBackend
            local_vega = VegaBackend(model_id=kwargs.pop("vega_model_id", "0.8b"), device=kwargs.get("device"))
        except Exception as e:
            logger.debug("Omega local Vega unavailable: %s", e)
        t2_mid = model_id or "gemini-3.5-flash-lite"
        t2 = GeminiDecisionEngine(model_id=t2_mid, max_budget_usd=float(kwargs.pop("max_budget_usd", 1.80)))
        return OmegaDecisionEngine(local_laya=local_laya, local_vega=local_vega, tier2_engine=t2, **kwargs)

    if backend == "torch":
        from .torch_backend import TorchBackend
        mid = model_id or "convaiinnovations/laya"
        return TorchBackend(model_id=mid, **kwargs)

    if backend == "emulator":
        mid = model_id or "convaiinnovations/laya-emulator"
        return EmulatorBackend(model_id=mid, **kwargs)

    if backend == "auto":
        # 1. Check for Apple Silicon MLX
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                from .mlx_backend import MLXBackend
                mid = model_id or "aac6fef/laya-mlx"
                logger.info("Auto-detected Apple Silicon: Initializing native MLX backend.")
                return MLXBackend(model_id=mid, **kwargs)
            except Exception as e:
                logger.debug("MLX backend not available on Darwin: %s", e)

        # 2. Check for upstream Laya runtime
        try:
            import laya  # noqa: F401
            from .torch_backend import TorchBackend
            mid = model_id or "convaiinnovations/laya"
            logger.info("Auto-detected Laya runtime: Initializing Torch/Laya backend.")
            return TorchBackend(model_id=mid, **kwargs)
        except Exception as e:
            logger.debug("Laya runtime not available: %s", e)

        # 3. Fallback to the heuristic EmulatorBackend (no model weights involved)
        global _EMULATOR_NOTICE_SHOWN
        if not _EMULATOR_NOTICE_SHOWN:
            _EMULATOR_NOTICE_SHOWN = True
            logger.warning(
                "laya-as-judge: no Laya model runtime found; using EmulatorBackend. "
                "The emulator is a calibrated keyword/heuristic backend, NOT the full Laya neural model: its "
                "verdicts and probabilities are illustrative only. Install "
                "'laya-as-judge[mlx]' on Apple Silicon or 'laya' for real model inference."
            )
        mid = model_id or "convaiinnovations/laya-emulator"
        return EmulatorBackend(model_id=mid, **kwargs)

    raise ValueError(
        f"Unknown backend '{backend}'; expected 'auto', 'mlx', 'torch', 'vega', 'gemini', 'omega', or 'emulator'"
    )
