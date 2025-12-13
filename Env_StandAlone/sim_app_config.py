"""
Centralized Isaac Sim `SimulationApp` creation.

Why:
- Default Isaac Sim renderer may trigger RTX PSO (RtPso) compilation which can appear "stuck"
  and can also lead to very high peak RAM usage (and OOM kills) on some setups.
- We provide a safer default renderer (HydraStorm) and allow opting back into RTX via env vars.

Env vars:
- DEX_HEADLESS: "1"/"true" to force headless.
- DEX_RENDERER: renderer string, e.g. "HydraStorm", "RayTracedLighting", "PathTracing".
- DEX_ENABLE_RTX: "1"/"true" to force "RayTracedLighting".
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from isaacsim import SimulationApp


def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def create_sim_app(
    *,
    headless: bool = False,
    renderer: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> SimulationApp:
    """
    Create `SimulationApp` with a safer default renderer.

    Notes:
    - If `DEX_ENABLE_RTX` is set, we force RTX real-time renderer.
    - Otherwise default renderer is HydraStorm to avoid RTX RtPso compilation on startup.
    """

    # Allow env to override headless (useful for remote/headless runs).
    if _env_flag("DEX_HEADLESS"):
        headless = True

    # Renderer selection order:
    # 1) explicit function arg
    # 2) DEX_ENABLE_RTX -> RayTracedLighting
    # 3) DEX_RENDERER env var
    # 4) safe default -> HydraStorm
    if renderer is None:
        if _env_flag("DEX_ENABLE_RTX"):
            renderer = "RayTracedLighting"
        else:
            renderer = os.environ.get("DEX_RENDERER", "").strip() or "HydraStorm"

    config: Dict[str, Any] = {
        "headless": headless,
        "renderer": renderer,
    }
    if extra_config:
        config.update(extra_config)

    return SimulationApp(config)


