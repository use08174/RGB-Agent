"""Analyzer factory for choosing between Docker/OpenCode and direct model backends."""
from __future__ import annotations

import os
from typing import Any

from rgb_agent.agent.direct_analyzer import DirectAnalyzerAgent
from rgb_agent.agent.opencode_agent import OpenCodeAgent


def _should_use_direct_backend(model: str) -> bool:
    if os.environ.get("ANALYZER_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("RGB_OPENAI_BASE_URL"):
        return True
    if model.startswith(("openai/", "openrouter/", "local/")):
        return True
    return False


def create_analyzer(
    *,
    model: str,
    plan_size: int,
    backend: str = "auto",
    timeout: int | None = None,
    resume_session: bool = True,
) -> Any:
    if backend == "direct":
        return DirectAnalyzerAgent(
            model=model,
            plan_size=plan_size,
            timeout=timeout,
            resume_session=resume_session,
        )
    if backend == "opencode":
        return OpenCodeAgent(
            model=model,
            plan_size=plan_size,
            timeout=timeout,
            resume_session=resume_session,
        )
    if _should_use_direct_backend(model):
        return DirectAnalyzerAgent(
            model=model,
            plan_size=plan_size,
            timeout=timeout,
            resume_session=resume_session,
        )
    return OpenCodeAgent(
        model=model,
        plan_size=plan_size,
        timeout=timeout,
        resume_session=resume_session,
    )
