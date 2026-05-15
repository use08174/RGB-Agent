"""Generate an agent file that plugs RGB-Agent into the official ARC agent repo."""
from __future__ import annotations


def build_submission_agent_source(class_name: str = "RGBSubmissionAgent") -> str:
    return f'''"""Submission adapter that exposes RGB-Agent through the official Agent interface."""
from __future__ import annotations

import os
from typing import Any

from .agent import Agent
from .structs import FrameData, GameState, GameAction

from rgb_agent.submission.competition_core import RGBSubmissionCore


class {class_name}(Agent):
    """Wrap RGB-Agent's planning loop behind the ARC-AGI-3-Agents interface."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._core = RGBSubmissionCore(
            game_id=self.game_id,
            agent_name=os.environ.get("RGB_AGENT_NAME", "rgb_submission"),
            analyzer_model=os.environ.get("RGB_MODEL", "openai/Qwen2.5-72B-Instruct"),
            analyzer_backend=os.environ.get("RGB_ANALYZER_BACKEND", "direct"),
            analyzer_interval=int(os.environ.get("RGB_ANALYZER_INTERVAL", "10")),
            analyzer_retries=int(os.environ.get("RGB_ANALYZER_RETRIES", "5")),
            max_actions=int(os.environ.get("RGB_MAX_ACTIONS", "500")),
            logs_root=os.environ.get("RGB_LOG_DIR", "/kaggle/working/rgb_submission_logs"),
        )

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self._core.should_stop(frames, latest_frame)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        return self._core.choose_action(frames, latest_frame)
'''
