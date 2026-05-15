"""Submission helpers for adapting RGB-Agent to official ARC competition runners."""

from rgb_agent.submission.competition_core import RGBSubmissionCore
from rgb_agent.submission.render_submission_agent import build_submission_agent_source

__all__ = ["RGBSubmissionCore", "build_submission_agent_source"]
