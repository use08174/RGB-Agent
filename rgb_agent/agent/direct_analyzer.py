"""Direct analyzer that calls an OpenAI-compatible chat endpoint without Docker."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import requests

from rgb_agent.agent.prompts import ACTIONS_ADDENDUM

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = int(os.environ.get("ANALYZER_TIMEOUT_SECONDS", "180"))
_DEFAULT_MAX_HEAD_CHARS = int(os.environ.get("ANALYZER_HEAD_CHARS", "12000"))
_DEFAULT_MAX_TAIL_CHARS = int(os.environ.get("ANALYZER_TAIL_CHARS", "42000"))
_DEFAULT_RECENT_BLOCKS = int(os.environ.get("ANALYZER_RECENT_BLOCKS", "40"))

_SYSTEM_PROMPT = """\
You are the analyzer for an AI agent playing a grid-based puzzle game.

You will be given:
- a structured summary of the run
- selected log excerpts from the agent's prompt log
- the latest board state extracted programmatically
- optionally your previous analysis for continuity

Your response MUST contain ALL sections below in this exact order:
1. A detailed strategic briefing (be concrete, reference coordinates when useful)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>

3. Followed by exactly this separator and a single valid JSON object:

[ACTIONS]
{"plan": [{"action": "ACTION1"}, {"action": "ACTION6", "x": 3, "y": 7}], "reasoning": "why these steps"}

Do not omit [ACTIONS]. Do not wrap the JSON in Markdown fences. Do not output any extra sections after [ACTIONS].
"""


@dataclass
class _SessionState:
    previous_response: str = ""


class DirectAnalyzerAgent:
    """Runs the analyzer directly against an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        model: str,
        plan_size: int = 5,
        timeout: Optional[int] = None,
        resume_session: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._provider, self._model_name = self._normalize_model(model)
        self._plan_size = plan_size
        self._timeout = timeout or _DEFAULT_TIMEOUT
        self._resume_session = resume_session
        self._base_url = self._resolve_base_url(base_url, self._provider)
        self._api_key = api_key or os.environ.get("ANALYZER_API_KEY") or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self._sessions: dict[str, _SessionState] = {}
        self._lock = Lock()

    @staticmethod
    def _normalize_model(model: str) -> tuple[str, str]:
        if "/" not in model:
            return "openai", model
        provider, raw_model = model.split("/", 1)
        if provider in {"openai", "openrouter", "local"}:
            return provider, raw_model
        return provider, model

    @staticmethod
    def _resolve_base_url(base_url: str | None, provider: str) -> str:
        if base_url:
            return base_url.rstrip("/")
        if provider == "openrouter":
            return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        resolved = (
            os.environ.get("ANALYZER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("RGB_OPENAI_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        )
        return resolved.rstrip("/")

    def analyze(self, log_path: Path, action_num: int, retry_nudge: str = "") -> Optional[str]:
        if not log_path.exists():
            return None

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        path_key = str(log_path)
        analyzer_log = log_path.parent / f"{log_path.stem}_analyzer.txt"

        with self._lock:
            previous = self._sessions.get(path_key, _SessionState()).previous_response

        prompt = self._build_prompt(
            log_text=log_text,
            is_first=not bool(previous),
            previous_response=previous,
            retry_nudge=retry_nudge,
        )

        response_text = self._call_model(prompt)
        response_text = self._normalize_response(response_text)
        self._write_analyzer_log(analyzer_log, action_num, prompt, response_text)

        if not response_text:
            return None

        if self._resume_session:
            with self._lock:
                self._sessions[path_key] = _SessionState(previous_response=response_text)

        return response_text

    def _normalize_response(self, response_text: str | None) -> str | None:
        """Best-effort repair so downstream code sees [PLAN] and [ACTIONS]."""
        if not response_text:
            return None

        text = response_text.replace("\r\n", "\n").strip()
        if "\n[ACTIONS]\n" in text and "\n[PLAN]\n" in text:
            return text

        actions_payload = self._extract_actions_payload(text)
        if not actions_payload:
            return text

        plan_text = self._extract_plan_text(text)

        briefing = text
        if "\n[PLAN]\n" in briefing:
            briefing = briefing.split("\n[PLAN]\n", 1)[0].strip()
        elif "\n[ACTIONS]\n" in briefing:
            briefing = briefing.split("\n[ACTIONS]\n", 1)[0].strip()

        rebuilt = (
            f"{briefing}\n\n"
            f"[PLAN]\n{plan_text}\n\n"
            f"[ACTIONS]\n{json.dumps(actions_payload, ensure_ascii=False, indent=2)}"
        )
        return rebuilt.strip()

    def _build_prompt(
        self,
        *,
        log_text: str,
        is_first: bool,
        previous_response: str,
        retry_nudge: str,
    ) -> str:
        summary = self._build_log_summary(log_text)
        instructions = self._build_instructions(is_first)
        parts = [instructions, summary, ACTIONS_ADDENDUM.format(plan_size=self._plan_size)]

        if self._resume_session and previous_response:
            parts.append("[PREVIOUS ANALYSIS]\n" + previous_response.strip())

        if retry_nudge:
            parts.append(retry_nudge.strip())

        return "\n\n".join(part for part in parts if part.strip())

    def _build_instructions(self, is_first: bool) -> str:
        if is_first:
            return _SYSTEM_PROMPT + "\nWork from the supplied summaries and excerpts to infer mechanics and choose the next short action batch."
        return (
            _SYSTEM_PROMPT
            + "\nUpdate your briefing based on what changed since the prior analysis."
            + "\nFocus on score transitions, board deltas, and whether the recent actions helped."
        )

    def _build_log_summary(self, log_text: str) -> str:
        latest_board = self._extract_latest_board(log_text)
        previous_board = self._extract_previous_board(log_text)
        recent_blocks = self._extract_recent_action_blocks(log_text, _DEFAULT_RECENT_BLOCKS)
        score_summary = self._extract_score_summary(log_text)
        head = log_text[:_DEFAULT_MAX_HEAD_CHARS].strip()
        tail = log_text[-_DEFAULT_MAX_TAIL_CHARS:].strip()

        parts = [
            "[RUN SUMMARY]",
            score_summary,
        ]
        if latest_board:
            parts.append("[LATEST BOARD STATE]\n" + latest_board)
        if previous_board:
            parts.append("[PREVIOUS BOARD STATE]\n" + previous_board)
        if recent_blocks:
            parts.append("[RECENT ACTION BLOCKS]\n" + recent_blocks)
        if head:
            parts.append("[LOG HEAD]\n" + head)
        if tail and tail != head:
            parts.append("[LOG TAIL]\n" + tail)
        return "\n\n".join(parts)

    def _extract_plan_text(self, text: str) -> str:
        match = re.search(r"\[PLAN\]\s*(.*?)(?:\n\s*\[[A-Z]+\]|\Z)", text, re.S)
        if match:
            plan = match.group(1).strip()
            if plan:
                return plan

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return " ".join(lines[-2:])[:400]
        return "Execute the proposed short batch, then re-evaluate based on board changes and score transitions."

    def _extract_actions_payload(self, text: str) -> dict | None:
        for candidate in reversed(list(self._iter_json_candidates(text))):
            if not self._looks_like_action_payload(candidate):
                continue
            if isinstance(candidate, list):
                return {"plan": candidate, "reasoning": ""}
            plan = candidate.get("plan", candidate.get("actions"))
            if isinstance(plan, list) and plan:
                return {
                    "plan": plan,
                    "reasoning": str(candidate.get("reasoning", "")),
                }
        return None

    def _iter_json_candidates(self, text: str):
        clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
        decoder = json.JSONDecoder()
        for idx, char in enumerate(clean):
            if char not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(clean, idx)
            except json.JSONDecodeError:
                continue
            yield parsed

    def _looks_like_action_payload(self, candidate: object) -> bool:
        if isinstance(candidate, list):
            return bool(candidate)
        if not isinstance(candidate, dict):
            return False
        plan = candidate.get("plan", candidate.get("actions"))
        return isinstance(plan, list) and bool(plan)

    def _extract_score_summary(self, log_text: str) -> str:
        action_headers = re.findall(r"Action (\d+) \| Level (\d+) \| Attempt (\d+)", log_text)
        score_lines = re.findall(r"Score: (\d+) \| State: ([A-Z_]+)", log_text)
        scores = [int(score) for score, _ in score_lines]
        transitions = []
        prev = None
        for score in scores:
            if prev is None or score != prev:
                transitions.append(score)
            prev = score

        latest_state = score_lines[-1][1] if score_lines else "UNKNOWN"
        latest_score = scores[-1] if scores else 0
        return (
            f"Total logged action headers: {len(action_headers)}\n"
            f"Observed score transitions: {transitions[-12:]}\n"
            f"Latest score: {latest_score}\n"
            f"Latest state: {latest_state}"
        )

    def _extract_recent_action_blocks(self, log_text: str, limit: int) -> str:
        matches = list(re.finditer(r"^={80}\nAction .+?(?=^={80}\nAction |\Z)", log_text, re.M | re.S))
        if not matches:
            return ""
        return "\n\n".join(match.group(0).strip() for match in matches[-limit:])

    def _extract_latest_board(self, log_text: str) -> str:
        return self._extract_board_by_index(log_text, -1)

    def _extract_previous_board(self, log_text: str) -> str:
        return self._extract_board_by_index(log_text, -2)

    def _extract_board_by_index(self, log_text: str, index: int) -> str:
        parts = re.split(r"\[(?:POST-ACTION|INITIAL) BOARD STATE\]\n", log_text)
        if len(parts) <= abs(index):
            return ""
        block = parts[index].strip()
        lines = block.splitlines()
        if lines and lines[0].startswith("Score:"):
            lines = lines[1:]

        board_lines: list[str] = []
        for line in lines:
            if not line.strip() or line.startswith("[") or line.startswith("="):
                break
            board_lines.append(line)
        return "\n".join(board_lines).strip()

    def _call_model(self, prompt: str) -> Optional[str]:
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": "Return plain text only. You must include [PLAN] and [ACTIONS]."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        if self._provider == "openrouter":
            headers.setdefault("HTTP-Referer", os.environ.get("OPENROUTER_HTTP_REFERER", "https://github.com/alexisfox7/RGB-Agent"))
            headers.setdefault("X-Title", os.environ.get("OPENROUTER_X_TITLE", "RGB-Agent"))

        try:
            response = requests.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            log.error("direct analyzer request failed: %s", exc, exc_info=True)
            return None

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            log.error("unexpected direct analyzer response: %s", json.dumps(data)[:1000])
            return None

    def _write_analyzer_log(self, analyzer_log: Path, action_num: int, prompt: str, response_text: str | None) -> None:
        with open(analyzer_log, "a", encoding="utf-8") as handle:
            handle.write(f"\n--- action={action_num} | {datetime.now().strftime('%H:%M:%S')} | direct ---\n")
            handle.write(f"[PROMPT]\n{prompt}\n\n")
            if response_text:
                handle.write(f"[ASSISTANT]\n{response_text}\n\n")
            else:
                handle.write("[ERROR]\nNo response returned.\n\n")
