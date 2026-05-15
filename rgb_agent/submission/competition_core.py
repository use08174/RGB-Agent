"""Competition-facing adapter that reuses RGB-Agent's analyzer and action queue."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from arcengine import GameAction

from rgb_agent.agent import ActionQueue, GameState, QueueExhausted, create_analyzer

log = logging.getLogger(__name__)

_RETRY_NUDGE = (
    "CRITICAL: Your previous response was missing the [ACTIONS] section. "
    "You MUST end your response with an [ACTIONS] section containing a JSON action plan. "
    "Do NOT write actions to a file — output them directly in your response text."
)


class RGBSubmissionCore:
    """Wrap RGB-Agent planning logic behind a single-step action interface."""

    def __init__(
        self,
        *,
        game_id: str,
        agent_name: str = "rgb_submission",
        analyzer_model: str = "openai/Qwen2.5-72B-Instruct",
        analyzer_backend: str = "direct",
        analyzer_interval: int = 10,
        analyzer_retries: int = 5,
        max_actions: int = 500,
        logs_root: str | Path | None = None,
    ) -> None:
        self.game_id = game_id
        self.agent_name = agent_name
        self.analyzer_retries = analyzer_retries
        self.max_actions = max_actions

        self._state = GameState(name=agent_name, plan_size=analyzer_interval)
        self._queue = ActionQueue()
        self._analyzer = create_analyzer(
            model=analyzer_model,
            plan_size=analyzer_interval,
            backend=analyzer_backend,
        )

        self._initialized = False
        self._frames_seen = 0
        self._level_num = 1
        self._attempt_num = 1
        self._current_score = 0
        self._current_state = "NOT_PLAYED"

        root = Path(logs_root or os.environ.get("RGB_LOG_DIR", "/kaggle/working/rgb_submission_logs"))
        game_dir = root / game_id.split("-")[0]
        game_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_log_path = game_dir / "logs.txt"
        if not self.prompts_log_path.exists():
            self.prompts_log_path.write_text("")

    def choose_action(self, frames: list[Any], latest_frame: Any) -> GameAction:
        self._sync_frame(frames, latest_frame)

        try:
            action_dict = self._next_action()
        except QueueExhausted:
            loaded = False
            for attempt in range(self.analyzer_retries):
                nudge = _RETRY_NUDGE if attempt > 0 or self._state.action_counter > 0 else ""
                if self._fire_analyzer(retry_nudge=nudge):
                    loaded = True
                    break
            if not loaded:
                return self._safe_fallback(latest_frame)
            try:
                action_dict = self._next_action()
            except QueueExhausted:
                return self._safe_fallback(latest_frame)

        self._state.record_action(action_dict)
        return self._to_game_action(action_dict)

    def should_stop(self, frames: list[Any], latest_frame: Any) -> bool:
        if self._state.action_counter >= self.max_actions:
            return True
        state_name = self._state_name(getattr(latest_frame, "state", ""))
        return state_name == "WIN"

    def _state_name(self, state: Any) -> str:
        return state.name if hasattr(state, "name") else str(state)

    def _frame_to_observation(self, frame: Any) -> dict[str, Any]:
        raw_frame = getattr(frame, "frame", [])
        if hasattr(raw_frame, "tolist"):
            raw_frame = raw_frame.tolist()
        else:
            raw_frame = [layer.tolist() if hasattr(layer, "tolist") else layer for layer in raw_frame]

        return {
            "game_id": getattr(frame, "game_id", self.game_id),
            "state": self._state_name(getattr(frame, "state", "UNKNOWN")),
            "score": int(getattr(frame, "levels_completed", 0) or 0),
            "frame": raw_frame,
            "guid": getattr(frame, "guid", None),
        }

    def _sync_frame(self, frames: list[Any], latest_frame: Any) -> None:
        obs = self._frame_to_observation(latest_frame)
        done = obs["state"] in ("WIN", "GAME_OVER")

        if not self._initialized:
            self._state.reset()
            self._queue.reset()
            self._state.record_env_update(observation=obs, reward=0.0, done=done)
            self._write_initial_board(obs["score"], obs["state"])
            self._current_score = obs["score"]
            self._current_state = obs["state"]
            self._frames_seen = len(frames)
            self._initialized = True
            return

        if len(frames) <= self._frames_seen:
            return

        self._state.record_env_update(observation=obs, reward=0.0, done=done)
        self._queue.check_score(obs["score"])
        self._log_action(self._state.action_counter, self._level_num, self._attempt_num, obs["score"], obs["state"])

        grid = self._state.render_board()
        if grid:
            with open(self.prompts_log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[POST-ACTION BOARD STATE]\nScore: {obs['score']}\n{grid}\n\n")

        if obs["score"] > self._current_score and obs["state"] not in ("WIN", "GAME_OVER"):
            self._level_num += 1
            self._attempt_num = 1
        elif obs["state"] == "GAME_OVER":
            self._attempt_num += 1

        self._current_score = obs["score"]
        self._current_state = obs["state"]
        self._frames_seen = len(frames)

    def _write_initial_board(self, score: int, state: str) -> None:
        grid = self._state.render_board()
        if not grid:
            return
        with open(self.prompts_log_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n{'='*80}\n")
            handle.write("Action 0 | Level 1 | Attempt 1 | INITIAL STATE\n")
            handle.write(f"Score: {score} | State: {state}\n")
            handle.write(f"{'='*80}\n\n")
            handle.write(f"[INITIAL BOARD STATE]\n{grid}\n\n")

    def _log_action(self, action_num: int, level: int, attempt: int, score: int, state: str) -> None:
        if not self._state.trajectory.steps:
            return
        last_step = self._state.trajectory.steps[-1]
        with open(self.prompts_log_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n{'='*80}\n")
            plan_info = (
                f" | Plan Step {self._queue.plan_index}/{self._queue.plan_total}"
                if self._queue.plan_total > 0 else ""
            )
            handle.write(f"Action {action_num} | Level {level} | Attempt {attempt}{plan_info}\n")
            handle.write(f"Score: {score} | State: {state}\n")
            handle.write(f"{'='*80}\n\n")
            for msg in last_step.chat_completions:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                handle.write(f"[{role.upper()}]\n")
                if content:
                    handle.write(f"{content}\n")
                handle.write("\n")

    def _next_action(self) -> dict[str, Any]:
        obs = self._state.last_observation or {}
        state = obs.get("state", "NOT_PLAYED")

        if state in ("NOT_PLAYED", "GAME_OVER") and self._state.last_executed_action != "RESET":
            return {"name": "RESET", "data": {}, "obs_text": "Resetting game state.", "action_text": ""}

        grid_raw, grid_text = self._state.process_frame(obs)
        score = obs.get("score", 0)

        use_queued = bool(self._queue and not self._queue.score_changed)
        if not use_queued:
            self._queue.score_changed = False

        self._state.build_observation_context(
            grid_text,
            score,
            grid_raw,
            use_queued=use_queued,
            queue=self._queue,
        )

        if use_queued and self._queue:
            action = self._queue.pop()
            label = f"plan step {self._queue.plan_index}/{self._queue.plan_total}"
            action["obs_text"] = ""
            action["action_text"] = f"[queued {label}]"
            self._state._last_action_prompt = f"[Queued {label} — no model call]"
            self._state._last_action_response = (
                f"Tool Call: {action['name']}({json.dumps(action['data'])})\n"
                f"Content: Executing pre-planned action ({label})"
            )
            return action

        raise QueueExhausted("Queue empty, no actions from analyzer")

    def _fire_analyzer(self, retry_nudge: str = "") -> bool:
        hint = self._analyzer.analyze(self.prompts_log_path, self._state.action_counter, retry_nudge=retry_nudge)
        if not hint:
            return False

        hint = "\n".join(line.rstrip() for line in hint.split("\n"))
        actions_text = None
        if "\n[ACTIONS]\n" in hint:
            hint, actions_text = hint.split("\n[ACTIONS]\n", 1)
            actions_text = actions_text.strip()

        if "\n[PLAN]\n" in hint:
            full_hint, plan = hint.split("\n[PLAN]\n", 1)
            full_hint, plan = full_hint.strip(), plan.strip()
        else:
            full_hint = plan = hint

        self._state.set_external_hint(full_hint)
        self._state.set_persistent_hint(plan)

        if actions_text:
            return self._queue.load(actions_text)
        return False

    def _safe_fallback(self, latest_frame: Any) -> GameAction:
        state_name = self._state_name(getattr(latest_frame, "state", "UNKNOWN"))
        if state_name in ("NOT_PLAYED", "GAME_OVER"):
            return GameAction.RESET

        for candidate in ("ACTION5", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION7"):
            try:
                action = GameAction.from_name(candidate)
                action.reasoning = "Fallback action after analyzer failure."
                return action
            except Exception:
                continue
        return GameAction.RESET

    def _to_game_action(self, action_dict: dict[str, Any]) -> GameAction:
        action = GameAction.from_name(action_dict["name"])
        if action.is_complex():
            x = max(0, min(63, int(action_dict.get("data", {}).get("x", 0))))
            y = max(0, min(63, int(action_dict.get("data", {}).get("y", 0))))
            action.set_data({"x": x, "y": y})
        action.reasoning = action_dict.get("obs_text", "") or action_dict.get("action_text", "") or "RGB-Agent action"
        try:
            action.action_data.reasoning = action.reasoning
        except Exception:
            pass
        return action
