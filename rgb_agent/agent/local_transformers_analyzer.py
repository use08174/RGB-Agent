"""Local analyzer that calls a Hugging Face Transformers model directly."""
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

from rgb_agent.agent.prompts import ACTIONS_ADDENDUM

log = logging.getLogger(__name__)

_DEFAULT_MAX_HEAD_CHARS = int(os.environ.get("ANALYZER_HEAD_CHARS", "1000"))
_DEFAULT_MAX_TAIL_CHARS = int(os.environ.get("ANALYZER_TAIL_CHARS", "4000"))
_DEFAULT_RECENT_BLOCKS = int(os.environ.get("ANALYZER_RECENT_BLOCKS", "4"))
_DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("ANALYZER_MAX_OUTPUT_TOKENS", "256"))

_SYSTEM_PROMPT = """\
You are the analyzer for an AI agent playing a grid-based puzzle game.

Your response MUST contain ALL sections below in this exact order:
1. A detailed strategic briefing
2. A [PLAN] section with a short 2-3 sentence plan
3. An [ACTIONS] section containing one valid JSON object with a non-empty "plan" array

Do not omit [ACTIONS]. Do not wrap JSON in Markdown fences.
"""


@dataclass
class _SessionState:
    previous_response: str = ""


class LocalTransformersAnalyzerAgent:
    """Runs the analyzer against a locally loaded Transformers model."""

    def __init__(
        self,
        *,
        model: str,
        plan_size: int = 5,
        timeout: Optional[int] = None,
        resume_session: bool = True,
    ) -> None:
        del timeout
        self._model_ref = self._resolve_model_ref(model)
        self._plan_size = plan_size
        self._resume_session = resume_session
        self._sessions: dict[str, _SessionState] = {}
        self._lock = Lock()
        self._model_lock = Lock()
        self._inference_lock = Lock()
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _resolve_model_ref(self, model: str) -> str:
        env_path = os.environ.get("LOCAL_TRANSFORMERS_MODEL_PATH")
        if env_path:
            return env_path
        if model.startswith("transformers/"):
            return model.split("/", 1)[1]
        if model.startswith("local/"):
            return model.split("/", 1)[1]
        return model

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return
        with self._model_lock:
            if self._model is not None and self._tokenizer is not None and self._torch is not None:
                return

            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_ref, trust_remote_code=True)
            config = AutoConfig.from_pretrained(self._model_ref, trust_remote_code=True)

            load_in_4bit = os.environ.get("TRANSFORMERS_LOAD_IN_4BIT", "1") == "1"
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
            }

            dtype_name = os.environ.get("TRANSFORMERS_TORCH_DTYPE", "bfloat16")
            if dtype_name == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16

            model_quant_config = getattr(config, "quantization_config", None)
            has_embedded_quant = model_quant_config is not None

            if load_in_4bit and not has_embedded_quant:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=os.environ.get("TRANSFORMERS_4BIT_QUANT", "nf4"),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
                )

            if has_embedded_quant:
                log.info(
                    "Model %s already provides quantization_config=%s; skipping BitsAndBytesConfig override.",
                    self._model_ref,
                    type(model_quant_config).__name__,
                )

            self._model = AutoModelForCausalLM.from_pretrained(self._model_ref, **model_kwargs)
            self._model.eval()

    def analyze(self, log_path: Path, action_num: int, retry_nudge: str = "") -> Optional[str]:
        if not log_path.exists():
            return None

        self._ensure_model()

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
        return _SYSTEM_PROMPT + "\nUpdate your briefing based on what changed since the prior analysis."

    def _build_log_summary(self, log_text: str) -> str:
        latest_board = self._extract_board_by_index(log_text, -1)
        previous_board = self._extract_board_by_index(log_text, -2)
        recent_blocks = self._extract_recent_action_blocks(log_text, _DEFAULT_RECENT_BLOCKS)
        head = log_text[:_DEFAULT_MAX_HEAD_CHARS].strip()
        tail = log_text[-_DEFAULT_MAX_TAIL_CHARS:].strip()

        parts = ["[RUN SUMMARY]"]
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

    def _extract_recent_action_blocks(self, log_text: str, limit: int) -> str:
        matches = list(re.finditer(r"^={80}\nAction .+?(?=^={80}\nAction |\Z)", log_text, re.M | re.S))
        if not matches:
            return ""
        return "\n\n".join(match.group(0).strip() for match in matches[-limit:])

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
        messages = [
            {"role": "system", "content": "Return plain text only. You must include [PLAN] and [ACTIONS]."},
            {"role": "user", "content": prompt},
        ]
        try:
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered = (
                "System: Return plain text only. You must include [PLAN] and [ACTIONS].\n\n"
                f"User: {prompt}\n\nAssistant:"
            )

        with self._inference_lock:
            inputs = self._tokenizer(rendered, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _normalize_response(self, response_text: str | None) -> str | None:
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

        return (
            f"{briefing}\n\n"
            f"[PLAN]\n{plan_text}\n\n"
            f"[ACTIONS]\n{json.dumps(actions_payload, ensure_ascii=False, indent=2)}"
        ).strip()

    def _extract_plan_text(self, text: str) -> str:
        match = re.search(r"\[PLAN\]\s*(.*?)(?:\n\s*\[[A-Z]+\]|\Z)", text, re.S)
        if match:
            plan = match.group(1).strip()
            if plan:
                return plan
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return " ".join(lines[-2:])[:400] if lines else "Execute the proposed short batch, then re-evaluate."

    def _extract_actions_payload(self, text: str) -> dict | None:
        for candidate in reversed(list(self._iter_json_candidates(text))):
            if not self._looks_like_action_payload(candidate):
                continue
            if isinstance(candidate, list):
                return {"plan": candidate, "reasoning": ""}
            plan = candidate.get("plan", candidate.get("actions"))
            if isinstance(plan, list) and plan:
                return {"plan": plan, "reasoning": str(candidate.get("reasoning", ""))}
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

    def _write_analyzer_log(self, analyzer_log: Path, action_num: int, prompt: str, response_text: str | None) -> None:
        with open(analyzer_log, "a", encoding="utf-8") as handle:
            handle.write(f"\n--- action={action_num} | {datetime.now().strftime('%H:%M:%S')} | transformers ---\n")
            handle.write(f"[PROMPT]\n{prompt}\n\n")
            if response_text:
                handle.write(f"[ASSISTANT]\n{response_text}\n\n")
            else:
                handle.write("[ERROR]\nNo response returned.\n\n")
