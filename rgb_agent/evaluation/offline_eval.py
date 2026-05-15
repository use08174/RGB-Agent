"""Offline evaluation entrypoint for RGB-Agent.

This module adapts the evaluation/result-collection ideas from the notebook
`arc3-agent-evaluation-and-recording-viewer (1).ipynb` to the native RGB-Agent
codebase, without importing any notebook-only baseline agent logic.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import arc_agi
from arc_agi import OperationMode

from rgb_agent.agent import create_analyzer
from rgb_agent.environment.config import EVALUATION_GAMES
from rgb_agent.environment.swarm import Swarm
from rgb_agent.metrics.reporting import calculate_stats, generate_console_report, save_summary_report

log = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env() -> None:
    root = _project_root()
    load_dotenv(dotenv_path=root / ".env.example")
    load_dotenv(dotenv_path=root / ".env", override=True)


def _discover_local_games(env_root: Path) -> list[str]:
    games: list[str] = []
    for metadata_path in sorted(env_root.glob("*/*/metadata.json")):
        try:
            payload = json.loads(metadata_path.read_text())
        except Exception:
            continue
        game_id = payload.get("game_id")
        if game_id:
            games.append(game_id)
    return games


def _resolve_games(game: str | None, suite: str | None, env_root: Path) -> list[str]:
    local_games = _discover_local_games(env_root)
    known = {gid for ids in EVALUATION_GAMES.values() for gid in ids}
    known.update(local_games)
    prefix_map = {gid.split("-")[0]: gid for gid in known}

    if game:
        raw = [g.strip() for g in game.split(",") if g.strip()]
        return [prefix_map.get(g, g) for g in raw]
    if suite:
        return EVALUATION_GAMES[suite]
    return local_games


def _make_run_dir(output_root: Path, agent_name: str, description: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{timestamp}_offline_eval_{agent_name}"
    if description:
        safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in description).strip("-")
        safe = "-".join(part for part in safe.split("-") if part)
        if safe:
            base = f"{base}_{safe}"

    run_dir = output_root / base
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{base}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "game",
        "status",
        "final_score",
        "highest_level_reached",
        "run_total_actions",
        "run_duration_seconds",
        "total_game_overs",
        "replay_url",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_result_rows(results: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for res in sorted(results, key=lambda r: r.game_id):
        rows.append(
            {
                "game": res.game_id,
                "status": getattr(res.status, "value", str(res.status)),
                "final_score": res.final_score,
                "highest_level_reached": res.highest_level_reached,
                "run_total_actions": res.run_total_actions,
                "run_duration_seconds": round(res.run_duration_seconds, 3),
                "total_game_overs": res.total_game_overs_across_run,
                "replay_url": res.replay_url or "",
                "error_message": res.error_message or "",
            }
        )
    return rows


def _write_results_json(path: Path, results: list[Any]) -> None:
    payload = []
    for res in sorted(results, key=lambda r: r.game_id):
        item = asdict(res)
        item["status"] = getattr(res.status, "value", str(res.status))
        item["level_metrics"] = {
            str(level): {
                "level_number": lm.level_number,
                "status": getattr(lm.status, "value", str(lm.status)),
                "attempts": [
                    {
                        "attempt_number": att.attempt_number,
                        "actions": att.actions,
                        "duration_seconds": att.duration_seconds,
                        "state_changes": att.state_changes,
                        "game_overs": att.game_overs,
                        "status": getattr(att.status, "value", str(att.status)),
                    }
                    for att in lm.attempts
                ],
            }
            for level, lm in res.level_metrics.items()
        }
        payload.append(item)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_run_info(path: Path, info: dict[str, Any]) -> None:
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def run_offline_evaluation(
    *,
    agent_name: str = "rgb_agent",
    game: str | None = None,
    suite: str | None = "all",
    max_actions: int = 500,
    analyzer_interval: int = 10,
    analyzer_model: str = "claude-opus-4-6",
    analyzer_backend: str = "auto",
    analyzer_retries: int = 5,
    parallel_games: int | None = None,
    description: str | None = None,
    output_root: str | Path = "evaluation_results",
    environments_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run RGB-Agent evaluation in offline mode and write notebook-style artifacts."""
    _load_env()

    env_root = Path(environments_dir or os.environ.get("ENVIRONMENTS_DIR", "environment_files"))
    if not env_root.exists():
        raise FileNotFoundError(f"Local environments not found: {env_root}")

    games = _resolve_games(game, suite, env_root)
    if not games:
        raise RuntimeError("No local games found for offline evaluation.")

    output_root = Path(output_root)
    run_dir = _make_run_dir(output_root, agent_name, description)

    tags = [f"offline-eval-{agent_name}"]
    if description:
        tags.append(description)

    arcade = arc_agi.Arcade(
        arc_api_key=os.getenv("ARC_API_KEY", ""),
        arc_base_url=os.environ.get("ROOT_URL", "https://three.arcprize.org"),
        operation_mode=OperationMode.OFFLINE,
    )

    analyzer = create_analyzer(
        model=analyzer_model,
        plan_size=analyzer_interval,
        backend=analyzer_backend,
    )
    if parallel_games is None and analyzer_backend == "transformers":
        parallel_games = 1

    swarm = Swarm(
        inner_agent_kwargs={"name": agent_name, "plan_size": analyzer_interval},
        arcade=arcade,
        games=games,
        tags=tags,
        max_actions=max_actions,
        analyzer_hook=analyzer.analyze,
        prompts_log_dir=run_dir,
        log_post_board=True,
        analyzer_retries=analyzer_retries,
        max_parallel_games=parallel_games,
    )
    results = swarm.run()
    results_list = list(results.values())

    scorecard_path = run_dir / "scorecard.json"
    summary_txt_path = run_dir / "summary.txt"
    summary_csv_path = run_dir / "summary.csv"
    results_json_path = run_dir / "results.json"
    run_info_path = run_dir / "run_info.json"

    if swarm.scorecard:
        scorecard_path.write_text(swarm.scorecard.model_dump_json(indent=2), encoding="utf-8")

    rows = _build_result_rows(results_list)
    _write_summary_csv(summary_csv_path, rows)
    _write_results_json(results_json_path, results_list)

    if results_list:
        generate_console_report(results_list, suite or "offline", agent_name, 1, scorecard=swarm.scorecard)
        game_stats, overall = calculate_stats(results_list)
        save_summary_report(
            str(summary_txt_path),
            game_stats,
            overall,
            results_list,
            agent_name,
            suite or "offline",
            1,
            scorecard=swarm.scorecard,
        )
    else:
        summary_txt_path.write_text("No results collected.\n", encoding="utf-8")
        overall = {}

    overall_score = float(swarm.scorecard.score) if swarm.scorecard else None
    run_info = {
        "agent_name": agent_name,
        "games": games,
        "suite": suite,
        "game_arg": game,
        "max_actions": max_actions,
        "analyzer_interval": analyzer_interval,
        "analyzer_model": analyzer_model,
        "analyzer_backend": analyzer_backend,
        "analyzer_retries": analyzer_retries,
        "description": description,
        "parallel_games": parallel_games or len(games),
        "operation_mode": "offline",
        "environments_dir": str(env_root),
        "run_dir": str(run_dir),
        "scorecard_json": str(scorecard_path) if scorecard_path.exists() else None,
        "summary_txt": str(summary_txt_path),
        "summary_csv": str(summary_csv_path),
        "results_json": str(results_json_path),
        "overall_score": overall_score,
        "num_results": len(results_list),
    }
    _write_run_info(run_info_path, run_info)

    return run_info


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run offline RGB-Agent evaluation and save notebook-style artifacts.")
    parser.add_argument("--agent", "-a", default="rgb_agent")
    parser.add_argument("--game", "-g")
    parser.add_argument("--suite", "-s", choices=list(EVALUATION_GAMES.keys()), default="all")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--interval", "-n", dest="analyzer_interval", type=int, default=10)
    parser.add_argument("--model", "-m", dest="analyzer_model", default="claude-opus-4-6")
    parser.add_argument("--analyzer-backend", choices=["auto", "opencode", "direct", "transformers"], default=os.environ.get("ANALYZER_BACKEND", "auto"))
    parser.add_argument("--retries", dest="analyzer_retries", type=int, default=5)
    parser.add_argument("--parallel-games", type=int, default=None)
    parser.add_argument("--description")
    parser.add_argument("--output-root", default="evaluation_results")
    parser.add_argument("--environments-dir")
    args = parser.parse_args()

    result = run_offline_evaluation(
        agent_name=args.agent,
        game=args.game,
        suite=args.suite,
        max_actions=args.max_actions,
        analyzer_interval=args.analyzer_interval,
        analyzer_model=args.analyzer_model,
        analyzer_backend=args.analyzer_backend,
        analyzer_retries=args.analyzer_retries,
        parallel_games=args.parallel_games,
        description=args.description,
        output_root=args.output_root,
        environments_dir=args.environments_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
