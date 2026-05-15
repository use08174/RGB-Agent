# Read-Grep-Bash Agent

An agent for [ARC-AGI-3](https://three.arcprize.org/) that completes all three preview games in 1,069 actions, the lowest publicly reported count.

For details on approach and findings, see our [blog post](https://blog.alexisfox.dev/arcagi3).

![Architecture](assets/architecture.png)

## Setup

Requires Python (3.12 recommended). Docker is only required for the original `opencode` analyzer backend.

```bash
git clone git@github.com:alexisfox7/RGB-Agent.git
cd RGB-Agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
cd docker/opencode-sandbox && bash build.sh   # build analyzer sandbox image (only for opencode backend)
```

Create a `.env` file:

```
ARC_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Usage

```bash
rgb-swarm --suite all --max-actions 500
rgb-swarm --game ls20,ft09
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--suite` | — | Predefined game suites (e.g. `ls20`, `vc33`, `ft09`, or `all`) |
| `--game` | — | Comma-separated game names or IDs (alternative to `--suite`) |
| `--max-actions` | 500 | Max actions per game |
| `--interval`, `-n` | 10 | Actions per analyzer batch plan |
| `--model`, `-m` | `claude-opus-4-6` | Analyzer model (see below) |
| `--operation-mode` | `online` | `online` / `offline` / `normal` |
| `--analyzer-backend` | `auto` | `opencode` / `direct` / `auto` |

### Models

Anthropic models can be passed without a prefix. For other providers, use `provider/model`.

| Model | `--model` value |
|-------|-----------------|
| Claude Opus 4.6 | `claude-opus-4-6` (default) |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| GPT 5.2 | `openai/gpt-5.2` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |

Any model available via OpenRouter can also be used with the `openrouter/` prefix (e.g. `openrouter/google/gemini-3.1-pro-preview`).

Set the matching API key in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `OPENROUTER_API_KEY`).

### Using a local model

This agent can also use a local OpenAI-compatible endpoint such as `vLLM` for models like `Qwen 2.5 72B Instruct`.

There are now two analyzer backends:

- `opencode`: original Docker/OpenCode analyzer
- `direct`: Docker-free analyzer that calls an OpenAI-compatible chat endpoint directly

For Kaggle or any Docker-less environment, use `direct`.

Example `.env`:

```bash
ARC_API_KEY=...
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

Example run:

```bash
rgb-swarm --suite all --model openai/Qwen2.5-72B-Instruct --analyzer-backend direct
```

If your local server uses a different model ID, pass that exact value after the `openai/` prefix.

If you still use the original Docker/OpenCode path, the analyzer runs inside a container, so your host model server usually needs:

```bash
OPENAI_BASE_URL=http://host.docker.internal:8000/v1
```

### Kaggle / Offline Usage

The repository already includes local `environment_files/` for the preview games, so you can avoid ARC API game discovery and run offline:

```bash
rgb-swarm --suite all --operation-mode offline --analyzer-backend direct --model openai/Qwen2.5-72B-Instruct
```

In `offline` mode the runner uses locally bundled environments instead of fetching the game list from the remote API.

Results are saved to `evaluation_results/`.

For competition-specific guidance on local development vs Kaggle submission constraints, see [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md).

## Architecture

The analyzer agent can run either through [OpenCode](https://github.com/opencode-ai/opencode) in a sandboxed Docker container or through the Docker-free `direct` backend that calls an OpenAI-compatible endpoint. In both cases, the analyzer reads the game's prompt log and outputs a JSON action plan. The action queue drains these one per step with zero LLM calls. When the queue empties or the score changes, the analyzer re-fires.

```
rgb_agent/
├── agent/              
│   ├── opencode_agent.py # Runs OpenCode in Docker to produce action plans
│   ├── action_queue.py # Drains one action per step (to support batched action plans + score-change flush)
│   ├── game_state.py   # Formatting
│   └── prompts.py      
├── environment/        
│   ├── arcagi3.py      # ARC-AGI-3 API wrapper (reset, step, scoring)
│   ├── runner.py       # Per-game orchestration loop
│   ├── swarm.py        # Runs multiple games in parallel on a scorecard
│   └── config.py     
├── metrics/            
└── utils/            
```
