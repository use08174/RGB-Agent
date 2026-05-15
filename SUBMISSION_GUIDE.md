# Submission Guide

This project now supports two distinct workflows:

1. Local development on your workstation
2. Kaggle competition submission

For ARC-AGI-3, these are not the same thing.

## Local Development

Recommended setup:

- Run the analyzer with `--analyzer-backend direct`
- Serve your model from the same machine with an OpenAI-compatible endpoint such as `vLLM`
- Use `rgb-offline-eval` or `rgb-swarm --operation-mode offline` for fast iteration

Typical flow:

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1

rgb-offline-eval \
  --suite all \
  --model openai/Qwen2.5-72B-Instruct \
  --analyzer-backend direct \
  --description local-dev
```

For a machine with `RTX PRO 6000 Blackwell 96GB`, serving `Qwen2.5-72B-Instruct` locally with `vLLM` is realistic.

## Kaggle Submission

The ARC-AGI-3 docs state:

- Kaggle competition runs in `Competition Mode`
- Kaggle uses the `listen_and_serve(..., competition_mode=True)` style environment

Implications:

- Your submission must be self-contained inside the Kaggle runtime
- Do not rely on a model server running on your external workstation
- `127.0.0.1` inside Kaggle refers to the Kaggle container itself, not your own machine
- If you use `vLLM`, it should run inside the Kaggle notebook runtime
- Model weights and Python wheels should be available as Kaggle input datasets so the notebook can install and load them without internet

## Is vLLM Allowed?

Based on the public ARC docs, `vLLM` is a tooling choice, not a forbidden API.

The safe interpretation is:

- `vLLM` running inside the Kaggle notebook/runtime: likely acceptable
- `vLLM` running on your own remote workstation and queried from Kaggle: risky and not recommended for submission

Competition docs to review:

- Competition Mode: https://docs.arcprize.org/toolkit/competition_mode
- Listen And Serve: https://docs.arcprize.org/toolkit/listen_and_serve

## Practical Recommendation

Use a split workflow:

### Development

- Use your `RTX PRO 6000 Blackwell`
- Run `vLLM` locally
- Tune prompts, models, and batching offline

### Submission

- Package dependencies as offline wheels
- Package model weights as a Kaggle dataset
- Start `vLLM` inside the Kaggle notebook
- Point `OPENAI_BASE_URL` to the local notebook server
- Run the agent in the Kaggle competition runtime only

## Project-Specific Notes

- This project now supports a Docker-free analyzer backend called `direct`
- The analyzer accepts OpenAI-compatible endpoints through `OPENAI_BASE_URL`
- The action planner now supports `ACTION7`, which is part of ARC-AGI-3's standard action set

## Current Gap

This repository is optimized for local experimentation and offline evaluation.
It is not yet a full drop-in replacement for the official ARC-AGI-3-Agents submission repository structure.

That means:

- local benchmarking is ready
- Kaggle submission packaging still needs a submission notebook that starts the local model server and runs the competition entrypoint under Kaggle constraints

The simplest path is usually:

1. Use this repo to develop the policy
2. Port the final agent logic into the official Kaggle submission structure
3. Keep the model-serving strategy local to the Kaggle runtime

## Recommended 96GB Config

For your `RTX PRO 6000 Blackwell 96GB`, the most practical single-GPU submission target is:

- Model: `Qwen/Qwen2.5-72B-Instruct-AWQ`
- Backend: `vLLM`
- Analyzer backend in this repo: `direct`
- OpenAI base URL inside the runtime: `http://127.0.0.1:8000/v1`

Suggested starting server settings:

```bash
vllm serve /path/to/Qwen2.5-72B-Instruct-AWQ \
  --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --generation-config vllm \
  --trust-remote-code
```

Why these defaults:

- `AWQ` is supported by vLLM on Ada-class GPUs and newer
- a 16k max context is much safer than trying to run the full model context in competition
- `generation-config vllm` avoids unexpected sampling defaults from the model repo

Relevant docs:

- vLLM OpenAI server quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart/
- vLLM quantization support: https://docs.vllm.ai/en/latest/features/quantization/
- Qwen AWQ model card: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ
