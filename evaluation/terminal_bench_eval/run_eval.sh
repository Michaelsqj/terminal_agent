#!/bin/bash

export LITELLM_MODEL="${LITELLM_MODEL:-"openai/Qwen/Qwen3-32B:nebius"}"
export LITELLM_TEMPERATURE="${LITELLM_TEMPERATURE:-0.1}"

export LITE_LLM_API_KEY="${LITE_LLM_API_KEY}"
export LITE_LLM_API_BASE="${LITE_LLM_API_BASE}"

uv run tb run \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --agent-import-path evaluation.terminal_bench_eval.terminal_agent:TerminalBenchAgent \
    --n-concurrent-trials 4 \
    --n-attempts 5
