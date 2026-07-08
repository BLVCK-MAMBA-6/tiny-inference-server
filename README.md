# Tiny Inference Server

A minimal LLM inference server built from scratch in raw Python, implementing 
the core techniques that power production serving systems like vLLM and TGI:

- **KV-caching** — avoid recomputing attention over the full sequence at every step
- **Dynamic batching** — serve multiple concurrent requests efficiently on shared hardware
- **Speculative decoding** — use a cheap draft model to propose tokens, verified in parallel by the target model

Target model: Qwen 0.5B (CPU-only, no GPU required)

## Why this exists

Most AI engineering work treats inference as a black box behind an API call. 
This project builds that box from the ground up to understand the actual 
systems problems involved: memory management (KV-cache), scheduling 
(batching), and pipelining/prediction (speculative decoding).

## Project stages

- [ ] **Stage 0** — Naive generate loop, no cache (baseline, tokens/sec)
- [ ] **Stage 1** — Manual KV-cache implementation, measure speedup
- [ ] **Stage 2** — Request queue + dynamic/continuous batching
- [ ] **Stage 3** — Speculative decoding (draft model or n-gram draft)
- [ ] **Stage 4** — Expose over HTTP via FastAPI

## Setup

CPU-only, no GPU or paid API required. Built and run on GitHub Codespaces.

```bash
pip install -r requirements.txt
```

## Status

🚧 In progress — Stage 0

## Notes

Build log and learnings tracked in `NOTES.md` as stages are completed.
