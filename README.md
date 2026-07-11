# Tiny Inference Server

A minimal LLM inference server built from scratch in raw Python, implementing
the core techniques that power production serving systems like vLLM and TGI:
KV-caching, batching (static and continuous), and speculative decoding —
all benchmarked and documented, running on CPU with Qwen2.5-0.5B.

## Why this exists

Most AI engineering work treats inference as a black box behind an API call.
This project builds that box from the ground up to understand the actual
systems problems involved: memory management (KV-cache), scheduling
(batching), and pipelining/prediction (speculative decoding). No GPU, no
paid API — everything here runs on a standard CPU Codespace.

## Results

All benchmarks: Qwen2.5-0.5B, CPU, float32, greedy decoding.

| Stage | Technique | Tokens/sec | Notes |
|-------|-----------|-----------:|-------|
| 0 | Naive loop, no cache | 2.27 | Baseline — recomputes full attention every step |
| 1 | KV-cache | 6.71 | ~3× speedup, identical output to Stage 0 (correctness check) |
| 2a | Static batching (3 requests) | 12.79 aggregate | ~1.9× throughput vs single-request cached |
| 2b | Continuous batching, no cache | 3.60 aggregate | Scheduler correctly recycles slots; slower than 2a because cache and scheduling are solved separately here |
| 3 | Speculative decoding, repetitive prompt | 7.10 | 4/4 draft tokens accepted every round — best case |
| 3 | Speculative decoding, novel prompt | 2.33 | 0 draft tokens ever proposed — confirms the technique's limitation |
| 4 | FastAPI server (KV-cache) | 6.26 | Real HTTP service, model loaded once at startup |

## Key learnings

- **KV-caching** is a pure efficiency win — it never changes model output,
  only how fast it's produced. Verified by comparing Stage 0 and Stage 1
  text output (identical).

- **Batching** trades some per-request efficiency (padding waste on
  short sequences) for much higher aggregate throughput — a real gain even
  on CPU, and one that scales further on GPU.

- **Continuous batching (scheduling) and KV-caching (memory reuse) are
  separate concerns** that production systems must solve *together* to
  achieve maximum performance. This project implements them independently
  to keep each concept understandable. Combining them (as vLLM does via
  PagedAttention) is the natural next step.

- **Speculative decoding via n-gram/prompt-lookup matching** provides a
  real ~3× speedup on repetitive or structured text (code, JSON,
  templated outputs). On novel prose it offers no benefit and cleanly
  falls back to standard generation with no correctness penalty. This
  illustrates why production systems typically use a lightweight draft
  *model* instead of pattern matching alone.

## Project structure

```text
tiny-inference-server/
├── stages/
│   ├── stage0_naive_loop.py        # Baseline, no optimization
│   ├── stage1_kv_cache.py          # KV-cache
│   ├── stage2a_static_batch.py     # Fixed-size batching
│   ├── stage2b_continuous_batch.py # Dynamic scheduler, slot recycling
│   └── stage3_speculative.py       # N-gram draft + verification
├── server.py                       # FastAPI server (KV-cache path)
├── requirements.txt
├── NOTES.md                        # Detailed implementation notes
└── README.md
```

## Setup

CPU-only. No GPU or paid API required.
Built and tested on GitHub Codespaces.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run any stage individually:

```bash
python stages/stage1_kv_cache.py
```

Run the FastAPI server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation:

```text
http://localhost:8000/docs
```

Or send a request directly:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of Nigeria is","max_new_tokens":20}'
```

## What's not built

Known limitations and natural extensions:

- Continuous batching + KV-cache combined
  - Would require per-request cache management as batch membership changes
    during decoding, essentially the problem solved by vLLM's
    PagedAttention.

- Speculative decoding with a real draft model
  - Replacing n-gram matching with a lightweight language model would
    generalize beyond repetitive text.

- Quantization (INT8/INT4)
  - Would significantly improve CPU throughput and reduce memory usage.

## Status

✅ Complete.

The project implements the core techniques behind modern inference
systems—KV-caching, static batching, continuous batching, and speculative
decoding—with independent benchmarks, correctness checks, and
documentation explaining the engineering tradeoffs.

See **NOTES.md** for the complete implementation walkthrough.