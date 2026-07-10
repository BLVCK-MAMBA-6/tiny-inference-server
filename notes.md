# Build Notes

## Stage 0 — Naive loop (no KV-cache)
- Model: Qwen2.5-0.5B, CPU, float32
- Prompt: "The capital of Nigeria is"
- Generated: 30 tokens
- Time: 13.20s
- **Tokens/sec: 2.27**
- Note: every step recomputes attention over the ENTIRE sequence so far,
  including tokens already processed in prior steps. This is the
  inefficiency KV-caching fixes.

## Stage 1 — KV-cache
- Same model, same prompt, same 30 tokens
- Time: 4.47s
- **Tokens/sec: 6.71** (up from 2.27 — ~3x speedup)
- Output text identical to Stage 0 (correctness check — caching should
  never change *what* is generated, only *how fast*)
- Mechanism: past_key_values carries forward each layer's K/V tensors,
  so each step only computes K/V for the new token and reuses the rest

## Stage 2a — Static batching
- 3 prompts, different lengths, batched together with left-padding
- Time: 4.69s for 20 tokens × 3 prompts (60 total)
- Aggregate tokens/sec: 12.79 (vs 6.71 single-request in Stage 1)
- ~1.9x throughput for 3 concurrent requests — not full 3x because:
  1. CPU has less parallel headroom than GPU
  2. Short prompts (e.g. "Hi") waste compute attending over padding
     from longer prompts in the same batch
- Left-padding required so generated tokens stay contiguous at the
  sequence end (right-padding breaks positional continuity)

