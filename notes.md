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

## Stage 2b — Continuous batching (no cache, scheduler isolated)
- MAX_BATCH_SIZE = 2, 5 queued requests, different lengths
- Scheduler correctly recycles slots as requests finish (verified in logs:
  request 1 finishes at step 4, request 2 fills the freed slot at step 5)
- Aggregate tokens/sec: 3.60 — WORSE than Stage 1 (6.71) and Stage 2a (12.79)
- Why: this version has no KV-cache, so every step recomputes full
  attention for all active requests (Stage 0's inefficiency, wrapped
  in a scheduler)
- Key insight: continuous batching (scheduling) and KV-caching (memory
  reuse) are separate concerns. Production systems (vLLM, TGI) need
  BOTH simultaneously — that's genuinely the hard systems-engineering
  problem (vLLM's PagedAttention exists to solve exactly this: giving
  each request its own persistent, resizable cache inside a shared batch)

  ## Stage 3 — Speculative decoding (n-gram/prompt-lookup draft, no cache)
- Repetitive prompt: "one two three four" x3
- 30 tokens, 6 draft rounds, 4/4 accepted EVERY round (best case)
- Time: 4.23s, Tokens/sec: 7.10 (vs Stage 0's 2.27 -- ~3x speedup, no cache)
- Mechanism: n-gram match on already-seen text proposes draft tokens,
  ONE forward pass verifies all of them, accepted prefix + model's own
  correction token both come "free" from that single pass
- LIMITATION (important): only works when text is repetitive/structured.
  Non-repetitive prompts (e.g. general prose) get near-zero draft
  acceptance since there's no earlier occurrence to match against.
  Production systems use a real small draft MODEL instead, which can
  guess plausible tokens for ANY text, not just previously-seen text.
- Not yet combined with Stage 1's KV-cache (isolated for clarity, same
  as Stage 2b's tradeoff)

## Stage 3 — Limitation confirmed (non-repetitive prompt)
- Prompt: "The history of the Roman Empire began when"
- Draft rounds attempted: 0 (no repeated 3-gram found anywhere in the text)
- Tokens/sec: 2.33 -- statistically identical to Stage 0's naive 2.27
- Confirms: n-gram/prompt-lookup drafting provides ZERO benefit on novel
  prose, falling back cleanly to plain token-by-token generation.
  No correctness cost, but no speedup either -- this is why production
  systems use a real draft MODEL (can generalize to any text) rather
  than pure text pattern-matching (only works on repeated content:
  code, JSON, structured/templated output).