import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()

def n_gram_draft(input_ids, n=3, num_draft_tokens=4):
    """
    Look for the last n tokens somewhere earlier in the sequence.
    If found, propose whatever tokens followed that earlier occurrence
    as our cheap 'draft' guess for what comes next.
    """
    seq = input_ids[0].tolist()
    if len(seq) < n:
        return []
    suffix = seq[-n:]
    for i in range(len(seq) - n):
        if seq[i:i + n] == suffix:
            start = i + n
            return seq[start:start + num_draft_tokens]
    return []

# Repetitive on purpose -- this is what lets n-gram matching actually work
prompt = "The history of the Roman Empire began when"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

NUM_TOKENS_TO_GENERATE = 30
N_GRAM = 3
DRAFT_LEN = 4

print(f"\nGenerating {NUM_TOKENS_TO_GENERATE} tokens WITH speculative decoding (n-gram draft)...\n")

start_time = time.time()
tokens_generated = 0
total_accepted = 0
total_draft_rounds = 0

while tokens_generated < NUM_TOKENS_TO_GENERATE:
    draft_tokens = n_gram_draft(input_ids, n=N_GRAM, num_draft_tokens=DRAFT_LEN)

    if not draft_tokens:
        # no repeated pattern found -- fall back to plain single-token step
        with torch.no_grad():
            outputs = model(input_ids)
        next_id = torch.argmax(outputs.logits[0, -1, :]).item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        tokens_generated += 1
        continue

    total_draft_rounds += 1

    draft_tensor = torch.tensor([draft_tokens])
    candidate = torch.cat([input_ids, draft_tensor], dim=1)

    # ONE forward pass verifies the ENTIRE draft -- this is the whole payoff.
    # Normally getting len(draft_tokens) tokens costs that many forward passes.
    with torch.no_grad():
        outputs = model(candidate)

    context_len = input_ids.shape[1]
    accepted = 0
    for i, draft_tok in enumerate(draft_tokens):
        predicted = torch.argmax(outputs.logits[0, context_len - 1 + i, :]).item()
        if predicted == draft_tok:
            accepted += 1
        else:
            break  # stop at first disagreement

    total_accepted += accepted

    if accepted > 0:
        input_ids = torch.cat([input_ids, torch.tensor([draft_tokens[:accepted]])], dim=1)
        tokens_generated += accepted

    # the model's own prediction at the divergence point is correct regardless
    # -- we already computed it in the verify pass, so it's free
    if tokens_generated < NUM_TOKENS_TO_GENERATE:
        correction = torch.argmax(outputs.logits[0, context_len - 1 + accepted, :]).item()
        input_ids = torch.cat([input_ids, torch.tensor([[correction]])], dim=1)
        tokens_generated += 1

end_time = time.time()
total_time = end_time - start_time

print(f"--- Stats ---")
print(f"Total time: {total_time:.2f}s")
print(f"Tokens/sec: {tokens_generated / total_time:.2f}")
print(f"Draft rounds attempted: {total_draft_rounds}")
print(f"Total draft tokens accepted: {total_accepted}")
if total_draft_rounds > 0:
    print(f"Avg accepted per round: {total_accepted / total_draft_rounds:.2f} / {DRAFT_LEN}")
print(f"\nFull output: {tokenizer.decode(input_ids[0])}")