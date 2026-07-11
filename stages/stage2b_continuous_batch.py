import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()

MAX_BATCH_SIZE = 2  # deliberately small so you SEE slot recycling happen

# Simulate a queue of incoming requests — different lengths, different
# max_new_tokens, to force requests to finish at different times
request_queue = [
    {"id": 0, "prompt": "The capital of Nigeria is", "max_new_tokens": 15},
    {"id": 1, "prompt": "Hi", "max_new_tokens": 5},
    {"id": 2, "prompt": "In the field of machine learning, backpropagation refers to", "max_new_tokens": 20},
    {"id": 3, "prompt": "Once upon a time", "max_new_tokens": 10},
    {"id": 4, "prompt": "The best way to learn programming is", "max_new_tokens": 12},
]

active = []  # requests currently occupying a batch slot
completed = []  # finished requests, with final text + timing

start_time = time.time()
step = 0

def build_batch(active_requests):
    """Turn the list of active requests into one padded batch tensor."""
    texts = [r["input_ids"] for r in active_requests]
    max_len = max(t.shape[1] for t in texts)
    padded = []
    masks = []
    for t in texts:
        pad_len = max_len - t.shape[1]
        pad = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
        padded.append(torch.cat([pad, t], dim=1))
        mask = torch.cat([torch.zeros((1, pad_len), dtype=torch.long),
                           torch.ones((1, t.shape[1]), dtype=torch.long)], dim=1)
        masks.append(mask)
    return torch.cat(padded, dim=0), torch.cat(masks, dim=0)

while request_queue or active:
    # --- SCHEDULER: fill free slots from the waiting queue ---
    while len(active) < MAX_BATCH_SIZE and request_queue:
        req = request_queue.pop(0)
        req["input_ids"] = tokenizer(req["prompt"], return_tensors="pt")["input_ids"]
        req["generated"] = 0
        req["arrival_step"] = step
        active.append(req)
        print(f"[step {step}] slot filled -> request {req['id']} ('{req['prompt'][:20]}...')")

    if not active:
        break

    # --- forward pass on whatever is currently active (full recompute, no cache) ---
    batch_input_ids, batch_mask = build_batch(active)
    with torch.no_grad():
        outputs = model(batch_input_ids, attention_mask=batch_mask)

    next_token_logits = outputs.logits[:, -1, :]
    next_token_ids = torch.argmax(next_token_logits, dim=-1)

    still_active = []
    for i, req in enumerate(active):
        new_token = next_token_ids[i].unsqueeze(0).unsqueeze(0)
        req["input_ids"] = torch.cat([req["input_ids"], new_token], dim=1)
        req["generated"] += 1

        if req["generated"] >= req["max_new_tokens"]:
            req["finish_step"] = step
            req["text"] = tokenizer.decode(req["input_ids"][0], skip_special_tokens=True)
            completed.append(req)
            print(f"[step {step}] request {req['id']} FINISHED after "
                  f"{step - req['arrival_step'] + 1} steps -> slot freed")
        else:
            still_active.append(req)

    active = still_active
    step += 1

end_time = time.time()
total_time = end_time - start_time
total_tokens = sum(r["max_new_tokens"] for r in completed)

print(f"\n--- Stats ---")
print(f"Total wall-clock time: {total_time:.2f}s")
print(f"Total tokens across all requests: {total_tokens}")
print(f"Aggregate tokens/sec: {total_tokens / total_time:.2f}")
print(f"Total steps taken: {step}")

print("\n--- Completed requests (in finish order) ---")
for r in sorted(completed, key=lambda x: x["finish_step"]):
    print(f"\n[id {r['id']}] finished at step {r['finish_step']}: {r['text']}")