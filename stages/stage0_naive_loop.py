import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()

prompt = "The capital of Nigeria is"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

NUM_TOKENS_TO_GENERATE = 30

print(f"\nGenerating {NUM_TOKENS_TO_GENERATE} tokens, naive style (no cache)...\n")

start_time = time.time()

for step in range(NUM_TOKENS_TO_GENERATE):
    with torch.no_grad():
        # NOTICE: we feed the ENTIRE sequence every time, including
        # tokens the model has already "seen" in previous steps.
        # This is the wasteful part Stage 1 will fix.
        outputs = model(input_ids)

    last_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(last_token_logits).item()

    # glue the new token onto the sequence
    next_token_tensor = torch.tensor([[next_token_id]])
    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    # optional: print progress as it generates
    print(tokenizer.decode([next_token_id]), end="", flush=True)

end_time = time.time()

total_time = end_time - start_time
tokens_per_sec = NUM_TOKENS_TO_GENERATE / total_time

print(f"\n\n--- Stats ---")
print(f"Total time: {total_time:.2f}s")
print(f"Tokens/sec: {tokens_per_sec:.2f}")
print(f"\nFinal sequence length: {input_ids.shape[1]} tokens")
print(f"Full output: {tokenizer.decode(input_ids[0])}")