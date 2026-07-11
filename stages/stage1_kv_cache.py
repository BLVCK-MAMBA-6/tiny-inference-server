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

print(f"\nGenerating {NUM_TOKENS_TO_GENERATE} tokens, WITH KV-cache...\n")

start_time = time.time()

past_key_values = None  # this will hold the cached K/V tensors, layer by layer
current_input = input_ids  # first call: full prompt. after that: just 1 new token

for step in range(NUM_TOKENS_TO_GENERATE):
    with torch.no_grad():
        outputs = model(
            current_input,
            past_key_values=past_key_values,  # feed in what we've cached so far
            use_cache=True,                    # tell the model to return the cache
        )

    # grab the updated cache — now includes K/V for the token(s) we just processed
    past_key_values = outputs.past_key_values

    last_token_logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(last_token_logits).item()

    # KEY DIFFERENCE from Stage 0: next input is JUST the new token,
    # not the whole sequence — the cache already "remembers" everything before it
    current_input = torch.tensor([[next_token_id]])

    input_ids = torch.cat([input_ids, current_input], dim=1)  # just for tracking full text
    print(tokenizer.decode([next_token_id]), end="", flush=True)

end_time = time.time()

total_time = end_time - start_time
tokens_per_sec = NUM_TOKENS_TO_GENERATE / total_time

print(f"\n\n--- Stats ---")
print(f"Total time: {total_time:.2f}s")
print(f"Tokens/sec: {tokens_per_sec:.2f}")
print(f"\nFull output: {tokenizer.decode(input_ids[0])}")