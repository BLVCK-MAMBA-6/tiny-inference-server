import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Qwen's tokenizer has no pad token by default — we need one for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()

# Different length prompts on purpose — this is the whole point
prompts = [
    "The capital of Nigeria is",
    "Hi",
    "In the field of machine learning, the term backpropagation refers to",
]

# tokenizer handles padding for us when we pass a list + padding=True
# padding_side="left" matters for generation: we want new tokens appended
# on the right, so padding must sit on the left of the real content
tokenizer.padding_side = "left"
batch = tokenizer(prompts, return_tensors="pt", padding=True)

print("\nInput IDs (padded, note the pad tokens):")
print(batch["input_ids"])
print("\nAttention mask (1 = real token, 0 = padding):")
print(batch["attention_mask"])

input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]

NUM_TOKENS_TO_GENERATE = 20

print(f"\nGenerating {NUM_TOKENS_TO_GENERATE} tokens for all {len(prompts)} prompts AS ONE BATCH...\n")

start_time = time.time()

past_key_values = None
current_input = input_ids
current_mask = attention_mask

for step in range(NUM_TOKENS_TO_GENERATE):
    with torch.no_grad():
        outputs = model(
            current_input,
            attention_mask=current_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

    past_key_values = outputs.past_key_values

    # logits shape now: [batch_size, seq_len, vocab_size]
    # grab the last token's logits for EACH item in the batch
    next_token_logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]
    next_token_ids = torch.argmax(next_token_logits, dim=-1)  # shape: [batch_size]

    current_input = next_token_ids.unsqueeze(-1)  # shape: [batch_size, 1]

    # extend the attention mask with a 1 for each new real token generated
    current_mask = torch.cat(
        [current_mask, torch.ones((current_mask.shape[0], 1), dtype=torch.long)],
        dim=1,
    )

    input_ids = torch.cat([input_ids, current_input], dim=1)

end_time = time.time()
total_time = end_time - start_time
total_tokens = NUM_TOKENS_TO_GENERATE * len(prompts)

print(f"--- Stats ---")
print(f"Total time: {total_time:.2f}s")
print(f"Total tokens generated (across batch): {total_tokens}")
print(f"Tokens/sec (aggregate): {total_tokens / total_time:.2f}")

print("\n--- Outputs ---")
for i, ids in enumerate(input_ids):
    print(f"\n[{i}] {tokenizer.decode(ids, skip_special_tokens=True)}")