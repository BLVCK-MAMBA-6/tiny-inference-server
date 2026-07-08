import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Qwen2.5-0.5B — small enough to run comfortably on CPU
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading tokenizer and model... (first run downloads ~1GB, be patient)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()  # turns off dropout etc. — we're not training, just running inference

# --- Step A: turn text into token IDs ---
prompt = "The capital of Nigeria is"
inputs = tokenizer(prompt, return_tensors="pt")  # "pt" = return PyTorch tensors
print("\nToken IDs:", inputs["input_ids"])
print("Decoded back:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# --- Step B: one forward pass, no gradient tracking (we're not training) ---
with torch.no_grad():
    outputs = model(**inputs)

# --- Step C: inspect the raw output ---
logits = outputs.logits
print("\nLogits shape:", logits.shape)
# shape is [batch_size, sequence_length, vocab_size]
# batch_size=1 (one prompt), sequence_length = number of input tokens,
# vocab_size = ~151k possible tokens Qwen can choose from

# We only care about the LAST position — that's the model's prediction
# for "what comes after everything I've seen so far"
last_token_logits = logits[0, -1, :]
print("Last token logits shape:", last_token_logits.shape)

# --- Step D: turn logits into an actual next token ---
next_token_id = torch.argmax(last_token_logits).item()  # greedy: pick highest score
next_token = tokenizer.decode([next_token_id])

print(f"\nPrompt: '{prompt}'")
print(f"Model's predicted next token: '{next_token}'")