import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# global dict to hold the model + tokenizer once loaded --
# every request handler reaches into this instead of reloading
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP: runs once, before the server accepts any requests ---
    print("Loading model into memory (once)...")
    ml_models["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
    ml_models["model"] = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    ml_models["model"].eval()
    print("Model loaded. Server ready.")
    yield
    # --- SHUTDOWN: runs once, when the server stops ---
    ml_models.clear()

app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30


class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_sec: float


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    tokenizer = ml_models["tokenizer"]
    model = ml_models["model"]

    input_ids = tokenizer(request.prompt, return_tensors="pt")["input_ids"]

    start_time = time.time()

    # this is Stage 1's KV-cache loop -- our best clean single-request path
    past_key_values = None
    current_input = input_ids

    for _ in range(request.max_new_tokens):
        with torch.no_grad():
            outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
        current_input = torch.tensor([[next_token_id]])
        input_ids = torch.cat([input_ids, current_input], dim=1)

    elapsed = time.time() - start_time
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return GenerateResponse(
        generated_text=generated_text,
        tokens_generated=request.max_new_tokens,
        time_seconds=round(elapsed, 2),
        tokens_per_sec=round(request.max_new_tokens / elapsed, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml_models}