# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from inference import get_backend
import config

app = FastAPI()
inference_backend = get_backend(config.INFERENCE_BACKEND)


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 1.0
    max_new_tokens: int = 50
    top_k: int = 5


@app.post("/generate")
async def generate_text(req: GenerateRequest):
    return inference_backend.generate(req.prompt, req.temperature, req.max_new_tokens, req.top_k)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)