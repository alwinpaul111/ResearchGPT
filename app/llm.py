"""
LLM wrapper. Two free-tier-friendly backends are supported:

1. Groq (default) - free API key from console.groq.com, extremely fast
   Llama-3 inference, no local GPU needed. Set GROQ_API_KEY.
2. HuggingFace Inference API - uses your existing `alwinn` HF account.
   Set HUGGINGFACEHUB_API_TOKEN.

"""
import os
from app.config import LLM_PROVIDER, GROQ_MODEL, HF_LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


def _call_groq(prompt: str) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
            "and set it as an environment variable."
        )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def _call_huggingface(prompt: str) -> str:
    from huggingface_hub import InferenceClient

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set.")
    client = InferenceClient(model=HF_LLM_MODEL, token=token)
    response = client.text_generation(
        prompt, max_new_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE
    )
    return response.strip()


def generate_answer(prompt: str) -> str:
    if LLM_PROVIDER == "huggingface":
        return _call_huggingface(prompt)
    return _call_groq(prompt)
