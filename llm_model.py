"""
LLM client:
- Prefer Gemini (Google Generative AI) if GEMINI_API_KEY or GOOGLE_API_KEY/GEMINI_API_KEY is set.
- Fallback to Hugging Face Inference API if HF_API_KEY is set.
- Final fallback to local `transformers` pipeline (LOCAL_LLM_MODEL env var).
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-small")

LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-small")
HF_API_URL = "https://api-inference.huggingface.co/models/"

# Helper: try multiple Gemini client styles (docs show multiple import patterns)
def _call_gemini(prompt: str, max_tokens: int = 256) -> str | None:
    if not GEMINI_API_KEY:
        return None

    # Try "from google import genai" style (newer).
    try:
        from google import genai  # type: ignore
        # Client picks up env var automatically; create client
        client = genai.Client()
        # Many doc examples use client.models.generate_content(..., contents=...)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        # response object usually has .text
        return getattr(resp, "text", str(resp))
    except Exception:
        pass

    # Try "import google.generativeai as genai" style
    try:
        import google.generativeai as genai  # type: ignore
        # configure if possible
        try:
            # older SDK uses genai.configure
            if hasattr(genai, "configure"):
                genai.configure(api_key=GEMINI_API_KEY)
        except Exception:
            pass

        # Try Client if provided
        if hasattr(genai, "Client"):
            try:
                client = genai.Client()
                resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                return getattr(resp, "text", str(resp))
            except Exception:
                pass

        # Try high-level generate function if present
        # (some SDKs expose genai.generate_text or genai.generate)
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=GEMINI_MODEL, prompt=prompt)
            return getattr(resp, "text", str(resp))
        if hasattr(genai, "generate"):
            resp = genai.generate(model=GEMINI_MODEL, messages=[{"role": "user", "content": prompt}])
            # try to extract candidate text
            if isinstance(resp, dict):
                return json.dumps(resp)
            return str(resp)
    except Exception:
        pass

    return None


def _call_hf_inference(prompt: str, max_tokens: int = 256) -> str | None:
    if not HF_API_KEY:
        return None
    model = HF_MODEL
    url = HF_API_URL + model
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            j = r.json()
            # Many HF text models return an array of {generated_text: "..."}
            if isinstance(j, list) and len(j) > 0 and isinstance(j[0], dict) and "generated_text" in j[0]:
                return j[0]["generated_text"]
            if isinstance(j, dict) and "generated_text" in j:
                return j["generated_text"]
            # Some models return { 'error': ... } or other shapes
            return json.dumps(j)
        else:
            return f"[HF API error {r.status_code}] {r.text}"
    except Exception as e:
        return f"[HF API call failed] {e}"


# Local transformers fallback
_local_pipe = None
def _call_local_transformers(prompt: str, max_tokens: int = 256) -> str | None:
    global _local_pipe
    try:
        if _local_pipe is None:
            from transformers import pipeline  # type: ignore
            # text2text-generation suits instruction-tuned models like flan
            _local_pipe = pipeline("text2text-generation", model=LOCAL_MODEL)
        out = _local_pipe(prompt, max_length=max_tokens)
        if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
            return out[0]["generated_text"]
        return str(out)
    except Exception as e:
        return f"[Local transformers error] {e}"


def ask_llm(prompt: str, max_tokens: int = 256, prefer_gemini: bool = True) -> str:
    """
    Main function to ask the LLM. Order:
      1) Gemini (if configured)
      2) Hugging Face Inference API (if HF_API_KEY)
      3) Local transformers pipeline
    Returns a string (may contain error messages if something went wrong).
    """
    # 1) Gemini
    if prefer_gemini:
        gem = _call_gemini(prompt, max_tokens=max_tokens)
        if gem:
            return gem

    # 2) HF
    hf = _call_hf_inference(prompt, max_tokens=max_tokens)
    if hf:
        return hf

    # 3) Local
    local = _call_local_transformers(prompt, max_tokens=max_tokens)
    if local:
        return local

    return "[No LLM available: set GEMINI_API_KEY or HF_API_KEY or install transformers with a local model]"
