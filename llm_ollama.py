# llm_ollama.py — local inference via Ollama (no OpenAI)
# ------------------------------------------------------
# Prereqs:
#   - Windows: winget install Ollama.Ollama (then restart terminal)
#   - Start Ollama service:       ollama serve
#   - Pull a model (examples):    ollama pull gemma3:4b-it-q4_K_M
#                                  or    ollama pull llama3.1:8b-instruct
#
# Usage:
#   from llm_ollama import apply_prompt
#   df_out = apply_prompt(df, content_function=..., response_format=..., model="gemma3:4b-it-q4_K_M")

import json
import random
import pandas as pd
import requests
from time import monotonic, sleep
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ----------------- Settings -----------------
# Use 127.0.0.1 to avoid rare localhost resolution issues on some systems.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# Use a valid tag; previous "gemma3:4b-instruct" does not exist in Ollama.
DEFAULT_MODEL   = "gemma3:4b-it-q4_K_M"  # e.g., "llama3.1:8b-instruct", "mistral:7b-instruct"
# --------------------------------------------

# Custom exceptions referenced by the retry policy
class OllamaConnectionError(Exception):
    pass

class OllamaServerError(Exception):
    pass


class RateLimiter:
    """
    Simple RPM-based limiter. If rpm <= 0 (or None), the limiter is disabled.
    """
    def __init__(self, rpm=60, jitter=0.0):
        self.min_interval = 0.0 if (rpm is None or rpm <= 0) else 60.0 / max(1, rpm)
        self.jitter = float(jitter)
        self._last = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        now = monotonic()
        delta = now - self._last
        wait_for = self.min_interval - delta
        if wait_for > 0:
            sleep(wait_for + random.uniform(0, self.jitter))
        self._last = monotonic()


@retry(
    reraise=True,
    wait=wait_exponential(multiplier=2, min=1, max=20),
    stop=stop_after_attempt(4),
    retry=(retry_if_exception_type(OllamaConnectionError) | retry_if_exception_type(OllamaServerError)),
)
def _ollama_chat(*, model, messages, max_tokens=64, temperature=0.0):
    """
    Direct call to Ollama /api/chat
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#chat
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
    except requests.exceptions.RequestException as e:
        raise OllamaConnectionError(f"Cannot reach Ollama at {url}: {e}") from e

    if r.status_code >= 500:
        raise OllamaServerError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
    if r.status_code >= 400:
        # clearer message for missing model
        if r.status_code == 404 and "not found" in r.text.lower():
            raise RuntimeError(
                f"Ollama HTTP 404 — Model not found: '{model}'. "
                f"Pull it first via API: POST /api/pull {{'name':'{model}'}} "
                f"or CLI: `ollama pull {model}`"
            )
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()
    # expected format: {'message': {'role': 'assistant', 'content': '...'}}
    content = ((data or {}).get("message") or {}).get("content", "") or ""
    return content


def apply_prompt_single_text(
    text: str,
    content_function,
    response_format,              # ignored by server; JSON is enforced via prompt
    nested_response=False,        # not used here
    model: str = DEFAULT_MODEL,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
):
    # Build "system"/"user" messages
    messages = [
        {"role": "system", "content": "Assume the role of a financial analyst. Reply ONLY with valid JSON."},
        {"role": "user",   "content": content_function(text)},
    ]
    try:
        content = _ollama_chat(
            model=model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        # Try strict JSON parse; otherwise wrap as {"s": "..."}
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"s": content}
        return parsed
    except (OllamaConnectionError, OllamaServerError) as e:
        return {"error": type(e).__name__, "detail": str(e)}
    except Exception as e:
        return {"error": type(e).__name__, "detail": str(e)}


def apply_prompt(
    dataframe: pd.DataFrame,
    content_function,
    response_format,
    nested_response: bool = False,
    model: str = DEFAULT_MODEL,
    text_column: str = "text",
    max_workers: int = 1,         # unused (sequential)
    sequential: bool = True,
    return_dataframe: bool = True,
    stacked_dataframe: bool = False,
    api_key=None,                 # ignored (no API key needed)
    RPM_BUDGET: int = 0,          # 0 = no artificial wait
    TEMPERATURE: float = 0.0,
    MAX_TOKENS: int = 64,         # enough for one short sentence
):
    # -- text normalization --
    def _normalize_text(x):
        if x is None or (isinstance(x, float) and pd.isna(x)): return ""
        if hasattr(x, "tolist"): x = x.tolist()
        if isinstance(x, (list, tuple)): return " ".join(map(str, x))
        return str(x)

    if text_column not in dataframe.columns:
        raise KeyError(f"Column '{text_column}' not in dataframe")

    texts = [_normalize_text(x) for x in dataframe[text_column].values]

    limiter = RateLimiter(rpm=RPM_BUDGET)
    results = []
    for idx, t in enumerate(texts):
        limiter.wait()
        out = apply_prompt_single_text(
            t,
            content_function=content_function,
            response_format=response_format,
            nested_response=nested_response,
            model=model,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        if isinstance(out, dict) and "error" in out:
            print(f"[warn] row={idx} failed: {out.get('error')} — {out.get('detail','')[:120]}")
        results.append(out)

    def _coerce_row(r): return r if isinstance(r, dict) else {"_value": r}
    rows = list(map(_coerce_row, results))

    if not return_dataframe:
        return rows

    if stacked_dataframe:
        frames = {dataframe.index[i]: pd.DataFrame([rows[i]]) for i in range(len(rows))}
        return pd.concat(frames)
    else:
        return pd.DataFrame(rows, index=dataframe.index)
