"""Cached Groq wrapper.

Every call is stored on disk (SQLite) keyed by a hash of (model, system, prompt,
temperature, max_tokens, sample_id), so a crashed or repeated run never pays for
the same call twice. It also retries on rate limits, keeps under the tokens-per-minute
limit, and refuses to exceed the daily request budget.

Usage:
    from llm import chat
    text = chat("Rewrite as one sentence: ...", role="rewrite")
    # SelfCheckGPT: same prompt, k different samples -> use sample_id=0..k-1
    s = chat(prompt, role="sampler", temperature=1.0, sample_id=2)

Needs GROQ_API_KEY in .env (project root) and the `llm:` section of config.yaml.
"""
import hashlib
import json
import os
import sqlite3
import time
from collections import deque
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
LLM = CFG["llm"]
CACHE_DIR = ROOT / CFG["project"]["paths"]["cache"]
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_db = None
_client = None
_window = deque()  # (timestamp, tokens) of calls in the last 60 s


def _load_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GROQ_API_KEY not found. Put it in .env in the project root.")


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=_load_key())
    return _client


def _get_db():
    global _db
    if _db is None:
        _db = sqlite3.connect(CACHE_DIR / "llm_cache.sqlite")
        _db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, response TEXT)")
        _db.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT, model TEXT, calls INTEGER, PRIMARY KEY (day, model))")
        _db.commit()
    return _db


def _key(model, system, prompt, temperature, max_tokens, sample_id):
    blob = json.dumps([model, system, prompt, temperature, max_tokens, sample_id], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _wait_for_token_budget(est_tokens):
    """Stay under tokens_per_minute by sleeping until the last-60s window has room."""
    limit = LLM["tokens_per_minute"]
    est_tokens = min(est_tokens, limit)
    while True:
        now = time.time()
        while _window and now - _window[0][0] > 60:
            _window.popleft()
        used = sum(t for _, t in _window)
        if used + est_tokens <= limit:
            _window.append((now, est_tokens))
            return
        time.sleep(max(0.5, 60 - (now - _window[0][0]) + 0.1))


def _count_call(model):
    db = _get_db()
    today = date.today().isoformat()
    row = db.execute("SELECT calls FROM usage WHERE day=? AND model=?", (today, model)).fetchone()
    calls = row[0] if row else 0
    if calls >= LLM["daily_request_limit"]:
        raise RuntimeError(
            "Daily request limit reached for %s (%d calls today). Run again tomorrow, "
            "or use another account's key." % (model, calls)
        )
    db.execute("INSERT OR REPLACE INTO usage VALUES (?,?,?)", (today, model, calls + 1))
    db.commit()


def chat(prompt, role="rewrite", system=None, temperature=None, max_tokens=400, sample_id=0, model=None):
    """Return the model's reply text. role: 'rewrite' (also mitigation) or 'sampler'."""
    import groq

    if model is None:
        model = LLM["sampler_model"] if role == "sampler" else LLM["rewrite_model"]
    if temperature is None:
        temperature = LLM["temperature_rewrite"]

    key = _key(model, system, prompt, temperature, max_tokens, sample_id)
    db = _get_db()
    hit = db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
    if hit:
        return hit[0]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    prompt_tokens = (len(prompt) + len(system or "")) // 4
    mt = max_tokens  # doubled automatically if the reasoning model uses it all on thinking

    extra = {}
    if LLM.get("reasoning_effort"):
        extra["reasoning_effort"] = LLM["reasoning_effort"]  # gpt-oss: keep hidden thinking short

    last_err = None
    for attempt in range(LLM["max_retries"]):
        _wait_for_token_budget(prompt_tokens + mt)
        try:
            _count_call(model)
            resp = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=mt,
                extra_body=extra or None,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                # reasoning used up max_tokens: do not cache an empty answer, retry with more room
                last_err = ValueError("empty reply (finish_reason=%s) at max_tokens=%d" % (resp.choices[0].finish_reason, mt))
                mt = min(mt * 2, 2000)
                continue
            db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, model, text))
            db.commit()
            return text
        except groq.RateLimitError as e:
            last_err = e
            retry_after = getattr(getattr(e, "response", None), "headers", {}).get("retry-after")
            time.sleep(float(retry_after) if retry_after else min(60, 2 ** attempt * 2))
        except (groq.APIConnectionError, groq.InternalServerError) as e:
            last_err = e
            time.sleep(min(60, 2 ** attempt * 2))
    raise RuntimeError("Groq call failed after %d tries: %s" % (LLM["max_retries"], last_err))


def usage_today():
    rows = _get_db().execute("SELECT model, calls FROM usage WHERE day=?", (date.today().isoformat(),)).fetchall()
    return dict(rows)


if __name__ == "__main__":
    print(chat("Say hi in 3 words.", role="sampler", max_tokens=200))
    print("calls today:", usage_today())
    print(chat("Say hi in 3 words.", role="sampler", max_tokens=200), "(second call comes from cache)")
    print("calls today:", usage_today())