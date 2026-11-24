"""LLM interface (OpenAI GPT-4o-mini) with an on-disk response cache.

Purpose
-------
Single entry point for querying the voting LLM with a causal prompt and
reducing its reply to a binary 1/0 vote, following the skeleton in PLAN.md
section 8.3 and matching the base paper's setup: GPT-4o-mini, temperature 0,
a short Yes/No answer mapped to 1/0.

The API key is loaded from the repo-root .env (OPENAI_API_KEY; never committed,
see .env.example). Responses are cached on disk keyed on the exact content
string sent (plus the model), so repeated queries do not re-hit the API — this
is the cost-control mitigation in PLAN.md section 11 (budget ~$30) and keeps
runs deterministic alongside temperature=0.

Maps to: Week 2, Days 3-4 (PLAN.md section 10) — "Implement LLM interface
(OpenAI GPT-4o-mini wrapper)."
"""

import hashlib
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Repo-root-relative paths: .../divvote
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
CACHE_PATH = REPO_ROOT / "data" / "llm_cache.json"

DEFAULT_MODEL = "gpt-4o-mini"
ANSWER_SUFFIX = " Answer only Yes or No."

_client = None
_cache_lock = threading.Lock()


def _get_client():
    """Lazily build the OpenAI client, loading the key from .env on first use."""
    global _client
    if _client is None:
        load_dotenv(ENV_PATH)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OPENAI_API_KEY not found. Copy .env.example to {ENV_PATH} "
                "and set your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def _cache_key(content, model):
    """Stable key for a cache entry: hash of the exact content string + model."""
    raw = f"{model}\x00{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_cache():
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _parse_vote(text):
    """Map a raw model reply to a binary vote (1 if it says yes, else 0)."""
    return 1 if "yes" in text.lower() else 0


def llm_vote(prompt: str, model: str = DEFAULT_MODEL, use_cache: bool = True) -> int:
    """Query the LLM with `prompt` and return a binary vote (1 = Yes, 0 = No).

    Matches PLAN.md section 8.3: appends a Yes/No instruction, temperature=0,
    max_tokens=5. Results are cached on disk keyed on the exact content sent.
    """
    content = prompt + ANSWER_SUFFIX
    key = _cache_key(content, model)

    if use_cache:
        with _cache_lock:
            cache = _load_cache()
        if key in cache:
            return int(cache[key]["vote"])

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=5,
    )
    text = resp.choices[0].message.content or ""
    vote = _parse_vote(text)

    if use_cache:
        with _cache_lock:
            cache = _load_cache()
            cache[key] = {"prompt": content, "model": model,
                          "response": text, "vote": vote}
            _save_cache(cache)

    return vote


def llm_generate(user: str, system: str = None, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, max_tokens: int = 400,
                 use_cache: bool = True) -> str:
    """Free-form text completion (used for one-time prompt-pool generation).

    Defaults to temperature=0 for reproducibility (CLAUDE.md). Responses are
    cached on disk keyed on the messages + model + sampling params.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    key = _cache_key(
        json.dumps({"messages": messages, "t": temperature, "mt": max_tokens},
                   sort_keys=True),
        model)

    if use_cache:
        with _cache_lock:
            cache = _load_cache()
        if key in cache:
            return cache[key]["response"]

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""

    if use_cache:
        with _cache_lock:
            cache = _load_cache()
            cache[key] = {"messages": messages, "model": model,
                          "response": text}
            _save_cache(cache)

    return text


def llm_vote_messages(messages, positive_token: str = "yes",
                      model: str = DEFAULT_MODEL, use_cache: bool = True,
                      max_tokens: int = 5) -> int:
    """Query the LLM with an explicit message list and return a binary vote.

    Used for the base paper's exact prompt (a system + human message pair with
    True/False output; set positive_token="true"). temperature=0; responses
    cached on disk keyed on the exact messages + model.
    """
    key = _cache_key(json.dumps(messages, sort_keys=True), model)

    if use_cache:
        with _cache_lock:
            cache = _load_cache()
        if key in cache:
            return int(cache[key]["vote"])

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    vote = 1 if positive_token.lower() in text.lower() else 0

    if use_cache:
        with _cache_lock:
            cache = _load_cache()
            cache[key] = {"messages": messages, "model": model,
                          "response": text, "vote": vote}
            _save_cache(cache)

    return vote
