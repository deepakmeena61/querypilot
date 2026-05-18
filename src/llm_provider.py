"""
Multi-provider LLM abstraction layer.
Priority: Anthropic > OpenAI > Google Gemini > Groq
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Session-level call counter (reset on app restart)
_api_call_count = 0
_active_provider: str | None = None

# Providers whose daily quota is exhausted — skip them until process restarts
_exhausted_providers: set[str] = set()

# Token-bucket rate limiter — keeps Google Gemini free tier under 15 RPM
_rl_lock = threading.Lock()
_rl_timestamps: list[float] = []
_RL_WINDOW = 60.0   # seconds
_RL_MAX = 14        # max calls per window (1 below the 15 RPM hard limit)


def _rate_limit(provider: str) -> None:
    """Block briefly if we're about to exceed the Gemini free-tier rate limit."""
    if provider != "google":
        return
    with _rl_lock:
        now = time.time()
        _rl_timestamps[:] = [t for t in _rl_timestamps if now - t < _RL_WINDOW]
        if len(_rl_timestamps) >= _RL_MAX:
            wait = _RL_WINDOW - (now - _rl_timestamps[0]) + 0.2
            if wait > 0:
                logger.info(f"Rate limiter: sleeping {wait:.1f}s to stay under 15 RPM")
                time.sleep(wait)
            _rl_timestamps[:] = [t for t in _rl_timestamps if time.time() - t < _RL_WINDOW]
        _rl_timestamps.append(time.time())


def _detect_provider() -> str:
    """Detect which provider to use based on available API keys, skipping exhausted ones."""
    candidates = [
        ("google", "GOOGLE_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ]
    for name, env_var in candidates:
        if os.environ.get(env_var) and name not in _exhausted_providers:
            return name
    raise EnvironmentError(
        "No LLM API key found (or all providers exhausted). Set at least one of: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY"
    )


def get_provider_info() -> dict:
    """Return display info about the active provider."""
    try:
        provider = _detect_provider()
    except EnvironmentError:
        return {"name": "None", "label": "❌ No API key set", "color": "red", "provider": None}

    info = {
        "anthropic": {
            "name": "anthropic",
            "label": "Claude Sonnet 4 (Anthropic)",
            "model": "claude-sonnet-4-20250514",
            "color": "green",
        },
        "openai": {
            "name": "openai",
            "label": "GPT-4o (OpenAI)",
            "model": "gpt-4o",
            "color": "green",
        },
        "google": {
            "name": "google",
            "label": "Gemini 2.5 Flash Lite (free tier)",
            "model": "gemini-2.5-flash-lite",
            "color": "green",
        },
        "groq": {
            "name": "groq",
            "label": "Llama 3.3 70B (Groq)",
            "model": "llama-3.3-70b-versatile",
            "color": "green",
        },
    }
    return info[provider]


def get_api_call_count() -> int:
    return _api_call_count


def _extract_json_from_text(text: str) -> dict:
    """Try multiple strategies to extract JSON from LLM response text."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"`([\s\S]*?)`",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Try finding the first { ... } block
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response: {text[:500]}")


def _with_backoff(fn, max_retries: int = 3, initial_delay: float = 5.0):
    """Retry with exponential backoff for rate limit errors."""
    delay = initial_delay
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_rate_limit = any(
                kw in err_str
                for kw in ["rate limit", "quota", "429", "resource_exhausted", "too many requests"]
            )
            # Daily/billing quota exhaustion won't recover with short waits — don't retry
            is_permanent = any(kw in err_str for kw in [
                "per_day", "per day", "limit: 0", "perday",
                "check your plan", "billing details",
            ])
            if is_rate_limit and not is_permanent and attempt < max_retries - 1:
                logger.warning(f"Rate limited, waiting {delay:.1f}s before retry {attempt + 1}...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise last_exc


# ── Provider-specific callers ──────────────────────────────────────────────


def _call_anthropic(system_prompt: str, user_prompt: str, temperature: float) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _do():
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return response.content[0].text

    return _with_backoff(_do)


def _call_openai(system_prompt: str, user_prompt: str, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _do():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    return _with_backoff(_do)


def _call_google(system_prompt: str, user_prompt: str, temperature: float) -> str:
    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def _do():
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": temperature,
            },
        )
        return response.text

    return _with_backoff(_do)


def _call_groq(system_prompt: str, user_prompt: str, temperature: float) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _do():
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    return _with_backoff(_do)


# ── JSON-mode callers ──────────────────────────────────────────────────────


def _call_anthropic_json(
    system_prompt: str, user_prompt: str, temperature: float
) -> dict:
    text = _call_anthropic(
        system_prompt + "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no explanation outside the JSON object.",
        user_prompt,
        temperature,
    )
    return _extract_json_from_text(text)


def _call_openai_json(system_prompt: str, user_prompt: str, temperature: float) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _do():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    return _with_backoff(_do)


def _call_google_json(
    system_prompt: str,
    user_prompt: str,
    response_schema: dict,
    temperature: float,
) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    config: dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": temperature,
        "response_mime_type": "application/json",
    }
    if response_schema:
        config["response_schema"] = response_schema

    def _do():
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_prompt,
            config=config,
        )
        return _extract_json_from_text(response.text)

    return _with_backoff(_do)


def _call_groq_json(system_prompt: str, user_prompt: str, temperature: float) -> dict:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _do():
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    return _with_backoff(_do)


# ── Public API ─────────────────────────────────────────────────────────────


def _is_daily_exhausted(e: Exception) -> bool:
    s = str(e).lower()
    return any(kw in s for kw in [
        "per_day", "per day", "limit: 0", "perday",
        "check your plan",    # Google free-tier billing quota exhausted
        "billing details",    # Google free-tier billing quota exhausted
    ])


def _is_unavailable(e: Exception) -> bool:
    s = str(e).lower()
    return any(kw in s for kw in ["503", "unavailable", "high demand", "overloaded"])


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """Call whichever LLM provider is available, auto-falling back on daily quota exhaustion."""
    global _api_call_count
    callers = {
        "anthropic": _call_anthropic,
        "openai": _call_openai,
        "google": _call_google,
        "groq": _call_groq,
    }
    while True:
        provider = _detect_provider()
        _rate_limit(provider)
        logger.info(f"LLM call via {provider}")
        try:
            result = callers[provider](system_prompt, user_prompt, temperature)
            _api_call_count += 1
            return result
        except Exception as e:
            if _is_daily_exhausted(e):
                logger.warning(f"{provider} daily quota exhausted — falling back to next provider")
                _exhausted_providers.add(provider)
            elif _is_unavailable(e):
                logger.warning(f"{provider} unavailable (503) — falling back to next provider")
                _exhausted_providers.add(provider)
            else:
                raise


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    response_schema: dict | None = None,
    temperature: float = 0.0,
) -> dict:
    """
    Call the LLM and parse the response as JSON, auto-falling back on daily quota exhaustion.
    """
    global _api_call_count
    while True:
        provider = _detect_provider()
        _rate_limit(provider)
        logger.info(f"LLM JSON call via {provider}")
        try:
            if provider == "anthropic":
                result = _call_anthropic_json(system_prompt, user_prompt, temperature)
            elif provider == "openai":
                result = _call_openai_json(system_prompt, user_prompt, temperature)
            elif provider == "google":
                result = _call_google_json(system_prompt, user_prompt, response_schema or {}, temperature)
            else:  # groq
                result = _call_groq_json(system_prompt, user_prompt, temperature)
            _api_call_count += 1
            return result
        except Exception as e:
            if _is_daily_exhausted(e):
                logger.warning(f"{provider} daily quota exhausted — falling back to next provider")
                _exhausted_providers.add(provider)
            elif _is_unavailable(e):
                logger.warning(f"{provider} unavailable (503) — falling back to next provider")
                _exhausted_providers.add(provider)
            else:
                raise
