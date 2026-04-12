from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import requests


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EVENT_FALLBACK = 0.02
REQUEST_TIMEOUT_SECONDS = 20
_RESPONSE_CACHE: dict[str, str] = {}
_LAST_LLM_ERROR: str | None = None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _set_last_llm_error(message: str | None) -> None:
    global _LAST_LLM_ERROR
    _LAST_LLM_ERROR = message


def clear_last_llm_error() -> None:
    _set_last_llm_error(None)


def get_last_llm_error() -> str | None:
    return _LAST_LLM_ERROR


def llm_available(api_key: str | None = None) -> bool:
    return bool(api_key or os.getenv("OPENAI_API_KEY"))


def test_openai_connectivity(api_key: str | None = None) -> dict[str, Any]:
    clear_last_llm_error()
    content = _chat_completion(
        api_key=api_key,
        temperature=0.0,
        max_tokens=32,
        cache_namespace=None,
        cache_payload=None,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return strict JSON with keys ok and message. "
                    "Set ok to true and message to a short connectivity confirmation."
                ),
            },
            {
                "role": "user",
                "content": "Run a connectivity check.",
            },
        ],
    )
    error = get_last_llm_error()
    if not content:
        return {
            "ok": False,
            "message": error or "OpenAI request failed.",
            "raw": None,
        }

    payload = _extract_json_object(content)
    if payload is None:
        return {
            "ok": False,
            "message": "OpenAI responded, but the payload was not valid JSON.",
            "raw": content,
        }

    return {
        "ok": bool(payload.get("ok", True)),
        "message": str(payload.get("message", "OpenAI connectivity check succeeded.")),
        "raw": payload,
    }


def _cache_key(namespace: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return f"{namespace}:{digest}"


def _chat_completion(
    *,
    messages: list[dict[str, str]],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 200,
    cache_namespace: str | None = None,
    cache_payload: dict[str, Any] | None = None,
) -> str | None:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        _set_last_llm_error("OPENAI_API_KEY is missing.")
        return None

    cache_id = None
    if cache_namespace and cache_payload is not None:
        cache_id = _cache_key(cache_namespace, cache_payload)
        cached = _RESPONSE_CACHE.get(cache_id)
        if cached is not None:
            return cached

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = ""
        if exc.response is not None and exc.response.text:
            body = exc.response.text.strip().replace("\n", " ")[:180]
        suffix = f" Response: {body}" if body else ""
        _set_last_llm_error(f"OpenAI HTTP {status}.{suffix}")
        return None
    except requests.RequestException as exc:
        _set_last_llm_error(f"OpenAI request failed: {exc}")
        return None
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        _set_last_llm_error(f"OpenAI response parsing failed: {exc}")
        return None

    if cache_id is not None:
        _RESPONSE_CACHE[cache_id] = content
    _set_last_llm_error(None)
    return content


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def get_event_score_llm(event: str, api_key: str | None = None) -> float:
    if not event.strip():
        return DEFAULT_EVENT_FALLBACK

    content = _chat_completion(
        api_key=api_key,
        temperature=0.2,
        max_tokens=24,
        cache_namespace="event-score",
        cache_payload={"event": event},
        messages=[
            {
                "role": "system",
                "content": (
                    "You score market reaction to a headline. "
                    "Return JSON with one key: score. "
                    "Score must be a float between -0.3 and 0.3."
                ),
            },
            {
                "role": "user",
                "content": f"Headline: {event}",
            },
        ],
    )
    if not content:
        return DEFAULT_EVENT_FALLBACK

    payload = _extract_json_object(content)
    if payload is None:
        match = re.search(r"-?\d+(?:\.\d+)?", content)
        if not match:
            _set_last_llm_error("Event-score LLM returned no parseable score.")
            return DEFAULT_EVENT_FALLBACK
        return clamp(float(match.group()), -0.3, 0.3)

    score = payload.get("score", DEFAULT_EVENT_FALLBACK)
    try:
        return clamp(float(score), -0.3, 0.3)
    except (TypeError, ValueError):
        _set_last_llm_error("Event-score LLM returned a non-numeric score.")
        return DEFAULT_EVENT_FALLBACK


def summarize_agent_round(agent_rows: list[dict[str, Any]]) -> str:
    if not agent_rows:
        return "No prior agent reactions."

    ordered = sorted(agent_rows, key=lambda row: row["sentiment"], reverse=True)
    average_sentiment = sum(row["sentiment"] for row in ordered) / len(ordered)
    bullish = ", ".join(
        f"{row['name']} ({row['sentiment']:+.2f})"
        for row in ordered[:2]
    )
    bearish = ", ".join(
        f"{row['name']} ({row['sentiment']:+.2f})"
        for row in ordered[-2:]
    )
    return (
        f"Average sentiment {average_sentiment:+.2f}. "
        f"Most bullish: {bullish}. Most bearish: {bearish}."
    )


def get_agent_round_llm(
    *,
    personas: list[dict[str, Any]],
    event_text: str,
    market_name: str,
    market_description: str,
    market_probability: float,
    topics: list[str],
    hybrid_score: float,
    linked_headlines: list[str],
    round_index: int,
    prior_memories: dict[str, str] | None = None,
    peer_summary: str = "",
    api_key: str | None = None,
) -> list[dict[str, Any]] | None:
    if not personas or not event_text.strip():
        return None

    payload = {
        "personas": personas,
        "event_text": event_text,
        "market_name": market_name,
        "market_description": market_description,
        "market_probability": round(market_probability, 4),
        "topics": topics,
        "hybrid_score": round(hybrid_score, 4),
        "linked_headlines": linked_headlines[:5],
        "round_index": round_index,
        "prior_memories": prior_memories or {},
        "peer_summary": peer_summary,
    }

    content = _chat_completion(
        api_key=api_key,
        temperature=0.35,
        max_tokens=900,
        cache_namespace="agent-round",
        cache_payload=payload,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are simulating a panel of prediction-market agents. "
                    "Stay faithful to each persona. "
                    "Return strict JSON with one key named agents. "
                    "agents must be a list of objects with keys: "
                    "name, sentiment, confidence, narrative. "
                    "sentiment must be between -1 and 1. "
                    "confidence must be between 0 and 1. "
                    "narrative must be under 28 words."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=True),
            },
        ],
    )
    if not content:
        return None

    parsed = _extract_json_object(content)
    if parsed is None:
        _set_last_llm_error(f"Agent round {round_index} returned invalid JSON.")
        return None

    rows = parsed.get("agents")
    if not isinstance(rows, list):
        _set_last_llm_error(f"Agent round {round_index} response is missing the agents list.")
        return None

    persona_names = [persona["name"] for persona in personas]
    rows_by_name = {
        str(row.get("name", "")).strip(): row
        for row in rows
        if isinstance(row, dict)
    }

    normalized: list[dict[str, Any]] = []
    for persona in personas:
        raw = rows_by_name.get(persona["name"], {})
        try:
            sentiment = clamp(float(raw.get("sentiment", 0.0)), -1.0, 1.0)
        except (TypeError, ValueError):
            sentiment = 0.0
        try:
            confidence = clamp(float(raw.get("confidence", 0.5)), 0.0, 1.0)
        except (TypeError, ValueError):
            confidence = 0.5
        narrative = str(raw.get("narrative", "")).strip()
        if not narrative:
            narrative = f"{persona['name']} keeps a neutral stance."
        normalized.append(
            {
                "name": persona["name"],
                "sentiment": sentiment,
                "confidence": confidence,
                "narrative": narrative,
                "source": "llm",
            }
        )

    if [row["name"] for row in normalized] != persona_names:
        _set_last_llm_error(
            f"Agent round {round_index} returned mismatched agent names."
        )
        return None
    _set_last_llm_error(None)
    return normalized
