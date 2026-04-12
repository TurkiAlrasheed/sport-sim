from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from utils import add_edge, save_state

_NEWS_SYSTEM_PROMPT = """\
You are an expert prediction-market analyst. Given a real-world news headline \
and a list of prediction markets, determine which markets are materially \
affected by the news.

For each affected market return a JSON object with:
- "target_id": the market's id (must match exactly)
- "direction": 1 if the news INCREASES the probability the market resolves YES, \
-1 if it DECREASES it
- "strength": float 0.1–1.0 indicating how strong the influence is \
(0.1 = barely relevant, 1.0 = directly decisive)
- "reason": one-sentence explanation of why this news affects that market

Rules:
- Only include markets that are genuinely affected. It is fine to return an \
empty list if the news is irrelevant to all markets.
- Be precise with direction: +1 means the headline makes the market outcome \
MORE likely, -1 means LESS likely.
- Strength should reflect causal proximity: direct evidence > indirect \
implication > loose correlation.

Respond with a JSON object containing a single key "edges" whose value is the \
array of edge objects. Example:
{"edges": [{"target_id": "fed-rate-cut", "direction": 1, "strength": 0.7, \
"reason": "Dovish language raises probability of a rate cut"}]}
"""

_MARKET_SYSTEM_PROMPT = """\
You are an expert prediction-market analyst. Given a SOURCE prediction market \
and a list of OTHER prediction markets, determine which other markets are \
causally influenced by the source market's outcome.

For each influenced market return a JSON object with:
- "target_id": the target market's id (must match exactly)
- "direction": 1 if the source market resolving YES would INCREASE the \
target's probability, -1 if it would DECREASE it
- "strength": float 0.1–1.0 indicating how strong the causal link is \
(0.1 = loose correlation, 1.0 = directly dependent)
- "reason": one-sentence explanation of the causal relationship

Rules:
- Only include markets with a genuine causal or strong correlative link. \
Most market pairs are unrelated — return an empty list when that is the case.
- Think about economic causation: e.g. a rate cut boosts equities, crypto \
regulation affects crypto prices, tariffs affect trade-sensitive sectors.
- Avoid spurious connections. Two markets in the same category are NOT \
automatically linked.
- Do NOT include the source market itself as a target.

Respond with a JSON object containing a single key "edges" whose value is the \
array of edge objects. Example:
{"edges": [{"target_id": "sp500-above-6000", "direction": 1, "strength": 0.6, \
"reason": "Rate cuts lower discount rates, boosting equity valuations"}]}
"""

_MODEL = "gpt-4o-mini"


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return OpenAI(api_key=api_key)


def _format_market_list(markets: list[dict[str, Any]]) -> str:
    lines = []
    for m in markets:
        lines.append(
            f"- id={m['id']}  name={m['name']}  "
            f"description={m['description']}  category={m['category']}  "
            f"probability={m['market_probability']}"
        )
    return "\n".join(lines)


def _parse_edges(
    raw_json: str,
    valid_target_ids: set[str],
    source_id: str,
    source_type: str,
) -> list[dict[str, Any]]:
    """Parse and validate the JSON response into edge dicts."""
    data = json.loads(raw_json)
    raw_edges: list[dict] = data.get("edges", [])

    edges: list[dict[str, Any]] = []
    for e in raw_edges:
        target_id = e.get("target_id", "")
        if target_id not in valid_target_ids or target_id == source_id:
            continue
        direction = int(e.get("direction", 1))
        if direction not in (1, -1):
            direction = 1
        strength = float(e.get("strength", 0.5))
        strength = max(0.1, min(1.0, strength))
        edges.append({
            "source_id": source_id,
            "source_type": source_type,
            "target_id": target_id,
            "direction": direction,
            "strength": round(strength, 2),
            "reason": str(e.get("reason", "")),
        })
    return edges


# ── News → Market edges ──────────────────────────────────────────────────


def generate_news_edges(
    news: dict[str, Any],
    markets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Call GPT-4o-mini to determine which markets a news headline affects."""
    client = _get_client()
    user_prompt = (
        f"NEWS HEADLINE: {news['headline']}\n"
        f"CATEGORY: {news.get('category', 'unknown')}\n"
        f"DATE: {news.get('timestamp', 'unknown')}\n\n"
        f"MARKETS:\n{_format_market_list(markets)}"
    )
    response = client.chat.completions.create(
        model=_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _NEWS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    raw = response.choices[0].message.content or "{}"
    valid_ids = {m["id"] for m in markets}
    return _parse_edges(raw, valid_ids, news["id"], "news")


def generate_all_news_edges(
    state: dict[str, list],
) -> list[dict[str, Any]]:
    """Generate AI edges for every news event. Does NOT mutate state."""
    all_edges: list[dict[str, Any]] = []
    for news in state["news"]:
        all_edges.extend(generate_news_edges(news, state["markets"]))
    return all_edges


# ── Market → Market edges ────────────────────────────────────────────────


def generate_market_edges(
    source_market: dict[str, Any],
    all_markets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Call GPT-4o-mini to determine which other markets a source market influences."""
    other_markets = [m for m in all_markets if m["id"] != source_market["id"]]
    if not other_markets:
        return []

    client = _get_client()
    user_prompt = (
        f"SOURCE MARKET:\n"
        f"  id={source_market['id']}  name={source_market['name']}  "
        f"description={source_market['description']}  "
        f"category={source_market['category']}  "
        f"probability={source_market['market_probability']}\n\n"
        f"OTHER MARKETS:\n{_format_market_list(other_markets)}"
    )
    response = client.chat.completions.create(
        model=_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _MARKET_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    raw = response.choices[0].message.content or "{}"
    valid_ids = {m["id"] for m in other_markets}
    return _parse_edges(raw, valid_ids, source_market["id"], "market")


def generate_all_market_edges(
    state: dict[str, list],
) -> list[dict[str, Any]]:
    """Generate AI edges for every market pair. Does NOT mutate state."""
    all_edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for market in state["markets"]:
        edges = generate_market_edges(market, state["markets"])
        for edge in edges:
            pair = (edge["source_id"], edge["target_id"])
            if pair not in seen:
                seen.add(pair)
                all_edges.append(edge)
    return all_edges


# ── State-mutating helpers ────────────────────────────────────────────────


def _apply_edges(state: dict[str, list], edges: list[dict[str, Any]]) -> int:
    """Deduplicate *edges* against state, add new ones, and persist.

    Returns the number of edges actually added.
    """
    existing_pairs = {
        (e["source_id"], e["target_id"]) for e in state["edges"]
    }
    added = 0
    for edge in edges:
        pair = (edge["source_id"], edge["target_id"])
        if pair not in existing_pairs:
            add_edge(state, edge)
            existing_pairs.add(pair)
            added += 1
    if added:
        save_state(state)
    return added


def apply_news_edges(
    state: dict[str, list],
    news: dict[str, Any],
) -> int:
    """Generate news→market edges and persist them. Returns count added."""
    edges = generate_news_edges(news, state["markets"])
    return _apply_edges(state, edges)


def apply_market_edges(
    state: dict[str, list],
    source_market: dict[str, Any],
) -> int:
    """Generate market→market edges and persist them. Returns count added."""
    edges = generate_market_edges(source_market, state["markets"])
    return _apply_edges(state, edges)


def apply_all_news_edges(state: dict[str, list]) -> int:
    """Regenerate all news→market edges: clears existing, generates fresh, persists.

    Returns the total number of edges added.
    """
    state["edges"] = [e for e in state["edges"] if e["source_type"] != "news"]
    all_edges: list[dict[str, Any]] = []
    for news in state["news"]:
        all_edges.extend(generate_news_edges(news, state["markets"]))
    return _apply_edges(state, all_edges)


def apply_all_market_edges(state: dict[str, list]) -> int:
    """Regenerate all market→market edges: clears existing, generates fresh, persists.

    Returns the total number of edges added.
    """
    state["edges"] = [e for e in state["edges"] if e["source_type"] != "market"]
    all_edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for market in state["markets"]:
        edges = generate_market_edges(market, state["markets"])
        for edge in edges:
            pair = (edge["source_id"], edge["target_id"])
            if pair not in seen:
                seen.add(pair)
                all_edges.append(edge)
    return _apply_edges(state, all_edges)
