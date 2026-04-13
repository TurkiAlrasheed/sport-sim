from __future__ import annotations

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Human-readable category names → Kalshi event category strings
# "Trending" is special: pulls across all categories, sorted by volume
KALSHI_CATEGORIES = [
    "Trending",
    "Politics",
    "Elections",
    "Economics",
    "Entertainment",
    "Sports",
    "Science and Technology",
    "Climate and Weather",
    "Financials",
    "Companies",
]


def _fetch_all_events(limit: int = 200) -> list[dict]:
    """Fetch open events with their nested markets in a single API call."""
    resp = requests.get(
        f"{BASE_URL}/events",
        params={"status": "open", "limit": limit, "with_nested_markets": "true"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("events", [])


def _event_volume(event: dict) -> float:
    """Sum 24h volume across all markets in an event."""
    return sum(
        float(m.get("volume_24h_fp") or 0)
        for m in event.get("markets", [])
    )


def _best_market(event: dict) -> dict | None:
    """Return the market with the highest 24h volume, or the first market."""
    markets = event.get("markets", [])
    if not markets:
        return None
    return max(markets, key=lambda m: float(m.get("volume_24h_fp") or 0))


def _market_probability(market: dict) -> float:
    """Convert market ask price to a normalized probability between 0.01 and 0.99."""
    raw = market.get("yes_ask_dollars") or market.get("last_price_dollars") or "0.5"
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return 0.5

    if value > 1:
        # Handle percentage-style values like 50 -> 0.5
        if value <= 100:
            value /= 100
    return max(0.01, min(0.99, value))


def fetch_top_markets(category: str, limit: int = 10) -> list[dict]:
    """Return the top N Kalshi events for a given category, formatted for the UI."""
    events = _fetch_all_events()

    if category == "Trending":
        filtered = events
    else:
        filtered = [e for e in events if e.get("category") == category]

    filtered.sort(key=_event_volume, reverse=True)

    result = []
    for event in filtered:
        if len(result) >= limit:
            break
        market = _best_market(event)
        if market is None:
            continue
        title = event.get("title") or event.get("event_ticker", "")
        sub = market.get("yes_sub_title", "").strip()
        event_text = f"{title} — {sub}" if sub else title
        result.append({
            "title": title,
            "event": event_text,
            "market_probability": _market_probability(market),
            "ticker": market.get("ticker", ""),
        })

    return result
