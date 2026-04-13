from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from utils import slugify

_BASE_URL = "https://newsapi.org/v2/top-headlines"


def fetch_top_headlines(
    page_size: int = 20,
    country: str = "us",
) -> list[dict[str, Any]]:
    """Fetch top headlines from the past 24 hours via NewsAPI.

    Returns a list of news dicts ready to be added to app state.
    Deduplication and persistence are handled by the caller.
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "NEWSAPI_KEY is not set. Add it to your .env file."
        )

    resp = requests.get(
        _BASE_URL,
        params={
            "apiKey": api_key,
            "country": country,
            "pageSize": page_size,
        },
        timeout=15,
    )

    data = resp.json()

    if resp.status_code != 200 or data.get("status") != "ok":
        msg = data.get("message") or data.get("code") or f"HTTP {resp.status_code}"
        raise RuntimeError(f"NewsAPI error: {msg}")

    articles: list[dict] = data.get("articles", [])

    results: list[dict[str, Any]] = []
    for article in articles:
        title = (article.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue

        published = article.get("publishedAt", "")
        if published:
            try:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                date_str = pub_dt.strftime("%Y-%m-%d")
            except ValueError:
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        results.append({
            "id": slugify(title),
            "headline": title,
            "category": _infer_category(title, article),
            "timestamp": date_str,
            "source": (article.get("source") or {}).get("name", ""),
        })

    return results


_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "macro": [
        "fed", "federal reserve", "inflation", "gdp", "unemployment",
        "interest rate", "rate cut", "rate hike", "jobs report", "cpi",
        "ppi", "treasury", "debt ceiling", "fiscal",
    ],
    "markets": [
        "stock", "s&p", "dow", "nasdaq", "rally", "sell-off", "ipo",
        "earnings", "wall street", "equity", "index", "market",
    ],
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
        "token", "defi", "nft", "stablecoin", "binance", "coinbase",
    ],
    "policy": [
        "congress", "senate", "house", "legislation", "bill", "law",
        "regulation", "executive order", "white house", "supreme court",
        "sanction", "tariff", "election", "vote", "president", "governor",
    ],
    "company": [
        "apple", "google", "microsoft", "amazon", "meta", "tesla",
        "nvidia", "openai", "ceo", "acquisition", "merger", "layoff",
        "revenue", "profit",
    ],
    "culture": [
        "oscars", "grammy", "super bowl", "nfl", "nba", "mlb",
        "entertainment", "celebrity", "movie", "album", "sport",
        "game", "concert", "festival",
    ],
}


def _infer_category(title: str, article: dict) -> str:
    lower = title.lower()
    source_name = ((article.get("source") or {}).get("name") or "").lower()
    combined = f"{lower} {source_name}"

    scores: dict[str, int] = {cat: 0 for cat in _CATEGORY_KEYWORDS}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[cat] += 1

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] > 0:
        return best
    return "markets"
