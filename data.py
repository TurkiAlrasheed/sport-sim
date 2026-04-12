from __future__ import annotations

import os

import requests


FALLBACK_HEADLINES = [
    "Fed signals rate cuts could arrive sooner than expected",
    "Bitcoin rallies as ETF inflows accelerate again",
    "Nvidia jumps after strong AI demand commentary",
    "Oil slips as traders react to softer global growth outlook",
    "Apple unveils new product line ahead of earnings season",
    "Treasury yields fall after cooler inflation data",
    "Tesla shares drop after weaker delivery numbers",
    "Major retailer raises guidance on stronger consumer demand",
]


def fetch_news(api_key: str | None = None, limit: int = 8) -> list[str]:
    api_key = api_key or os.getenv("NEWSAPI_KEY")
    page_size = min(max(limit, 5), 10)

    if not api_key:
        return FALLBACK_HEADLINES[:page_size]

    try:
        response = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                "country": "us",
                "category": "business",
                "pageSize": page_size,
            },
            headers={"X-Api-Key": api_key},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        headlines: list[str] = []
        for article in payload.get("articles", []):
            title = (article.get("title") or "").strip()
            if not title or title == "[Removed]" or title in headlines:
                continue
            headlines.append(title)

        return headlines[:page_size] or FALLBACK_HEADLINES[:page_size]
    except (requests.RequestException, TypeError, ValueError):
        return FALLBACK_HEADLINES[:page_size]
