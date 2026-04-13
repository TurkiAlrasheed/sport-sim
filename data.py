from __future__ import annotations

import os
from datetime import date, timedelta

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

FALLBACK_PAST_HEADLINES = [
    "Fed officials signal rates may stay higher for longer",
    "Bitcoin rises as ETF demand remains strong",
    "Nvidia extends rally on AI infrastructure spending optimism",
    "Oil falls as recession fears pressure demand outlook",
    "Apple supplier warns of softer smartphone demand",
    "Treasury yields drop after cooler inflation report",
    "Tesla rebounds after update on lower-cost vehicle plans",
    "Retail sales top expectations, easing slowdown concerns",
    "Chip stocks slip after export restriction headlines",
    "Gold climbs as investors seek safety amid policy uncertainty",
]


def _build_fallback_past_headlines(days: int, headlines_per_day: int) -> list[str]:
    fallback_events: list[str] = []
    base_count = len(FALLBACK_PAST_HEADLINES)
    for day_index in range(days):
        for headline_index in range(headlines_per_day):
            template = FALLBACK_PAST_HEADLINES[(day_index + headline_index) % base_count]
            fallback_events.append(
                f"{template} [fallback day {day_index + 1} item {headline_index + 1}]"
            )
    return fallback_events


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


def fetch_past_news(
    api_key: str | None = None,
    days: int = 10,
    headlines_per_day: int = 10,
) -> list[str]:
    api_key = api_key or os.getenv("NEWSAPI_KEY")
    days = max(days, 1)
    headlines_per_day = min(max(headlines_per_day, 1), 20)
    fallback_events = _build_fallback_past_headlines(days, headlines_per_day)

    if not api_key:
        return fallback_events

    end_date = date.today()
    headlines: list[str] = []

    try:
        for offset in range(days - 1, -1, -1):
            day_start = end_date - timedelta(days=offset)
            day_end = day_start + timedelta(days=1)
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": "fed OR inflation OR stocks OR earnings OR bitcoin OR crypto OR rates OR economy",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "from": day_start.isoformat(),
                    "to": day_end.isoformat(),
                    "pageSize": headlines_per_day,
                },
                headers={"X-Api-Key": api_key},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()

            day_headlines: list[str] = []
            for article in payload.get("articles", []):
                title = (article.get("title") or "").strip()
                if not title or title == "[Removed]" or title in day_headlines:
                    continue
                day_headlines.append(title)

            if len(day_headlines) < headlines_per_day:
                start_index = len(headlines)
                needed = headlines_per_day - len(day_headlines)
                day_headlines.extend(fallback_events[start_index:start_index + needed])

            headlines.extend(day_headlines[:headlines_per_day])

        return headlines or fallback_events
    except (requests.RequestException, TypeError, ValueError):
        return fallback_events
