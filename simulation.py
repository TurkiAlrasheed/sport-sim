from __future__ import annotations

from dataclasses import dataclass
import os
import random
import re
from typing import Any, Iterable

import requests


# ── Keyword / phrase scoring tables ──────────────────────────────────────

POSITIVE_PHRASES = {
    "rate cut": 0.18,
    "cuts rates": 0.18,
    "fed cuts": 0.16,
    "beats earnings": 0.22,
    "record sales": 0.18,
    "etf approval": 0.24,
    "drops album": 0.18,
    "new album": 0.12,
    "partnership announced": 0.12,
    "ceasefire": 0.20,
}

NEGATIVE_PHRASES = {
    "rate hike": -0.22,
    "raises rates": -0.22,
    "misses earnings": -0.25,
    "album delayed": -0.18,
    "delays album": -0.18,
    "investigation launched": -0.28,
    "guidance cut": -0.22,
    "lawsuit filed": -0.24,
    "supply shock": -0.18,
    "recession fears": -0.22,
}

TOKEN_WEIGHTS = {
    "approval": 0.10,
    "approved": 0.10,
    "bullish": 0.14,
    "boost": 0.10,
    "beat": 0.12,
    "beats": 0.12,
    "growth": 0.10,
    "launch": 0.08,
    "surge": 0.12,
    "viral": 0.10,
    "win": 0.08,
    "wins": 0.08,
    "record": 0.08,
    "cut": 0.04,
    "cuts": 0.04,
    "bearish": -0.14,
    "delay": -0.12,
    "delayed": -0.12,
    "fear": -0.10,
    "fears": -0.10,
    "inflation": -0.10,
    "investigation": -0.16,
    "lawsuit": -0.16,
    "miss": -0.14,
    "misses": -0.14,
    "recession": -0.20,
    "risk": -0.10,
    "scandal": -0.20,
    "shock": -0.12,
    "tariffs": -0.12,
}

TOPIC_KEYWORDS = {
    "macro": {"fed", "rates", "inflation", "cpi", "jobs", "gdp", "recession", "treasury"},
    "markets": {"stocks", "equities", "earnings", "etf", "ipo", "guidance", "shares"},
    "crypto": {"bitcoin", "btc", "ethereum", "eth", "crypto", "token"},
    "culture": {"album", "drake", "tour", "movie", "streaming", "viral", "song"},
    "policy": {"election", "senate", "president", "tariffs", "regulation", "ban"},
    "company": {"apple", "tesla", "nvidia", "microsoft", "meta", "amazon", "google"},
}


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PersonaTemplate:
    name: str
    role: str
    base_bias: float
    volatility: float
    topic_tilts: dict[str, float]


@dataclass
class AgentReaction:
    name: str
    role: str
    sentiment: float
    narrative: str


PERSONA_TEMPLATES = [
    PersonaTemplate(
        name="Optimist",
        role="Looks for upside and momentum",
        base_bias=0.12,
        volatility=0.10,
        topic_tilts={"markets": 0.06, "culture": 0.08, "crypto": 0.05},
    ),
    PersonaTemplate(
        name="Pessimist",
        role="Overweights downside and second-order risks",
        base_bias=-0.12,
        volatility=0.10,
        topic_tilts={"macro": -0.05, "policy": -0.06, "markets": -0.04},
    ),
    PersonaTemplate(
        name="Macro Trader",
        role="Focuses on rate expectations and liquidity",
        base_bias=0.02,
        volatility=0.12,
        topic_tilts={"macro": 0.14, "policy": 0.05, "markets": 0.06},
    ),
    PersonaTemplate(
        name="Economist",
        role="Evaluates whether the event changes fundamentals",
        base_bias=0.00,
        volatility=0.08,
        topic_tilts={"macro": 0.16, "policy": 0.08, "culture": -0.04},
    ),
    PersonaTemplate(
        name="Retail Investor",
        role="Chases narratives that feel obvious and tradable",
        base_bias=0.06,
        volatility=0.13,
        topic_tilts={"markets": 0.09, "culture": 0.05, "company": 0.05},
    ),
    PersonaTemplate(
        name="Momentum Chaser",
        role="Buys what sounds like attention and flow",
        base_bias=0.08,
        volatility=0.14,
        topic_tilts={"markets": 0.08, "crypto": 0.10, "culture": 0.07},
    ),
    PersonaTemplate(
        name="Contrarian",
        role="Assumes the crowd is overreacting",
        base_bias=-0.03,
        volatility=0.10,
        topic_tilts={"markets": -0.08, "culture": -0.05, "crypto": -0.08},
    ),
    PersonaTemplate(
        name="Policy Wonk",
        role="Thinks in policy mechanics before price action",
        base_bias=-0.01,
        volatility=0.09,
        topic_tilts={"policy": 0.14, "macro": 0.08},
    ),
    PersonaTemplate(
        name="Crypto Degenerate",
        role="Maps headlines into speculative reflexes",
        base_bias=0.03,
        volatility=0.16,
        topic_tilts={"crypto": 0.18, "markets": 0.04},
    ),
    PersonaTemplate(
        name="Entertainment Fan",
        role="Overreacts to celebrity and culture signals",
        base_bias=0.05,
        volatility=0.11,
        topic_tilts={"culture": 0.18, "company": -0.03},
    ),
]


# ── Helpers ──────────────────────────────────────────────────────────────

def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def infer_topics(text: str) -> list[str]:
    tokens = set(tokenize(text))
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if tokens.intersection(keywords):
            topics.append(topic)
    return topics or ["markets"]


def score_event_text(text: str) -> tuple[float, list[str]]:
    lowered = text.lower()
    tokens = tokenize(lowered)
    score = 0.0
    signals: list[str] = []

    for phrase, weight in POSITIVE_PHRASES.items():
        if phrase in lowered:
            score += weight
            signals.append(f"{phrase} (+)")

    for phrase, weight in NEGATIVE_PHRASES.items():
        if phrase in lowered:
            score += weight
            signals.append(f"{phrase} (-)")

    for token in set(tokens):
        if token in TOKEN_WEIGHTS:
            score += TOKEN_WEIGHTS[token]
            signals.append(token)

    if not signals:
        score = 0.02
        signals.append("neutral headline baseline")

    return clamp(score, -0.35, 0.35), signals


def get_event_score_llm(event: str, api_key: str | None = None) -> float:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key or not event.strip():
        return 0.02

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "max_tokens": 12,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You score market reaction to a news headline. "
                    "Return only one float between -0.3 and 0.3. "
                    "-0.3 means strongly negative for market sentiment, "
                    "0.0 means neutral, 0.3 means strongly positive."
                ),
            },
            {
                "role": "user",
                "content": f"Headline: {event}",
            },
        ],
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        match = re.search(r"-?\d+(?:\.\d+)?", content)
        if not match:
            return 0.02
        return clamp(float(match.group()), -0.3, 0.3)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return 0.02


def get_hybrid_event_score(event: str, api_key: str | None = None) -> tuple[float, float, float, list[str]]:
    rule_score, signals = score_event_text(event)
    llm_score = get_event_score_llm(event, api_key=api_key)
    hybrid_score = clamp((0.6 * llm_score) + (0.4 * rule_score), -0.3, 0.3)
    return hybrid_score, rule_score, llm_score, signals


def select_templates(agent_count: int, rng: random.Random) -> Iterable[PersonaTemplate]:
    full_rounds, remainder = divmod(agent_count, len(PERSONA_TEMPLATES))
    selected = PERSONA_TEMPLATES * full_rounds + PERSONA_TEMPLATES[:remainder]
    if not selected:
        selected = PERSONA_TEMPLATES[:]
    rng.shuffle(selected)
    return selected[:agent_count]


def build_narrative(
    template: PersonaTemplate,
    topics: list[str],
    event_score: float,
    topic_effect: float,
    noise: float,
) -> str:
    tone = "bullish" if event_score >= 0 else "skeptical"
    parts = [f"Starts {tone} from the headline."]

    if topics:
        strongest_topic = max(topics, key=lambda topic: abs(template.topic_tilts.get(topic, 0.0)))
        tilt = template.topic_tilts.get(strongest_topic, 0.0)
        if tilt > 0.04:
            parts.append(f"Leans into the {strongest_topic} angle.")
        elif tilt < -0.04:
            parts.append(f"Distrusts the {strongest_topic} framing.")

    if topic_effect > 0.08:
        parts.append("Persona bias adds conviction.")
    elif topic_effect < -0.08:
        parts.append("Persona bias reduces confidence.")

    if noise > 0.10:
        parts.append("Crowd noise pushes the reaction stronger.")
    elif noise < -0.10:
        parts.append("Randomness tempers the reaction.")

    return " ".join(parts)


def _simulate_from_components(
    *,
    event_text: str,
    event_score: float,
    topics: list[str],
    signals: list[str],
    agent_count: int,
    randomness: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    chosen_templates = list(select_templates(agent_count, rng))

    reactions: list[AgentReaction] = []
    for index, template in enumerate(chosen_templates, start=1):
        topic_effect = sum(template.topic_tilts.get(topic, 0.0) for topic in topics)
        volatility_scale = 0.6 + template.volatility
        noise = rng.uniform(-randomness, randomness) * volatility_scale
        raw_sentiment = event_score + template.base_bias + topic_effect + noise
        sentiment = clamp(raw_sentiment, -1.0, 1.0)
        reactions.append(
            AgentReaction(
                name=f"{template.name} {index}",
                role=template.role,
                sentiment=sentiment,
                narrative=build_narrative(template, topics, event_score, topic_effect, noise),
            )
        )

    aggregate_sentiment = clamp(
        sum(reaction.sentiment for reaction in reactions) / max(len(reactions), 1),
        -0.49,
        0.49,
    )
    model_probability = clamp(0.5 + aggregate_sentiment, 0.0, 1.0)

    return {
        "event_text": event_text,
        "topics": topics,
        "signals": signals,
        "event_score": event_score,
        "hybrid_score": event_score,
        "agents": reactions,
        "aggregate_sentiment": aggregate_sentiment,
        "model_probability": model_probability,
    }


def classify_trade_signal(edge: float, threshold: float = 0.05) -> str:
    if edge >= threshold:
        return "BUY"
    if edge <= -threshold:
        return "SELL"
    return "HOLD"


# ── Legacy single-event simulation (kept for direct use) ────────────────

def generate_agents(
    event_text: str,
    agent_count: int = 8,
    randomness: float = 0.12,
    seed: int = 7,
    openai_api_key: str | None = None,
) -> dict:
    topics = infer_topics(event_text)
    event_score, rule_score, llm_score, signals = get_hybrid_event_score(
        event_text,
        api_key=openai_api_key,
    )
    result = _simulate_from_components(
        event_text=event_text,
        event_score=event_score,
        topics=topics,
        signals=signals,
        agent_count=agent_count,
        randomness=randomness,
        seed=seed,
    )
    result["rule_score"] = rule_score
    result["llm_score"] = llm_score
    return result


# ── Multi-market simulation ─────────────────────────────────────────────

def _composite_event_score(
    news_items: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[float, float, float, list[str], list[str], list[dict[str, Any]]]:
    """Score a market by combining all linked news headlines weighted by edges.

    Returns composite hybrid/rule/llm scores, signals, topics, and linked-news detail.
    """
    if not news_items or not edges:
        return 0.02, 0.02, 0.02, ["no linked news — neutral baseline"], ["markets"], []

    edge_map = {e["source_id"]: e for e in edges}
    total_hybrid_score = 0.0
    total_rule_score = 0.0
    total_llm_score = 0.0
    all_signals: list[str] = []
    topic_set: set[str] = set()
    linked_news: list[dict[str, Any]] = []

    for news in news_items:
        edge = edge_map.get(news["id"])
        if edge is None:
            continue
        hybrid_score, rule_score, llm_score, signals = get_hybrid_event_score(news["headline"])
        weighted_hybrid = hybrid_score * edge["direction"] * edge["strength"]
        weighted_rule = rule_score * edge["direction"] * edge["strength"]
        weighted_llm = llm_score * edge["direction"] * edge["strength"]
        total_hybrid_score += weighted_hybrid
        total_rule_score += weighted_rule
        total_llm_score += weighted_llm
        prefix = news["headline"][:40]
        for sig in signals:
            all_signals.append(f"[{prefix}...] {sig}")
        topic_set.update(infer_topics(news["headline"]))
        linked_news.append(
            {
                "headline": news["headline"],
                "timestamp": news.get("timestamp", ""),
                "direction": edge["direction"],
                "strength": edge["strength"],
                "reason": edge.get("reason", ""),
                "effect": weighted_hybrid,
            }
        )

    if not all_signals:
        return 0.02, 0.02, 0.02, ["no matched signals — neutral baseline"], ["markets"], linked_news

    return (
        clamp(total_hybrid_score, -0.35, 0.35),
        clamp(total_rule_score, -0.35, 0.35),
        clamp(total_llm_score, -0.3, 0.3),
        all_signals,
        sorted(topic_set) or ["markets"],
        linked_news,
    )


def simulate_market(
    market: dict[str, Any],
    news_items: list[dict[str, Any]],
    news_edges: list[dict[str, Any]],
    agent_count: int = 8,
    randomness: float = 0.12,
    seed: int = 7,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Run the full agent simulation for a single market.

    *news_items* should be only the news linked to this market.
    *news_edges* should be only the news→market edges for this market.
    """
    event_score, rule_score, llm_score, signals, topics, linked_news = _composite_event_score(news_items, news_edges)
    result = _simulate_from_components(
        event_text=" | ".join(news["headline"] for news in news_items[:3]) or market["description"],
        event_score=event_score,
        topics=topics,
        signals=signals,
        agent_count=agent_count,
        randomness=randomness,
        seed=seed,
    )
    model_probability = result["model_probability"]
    market_probability = market["market_probability"]
    edge_value = model_probability - market_probability
    signal = classify_trade_signal(edge_value, threshold)

    return {
        **result,
        "market": market,
        "rule_score": rule_score,
        "llm_score": llm_score,
        "linked_news": linked_news,
        "edge": edge_value,
        "signal": signal,
    }


def simulate_all(
    state: dict[str, list],
    agent_count: int = 8,
    randomness: float = 0.12,
    seed: int = 7,
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Run simulate_market for every market in the state, return list of results."""
    news_by_id: dict[str, dict] = {n["id"]: n for n in state["news"]}
    results: list[dict[str, Any]] = []

    for market in state["markets"]:
        news_edges = [
            e for e in state["edges"]
            if e["target_id"] == market["id"] and e["source_type"] == "news"
        ]
        linked_news = [
            news_by_id[e["source_id"]]
            for e in news_edges
            if e["source_id"] in news_by_id
        ]
        result = simulate_market(
            market=market,
            news_items=linked_news,
            news_edges=news_edges,
            agent_count=agent_count,
            randomness=randomness,
            seed=seed,
            threshold=threshold,
        )
        results.append(result)

    return results
