# AGENTS.md — Event Intelligence Terminal

## Vision

Bloomberg Terminal for Kalshi. Maintain a portfolio of prediction markets and a feed of real-world news. AI agent personas simulate how diverse market participants react to news, producing per-market BUY / SELL / HOLD signals before the market fully prices the information.

## Architecture

```
Markets (persistent list)          News events (persistent list)
      │                                   │
      │          Dependency edges          │
      ◄──────── (news → market,  ─────────►
      │           market → market)         │
      ▼                                    ▼
┌──────────────────────────────────────────────┐
│  simulate_market()                           │
│  For each market:                            │
│    1. Gather linked news via edges           │
│    2. Composite score = Σ(headline_score     │
│       × edge.direction × edge.strength)      │
│    3. N agent personas react (sentiment)     │
│    4. Aggregate → model_probability          │
│    5. edge = model_prob − market_prob        │
│    6. Signal: BUY / SELL / HOLD              │
└──────────────────────────────────────────────┘
      │
      ▼
Dashboard table with per-market signals
```

## File Map

| File | Purpose | Status |
|---|---|---|
| `app.py` | Streamlit dashboard — runs simulations across all markets, shows signal table, drill-down per market with agent chart + table | Active |
| `simulation.py` | Core engine — persona templates, keyword/phrase scoring, topic inference, `simulate_market()`, `simulate_all()`, trade signal classification | Active |
| `utils.py` | Persistence layer — JSON load/save, session state management, CRUD helpers for markets/news/edges, lookup utilities | Active |
| `data/state.json` | Persistent storage for markets, news events, and dependency edges | Active |
| `pages/1_Markets.py` | Streamlit page — add/view/remove prediction markets | Active |
| `pages/2_News.py` | Streamlit page — add/view/remove news events | Active |
| `pages/3_Dependencies.py` | Streamlit page — interactive dependency graph (streamlit-agraph), add/remove edges between news↔markets and markets↔markets | Active |
| `model.py` | Placeholder for ML / LLM probability model | Empty — extend here |
| `requirements.txt` | Python deps: `streamlit`, `pandas`, `streamlit-agraph` | Active |

## Key Concepts

### Data Model

Three entity types persisted in `data/state.json`:

- **Market** — a tradeable prediction contract with `id`, `name`, `description`, `category`, `market_probability`
- **News Event** — a real-world headline with `id`, `headline`, `category`, `timestamp`
- **Dependency Edge** — a directed relationship: `source_id` → `target_id` with `source_type` (news|market), `direction` (+1|-1), `strength` (0–1), `reason`

### Persona Templates (`simulation.py`)
Ten archetypes (Optimist, Pessimist, Macro Trader, Economist, Retail Investor, Momentum Chaser, Contrarian, Policy Wonk, Crypto Degenerate, Entertainment Fan). Each has:
- **base_bias** — intrinsic bullish/bearish lean
- **volatility** — scales the noise term
- **topic_tilts** — dict mapping topic → sentiment modifier

### Event Scoring
Two-tier keyword lookup:
1. **Phrase match** (`POSITIVE_PHRASES` / `NEGATIVE_PHRASES`) — multi-word patterns
2. **Token match** (`TOKEN_WEIGHTS`) — single tokens

Output is clamped to [-0.35, 0.35].

### Composite Scoring (Multi-News)
For a given market, the composite event score aggregates all linked news:
```
composite_score = Σ (score_event_text(headline) × edge.direction × edge.strength)
```
Clamped to [-0.35, 0.35].

### Agent Sentiment Calculation
```
topic_effect = Σ template.topic_tilts[topic] for each detected topic
noise = uniform(-randomness, randomness) × (0.6 + template.volatility)
sentiment = clamp(event_score + base_bias + topic_effect + noise, -1, 1)
```

### Probability & Signal
```
aggregate_sentiment = clamp(mean(agent sentiments), -0.49, 0.49)
model_probability = clamp(0.5 + aggregate_sentiment, 0, 1)
edge = model_probability − market_probability
signal = BUY if edge ≥ threshold else SELL if edge ≤ −threshold else HOLD
```

### Dependency Graph
Interactive visualization using `streamlit-agraph`:
- Market nodes = circles, colored by category
- News nodes = squares, colored by category
- Green edges = positive influence, red = negative
- Edge width proportional to strength

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pages:
- **Dashboard** (app.py) — run simulations, view signal table, drill into individual markets
- **Markets** — add/remove prediction markets
- **News** — add/remove news events
- **Dependencies** — visualize and manage the dependency graph

## Planned Extensions

- **LLM-powered agent reactions** — replace keyword scoring with LLM calls so agents produce richer, context-aware sentiment (use `model.py`)
- **Live market feed** — pull real-time Kalshi prices instead of mocked market probability
- **Agent memory & network effects** — let agents remember past events and influence each other across rounds
- **Event chains** — model cascading events (e.g. rate cut → equity rally → crypto surge)
- **News ingestion** — auto-fetch headlines from RSS / APIs and run the simulation continuously
- **GPU-accelerated model** — when training ML models for probability estimation, use GPU

## Conventions

- Python 3.12+, type hints everywhere, `from __future__ import annotations`
- Frozen dataclasses for immutable config (`PersonaTemplate`), mutable dataclasses for runtime state (`AgentReaction`)
- All randomness flows through a seeded `random.Random` instance for reproducibility
- Probabilities are floats in [0, 1]; the UI converts to/from percentages at the boundary
- Data persisted as JSON in `data/state.json`, loaded into `st.session_state` at startup
- No external API calls yet — everything is self-contained and deterministic given a seed
