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
| `app.py` | Streamlit dashboard — runs `simulate_all()` across all markets, shows signal overview table with colored BUY/SELL/HOLD, metric summary counts, drill-down per market with linked news, agent bar chart + detail table | Active |
| `simulation.py` | Core engine — persona templates, keyword/phrase scoring (`score_event_text`), topic inference (`infer_topics`), composite multi-news scoring (`_composite_event_score`), per-market simulation (`simulate_market`), batch simulation (`simulate_all`), legacy single-event simulation (`generate_agents`), trade signal classification (`classify_trade_signal`) | Active |
| `utils.py` | Persistence layer — `load_state`/`save_state`/`get_state`/`persist` for JSON I/O, `slugify` for ID generation, lookup helpers (`find_market`, `find_news`, `edges_targeting`, `edges_from`, `news_edges_for_market`, `market_edges_for_market`), CRUD helpers (`add_market`, `remove_market`, `add_news`, `remove_news`, `add_edge`, `remove_edge`), shared `CATEGORIES` list | Active |
| `data/state.json` | Persistent storage — 14 seed markets, 5 seed news events, 37 dependency edges (30 market→market from original `PRESET_EVENTS`, 7 news→market) | Active |
| `pages/1_Markets.py` | Streamlit page — expandable form to add markets (name, description, category, probability), auto-generates market→market edges via AI on add, dataframe listing all markets, expandable section to remove a market (cascades to delete its edges) | Active |
| `pages/2_News.py` | Streamlit page — expandable form to add news (headline, category, date), auto-generates news→market edges via AI on add, dataframe listing all news, expandable section to remove a news event (cascades to delete its edges) | Active |
| `pages/3_Dependencies.py` | Streamlit page — interactive directed graph via `streamlit-agraph`, AI bulk regeneration buttons for news→market and market→market edges (GPT-4o-mini), manual form to add/remove edges | Active |
| `edge_analysis.py` | AI edge generation — `generate_news_edges` (news→market), `generate_market_edges` (market→market), `generate_all_news_edges`, `generate_all_market_edges`; calls GPT-4o-mini with JSON structured output, validates target IDs, clamps strength/direction | Active |
| `requirements.txt` | Python deps: `streamlit >=1.44,<2`, `pandas >=2.2,<3`, `streamlit-agraph >=0.0.45`, `openai >=1.0,<2`, `python-dotenv >=1.0,<2` | Active |

## Data Model

Three entity types persisted in `data/state.json`:

```
Market
  id                  str         slug, max 64 chars
  name                str         human-readable market name
  description         str         full contract description
  category            str         macro | markets | crypto | culture | policy | company
  market_probability  float       0–1, the current market-implied probability

News Event
  id                  str         slug, max 64 chars
  headline            str         the real-world headline text
  category            str         same category set as markets
  timestamp           str         ISO date (YYYY-MM-DD)

Dependency Edge
  source_id           str         news ID or market ID
  source_type         str         "news" | "market"
  target_id           str         always a market ID
  direction           int         +1 (increases target probability) or −1 (decreases)
  strength            float       0.0–1.0, how strong the influence is
  reason              str         human-readable explanation
```

## Key Concepts

### Persona Templates (`simulation.py`)
Twenty archetypes: Optimist, Pessimist, Macro Trader, Economist, Retail Investor, Momentum Chaser, Contrarian, Policy Wonk, Crypto Degenerate, Entertainment Fan, Value Investor, Day Trader, Hedge Fund Manager, Social Media Influencer, Academic Researcher, Risk Manager, Speculator, Financial Analyst, Activist Investor, Global Investor. Each has:
- **base_bias** — intrinsic bullish/bearish lean (e.g. Optimist +0.12, Pessimist −0.12)
- **volatility** — scales the noise term (0.06–0.18)
- **topic_tilts** — dict mapping topic → sentiment modifier (e.g. Crypto Degenerate gets +0.18 on "crypto")

Agents are selected round-robin from templates, shuffled per seed, up to the configured `agent_count` (5–100).

### Event Scoring (`score_event_text`)
Two-tier keyword lookup on headline text:
1. **Phrase match** — `POSITIVE_PHRASES` (10 phrases, e.g. "rate cut" → +0.18) and `NEGATIVE_PHRASES` (10 phrases, e.g. "misses earnings" → −0.25)
2. **Token match** — `TOKEN_WEIGHTS` (30 tokens, e.g. "bullish" → +0.14, "recession" → −0.20)

Falls back to +0.02 "neutral headline baseline" when nothing matches. Output clamped to [−0.35, 0.35].

### Topic Inference (`infer_topics`)
Six topic buckets (`TOPIC_KEYWORDS`): macro, markets, crypto, culture, policy, company. Determined by token overlap with curated keyword sets. Falls back to `["markets"]` when nothing matches.

### Composite Scoring (`_composite_event_score`)
For a given market, aggregates all linked news headlines weighted by their edge:
```
composite_score = Σ (score_event_text(headline) × edge.direction × edge.strength)
```
Clamped to [−0.35, 0.35]. Topics are the union of all linked headlines' inferred topics.

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

### Dependency Graph Visualization
Interactive directed graph rendered by `streamlit-agraph` on the Dependencies page. Category colors:
- macro = blue (#3b82f6), markets = indigo (#6366f1), crypto = amber (#f59e0b)
- culture = pink (#ec4899), policy = emerald (#10b981), company = violet (#8b5cf6)

Market nodes are circles (size 25), news nodes are squares (size 18). Edge color: green (#22c55e) for positive, red (#ef4444) for negative. Edge width = `1 + strength × 3`.

### Persistence
All state lives in `data/state.json`, loaded into `st.session_state["app_state"]` on first access via `get_state()`. Mutations go through CRUD helpers in `utils.py` and are flushed to disk via `persist()`. Removing a market or news event cascades to delete all associated edges.

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Pages
- **Dashboard** (`app.py`) — configure agent count / randomness / seed / threshold in sidebar, run simulations, view signal table, drill into individual markets
- **Markets** (`pages/1_Markets.py`) — add/remove prediction markets with name, description, category, probability
- **News** (`pages/2_News.py`) — add/remove news events with headline, category, date
- **Dependencies** (`pages/3_Dependencies.py`) — interactive graph visualization, add/remove edges between news↔markets and markets↔markets

### Seed Data
Ships with 14 prediction markets across 6 categories (macro, markets, crypto, company, policy, culture), 5 sample news events, and 37 dependency edges (30 market→market, 7 news→market). Reset by restoring `data/state.json` from version control.

## Planned Extensions

- **LLM-powered agent reactions** — replace keyword scoring with LLM calls so agents produce richer, context-aware sentiment (use `model.py`)
- **Live market feed** — pull real-time Kalshi prices instead of mocked market probability
- **Agent memory & network effects** — let agents remember past events and influence each other across rounds
- **Event chains** — model cascading events via the dependency graph (e.g. rate cut → equity rally → crypto surge)
- **News ingestion** — auto-fetch headlines from RSS / APIs and run the simulation continuously
- **GPU-accelerated model** — when training ML models for probability estimation, use GPU
- **Market→market simulation propagation** — use market→market edges to propagate probability shifts across the graph

## Conventions

- Python 3.12+, type hints everywhere, `from __future__ import annotations`
- Frozen dataclasses for immutable config (`PersonaTemplate`), mutable dataclasses for runtime state (`AgentReaction`)
- All randomness flows through a seeded `random.Random` instance for reproducibility
- Probabilities are floats in [0, 1]; the UI converts to/from percentages at the boundary
- Data persisted as JSON in `data/state.json`, loaded into `st.session_state` at startup
- IDs are slugified from names (lowercase, alphanumeric + hyphens, max 64 chars)
- Removing an entity cascades to remove all edges referencing it
- No external API calls yet — everything is self-contained and deterministic given a seed
