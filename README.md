# Event Intelligence Terminal

Prototype "Bloomberg Terminal for prediction markets" built with Streamlit.

The app takes a real-world event, simulates how a small set of personas react to it, converts the resulting sentiment into a model probability, compares that against a market probability, and surfaces a simple BUY/SELL signal.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
set -a
source .env
set +a
streamlit run app.py
```

## What is included

- `app.py`: Streamlit UI for the prototype
- `simulation.py`: MiroFish-lite simulation engine
- `requirements.txt`: Python dependencies

## Current MVP behavior

- Accepts a custom event headline or uses a preset example
- Simulates 5-20 agents with persona bias and randomness
- Infers a simple headline score from keywords and themes
- Aggregates agent sentiment into a model probability
- Compares model probability vs market probability
- Displays edge and BUY/SELL/HOLD signal

## Notes

- Market probability is currently user-provided or mocked via presets
- Live Kalshi markets can be imported from the Markets page for real-time market probability data
- Event understanding is intentionally lightweight and heuristic-based
- This is meant to be a hackathon starter, not a production pricing model
- Real API keys should go in `.env`, which is ignored by git
