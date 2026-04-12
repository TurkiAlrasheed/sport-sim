from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st

from data import fetch_news
from simulation import classify_trade_signal, generate_agents, simulate_all
from utils import get_state, news_edges_for_market


PRESET_EVENTS = {
    "Custom": {
        "event": "Fed cuts rates by 25 bps at the next meeting",
        "market_probability": 44,
    },
    "Fed Cuts Rates": {
        "event": "Fed cuts rates by 25 bps at the next meeting",
        "market_probability": 44,
    },
    "Drake Drops Album": {
        "event": "Drake drops album this month and it goes viral",
        "market_probability": 58,
    },
    "Bitcoin ETF Approval": {
        "event": "Spot Bitcoin ETF approval drives crypto surge",
        "market_probability": 63,
    },
    "Company Misses Earnings": {
        "event": "Major tech company misses earnings and cuts guidance",
        "market_probability": 71,
    },
}


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


@st.cache_data(ttl=900)
def load_news() -> list[str]:
    return fetch_news()


def report_env_var_status() -> None:
    statuses = {
        "NEWSAPI_KEY": bool(os.getenv("NEWSAPI_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
    }
    last_reported = st.session_state.get("_reported_env_status")
    if last_reported == statuses:
        return

    for key, is_present in statuses.items():
        if is_present:
            print(f"[env] {key}: detected, using live service.")
        else:
            print(f"[env] {key}: missing, using fallback behavior.")

    st.session_state["_reported_env_status"] = statuses


st.set_page_config(page_title="Event Intelligence Terminal", page_icon=":bar_chart:", layout="wide")
report_env_var_status()

st.title("Event Intelligence Terminal")
st.caption("Simulates how diverse market personas react to news, surfaces BUY / SELL / HOLD signals per market.")



state = get_state()

if not state["markets"]:
    st.warning("No markets yet. Add some on the **Markets** page.")
    st.stop()

# ── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Scenario")
    input_mode = st.radio("Event source", ["Latest news", "Preset", "Manual"], index=0)

    selected_headline = ""
    default_market_probability = 44
    if input_mode == "Latest news":
        news_headlines = load_news()
        selected_headline = st.selectbox("Latest headlines", news_headlines)
        event_text = selected_headline
    elif input_mode == "Preset":
        preset_name = st.selectbox("Preset", list(PRESET_EVENTS.keys()))
        preset = PRESET_EVENTS[preset_name]
        default_market_probability = preset["market_probability"]
        event_text = st.text_area(
            "Event input",
            value=preset["event"],
            height=110,
            help="Describe the real-world event you want the agents to react to.",
        )
    else:
        event_text = st.text_area(
            "Event input",
            value="",
            height=110,
            help="Describe the real-world event you want the agents to react to.",
        )

    market_probability_pct = st.slider(
        "Market probability (%)",
        min_value=0,
        max_value=100,
        value=default_market_probability,
        help="Mocked market-implied probability for the event.",
    )
    agent_count = st.slider("Number of agents", min_value=5, max_value=20, value=8)
    randomness = st.slider("Agent randomness", min_value=0.0, max_value=0.35, value=0.12, step=0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
    threshold_pct = st.slider("Trade threshold (%)", min_value=1, max_value=20, value=5)
    run = st.button("Run all simulations", type="primary", use_container_width=True)

    if not os.getenv("NEWSAPI_KEY"):
        st.caption("`NEWSAPI_KEY` not set. Using fallback headlines.")
    if not os.getenv("OPENAI_API_KEY"):
        st.caption("`OPENAI_API_KEY` not set. LLM score falls back to `0.02`.")

    run_simulation = st.button("Run simulation", type="primary", use_container_width=True)

if not run and "sim_results" not in st.session_state:
    st.info("Configure settings in the sidebar and press **Run all simulations**.")
    st.stop()

result = generate_agents(
    event_text=event_text,
    agent_count=agent_count,
    randomness=randomness,
    seed=int(seed),
)

market_probability = market_probability_pct / 100
edge = result["model_probability"] - market_probability
signal = classify_trade_signal(edge=edge, threshold=signal_threshold_pct / 100)

signal_color = {
    "BUY": "green",
    "SELL": "red",
    "HOLD": "orange",
}[signal]

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Global Sentiment", f"{result['aggregate_sentiment']:+.2f}")
metric_2.metric("LLM + Rule Hybrid Score", f"{result['hybrid_score']:+.2f}")
metric_3.metric("Model Probability", format_percent(result["model_probability"]))
metric_4.metric("Edge", f"{edge * 100:+.1f}%")
if run:
    st.session_state.sim_results = simulate_all(
        state=state,
        agent_count=agent_count,
        randomness=randomness,
        seed=int(seed),
        threshold=threshold_pct / 100,
    )

results: list[dict] = st.session_state.sim_results

# ── Dashboard table ──────────────────────────────────────────────────────

SIGNAL_COLORS = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#f59e0b"}

rows = []
for r in results:
    m = r["market"]
    rows.append({
        "Market": m["name"],
        "Category": m["category"],
        "Market Prob": f"{m['market_probability']:.0%}",
        "Model Prob": f"{r['model_probability']:.0%}",
        "Edge": f"{r['edge']:+.1%}",
        "Sentiment": f"{r['aggregate_sentiment']:+.3f}",
        "Signal": r["signal"],
        "News Links": len(news_edges_for_market(state, m["id"])),
    })

df = pd.DataFrame(rows)

st.subheader("Market Signals Overview")
st.dataframe(
    df.style.applymap(
        lambda v: f"color: {SIGNAL_COLORS.get(v, 'inherit')}; font-weight: 700",
        subset=["Signal"],
    ),
    use_container_width=True,
    hide_index=True,
    height=min(len(df) * 38 + 40, 600),
)

# ── Signal summary ───────────────────────────────────────────────────────

buy_count = sum(1 for r in results if r["signal"] == "BUY")
sell_count = sum(1 for r in results if r["signal"] == "SELL")
hold_count = sum(1 for r in results if r["signal"] == "HOLD")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Markets", len(results))
col2.metric("BUY", buy_count)
col3.metric("SELL", sell_count)
col4.metric("HOLD", hold_count)

# ── Drill-down into a single market ──────────────────────────────────────

st.divider()
st.subheader("Market Drill-Down")

market_names = [m["name"] for m in state["markets"]]
selected_name = st.selectbox("Select a market to inspect", market_names)

if selected_name:
    selected_result = next((r for r in results if r["market"]["name"] == selected_name), None)
    if selected_result is None:
        st.warning("No simulation result for this market.")
        st.stop()

    m = selected_result["market"]
    st.markdown(f"**{m['name']}** — _{m['description']}_")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentiment", f"{selected_result['aggregate_sentiment']:+.3f}")
    m2.metric("Model Prob", f"{selected_result['model_probability']:.1%}")
    m3.metric("Market Prob", f"{m['market_probability']:.1%}")
    m4.metric("Edge", f"{selected_result['edge']:+.1%}")

    signal = selected_result["signal"]
    color = SIGNAL_COLORS[signal]
    st.markdown(
        f'<div style="padding:0.75rem;border-radius:0.5rem;background:{color};'
        f'color:white;font-weight:700;text-align:center;font-size:1.1rem;">'
        f'Signal: {signal}</div>',
        unsafe_allow_html=True,
    )

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Simulation Summary")
    if selected_headline:
        st.write(f"**Selected headline:** {selected_headline}")
    st.write(f"**Detected topics:** {', '.join(result['topics'])}")
    st.write(f"**Rule score:** {result['rule_score']:+.2f}")
    st.write(f"**LLM score:** {result['llm_score']:+.2f}")
    st.write(f"**LLM + Rule Hybrid Score:** {result['hybrid_score']:+.2f}")
    st.write(f"**Matched signals:** {', '.join(result['signals'])}")
    st.code(
        "Event -> Agent Simulation -> Sentiment -> Probability -> Market Comparison -> Trade Signal",
        language="text",
    )

with right_col:
    st.subheader("Probability Framework")
    st.write("The prototype maps hybrid event understanding into agent reactions and a model probability.")
    st.code(
        "hybrid_score = 0.6 * llm_score + 0.4 * rule_score\n"
        "aggregate_sentiment = average(agent_sentiments)\n"
        "model_probability = clamp(0.5 + aggregate_sentiment, 0, 1)\n"
        "edge = model_probability - market_probability",
        language="python",
    )

    agent_rows = [
        {
            "Agent": a.name,
            "Role": a.role,
            "Sentiment": round(a.sentiment, 3),
            "Narrative": a.narrative,
        }
        for a in selected_result["agents"]
    ]
    agents_df = pd.DataFrame(agent_rows).sort_values("Sentiment", ascending=False)

    chart_col, table_col = st.columns([1, 1.4])
    with chart_col:
        st.write("**Agent Reactions**")
        st.bar_chart(agents_df.set_index("Agent")[["Sentiment"]], height=360)
    with table_col:
        st.write("**Agent Detail**")
        st.dataframe(agents_df, use_container_width=True, hide_index=True)
