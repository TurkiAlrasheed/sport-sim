from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from data import fetch_news
from simulation import classify_trade_signal, generate_agents


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
st.caption("Prototype terminal for simulating how people react to events before markets fully price them.")

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
    signal_threshold_pct = st.slider("Trade threshold (%)", min_value=1, max_value=20, value=5)

    if not os.getenv("NEWSAPI_KEY"):
        st.caption("`NEWSAPI_KEY` not set. Using fallback headlines.")
    if not os.getenv("OPENAI_API_KEY"):
        st.caption("`OPENAI_API_KEY` not set. LLM score falls back to `0.02`.")

    run_simulation = st.button("Run simulation", type="primary", use_container_width=True)

if not run_simulation:
    st.info("Choose a scenario in the sidebar and run the simulation.")
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

st.markdown(
    f"""
    <div style="padding: 1rem; border-radius: 0.75rem; background-color: {signal_color}; color: white; font-weight: 600;">
        Signal: {signal}
    </div>
    """,
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
        "Agent": agent.name,
        "Role": agent.role,
        "Sentiment": round(agent.sentiment, 3),
        "Narrative": agent.narrative,
    }
    for agent in result["agents"]
]
agents_df = pd.DataFrame(agent_rows).sort_values("Sentiment", ascending=False)

chart_df = agents_df.set_index("Agent")[["Sentiment"]]

chart_col, table_col = st.columns([1, 1.4])

with chart_col:
    st.subheader("Agent Reactions")
    st.bar_chart(chart_df, height=360)

with table_col:
    st.subheader("Agent Detail")
    st.dataframe(agents_df, use_container_width=True, hide_index=True)

with st.expander("How to extend this starter"):
    st.write("Replace keyword scoring with LLM-generated reactions or live news ingestion.")
    st.write("Swap mocked market probability for Kalshi or another market feed.")
    st.write("Add agent memory, network effects, and event chains for richer dynamics.")
