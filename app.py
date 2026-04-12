from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from simulation import simulate_all, simulate_market
from utils import get_state, news_edges_for_market


def load_local_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


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


SIGNAL_COLORS = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#f59e0b"}


load_local_env()
st.set_page_config(page_title="Event Intelligence Terminal", page_icon=":bar_chart:", layout="wide")
report_env_var_status()

st.title("Event Intelligence Terminal")
st.caption("Simulates how linked news affects prediction markets and surfaces BUY / SELL / HOLD signals.")

state = get_state()

if not state["markets"]:
    st.warning("No markets yet. Add some on the **Markets** page.")
    st.stop()

with st.sidebar:
    st.header("Simulation")
    agent_count = st.slider("Number of agents", min_value=5, max_value=20, value=8)
    randomness = st.slider("Agent randomness", min_value=0.0, max_value=0.35, value=0.12, step=0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
    threshold_pct = st.slider("Trade threshold (%)", min_value=1, max_value=20, value=5)
    detail_agent_engine = st.radio(
        "Drill-down agent engine",
        ["Heuristic", "AI agents"],
        help="Overview remains heuristic. AI agents run only for the selected market drill-down to keep cost and latency reasonable.",
    )

    if not os.getenv("NEWSAPI_KEY"):
        st.caption("`NEWSAPI_KEY` not set. Stored news in the app still works.")
    if not os.getenv("OPENAI_API_KEY"):
        st.caption("`OPENAI_API_KEY` not set. Hybrid scoring falls back to `0.02` and AI agents degrade to heuristics.")

    run = st.button("Run all simulations", type="primary", use_container_width=True)

if run or "sim_results" not in st.session_state:
    st.session_state.sim_results = simulate_all(
        state=state,
        agent_count=agent_count,
        randomness=randomness,
        seed=int(seed),
        threshold=threshold_pct / 100,
    )

results: list[dict] = st.session_state.sim_results

rows = []
for result in results:
    market = result["market"]
    rows.append(
        {
            "Market": market["name"],
            "Category": market["category"],
            "Market Prob": f"{market['market_probability']:.0%}",
            "Model Prob": f"{result['model_probability']:.0%}",
            "Edge": f"{result['edge']:+.1%}",
            "Sentiment": f"{result['aggregate_sentiment']:+.3f}",
            "Signal": result["signal"],
            "News Links": len(news_edges_for_market(state, market["id"])),
        }
    )

df = pd.DataFrame(rows)

st.subheader("Market Signals Overview")
st.dataframe(
    df.style.applymap(
        lambda value: f"color: {SIGNAL_COLORS.get(value, 'inherit')}; font-weight: 700",
        subset=["Signal"],
    ),
    use_container_width=True,
    hide_index=True,
    height=min(len(df) * 38 + 40, 600),
)

buy_count = sum(1 for result in results if result["signal"] == "BUY")
sell_count = sum(1 for result in results if result["signal"] == "SELL")
hold_count = sum(1 for result in results if result["signal"] == "HOLD")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Markets", len(results))
col2.metric("BUY", buy_count)
col3.metric("SELL", sell_count)
col4.metric("HOLD", hold_count)

st.divider()
st.subheader("Market Drill-Down")
if detail_agent_engine == "AI agents":
    st.caption("Selected market is being re-simulated with two LLM agent rounds. The overview table above is still heuristic.")

market_names = [market["name"] for market in state["markets"]]
selected_name = st.selectbox("Select a market to inspect", market_names)
selected_result = next((result for result in results if result["market"]["name"] == selected_name), None)

if selected_result is None:
    st.warning("No simulation result for this market.")
    st.stop()

market = selected_result["market"]
if detail_agent_engine == "AI agents":
    news_edges = news_edges_for_market(state, market["id"])
    news_by_id = {news["id"]: news for news in state["news"]}
    linked_news_items = [
        news_by_id[edge["source_id"]]
        for edge in news_edges
        if edge["source_id"] in news_by_id
    ]
    llm_cache_key = (
        market["id"],
        agent_count,
        round(randomness, 4),
        int(seed),
        threshold_pct,
        bool(os.getenv("OPENAI_API_KEY")),
        tuple(
            (
                edge["source_id"],
                edge["direction"],
                edge["strength"],
                edge.get("reason", ""),
            )
            for edge in news_edges
        ),
        tuple(news["headline"] for news in linked_news_items),
    )
    if st.session_state.get("llm_detail_key") != llm_cache_key:
        with st.spinner("Running AI-agent drill-down..."):
            st.session_state.llm_detail_result = simulate_market(
                market=market,
                news_items=linked_news_items,
                news_edges=news_edges,
                agent_count=agent_count,
                randomness=randomness,
                seed=int(seed),
                threshold=threshold_pct / 100,
                mode="llm_agents",
            )
            st.session_state.llm_detail_key = llm_cache_key
    selected_result = st.session_state.llm_detail_result

st.markdown(f"**{market['name']}** — _{market['description']}_")

llm_error = selected_result.get("llm_error")
if detail_agent_engine == "AI agents" and llm_error:
    st.warning(f"AI agent issue: {llm_error}")
    if st.session_state.get("_last_llm_error_reported") != llm_error:
        print(f"[llm_agents] {llm_error}")
        st.session_state["_last_llm_error_reported"] = llm_error

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sentiment", f"{selected_result['aggregate_sentiment']:+.3f}")
m2.metric("Model Prob", f"{selected_result['model_probability']:.1%}")
m3.metric("Market Prob", f"{market['market_probability']:.1%}")
m4.metric("Edge", f"{selected_result['edge']:+.1%}")

signal = selected_result["signal"]
color = SIGNAL_COLORS[signal]
st.markdown(
    f'<div style="padding:0.75rem;border-radius:0.5rem;background:{color};'
    f'color:white;font-weight:700;text-align:center;font-size:1.1rem;">'
    f'Signal: {signal}</div>',
    unsafe_allow_html=True,
)

linked_news = selected_result.get("linked_news", [])
if linked_news:
    st.write("**Linked news driving this market**")
    news_rows = [
        {
            "Headline": item["headline"],
            "Date": item.get("timestamp", ""),
            "Direction": "+" if item["direction"] > 0 else "-",
            "Strength": item["strength"],
            "Effect": round(item["effect"], 3),
            "Reason": item.get("reason", ""),
        }
        for item in linked_news
    ]
    st.dataframe(pd.DataFrame(news_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No linked news edges for this market. Falling back to the market description.")

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Simulation Summary")
    st.write(f"**Agent engine:** {selected_result.get('agent_backend', 'heuristic')}")
    st.write(f"**Detected topics:** {', '.join(selected_result['topics'])}")
    st.write(f"**Rule score:** {selected_result['rule_score']:+.2f}")
    st.write(f"**LLM score:** {selected_result['llm_score']:+.2f}")
    st.write(f"**LLM + Rule Hybrid Score:** {selected_result['hybrid_score']:+.2f}")
    st.write(f"**Matched signals:** {', '.join(selected_result['signals'])}")
    st.code(
        "linked_headlines -> hybrid_score -> agent reactions -> model_probability -> edge -> trade signal",
        language="text",
    )

with right_col:
    st.subheader("Probability Framework")
    st.write("The dashboard combines linked headlines into a hybrid event score, then simulates market personas.")
    st.code(
        "headline_score = 0.6 * llm_score + 0.4 * rule_score\n"
        "market_event_score = sum(headline_score * edge.direction * edge.strength)\n"
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
        "Confidence": round(agent.confidence, 2) if agent.confidence is not None else None,
        "Source": agent.source,
        "Narrative": agent.narrative,
    }
    for agent in selected_result["agents"]
]
agents_df = pd.DataFrame(agent_rows).sort_values("Sentiment", ascending=False)

chart_col, table_col = st.columns([1, 1.4])
with chart_col:
    st.write("**Agent Reactions**")
    st.bar_chart(agents_df.set_index("Agent")[["Sentiment"]], height=360)
with table_col:
    st.write("**Agent Detail**")
    st.dataframe(agents_df, use_container_width=True, hide_index=True)
