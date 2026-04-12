from __future__ import annotations

import pandas as pd
import streamlit as st

from simulation import simulate_all, simulate_market, classify_trade_signal
from utils import get_state, news_edges_for_market

st.set_page_config(page_title="Event Intelligence Terminal", page_icon=":bar_chart:", layout="wide")

st.title("Event Intelligence Terminal")
st.caption("Simulates how diverse market personas react to news, surfaces BUY / SELL / HOLD signals per market.")

state = get_state()

if not state["markets"]:
    st.warning("No markets yet. Add some on the **Markets** page.")
    st.stop()

# ── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Simulation Settings")
    agent_count = st.slider("Number of agents", min_value=5, max_value=20, value=8)
    randomness = st.slider("Agent randomness", min_value=0.0, max_value=0.35, value=0.12, step=0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
    threshold_pct = st.slider("Trade threshold (%)", min_value=1, max_value=20, value=5)
    run = st.button("Run all simulations", type="primary", use_container_width=True)

# ── Run simulation across all markets ────────────────────────────────────

if not run and "sim_results" not in st.session_state:
    st.info("Configure settings in the sidebar and press **Run all simulations**.")
    st.stop()

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

    st.write(f"**Detected topics:** {', '.join(selected_result['topics'])}")

    linked_news_edges = news_edges_for_market(state, m["id"])
    if linked_news_edges:
        news_by_id = {n["id"]: n for n in state["news"]}
        st.write("**Linked news:**")
        for e in linked_news_edges:
            n = news_by_id.get(e["source_id"])
            if n:
                direction = "+" if e["direction"] > 0 else "-"
                st.write(f"- {n['headline']} ({direction}, strength {e['strength']:.1f})")
    else:
        st.write("_No news linked to this market._")

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
