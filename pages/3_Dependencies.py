from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from streamlit_agraph import agraph, Config, Edge, Node

from edge_analysis import generate_market_edges, generate_news_edges
from utils import (
    add_edge,
    find_market,
    find_news,
    get_state,
    persist,
    remove_edge,
)

st.set_page_config(page_title="Dependency Graph", page_icon=":link:", layout="wide")

st.title("Dependency Graph")
st.caption("Visualize and manage how news and markets influence each other.")

state = get_state()

# ── Category colors ──────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "macro": "#3b82f6",
    "markets": "#6366f1",
    "crypto": "#f59e0b",
    "culture": "#ec4899",
    "policy": "#10b981",
    "company": "#8b5cf6",
}

# ── Build graph nodes and edges ──────────────────────────────────────────

nodes: list[Node] = []
edges_vis: list[Edge] = []

for m in state["markets"]:
    nodes.append(Node(
        id=m["id"],
        label=m["name"][:30],
        title=m["description"],
        color=CATEGORY_COLORS.get(m["category"], "#94a3b8"),
        size=25,
        shape="dot",
    ))

for n in state["news"]:
    nodes.append(Node(
        id=n["id"],
        label=n["headline"][:30],
        title=n["headline"],
        color=CATEGORY_COLORS.get(n["category"], "#94a3b8"),
        size=18,
        shape="square",
    ))

for e in state["edges"]:
    edge_color = "#22c55e" if e["direction"] > 0 else "#ef4444"
    edges_vis.append(Edge(
        source=e["source_id"],
        target=e["target_id"],
        label=f"{'+' if e['direction'] > 0 else '-'}{e['strength']:.1f}",
        color=edge_color,
        width=1 + e["strength"] * 3,
        title=e.get("reason", ""),
    ))

config = Config(
    width="100%",
    height=600,
    directed=True,
    physics=True,
    hierarchical=False,
    nodeHighlightBehavior=True,
    highlightColor="#f1fa8c",
    collapsible=False,
)

if nodes:
    st.subheader("Interactive Graph")
    st.write("Circles = markets, squares = news. Green edges = positive influence, red = negative.")
    agraph(nodes=nodes, edges=edges_vis, config=config)
else:
    st.info("Add some markets and news first to see the graph.")

# ── AI edge generation ────────────────────────────────────────────────────

st.divider()
st.subheader("AI Edge Generation")

has_key = bool(os.environ.get("OPENAI_API_KEY"))
has_markets = bool(state["markets"])

if not has_key:
    st.error("Set `OPENAI_API_KEY` in your `.env` file to enable AI edge generation.")
elif not has_markets:
    st.info("Add at least one market to use AI generation.")
else:
    current_news_edges = [e for e in state["edges"] if e["source_type"] == "news"]
    current_market_edges = [e for e in state["edges"] if e["source_type"] == "market"]

    col_news_btn, col_mkt_btn = st.columns(2)

    # ── News → Market ────────────────────────────────────────────────
    with col_news_btn:
        news_count = len(state["news"])
        st.markdown(
            f"**News → Market** edges  \n"
            f"{news_count} headline(s), {len(current_news_edges)} existing edge(s)"
        )
        news_disabled = news_count == 0
        regen_news = st.button(
            "Regenerate news edges",
            type="primary",
            use_container_width=True,
            disabled=news_disabled,
            help="No news events to analyze" if news_disabled else None,
        )

    # ── Market → Market ──────────────────────────────────────────────
    with col_mkt_btn:
        market_count = len(state["markets"])
        st.markdown(
            f"**Market → Market** edges  \n"
            f"{market_count} market(s), {len(current_market_edges)} existing edge(s)"
        )
        mkt_disabled = market_count < 2
        regen_markets = st.button(
            "Regenerate market edges",
            type="primary",
            use_container_width=True,
            disabled=mkt_disabled,
            help="Need at least 2 markets" if mkt_disabled else None,
        )

    if regen_news:
        state["edges"] = [e for e in state["edges"] if e["source_type"] != "news"]
        progress = st.progress(0, text="Generating news edges...")
        total = 0
        for i, news in enumerate(state["news"]):
            progress.progress(
                (i + 1) / news_count,
                text=f"Analyzing: {news['headline'][:60]}...",
            )
            try:
                edges = generate_news_edges(news, state["markets"])
                for edge in edges:
                    existing = any(
                        e["source_id"] == edge["source_id"]
                        and e["target_id"] == edge["target_id"]
                        for e in state["edges"]
                    )
                    if not existing:
                        add_edge(state, edge)
                        total += 1
            except Exception as exc:
                st.warning(f"Failed for \"{news['headline'][:40]}...\": {exc}")
        persist()
        progress.empty()
        st.success(f"Generated **{total}** news→market edge(s).")
        st.rerun()

    if regen_markets:
        state["edges"] = [e for e in state["edges"] if e["source_type"] != "market"]
        progress = st.progress(0, text="Generating market edges...")
        total = 0
        seen: set[tuple[str, str]] = set()
        for i, market in enumerate(state["markets"]):
            progress.progress(
                (i + 1) / market_count,
                text=f"Analyzing: {market['name'][:60]}...",
            )
            try:
                edges = generate_market_edges(market, state["markets"])
                for edge in edges:
                    pair = (edge["source_id"], edge["target_id"])
                    if pair in seen:
                        continue
                    seen.add(pair)
                    existing = any(
                        e["source_id"] == edge["source_id"]
                        and e["target_id"] == edge["target_id"]
                        for e in state["edges"]
                    )
                    if not existing:
                        add_edge(state, edge)
                        total += 1
            except Exception as exc:
                st.warning(f"Failed for \"{market['name'][:40]}...\": {exc}")
        persist()
        progress.empty()
        st.success(f"Generated **{total}** market→market edge(s).")
        st.rerun()

# ── Add edge form ────────────────────────────────────────────────────────

st.divider()
st.subheader("Add Dependency Edge")

all_sources: dict[str, tuple[str, str]] = {}
for m in state["markets"]:
    all_sources[f"[Market] {m['name']}"] = (m["id"], "market")
for n in state["news"]:
    all_sources[f"[News] {n['headline']}"] = (n["id"], "news")

target_options: dict[str, str] = {m["name"]: m["id"] for m in state["markets"]}

if all_sources and target_options:
    with st.form("add_edge", clear_on_submit=True):
        col_src, col_tgt = st.columns(2)
        source_label = col_src.selectbox("Source (news or market)", list(all_sources.keys()))
        target_label = col_tgt.selectbox("Target market", list(target_options.keys()))

        col_dir, col_str = st.columns(2)
        direction = col_dir.radio("Direction", ["+1 (increases)", "-1 (decreases)"], horizontal=True)
        strength = col_str.slider("Strength", 0.1, 1.0, 0.5, 0.1)
        reason = st.text_input("Reason", placeholder="Why does this source affect the target?")
        submitted = st.form_submit_button("Add edge", use_container_width=True)

    if submitted:
        src_id, src_type = all_sources[source_label]
        tgt_id = target_options[target_label]
        if src_id == tgt_id:
            st.error("Source and target cannot be the same.")
        else:
            existing = any(
                e["source_id"] == src_id and e["target_id"] == tgt_id
                for e in state["edges"]
            )
            if existing:
                st.error("This edge already exists.")
            else:
                dir_val = 1 if direction.startswith("+") else -1
                add_edge(state, {
                    "source_id": src_id,
                    "source_type": src_type,
                    "target_id": tgt_id,
                    "direction": dir_val,
                    "strength": strength,
                    "reason": reason.strip(),
                })
                persist()
                st.success("Edge added.")
                st.rerun()

# ── Remove edge ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Remove Dependency Edge")

if state["edges"]:
    edge_labels: dict[str, tuple[str, str]] = {}
    for e in state["edges"]:
        src_name = e["source_id"]
        tgt_name = e["target_id"]
        src_obj = find_market(state, e["source_id"]) or find_news(state, e["source_id"])
        tgt_obj = find_market(state, e["target_id"])
        if src_obj:
            src_name = src_obj.get("name", src_obj.get("headline", src_name))
        if tgt_obj:
            tgt_name = tgt_obj.get("name", tgt_name)
        dir_sign = "+" if e["direction"] > 0 else "-"
        label = f"{src_name} → {tgt_name} ({dir_sign}{e['strength']:.1f})"
        edge_labels[label] = (e["source_id"], e["target_id"])

    selected_edge = st.selectbox("Select edge to remove", list(edge_labels.keys()))
    if st.button("Remove edge", type="secondary"):
        src, tgt = edge_labels[selected_edge]
        remove_edge(state, src, tgt)
        persist()
        st.success("Edge removed.")
        st.rerun()
else:
    st.info("No edges to remove.")
