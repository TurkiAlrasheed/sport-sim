from __future__ import annotations

import streamlit as st
from streamlit_agraph import agraph, Config, Edge, Node

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
