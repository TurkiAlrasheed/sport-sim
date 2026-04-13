from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

try:
    from streamlit_agraph import agraph, Config, Edge, Node
except ImportError:
    agraph = None
    Config = Edge = Node = None

from edge_analysis import apply_market_edges, apply_news_edges
from utils import (
    add_edge,
    find_market,
    find_news,
    get_state,
    persist,
    remove_edge,
    save_state,
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

if agraph is None:
    st.warning("`streamlit-agraph` is not installed. Showing a dependency table instead.")
    if state["edges"]:
        rows = []
        for edge in state["edges"]:
            source_obj = find_market(state, edge["source_id"]) or find_news(state, edge["source_id"])
            target_obj = find_market(state, edge["target_id"])
            rows.append(
                {
                    "Source": (source_obj or {}).get("name", (source_obj or {}).get("headline", edge["source_id"])),
                    "Source Type": edge["source_type"],
                    "Target": (target_obj or {}).get("name", edge["target_id"]),
                    "Direction": "+" if edge["direction"] > 0 else "-",
                    "Strength": edge["strength"],
                    "Reason": edge.get("reason", ""),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Add some markets and news first to see dependencies.")
else:
    # ── Selection state ───────────────────────────────────────────────
    if "selected_graph_node" not in st.session_state:
        st.session_state.selected_graph_node = None

    sel_id: str | None = st.session_state.selected_graph_node

    # Precompute 1-hop neighbor set for the selected node
    highlight_ids: set[str] = set()
    highlight_edge_keys: set[tuple[str, str]] = set()
    if sel_id:
        highlight_ids.add(sel_id)
        for e in state["edges"]:
            if e["source_id"] == sel_id:
                highlight_ids.add(e["target_id"])
                highlight_edge_keys.add((e["source_id"], e["target_id"]))
            elif e["target_id"] == sel_id:
                highlight_ids.add(e["source_id"])
                highlight_edge_keys.add((e["source_id"], e["target_id"]))

    DIM_NODE_COLOR = "rgba(180,180,180,0.12)"
    DIM_FONT_COLOR = "rgba(160,160,160,0.25)"
    DIM_EDGE_COLOR = "rgba(180,180,180,0.08)"

    def _node_color(node_id: str, category: str) -> str | dict:
        base = CATEGORY_COLORS.get(category, "#94a3b8")
        if not sel_id or node_id in highlight_ids:
            if node_id == sel_id:
                return {"background": base, "border": "#facc15"}
            return base
        return {"background": DIM_NODE_COLOR, "border": DIM_NODE_COLOR}

    def _node_font(node_id: str) -> dict:
        if not sel_id or node_id in highlight_ids:
            return {"color": "#e2e8f0"}
        return {"color": DIM_FONT_COLOR}

    def _border_width(node_id: str) -> int:
        return 3 if node_id == sel_id else 1

    nodes: list[Node] = []
    edges_vis: list[Edge] = []

    for m in state["markets"]:
        nodes.append(Node(
            id=m["id"],
            label=m["name"][:30],
            title=m["description"],
            color=_node_color(m["id"], m["category"]),
            size=25,
            shape="dot",
            font=_node_font(m["id"]),
            borderWidth=_border_width(m["id"]),
        ))

    for n in state["news"]:
        nodes.append(Node(
            id=n["id"],
            label=n["headline"][:30],
            title=n["headline"],
            color=_node_color(n["id"], n["category"]),
            size=18,
            shape="square",
            font=_node_font(n["id"]),
            borderWidth=_border_width(n["id"]),
        ))

    for e in state["edges"]:
        active = not sel_id or (e["source_id"], e["target_id"]) in highlight_edge_keys
        if active:
            edge_color = "#22c55e" if e["direction"] > 0 else "#ef4444"
            edge_width = 1 + e["strength"] * 3
            edge_label = f"{'+' if e['direction'] > 0 else '-'}{e['strength']:.1f}"
            edge_font = {"color": "#cbd5e1", "strokeWidth": 0, "size": 10}
        else:
            edge_color = DIM_EDGE_COLOR
            edge_width = 0.5
            edge_label = ""
            edge_font = {"color": "rgba(0,0,0,0)"}
        edges_vis.append(Edge(
            source=e["source_id"],
            target=e["target_id"],
            label=edge_label,
            color=edge_color,
            width=edge_width,
            title=e.get("reason", "") if active else "",
            font=edge_font,
        ))

    config = Config(
        width="100%",
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
    )

    if nodes:
        st.subheader("Interactive Graph")

        # ── Color-code legend ─────────────────────────────────────────
        legend_html = "<div style='display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start;margin-bottom:12px'>"

        legend_html += "<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center'>"
        legend_html += "<span style='font-weight:600;margin-right:4px'>Categories:</span>"
        for cat, color in CATEGORY_COLORS.items():
            legend_html += (
                f"<span style='display:inline-flex;align-items:center;gap:4px'>"
                f"<span style='width:12px;height:12px;border-radius:3px;background:{color};display:inline-block'></span>"
                f"<span style='font-size:0.85rem'>{cat}</span></span>"
            )
        legend_html += "</div>"

        legend_html += (
            "<div style='display:flex;gap:8px;align-items:center'>"
            "<span style='font-weight:600;margin-right:4px'>Nodes:</span>"
            "<span style='display:inline-flex;align-items:center;gap:4px'>"
            "<span style='width:12px;height:12px;border-radius:50%;background:#94a3b8;display:inline-block'></span>"
            "<span style='font-size:0.85rem'>Market</span></span>"
            "<span style='display:inline-flex;align-items:center;gap:4px'>"
            "<span style='width:12px;height:12px;border-radius:1px;background:#94a3b8;display:inline-block'></span>"
            "<span style='font-size:0.85rem'>News</span></span>"
            "</div>"
        )

        legend_html += (
            "<div style='display:flex;gap:8px;align-items:center'>"
            "<span style='font-weight:600;margin-right:4px'>Edges:</span>"
            "<span style='display:inline-flex;align-items:center;gap:4px'>"
            "<span style='width:18px;height:3px;background:#22c55e;display:inline-block;border-radius:1px'></span>"
            "<span style='font-size:0.85rem'>Positive</span></span>"
            "<span style='display:inline-flex;align-items:center;gap:4px'>"
            "<span style='width:18px;height:3px;background:#ef4444;display:inline-block;border-radius:1px'></span>"
            "<span style='font-size:0.85rem'>Negative</span></span>"
            "</div>"
        )

        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

        returned_node = agraph(nodes=nodes, edges=edges_vis, config=config)

        # Handle selection change → rerun with updated dimming
        if returned_node and returned_node != sel_id:
            st.session_state.selected_graph_node = returned_node
            st.rerun()

        # ── 1-hop neighbor details on selection ───────────────────────
        if sel_id:
            node_lookup: dict[str, dict] = {}
            for m in state["markets"]:
                node_lookup[m["id"]] = {**m, "_type": "market"}
            for n in state["news"]:
                node_lookup[n["id"]] = {**n, "_type": "news"}

            selected = node_lookup.get(sel_id)
            if selected:
                sel_label = selected.get("name") or selected.get("headline", sel_id)

                hdr_col, btn_col = st.columns([5, 1])
                hdr_col.markdown(f"#### Selected: {sel_label}")
                if btn_col.button("Clear selection", use_container_width=True):
                    st.session_state.selected_graph_node = None
                    st.rerun()

                neighbors: list[dict] = []
                for e in state["edges"]:
                    if e["source_id"] == sel_id:
                        nbr = node_lookup.get(e["target_id"])
                        if nbr:
                            neighbors.append({
                                "Relation": "→ influences",
                                "Node": nbr.get("name") or nbr.get("headline", e["target_id"]),
                                "Type": nbr["_type"].title(),
                                "Category": nbr.get("category", ""),
                                "Direction": "+" if e["direction"] > 0 else "−",
                                "Strength": e["strength"],
                                "Reason": e.get("reason", ""),
                            })
                    if e["target_id"] == sel_id:
                        nbr = node_lookup.get(e["source_id"])
                        if nbr:
                            neighbors.append({
                                "Relation": "← influenced by",
                                "Node": nbr.get("name") or nbr.get("headline", e["source_id"]),
                                "Type": nbr["_type"].title(),
                                "Category": nbr.get("category", ""),
                                "Direction": "+" if e["direction"] > 0 else "−",
                                "Strength": e["strength"],
                                "Reason": e.get("reason", ""),
                            })

                if neighbors:
                    st.dataframe(neighbors, use_container_width=True, hide_index=True)
                else:
                    st.info("No edges connected to this node.")
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
        save_state(state)
        progress = st.progress(0, text="Generating news edges...")
        total = 0
        for i, news in enumerate(state["news"]):
            progress.progress(
                (i + 1) / news_count,
                text=f"Analyzing: {news['headline'][:60]}...",
            )
            try:
                total += apply_news_edges(state, news)
            except Exception as exc:
                st.warning(f"Failed for \"{news['headline'][:40]}...\": {exc}")
        progress.empty()
        st.success(f"Generated **{total}** news→market edge(s).")
        st.rerun()

    if regen_markets:
        state["edges"] = [e for e in state["edges"] if e["source_type"] != "market"]
        save_state(state)
        progress = st.progress(0, text="Generating market edges...")
        total = 0
        for i, market in enumerate(state["markets"]):
            progress.progress(
                (i + 1) / market_count,
                text=f"Analyzing: {market['name'][:60]}...",
            )
            try:
                total += apply_market_edges(state, market)
            except Exception as exc:
                st.warning(f"Failed for \"{market['name'][:40]}...\": {exc}")
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
