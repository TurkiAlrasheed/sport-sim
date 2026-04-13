from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st

from edge_analysis import apply_market_edges
from kalshi import fetch_top_markets, KALSHI_CATEGORIES
from utils import (
    CATEGORIES,
    add_market,
    get_state,
    persist,
    remove_market,
    slugify,
)

st.set_page_config(page_title="Markets", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Markets")
st.caption("Add, view, and remove prediction markets.")

state = get_state()

st.subheader("Import live Kalshi markets")
st.caption("Fetch open Kalshi prediction markets and add them directly to your state.")

col_kcat, col_klimit = st.columns([2, 1])
selected_category = col_kcat.selectbox("Kalshi category", KALSHI_CATEGORIES)
limit = col_klimit.slider("Max markets", min_value=1, max_value=20, value=10)
fetch_clicked = st.button("Fetch Kalshi markets", type="primary", use_container_width=True)

if fetch_clicked:
    with st.spinner("Loading Kalshi markets..."):
        try:
            markets = fetch_top_markets(selected_category, limit=limit)
        except Exception as exc:
            st.error(f"Failed to fetch Kalshi markets: {exc}")
            markets = []

    if not markets:
        st.warning("No markets were returned from Kalshi. Try again later.")
    else:
        existing_ids = {m["id"] for m in state["markets"]}
        added = 0
        imported_markets: list[dict] = []
        for km in markets:
            market_id = slugify(km["ticker"] or km["title"])
            if market_id in existing_ids:
                continue
            market_obj = {
                "id": market_id,
                "name": km["title"],
                "description": km["event"],
                "category": selected_category,
                "market_probability": km["market_probability"],
            }
            add_market(state, market_obj)
            imported_markets.append(market_obj)
            existing_ids.add(market_id)
            added += 1

        if added:
            persist()
            st.success(f"Added **{added}** Kalshi market(s) from {selected_category}.")

            if os.environ.get("OPENAI_API_KEY") and len(state["markets"]) >= 2:
                with st.spinner("Generating AI market relationships for imported markets..."):
                    for market in imported_markets:
                        try:
                            apply_market_edges(state, market)
                        except Exception:
                            pass
                persist()

            st.rerun()
        else:
            st.info("All fetched Kalshi markets are already present in your market list.")

# ── Add market form ──────────────────────────────────────────────────────

with st.expander("Add new market", expanded=False):
    with st.form("add_market", clear_on_submit=True):
        name = st.text_input("Market name", placeholder="e.g. Bitcoin above $100K on July 1")
        description = st.text_area(
            "Description",
            placeholder="Full description of the market contract",
            height=80,
        )
        col_cat, col_prob = st.columns(2)
        category = col_cat.selectbox("Category", CATEGORIES)
        probability_pct = col_prob.slider("Market probability (%)", 0, 100, 50)
        submitted = st.form_submit_button("Add market", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("Market name is required.")
        else:
            market_id = slugify(name)
            if any(m["id"] == market_id for m in state["markets"]):
                st.error(f"A market with id **{market_id}** already exists.")
            else:
                market_obj = {
                    "id": market_id,
                    "name": name.strip(),
                    "description": description.strip() or name.strip(),
                    "category": category,
                    "market_probability": probability_pct / 100,
                }
                add_market(state, market_obj)
                persist()
                st.success(f"Added market: **{name.strip()}**")

                if os.environ.get("OPENAI_API_KEY") and len(state["markets"]) >= 2:
                    try:
                        with st.spinner("Analyzing market relationships with AI..."):
                            added = apply_market_edges(state, market_obj)
                        if added:
                            st.success(
                                f"AI generated {added} market→market "
                                f"edge{'s' if added != 1 else ''}."
                            )
                        else:
                            st.info("AI found no causal links to other markets.")
                    except Exception as exc:
                        st.warning(f"AI edge generation failed: {exc}")

                st.rerun()

# ── Market list ──────────────────────────────────────────────────────────

st.subheader(f"All Markets ({len(state['markets'])})")

if not state["markets"]:
    st.info("No markets yet. Add one above.")
    st.stop()

rows = []
for m in state["markets"]:
    rows.append({
        "ID": m["id"],
        "Name": m["name"],
        "Category": m["category"],
        "Market Prob": f"{m['market_probability']:.0%}",
        "Description": m["description"],
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Delete market ────────────────────────────────────────────────────────

with st.expander("Remove a market"):
    market_names = {m["name"]: m["id"] for m in state["markets"]}
    to_delete = st.selectbox("Select market to remove", list(market_names.keys()))
    if st.button("Remove", type="secondary"):
        remove_market(state, market_names[to_delete])
        persist()
        st.success(f"Removed **{to_delete}** and its edges.")
        st.rerun()
