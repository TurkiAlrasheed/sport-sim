from __future__ import annotations

import os
from datetime import date

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st

from edge_analysis import apply_news_edges
from news_api import fetch_top_headlines
from utils import (
    CATEGORIES,
    add_news,
    get_state,
    persist,
    remove_news,
    slugify,
)

st.set_page_config(page_title="News Events", page_icon=":newspaper:", layout="wide")

st.title("News Events")
st.caption("Add, view, and remove real-world news that affects markets.")

state = get_state()

# ── Fetch live headlines ─────────────────────────────────────────────────

st.subheader("Fetch Live Headlines")

has_newsapi_key = bool(os.environ.get("NEWSAPI_KEY"))
if not has_newsapi_key:
    st.warning("Set `NEWSAPI_KEY` in your `.env` file to enable live headline fetching.")

col_fetch, col_info = st.columns([1, 2])
fetch_clicked = col_fetch.button(
    "Fetch latest headlines",
    type="primary",
    use_container_width=True,
    disabled=not has_newsapi_key,
)

if fetch_clicked:
    with st.spinner("Fetching top headlines from NewsAPI..."):
        try:
            headlines = fetch_top_headlines(page_size=20)
        except Exception as exc:
            st.error(f"Failed to fetch headlines: {exc}")
            headlines = []

    if not headlines:
        st.warning("NewsAPI returned no usable headlines. Try again later.")
    else:
        existing_ids = {n["id"] for n in state["news"]}
        new_headlines = [h for h in headlines if h["id"] not in existing_ids]

        if not new_headlines:
            st.info(
                f"Fetched {len(headlines)} headline(s), but all are already "
                "in your news list."
            )
        else:
            total_edges = 0
            for h in new_headlines:
                add_news(state, {
                    "id": h["id"],
                    "headline": h["headline"],
                    "category": h["category"],
                    "timestamp": h["timestamp"],
                })

            persist()
            st.success(f"Added **{len(new_headlines)}** new headline(s).")

            if os.environ.get("OPENAI_API_KEY") and state["markets"]:
                with st.spinner("Generating AI dependency edges for new headlines..."):
                    for h in new_headlines:
                        news_obj = next(
                            n for n in state["news"] if n["id"] == h["id"]
                        )
                        try:
                            added = apply_news_edges(state, news_obj)
                            total_edges += added
                        except Exception:
                            pass
                if total_edges:
                    st.success(
                        f"AI generated **{total_edges}** dependency "
                        f"edge{'s' if total_edges != 1 else ''}."
                    )

            st.rerun()

st.divider()

# ── Add news form ────────────────────────────────────────────────────────

with st.expander("Add new news event", expanded=False):
    with st.form("add_news", clear_on_submit=True):
        headline = st.text_input("Headline", placeholder="e.g. Fed announces surprise 50 bps rate cut")
        col_cat, col_date = st.columns(2)
        category = col_cat.selectbox("Category", CATEGORIES)
        timestamp = col_date.date_input("Date", value=date.today())
        submitted = st.form_submit_button("Add news event", use_container_width=True)

    if submitted:
        if not headline.strip():
            st.error("Headline is required.")
        else:
            news_id = slugify(headline)
            if any(n["id"] == news_id for n in state["news"]):
                st.error(f"A news event with id **{news_id}** already exists.")
            else:
                news_obj = {
                    "id": news_id,
                    "headline": headline.strip(),
                    "category": category,
                    "timestamp": str(timestamp),
                }
                add_news(state, news_obj)
                persist()
                st.success(f"Added news: **{headline.strip()}**")

                if os.environ.get("OPENAI_API_KEY") and state["markets"]:
                    try:
                        with st.spinner("Analyzing headline with AI..."):
                            added = apply_news_edges(state, news_obj)
                        if added:
                            st.success(
                                f"AI generated {added} dependency "
                                f"edge{'s' if added != 1 else ''}."
                            )
                        else:
                            st.info("AI found no relevant market connections for this headline.")
                    except Exception as exc:
                        st.warning(f"AI edge generation failed: {exc}")

                st.rerun()

# ── News list ────────────────────────────────────────────────────────────

st.subheader(f"All News Events ({len(state['news'])})")

if not state["news"]:
    st.info("No news events yet. Add one above.")
    st.stop()

rows = []
for n in state["news"]:
    rows.append({
        "ID": n["id"],
        "Headline": n["headline"],
        "Category": n["category"],
        "Date": n.get("timestamp", ""),
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Delete news ──────────────────────────────────────────────────────────

with st.expander("Remove a news event"):
    news_map = {n["headline"]: n["id"] for n in state["news"]}
    to_delete = st.selectbox("Select news event to remove", list(news_map.keys()))
    if st.button("Remove", type="secondary"):
        remove_news(state, news_map[to_delete])
        persist()
        st.success(f"Removed **{to_delete}** and its edges.")
        st.rerun()
