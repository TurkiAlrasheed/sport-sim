from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

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
                add_news(state, {
                    "id": news_id,
                    "headline": headline.strip(),
                    "category": category,
                    "timestamp": str(timestamp),
                })
                persist()
                st.success(f"Added news: **{headline.strip()}**")
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
