import os
import requests
import streamlit as st
from typing import List

API_HOST = os.getenv("API_HOST", "api")  # docker-compose service name
API_PORT = os.getenv("API_PORT", "8000")
API_BASE = f"http://{API_HOST}:{API_PORT}"

st.set_page_config(page_title="Yambda Recommender", page_icon="🎧", layout="wide")

st.title("🎧 Yambda Music Recommendation Demo")
st.caption("Interactively pick tracks you like and get recommendations (demo user id = -1)")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    top_n = st.slider("How many recommendations?", 5, 30, 10, 1)
    if st.button("🔄 Get new random tracks"):
        st.session_state.random_items = None  # force refresh

# Session state init
if "random_items" not in st.session_state:
    st.session_state.random_items = None
if "liked" not in st.session_state:
    st.session_state.liked = set()
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None


def fetch_random_items() -> List[int]:
    try:
        r = requests.get(f"{API_BASE}/recommend/get_random_items", timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("random_items", [])
    except Exception as e:
        st.error(f"Failed to fetch random items: {e}")
        return []


def fetch_recommendations(selected_items: List[int], top_n: int):
    payload = {"items": [{"item_id": it} for it in selected_items]}
    try:
        r = requests.post(
            f"{API_BASE}/recommend/-1?top_n={top_n}", json=payload, timeout=60
        )
        r.raise_for_status()
        data = r.json()
        return data.get("recommendations", [])
    except Exception as e:
        st.error(f"Failed to get recommendations: {e}")
        return []


# Load random items once (or after refresh)
if st.session_state.random_items is None:
    st.session_state.random_items = fetch_random_items()
    st.session_state.liked.clear()
    st.session_state.recommendations = None

random_items = st.session_state.random_items

if not random_items:
    st.warning("No random items returned. Try again.")
    if st.button("Retry"):
        st.session_state.random_items = None
        st.rerun()
    st.stop()

st.subheader("Random Tracks – click to like")
cols = st.columns(min(5, len(random_items)))
for idx, item_id in enumerate(random_items):
    col = cols[idx % len(cols)]
    liked = item_id in st.session_state.liked
    button_label = f"❤️ {item_id}" if liked else f"➕ {item_id}"
    help_txt = "Click to remove from liked" if liked else "Click to add to liked"
    if col.button(button_label, key=f"track_{item_id}", help=help_txt):
        if liked:
            st.session_state.liked.remove(item_id)
        else:
            st.session_state.liked.add(item_id)
        st.session_state.recommendations = None  # invalidate
        st.rerun()

with st.expander("Current liked items", expanded=True):
    if st.session_state.liked:
        st.write(sorted(list(st.session_state.liked)))
    else:
        st.caption("No items liked yet.")

# Recommend button
recommend_disabled = len(st.session_state.liked) == 0
if st.button("🎯 Get Recommendations", disabled=recommend_disabled):
    if recommend_disabled:
        st.info("Select at least one track first.")
    else:
        with st.spinner("Computing recommendations..."):
            recs = fetch_recommendations(list(st.session_state.liked), top_n)
            st.session_state.recommendations = recs

# Show recommendations
if st.session_state.recommendations is not None:
    st.subheader("Recommended Tracks")
    if not st.session_state.recommendations:
        st.info("No recommendations returned.")
    else:
        for rec in st.session_state.recommendations:
            st.write(f"• Track {rec['item_id']} (score: {rec['score']:.4f})")
