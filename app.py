"""
Streamlit UI for the Explainable and Reliable Music Recommender.
Run from the project root:  streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from recommender import Recommender, UserProfile, Recommendation
from loader import (
    load_spotify_dataset, load_songs,
    GENRE_GROUPS, derive_mood,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Catalog (cached — loads once per session)
# ---------------------------------------------------------------------------

_DATA = os.path.join(os.path.dirname(__file__), "data")
_SPOTIFY = os.path.join(_DATA, "spotify-tracks-dataset.csv")
_SMALL   = os.path.join(_DATA, "songs.csv")

@st.cache_data(show_spinner="Loading catalog …")
def get_songs():
    if os.path.exists(_SPOTIFY):
        return load_spotify_dataset(_SPOTIFY)
    return load_songs(_SMALL)

songs = get_songs()

# Derived UI option lists
GENRE_OPTIONS  = ["(any)"] + sorted(GENRE_GROUPS.keys()) + ["other"]
MOOD_OPTIONS   = ["(any)", "happy", "chill", "intense", "moody",
                  "relaxed", "energetic", "focused"]
ALL_ARTISTS    = sorted({s.artist for s in songs})
CATALOG_SIZE   = f"{len(songs):,}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TAG_STYLE = {
    "safe match":    ("🟢", "#d4edda", "#155724"),
    "explore pick":  ("🔵", "#cce5ff", "#004085"),
    "partial match": ("🟡", "#fff3cd", "#856404"),
}

def tag_badge(tag: str) -> str:
    emoji, bg, fg = TAG_STYLE.get(tag, ("⚪", "#f8f9fa", "#495057"))
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 12px;'
        f'border-radius:14px;font-size:0.78em;font-weight:700;">'
        f'{emoji} {tag}</span>'
    )

def conf_color(c: float) -> str:
    return "#28a745" if c >= 70 else "#fd7e14" if c >= 40 else "#dc3545"

def render_card(rank: int, r: Recommendation) -> None:
    col_main, col_metrics = st.columns([3, 1])
    with col_main:
        st.markdown(
            f"**#{rank} &nbsp; {r.song.title}** &nbsp; *by {r.song.artist}* "
            f"&nbsp;&nbsp; {tag_badge(r.tag)}",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.caption(f"🎵 **{r.song.genre}** ({r.song.genre_group})")
        c2.caption(f"😌 Mood: **{r.song.mood}**")
        c3.caption(f"⚡ Energy: **{r.song.energy:.2f}**")
        c4.caption(f"💃 Dance: **{r.song.danceability:.2f}**")
        st.markdown(f"_{r.explanation}_")
    with col_metrics:
        color = conf_color(r.confidence)
        st.markdown(
            f'<p style="margin:0;font-size:0.78em;color:#6c757d;">Confidence</p>'
            f'<p style="margin:0;font-size:1.8em;font-weight:700;color:{color};">'
            f'{r.confidence:.0f}%</p>',
            unsafe_allow_html=True,
        )
        st.caption(f"Score: {r.score:.2f} / 8.1")
        if r.song.popularity:
            st.caption(f"Popularity: {r.song.popularity}/100")
    st.progress(r.confidence / 100)
    st.divider()

# ---------------------------------------------------------------------------
# Sidebar — user profile
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"## 🎛️ Your Profile")
    st.caption(f"Catalog: **{CATALOG_SIZE} songs** · changes update instantly")
    st.divider()

    # --- Genre & mood ---
    genre = st.selectbox("Favorite Genre", GENRE_OPTIONS,
                         help="Songs from this genre group get priority.")
    mood = st.selectbox("Preferred Mood", MOOD_OPTIONS,
                        help="Derived from audio features (valence + energy).")

    st.divider()

    # --- Audio feature targets ---
    st.markdown("**Audio Feature Targets**")
    energy = st.slider("Energy", 0.0, 1.0, 0.6, 0.01,
                       help="0 = calm / ambient · 1 = intense / high-tempo")
    valence = st.slider("Mood Tone (Valence)", 0.0, 1.0, 0.5, 0.01,
                        help="0 = melancholic · 1 = uplifting / bright")
    danceability = st.slider("Groove (Danceability)", 0.0, 1.0, 0.5, 0.01,
                             help="0 = not for dancing · 1 = highly danceable")

    st.divider()

    # --- Preferences ---
    st.markdown("**Preferences**")
    likes_acoustic = st.toggle("Acoustic sounds 🎸", value=False)
    allow_explicit = st.toggle("Allow explicit content 🔞", value=True)

    st.divider()

    # --- Artists & discovery ---
    favorite_artists = st.multiselect(
        "Favorite Artists",
        options=ALL_ARTISTS,
        help="Songs by these artists get a score bonus.",
    )
    discovery = st.slider(
        "Discovery Preference", 0.0, 1.0, 0.5, 0.05,
        help="Low = stick to your genre · High = explore new genres",
    )
    dcols = st.columns(2)
    dcols[0].caption("🔒 Familiar")
    dcols[1].markdown('<p style="text-align:right;font-size:0.75em;color:#6c757d;">Exploratory 🔀</p>',
                      unsafe_allow_html=True)

    st.divider()
    k = st.slider("# Recommendations", 1, 20, 5)

    st.divider()
    st.markdown("**Legend**")
    for tag, (_, bg, fg) in TAG_STYLE.items():
        st.markdown(tag_badge(tag), unsafe_allow_html=True)
    st.caption("🟢 genre + mood match  ·  🔵 discovery pick  ·  🟡 partial")

# ---------------------------------------------------------------------------
# Build profile and run
# ---------------------------------------------------------------------------

user = UserProfile(
    favorite_genre="" if genre == "(any)" else genre,
    favorite_mood="" if mood == "(any)" else mood,
    target_energy=energy,
    target_valence=valence,
    target_danceability=danceability,
    likes_acoustic=likes_acoustic,
    allow_explicit=allow_explicit,
    favorite_artists=list(favorite_artists),
    discovery_preference=discovery,
)

engine  = Recommender(songs)
recs    = engine.recommend(user, k=k)
warnings = engine.run_guardrails(recs)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown("# 🎧 Music Recommender")
st.caption("Explainable song suggestions powered by a 5-feature audio vector.")

# Active profile summary bar
parts = []
if genre != "(any)": parts.append(f"**Genre:** {genre}")
if mood  != "(any)": parts.append(f"**Mood:** {mood}")
parts.append(f"**Energy:** {energy:.2f}")
parts.append(f"**Valence:** {valence:.2f}")
parts.append(f"**Dance:** {danceability:.2f}")
parts.append(f"**Discovery:** {'High 🔀' if discovery > 0.65 else 'Low 🔒' if discovery < 0.35 else 'Med ⚖️'}")
if not allow_explicit: parts.append("🔞 off")
if favorite_artists:   parts.append(f"**Favs:** {', '.join(favorite_artists)}")
st.info("  ·  ".join(parts))

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

st.markdown(f"### Top {len(recs)} Recommendations")
for i, r in enumerate(recs, 1):
    render_card(i, r)

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

if warnings:
    st.markdown("### ⚠️ Guardrail Notices")
    for w in warnings:
        st.warning(w)

# ---------------------------------------------------------------------------
# Genre breakdown
# ---------------------------------------------------------------------------

st.markdown("### 📊 Genre Breakdown of Results")
group_counts: dict = {}
for r in recs:
    group_counts[r.song.genre_group] = group_counts.get(r.song.genre_group, 0) + 1
st.bar_chart(pd.DataFrame({"Count": group_counts}))

# ---------------------------------------------------------------------------
# Catalog explorer (search + filter, paginated)
# ---------------------------------------------------------------------------

with st.expander(f"📀 Browse Catalog ({CATALOG_SIZE} songs)"):
    col_search, col_genre_f, col_mood_f = st.columns([2, 1, 1])
    search_q  = col_search.text_input("Search title or artist", key="cat_search")
    g_filter  = col_genre_f.selectbox("Genre group", ["all"] + sorted(GENRE_GROUPS.keys()),
                                      key="cat_genre")
    m_filter  = col_mood_f.selectbox("Mood", ["all"] + MOOD_OPTIONS[1:],
                                     key="cat_mood")

    filtered = songs
    if search_q:
        q = search_q.lower()
        filtered = [s for s in filtered
                    if q in s.title.lower() or q in s.artist.lower()]
    if g_filter != "all":
        filtered = [s for s in filtered if s.genre_group == g_filter]
    if m_filter != "all":
        filtered = [s for s in filtered if s.mood == m_filter]

    st.caption(f"Showing top 50 of {len(filtered):,} matching songs.")

    rows = filtered[:50]
    if rows:
        df = pd.DataFrame([{
            "Title":       s.title,
            "Artist":      s.artist,
            "Genre":       s.genre,
            "Group":       s.genre_group,
            "Mood":        s.mood,
            "Energy":      round(s.energy, 2),
            "Valence":     round(s.valence, 2),
            "Dance":       round(s.danceability, 2),
            "Popularity":  s.popularity,
            "Explicit":    s.explicit,
        } for s in rows])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No songs match your filters.")
