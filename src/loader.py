"""
Dataset loading and preprocessing.

Handles both the large Spotify tracks dataset and the small test catalog.
"""
from __future__ import annotations

import csv
from typing import Dict, List, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from recommender import Song

# ---------------------------------------------------------------------------
# Genre grouping: 114 raw Spotify genres → 14 human-facing groups
# ---------------------------------------------------------------------------

GENRE_GROUPS: Dict[str, List[str]] = {
    "pop": [
        "pop", "power-pop", "synth-pop", "indie-pop", "cantopop", "k-pop",
        "mandopop", "j-pop", "pop-film", "disney", "show-tunes", "children",
        "kids", "party", "romance", "happy",
    ],
    "rock": [
        "rock", "alt-rock", "alternative", "hard-rock", "punk-rock", "punk",
        "psych-rock", "garage", "grunge", "emo", "indie", "british",
        "rock-n-roll", "rockabilly", "j-rock",
    ],
    "electronic": [
        "electronic", "dance", "edm", "house", "chicago-house", "detroit-techno",
        "deep-house", "club", "dubstep", "drum-and-bass", "breakbeat",
        "minimal-techno", "trance", "electro", "disco", "techno",
        "progressive-house", "hardstyle", "idm", "groove", "j-dance",
    ],
    "hip-hop": [
        "hip-hop", "reggaeton", "dancehall", "dub", "trip-hop",
    ],
    "r&b": [
        "r-n-b", "soul", "funk",
    ],
    "metal": [
        "metal", "black-metal", "death-metal", "heavy-metal", "metalcore",
        "grindcore", "hardcore", "goth", "industrial",
    ],
    "classical": [
        "classical", "piano", "opera",
    ],
    "jazz": [
        "jazz", "blues", "gospel",
    ],
    "country": [
        "country", "bluegrass", "honky-tonk",
    ],
    "latin": [
        "latin", "latino", "brazil", "samba", "salsa", "bossanova", "mpb",
        "pagode", "sertanejo", "forro", "tango", "spanish",
    ],
    "ambient": [
        "ambient", "chill", "new-age", "sleep", "study", "sad",
    ],
    "folk": [
        "folk", "acoustic", "singer-songwriter", "songwriter", "guitar",
    ],
    "reggae": [
        "reggae", "ska", "afrobeat",
    ],
    "world": [
        "world-music", "swedish", "turkish", "iranian", "malay",
        "french", "german", "indian", "anime", "j-idol",
    ],
}

# Reverse lookup: raw genre → group name
GENRE_TO_GROUP: Dict[str, str] = {
    genre: group
    for group, genres in GENRE_GROUPS.items()
    for genre in genres
}

TEMPO_MAX: float = 243.37  # dataset observed maximum

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def map_genre_group(genre: str) -> str:
    return GENRE_TO_GROUP.get(genre.lower(), "other")


def derive_mood(valence: float, energy: float) -> str:
    """Map (valence, energy) → one of 7 mood labels."""
    if valence >= 0.6 and energy >= 0.6:
        return "happy"
    if valence >= 0.6 and energy < 0.6:
        return "chill"
    if valence < 0.4 and energy >= 0.6:
        return "intense"
    if valence < 0.4 and energy < 0.4:
        return "moody"
    if energy < 0.4:
        return "relaxed"
    if energy >= 0.7:
        return "energetic"
    return "focused"


# ---------------------------------------------------------------------------
# Spotify tracks dataset loader (~114k songs)
# ---------------------------------------------------------------------------

def load_spotify_dataset(csv_path: str) -> List[Song]:
    """
    Load, clean, and convert the Spotify tracks CSV into Song objects.

    Cleans:  drops junk index columns, removes null rows, deduplicates by track_id.
    Derives: genre_group, mood, tempo_norm.
    """
    from recommender import Song

    df = pd.read_csv(csv_path)

    # Drop pandas artifact index columns (Unnamed: 0, Unnamed: 0.1, …)
    df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True)

    # Drop rows missing fields we can't operate without
    df.dropna(subset=["track_name", "artists", "track_genre"], inplace=True)

    # One entry per track — same song can appear under multiple genres in source
    df.drop_duplicates(subset=["track_id"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Derived columns
    df["tempo_norm"] = (df["tempo"] / TEMPO_MAX).clip(0.0, 1.0)
    df["mood"] = [derive_mood(v, e) for v, e in zip(df["valence"], df["energy"])]
    df["genre_group"] = [map_genre_group(g) for g in df["track_genre"]]

    songs: List[Song] = []
    for i, row in enumerate(df.itertuples(index=False)):
        songs.append(
            Song(
                id=i,
                title=row.track_name,
                artist=row.artists,
                genre=row.track_genre,
                genre_group=row.genre_group,
                mood=row.mood,
                energy=float(row.energy),
                valence=float(row.valence),
                danceability=float(row.danceability),
                acousticness=float(row.acousticness),
                tempo_norm=float(row.tempo_norm),
                popularity=int(row.popularity),
                explicit=bool(row.explicit),
            )
        )

    return songs


# ---------------------------------------------------------------------------
# Small test-catalog loader (data/songs.csv)
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Song]:
    """Load the small hand-crafted catalog into Song objects."""
    from recommender import Song

    songs: List[Song] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(
                Song(
                    id=int(row["id"]),
                    title=row["title"],
                    artist=row["artist"],
                    genre=row["genre"],
                    genre_group=row["genre_group"],
                    mood=row["mood"],
                    energy=float(row["energy"]),
                    valence=float(row["valence"]),
                    danceability=float(row["danceability"]),
                    acousticness=float(row["acousticness"]),
                    tempo_norm=float(row["tempo_norm"]),
                    popularity=int(row["popularity"]),
                    explicit=row["explicit"].strip().lower() == "true",
                )
            )
    return songs
