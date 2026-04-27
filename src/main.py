"""
Command-line runner for the Explainable and Reliable Music Recommender.
Tries the large Spotify dataset first; falls back to the small catalog.

Run from the project root:  python -m src.main
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from loader import load_spotify_dataset, load_songs
from recommender import Recommender, UserProfile

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_SPOTIFY_CSV = os.path.join(_DATA, "spotify-tracks-dataset.csv")
_SMALL_CSV = os.path.join(_DATA, "songs.csv")


def get_songs():
    if os.path.exists(_SPOTIFY_CSV):
        print("Loading Spotify dataset …")
        songs = load_spotify_dataset(_SPOTIFY_CSV)
        print(f"  {len(songs):,} songs loaded.")
        return songs
    print("Spotify dataset not found, using small catalog.")
    return load_songs(_SMALL_CSV)


def print_recommendations(label, recs, warnings) -> None:
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    for i, r in enumerate(recs, 1):
        print(f"\n#{i}: {r.song.title} — {r.song.artist}  [{r.tag}]")
        print(f"     Genre: {r.song.genre} ({r.song.genre_group}) | Mood: {r.song.mood}")
        print(f"     Energy: {r.song.energy} | Valence: {r.song.valence:.2f} | Dance: {r.song.danceability:.2f}")
        print(f"     Score: {r.score:.2f} | Confidence: {r.confidence:.0f}%")
        print(f"     {r.explanation}")
    if warnings:
        print("\n  ⚠ Guardrails:")
        for w in warnings:
            print(f"    ! {w}")
    print()


def main() -> None:
    songs = get_songs()
    rec = Recommender(songs)

    profiles = [
        (
            "Rock fan — energetic, high discovery",
            UserProfile(
                favorite_genre="rock", favorite_mood="energetic",
                target_energy=0.85, target_valence=0.5, target_danceability=0.6,
                likes_acoustic=False, allow_explicit=True, discovery_preference=0.8,
            ),
        ),
        (
            "Chill ambient listener — no explicit",
            UserProfile(
                favorite_genre="ambient", favorite_mood="chill",
                target_energy=0.3, target_valence=0.65, target_danceability=0.4,
                likes_acoustic=True, allow_explicit=False, discovery_preference=0.3,
            ),
        ),
        (
            "Pop lover — upbeat & danceable",
            UserProfile(
                favorite_genre="pop", favorite_mood="happy",
                target_energy=0.8, target_valence=0.8, target_danceability=0.8,
                likes_acoustic=False, allow_explicit=True, discovery_preference=0.4,
            ),
        ),
        (
            "No preferences set (fallback test)",
            UserProfile(
                favorite_genre="", favorite_mood="",
                target_energy=0.6, target_valence=0.5, target_danceability=0.5,
                likes_acoustic=False, allow_explicit=True, discovery_preference=0.5,
            ),
        ),
    ]

    for label, user in profiles:
        recs = rec.recommend(user, k=5)
        print_recommendations(label, recs, rec.run_guardrails(recs))


if __name__ == "__main__":
    main()
