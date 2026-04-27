import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recommender import Recommender, UserProfile
from loader import load_spotify_dataset, load_songs

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
SPOTIFY = os.path.join(DATA, "spotify-tracks-dataset.csv")
SMALL = os.path.join(DATA, "songs.csv")


def load_data():
    if os.path.exists(SPOTIFY):
        print("Loading Spotify dataset...")
        return load_spotify_dataset(SPOTIFY)
    print("Using small dataset...")
    return load_songs(SMALL)


def evaluate_profile(label, user, recommender, k=5):
    recs = recommender.recommend(user, k=k)
    warnings = recommender.run_guardrails(recs)

    avg_conf = sum(r.confidence for r in recs) / len(recs)
    unique_genres = len(set(r.song.genre_group for r in recs))
    duplicates = len(recs) - len(set(r.song.id for r in recs))

    print(f"\n=== {label} ===")
    print(f"Recommendations: {len(recs)}")
    print(f"Avg Confidence: {avg_conf:.2f}%")
    print(f"Unique Genres: {unique_genres}")
    print(f"Duplicates: {duplicates}")

    if warnings:
        print("Guardrails triggered:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Guardrails: OK")

    # Simple pass/fail logic
    min_confidence = 25 if user.favorite_genre == "" else 40

    requires_diversity = (
        user.discovery_preference >= 0.75 or user.favorite_genre == ""
    )

    diversity_ok = unique_genres >= 2 if requires_diversity else unique_genres >= 1

    passed = (
        len(recs) == k and
        avg_conf >= min_confidence and
        duplicates == 0 and
        diversity_ok
    )

    print("Result:", "PASS ✅" if passed else "FAIL ❌")
    return passed


def main():
    songs = load_data()
    rec = Recommender(songs)

    tests = [
        ("Pop energetic user",
         UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8)),

        ("Chill ambient user",
         UserProfile(favorite_genre="ambient", favorite_mood="chill", target_energy=0.3,
                     allow_explicit=False)),

        ("Exploration user",
         UserProfile(favorite_genre="rock", favorite_mood="energetic",
                     target_energy=0.8, discovery_preference=0.9)),

        ("Sparse input user",
         UserProfile(favorite_genre="", favorite_mood="", target_energy=0.5)),

        ("Artist-focused user",
         UserProfile(favorite_genre="jazz", favorite_mood="", target_energy=0.5,
                     favorite_artists=["Neon Echo"])
        )       
    ]
    results = [evaluate_profile(label, user, rec) for label, user in tests]

    print("\n======================")
    print(f"Overall: {sum(results)}/{len(results)} passed")
    print("======================\n")


if __name__ == "__main__":
    main()