"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 40)
    print("  Top Recommendations")
    print("=" * 40)
    for i, rec in enumerate(recommendations, start=1):
        song, score, explanation = rec
        print(f"\n#{i}: {song['title']} by {song['artist']}")
        print(f"    Genre: {song['genre']} | Mood: {song['mood']} | Energy: {song['energy']}")
        print(f"    Score: {score:.2f}")
        print(f"    {explanation}")
    print("\n" + "=" * 40)


if __name__ == "__main__":
    main()
