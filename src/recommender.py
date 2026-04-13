from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score_song(self, user: UserProfile, song: Song) -> float:
        """Compute a numeric score for a Song against a UserProfile."""
        score = 0.0

        if song.genre.lower() == user.favorite_genre.lower():
            score += 2.0
        if song.mood.lower() == user.favorite_mood.lower():
            score += 1.0

        energy_similarity = 1.0 - abs(song.energy - user.target_energy)
        score += energy_similarity

        if user.likes_acoustic and song.acousticness >= 0.7:
            score += 0.5
        elif not user.likes_acoustic and song.acousticness >= 0.7:
            score -= 0.5

        return score

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs sorted from highest to lowest score."""
        scored = [(song, self._score_song(user, song)) for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why a song was recommended."""
        reasons = []

        if song.genre.lower() == user.favorite_genre.lower():
            reasons.append(f"matches your favorite genre ({song.genre})")
        if song.mood.lower() == user.favorite_mood.lower():
            reasons.append(f"matches your preferred mood ({song.mood})")

        energy_diff = abs(song.energy - user.target_energy)
        if energy_diff <= 0.1:
            reasons.append("energy level is very close to your target")
        elif energy_diff <= 0.25:
            reasons.append("energy level is fairly close to your target")

        if user.likes_acoustic and song.acousticness >= 0.7:
            reasons.append("has the acoustic sound you enjoy")

        if not reasons:
            reasons.append("has some features that partially align with your preferences")

        return "Recommended because it " + ", ".join(reasons) + "."


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return a list of dicts with typed values."""
    songs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                'id': int(row['id']),
                'title': row['title'],
                'artist': row['artist'],
                'genre': row['genre'],
                'mood': row['mood'],
                'energy': float(row['energy']),
                'tempo_bpm': float(row['tempo_bpm']),
                'valence': float(row['valence']),
                'danceability': float(row['danceability']),
                'acousticness': float(row['acousticness']),
            })
    print(f"Loaded songs: {len(songs)}")
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a single song dict against user_prefs; return (score, reasons)."""
    score = 0.0
    reasons = []

    # Genre match: +2.0 points
    if song['genre'].lower() == user_prefs.get('genre', '').lower():
        score += 2.0
        reasons.append(f"genre match (+2.0)")

    # Mood match: +1.0 point
    if song['mood'].lower() == user_prefs.get('mood', '').lower():
        score += 1.0
        reasons.append(f"mood match (+1.0)")

    # Energy similarity: up to +1.0 point (closer = higher score)
    target_energy = float(user_prefs.get('energy', 0.5))
    energy_sim = round(1.0 - abs(song['energy'] - target_energy), 2)
    score += energy_sim
    reasons.append(f"energy similarity (+{energy_sim})")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank all songs by score and return the top-k as (song, score, explanation) tuples."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "Because: " + ", ".join(reasons)
        scored.append((song, score, explanation))

    # sorted() returns a new list; .sort() modifies in place — using sorted() here
    # so the original list is not mutated
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return ranked[:k]
