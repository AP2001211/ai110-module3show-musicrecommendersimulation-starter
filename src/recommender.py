from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str        # raw genre (e.g. "alt-rock")
    genre_group: str  # mapped group (e.g. "rock")
    mood: str         # derived from valence + energy
    energy: float
    valence: float
    danceability: float
    acousticness: float
    tempo_norm: float  # tempo / 243, normalized 0–1
    popularity: int    # 0–100
    explicit: bool


@dataclass
class UserProfile:
    favorite_genre: str        # genre group name (e.g. "rock"), or "" for any
    favorite_mood: str         # one of the 7 derived moods, or "" for any
    target_energy: float       # 0–1
    target_valence: float = 0.5       # 0 = melancholic, 1 = uplifting
    target_danceability: float = 0.5  # 0 = calm, 1 = very danceable
    likes_acoustic: bool = False
    allow_explicit: bool = True
    favorite_artists: List[str] = field(default_factory=list)
    discovery_preference: float = 0.5  # 0 = familiar, 1 = exploratory


@dataclass
class Recommendation:
    song: Song
    score: float
    confidence: float   # 0–100
    explanation: str
    tag: str            # "safe match" | "explore pick" | "partial match"


# ---------------------------------------------------------------------------
# Retriever — Stage 1: candidate filtering
# ---------------------------------------------------------------------------

class Retriever:
    """
    Cuts the full catalog down to a manageable candidate pool before scoring.

    With a large catalog (100k+ songs), filtering by genre group + energy window
    reduces candidates to a few thousand, keeping recommendation latency low.
    """

    _MIN_POOL = 50          # widen search if genre pool is smaller than this
    _ENERGY_WINDOW = 0.30   # ±energy tolerance for candidate pre-filter
    
    def retrieve(self, user: UserProfile, songs: List[Song]) -> List[Song]:
        
        logger.info(f"Starting retrieval with {len(songs)} songs")

        pool = [s for s in songs if user.allow_explicit or not s.explicit]
        logger.info(f"After explicit filter: {len(pool)} songs")

        if user.favorite_genre:
            genre_pool = [s for s in pool if s.genre_group == user.favorite_genre]
            logger.info(f"Genre filter '{user.favorite_genre}': {len(genre_pool)} songs")

            if len(genre_pool) >= self._MIN_POOL:
                energy_pool = [
                    s for s in genre_pool
                    if abs(s.energy - user.target_energy) <= self._ENERGY_WINDOW
                ]
                logger.info(f"Energy filter: {len(energy_pool)} songs")
                pool = energy_pool if len(energy_pool) >= self._MIN_POOL else genre_pool
            else:
                logger.info("Genre too sparse, using fallback")
                fallback = [
                    s for s in pool
                    if (not user.favorite_mood or s.mood == user.favorite_mood)
                    and abs(s.energy - user.target_energy) <= self._ENERGY_WINDOW
                ]
                pool = fallback if len(fallback) >= self._MIN_POOL else pool

        logger.info(f"Final candidate pool size: {len(pool)}")
        return pool if pool else songs[: self._MIN_POOL]

# ---------------------------------------------------------------------------
# Ranker — Stage 2: feature-vector scoring
# ---------------------------------------------------------------------------

class Ranker:
    """
    Scores candidates using a 5-feature weighted vector plus genre, mood,
    acoustic, and artist bonuses.

    Max achievable score breakdown:
        genre exact match  +3.0
        mood match         +1.5
        energy similarity  +1.0   (weight 1.0)
        valence similarity +0.8   (weight 0.8)
        danceability sim.  +0.8   (weight 0.8)
        acoustic bonus     +0.5
        artist bonus       +0.5
        ──────────────────────
        MAX                 8.1
    """

    _MAX_RAW_SCORE: float = 8.1

    def score(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Return (raw_score, reasons) for one song."""
        s = 0.0
        reasons: List[str] = []

        # --- Genre (exact = +3.0, same group = +1.5) ---
        if user.favorite_genre:
            if song.genre.lower() == user.favorite_genre.lower():
                s += 3.0
                reasons.append(f"exact genre match ({song.genre})")
            elif song.genre_group == user.favorite_genre:
                s += 1.5
                reasons.append(f"related genre ({song.genre} ≈ {user.favorite_genre})")

        # --- Mood (+1.5) ---
        if user.favorite_mood and song.mood == user.favorite_mood:
            s += 1.5
            reasons.append(f"mood match ({song.mood})")

        # --- Energy similarity (0–1.0) ---
        e_sim = round(1.0 - abs(song.energy - user.target_energy), 2)
        s += e_sim
        if e_sim >= 0.85:
            reasons.append(f"energy is a strong fit ({e_sim:.0%})")
        elif e_sim >= 0.70:
            reasons.append(f"energy is close ({e_sim:.0%})")

        # --- Valence similarity (0–0.8) ---
        v_sim = round((1.0 - abs(song.valence - user.target_valence)) * 0.8, 2)
        s += v_sim
        if v_sim >= 0.68:
            reasons.append("mood tone is a strong fit")
        elif v_sim >= 0.56:
            reasons.append("mood tone is close")

        # --- Danceability similarity (0–0.8) ---
        d_sim = round((1.0 - abs(song.danceability - user.target_danceability)) * 0.8, 2)
        s += d_sim
        if d_sim >= 0.68:
            reasons.append("groove level matches well")

        # --- Acoustic preference (±0.5) ---
        if song.acousticness >= 0.7:
            if user.likes_acoustic:
                s += 0.5
                reasons.append("acoustic sound you enjoy")
            else:
                s -= 0.5

        # --- Favorite artist bonus (+0.5) ---
        if user.favorite_artists and song.artist in user.favorite_artists:
            s += 0.5
            reasons.append(f"by {song.artist}, one of your favorites")

        # --- Discovery boost (nudges non-genre songs up when user wants exploration) ---
        if user.discovery_preference > 0.5 and user.favorite_genre:
            if song.genre_group != user.favorite_genre:
                boost = round((user.discovery_preference - 0.5) * 0.6, 2)
                s += boost

        return s, reasons

    def rank(
        self, user: UserProfile, candidates: List[Song]
    ) -> List[Tuple[Song, float, List[str]]]:
        """Score and sort candidates, highest first."""
        scored = [(song, *self.score(user, song)) for song in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# Recommender — orchestrates the full pipeline
# ---------------------------------------------------------------------------

class Recommender:
    """retrieve → rank → diversity → annotate"""

    def __init__(self, songs: List[Song]) -> None:
        self.songs = songs
        self._retriever = Retriever()
        self._ranker = Ranker()

    def recommend(self, user: UserProfile, k: int = 5) -> List[Recommendation]:
        logger.info("Running recommendation pipeline")

        candidates = self._retriever.retrieve(user, self.songs)
        logger.info(f"Candidates retrieved: {len(candidates)}")

        ranked = self._ranker.rank(user, candidates)
        logger.info("Ranking completed")

        diverse = self._apply_diversity(ranked, max_per_genre=2, pool=k * 4)
        logger.info(f"After diversity filter: {len(diverse)}")

        results = [
            self._make_recommendation(user, song, score, reasons)
            for song, score, reasons in diverse[:k]
        ]

        logger.info(f"Returning {len(results)} recommendations")
        return results
    def run_guardrails(self, recs: List[Recommendation]) -> List[str]:
        warnings: List[str] = []

        if len(set(r.song.genre_group for r in recs)) == 1:
            warnings.append("All recommendations from same genre")

        low_conf = [r for r in recs if r.confidence < 30]
        if low_conf:
            warnings.append("Low confidence recommendations detected")

        if warnings:
            logger.warning(f"Guardrails triggered: {warnings}")
        else:
            logger.info("Guardrails passed")

        return warnings

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_diversity(
        self,
        ranked: List[Tuple[Song, float, List[str]]],
        max_per_genre: int,
        pool: int,
    ) -> List[Tuple[Song, float, List[str]]]:
        """Cap songs per genre_group to enforce variety."""
        genre_counts: Dict[str, int] = {}
        result: List[Tuple[Song, float, List[str]]] = []
        overflow: List[Tuple[Song, float, List[str]]] = []

        for item in ranked:
            g = item[0].genre_group
            count = genre_counts.get(g, 0)
            if count < max_per_genre:
                genre_counts[g] = count + 1
                result.append(item)
            else:
                overflow.append(item)
            if len(result) >= pool:
                break

        result.extend(overflow)
        return result

    def _make_recommendation(
        self,
        user: UserProfile,
        song: Song,
        score: float,
        reasons: List[str],
    ) -> Recommendation:
        confidence = round(min(score / Ranker._MAX_RAW_SCORE, 1.0) * 100, 1)
        tag = self._assign_tag(user, song, confidence)
        explanation = (
            "Recommended because: " + ", ".join(reasons) + "."
            if reasons
            else "Has some features that partially align with your preferences."
        )
        return Recommendation(
            song=song,
            score=round(score, 2),
            confidence=confidence,
            explanation=explanation,
            tag=tag,
        )

    def _assign_tag(self, user: UserProfile, song: Song, confidence: float) -> str:
        genre_match = bool(user.favorite_genre) and (
            song.genre.lower() == user.favorite_genre.lower()
            or song.genre_group == user.favorite_genre
        )
        mood_match = bool(user.favorite_mood) and song.mood == user.favorite_mood

        if genre_match and mood_match:
            return "safe match"
        if not genre_match and user.discovery_preference > 0.5:
            return "explore pick"
        if confidence >= 60:
            return "safe match"
        return "partial match"
