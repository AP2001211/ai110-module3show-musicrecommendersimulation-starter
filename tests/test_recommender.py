import pytest
from src.recommender import Song, UserProfile, Recommender, Retriever, Ranker, Recommendation
from src.loader import derive_mood, map_genre_group, GENRE_GROUPS


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_song(
    id=1, title="Test Track", artist="Test Artist",
    genre="pop", genre_group="pop", mood="happy",
    energy=0.8, valence=0.8, danceability=0.8,
    acousticness=0.2, tempo_norm=0.5, popularity=60, explicit=False,
) -> Song:
    return Song(
        id=id, title=title, artist=artist,
        genre=genre, genre_group=genre_group, mood=mood,
        energy=energy, valence=valence, danceability=danceability,
        acousticness=acousticness, tempo_norm=tempo_norm,
        popularity=popularity, explicit=explicit,
    )


def make_catalog() -> list:
    return [
        make_song(1,  "Sunrise City",       "Neon Echo",      "pop",       "pop",       "happy",    0.82, 0.84, 0.79, 0.18, 0.485, 72),
        make_song(2,  "Midnight Coding",    "LoRoom",         "lofi",      "ambient",   "focused",  0.42, 0.56, 0.62, 0.71, 0.321, 58),
        make_song(3,  "Storm Runner",       "Voltline",       "rock",      "rock",      "energetic",0.91, 0.48, 0.66, 0.10, 0.625, 65),
        make_song(4,  "Library Rain",       "Paper Lanterns", "lofi",      "ambient",   "chill",    0.35, 0.60, 0.58, 0.86, 0.296, 54),
        make_song(5,  "Gym Hero",           "Max Pulse",      "pop",       "pop",       "happy",    0.93, 0.77, 0.88, 0.05, 0.542, 68),
        make_song(6,  "Spacewalk Thoughts", "Orbit Bloom",    "ambient",   "ambient",   "chill",    0.28, 0.65, 0.41, 0.92, 0.247, 45),
        make_song(7,  "Coffee Shop Stories","Slow Stereo",    "jazz",      "jazz",      "chill",    0.37, 0.71, 0.54, 0.89, 0.370, 51),
        make_song(8,  "Night Drive Loop",   "Neon Echo",      "synthwave", "electronic","energetic",0.75, 0.49, 0.73, 0.22, 0.452, 63),
        make_song(9,  "Focus Flow",         "LoRoom",         "lofi",      "ambient",   "focused",  0.40, 0.59, 0.60, 0.78, 0.329, 55),
        make_song(10, "Rooftop Lights",     "Indigo Parade",  "indie-pop", "pop",       "happy",    0.76, 0.81, 0.82, 0.35, 0.510, 70),
        make_song(11, "Golden Flow",        "Metro Wave",     "hip-hop",   "hip-hop",   "happy",    0.85, 0.72, 0.90, 0.08, 0.390, 74),
        make_song(12, "Sunday Drive",       "Cactus Road",    "country",   "country",   "chill",    0.55, 0.78, 0.65, 0.55, 0.431, 60),
        make_song(13, "Velvet Throne",      "R&B Soul",       "r-n-b",     "r&b",       "happy",    0.65, 0.67, 0.77, 0.28, 0.362, 66),
        make_song(14, "Thunderclap",        "Iron Fist",      "metal",     "metal",     "intense",  0.97, 0.31, 0.55, 0.04, 0.740, 59),
        make_song(15, "Moonrise Waltz",     "Clara Strings",  "classical", "classical", "chill",    0.30, 0.82, 0.35, 0.95, 0.288, 48),
    ]


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

class TestLoaderHelpers:

    def test_derive_mood_happy(self):
        assert derive_mood(0.8, 0.8) == "happy"

    def test_derive_mood_chill(self):
        assert derive_mood(0.7, 0.4) == "chill"

    def test_derive_mood_intense(self):
        assert derive_mood(0.3, 0.8) == "intense"

    def test_derive_mood_moody(self):
        assert derive_mood(0.2, 0.2) == "moody"

    def test_derive_mood_relaxed(self):
        assert derive_mood(0.5, 0.3) == "relaxed"

    def test_derive_mood_energetic(self):
        assert derive_mood(0.5, 0.8) == "energetic"

    def test_derive_mood_focused(self):
        assert derive_mood(0.5, 0.55) == "focused"

    def test_map_genre_group_exact(self):
        assert map_genre_group("rock") == "rock"
        assert map_genre_group("alt-rock") == "rock"
        assert map_genre_group("pop") == "pop"
        assert map_genre_group("k-pop") == "pop"

    def test_map_genre_group_unknown_returns_other(self):
        assert map_genre_group("zydeco") == "other"

    def test_all_114_genres_mapped(self):
        all_mapped = set(g for genres in GENRE_GROUPS.values() for g in genres)
        # Every mapped genre should return a non-"other" group
        for genre in all_mapped:
            assert map_genre_group(genre) != "other", f"{genre} should not be 'other'"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:

    def test_genre_group_filter_returns_matching_songs(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="pop", favorite_mood="", target_energy=0.8)
        candidates = Retriever().retrieve(user, songs)
        assert any(s.genre_group == "pop" for s in candidates)

    def test_explicit_filter_removes_explicit_songs(self):
        songs = make_catalog()
        songs[0] = make_song(1, explicit=True)
        user = UserProfile(favorite_genre="", favorite_mood="", target_energy=0.5,
                           allow_explicit=False)
        candidates = Retriever().retrieve(user, songs)
        assert all(not s.explicit for s in candidates)

    def test_explicit_allowed_keeps_explicit_songs(self):
        songs = [make_song(1, genre="rock", genre_group="rock", explicit=True)]
        user = UserProfile(favorite_genre="rock", favorite_mood="", target_energy=0.8,
                           allow_explicit=True)
        candidates = Retriever().retrieve(user, songs)
        assert any(s.explicit for s in candidates)

    def test_artist_songs_always_included(self):
        songs = make_catalog()
        # "Neon Echo" songs are in pop and electronic group
        user = UserProfile(favorite_genre="jazz", favorite_mood="", target_energy=0.5,
                           favorite_artists=["Neon Echo"])
        candidates = Retriever().retrieve(user, songs)
        assert any(s.artist == "Neon Echo" for s in candidates)

    def test_fallback_never_returns_empty(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="zydeco", favorite_mood="volcanic",
                           target_energy=0.5)
        candidates = Retriever().retrieve(user, songs)
        assert len(candidates) > 0


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

class TestRanker:

    def test_exact_genre_scores_higher_than_group_match(self):
        ranker = Ranker()
        user = UserProfile(favorite_genre="pop", favorite_mood="", target_energy=0.8)
        exact = make_song(1, genre="pop", genre_group="pop", energy=0.8, valence=0.5, danceability=0.5)
        group = make_song(2, genre="indie-pop", genre_group="pop", energy=0.8, valence=0.5, danceability=0.5)
        s_exact, _ = ranker.score(user, exact)
        s_group, _ = ranker.score(user, group)
        assert s_exact > s_group

    def test_mood_match_raises_score(self):
        ranker = Ranker()
        user = UserProfile(favorite_genre="", favorite_mood="happy", target_energy=0.5)
        happy = make_song(1, mood="happy", energy=0.5, valence=0.5, danceability=0.5)
        chill = make_song(2, mood="chill", energy=0.5, valence=0.5, danceability=0.5)
        s_happy, _ = ranker.score(user, happy)
        s_chill, _ = ranker.score(user, chill)
        assert s_happy > s_chill

    def test_valence_similarity_scored(self):
        ranker = Ranker()
        user = UserProfile(favorite_genre="", favorite_mood="", target_energy=0.5,
                           target_valence=0.9)
        high_val = make_song(1, valence=0.9, energy=0.5, danceability=0.5)
        low_val  = make_song(2, valence=0.1, energy=0.5, danceability=0.5)
        s_high, _ = ranker.score(user, high_val)
        s_low,  _ = ranker.score(user, low_val)
        assert s_high > s_low

    def test_danceability_similarity_scored(self):
        ranker = Ranker()
        user = UserProfile(favorite_genre="", favorite_mood="", target_energy=0.5,
                           target_danceability=0.9)
        high_dance = make_song(1, danceability=0.9, energy=0.5, valence=0.5)
        low_dance  = make_song(2, danceability=0.1, energy=0.5, valence=0.5)
        s_high, _ = ranker.score(user, high_dance)
        s_low,  _ = ranker.score(user, low_dance)
        assert s_high > s_low

    def test_artist_bonus_raises_score(self):
        ranker = Ranker()
        user_fav   = UserProfile(favorite_genre="", favorite_mood="", target_energy=0.5,
                                 favorite_artists=["Neon Echo"])
        user_plain = UserProfile(favorite_genre="", favorite_mood="", target_energy=0.5)
        song = make_song(1, artist="Neon Echo")
        s_fav,   _ = ranker.score(user_fav, song)
        s_plain, _ = ranker.score(user_plain, song)
        assert s_fav > s_plain

    def test_acoustic_penalty_lowers_score(self):
        ranker = Ranker()
        song = make_song(1, acousticness=0.9)
        user_dislikes = UserProfile(favorite_genre="", favorite_mood="", target_energy=0.5,
                                    likes_acoustic=False)
        user_likes    = UserProfile(favorite_genre="", favorite_mood="", target_energy=0.5,
                                    likes_acoustic=True)
        s_dislike, _ = ranker.score(user_dislikes, song)
        s_like,    _ = ranker.score(user_likes, song)
        assert s_like > s_dislike

    def test_rank_returns_descending_order(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
        ranked = Ranker().rank(user, songs)
        scores = [item[1] for item in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_no_preference_skips_genre_and_mood_scoring(self):
        ranker = Ranker()
        user = UserProfile(favorite_genre="", favorite_mood="", target_energy=0.5)
        _, reasons = ranker.score(user, make_song(1))
        assert not any("genre match" in r for r in reasons)
        assert not any("mood match" in r for r in reasons)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class TestRecommender:

    def test_returns_k_results(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
        assert len(Recommender(songs).recommend(user, k=3)) == 3

    def test_top_result_is_best_genre_mood_match(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
        results = Recommender(songs).recommend(user, k=5)
        assert results[0].song.genre_group == "pop"
        assert results[0].song.mood == "happy"

    def test_diversity_caps_genre_at_two(self):
        songs = make_catalog()
        # ambient group has 3 songs (lofi×2, ambient×1)
        user = UserProfile(favorite_genre="ambient", favorite_mood="", target_energy=0.35)
        results = Recommender(songs).recommend(user, k=5)
        counts: dict = {}
        for r in results:
            counts[r.song.genre_group] = counts.get(r.song.genre_group, 0) + 1
        assert all(c <= 2 for c in counts.values())

    def test_every_rec_has_confidence_and_tag(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="rock", favorite_mood="", target_energy=0.8)
        for r in Recommender(songs).recommend(user, k=5):
            assert 0 <= r.confidence <= 100
            assert r.tag in {"safe match", "explore pick", "partial match"}

    def test_safe_match_tag_when_genre_and_mood_match(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
        results = Recommender(songs).recommend(user, k=5)
        assert results[0].tag == "safe match"

    def test_explore_pick_with_high_discovery(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8,
                           discovery_preference=0.9)
        tags = [r.tag for r in Recommender(songs).recommend(user, k=5)]
        assert "explore pick" in tags

    def test_explanation_non_empty(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
        for r in Recommender(songs).recommend(user, k=5):
            assert r.explanation.strip() != ""

    def test_explicit_filter_honoured(self):
        songs = make_catalog()
        songs[0] = make_song(1, genre="pop", genre_group="pop", mood="happy",
                             energy=0.82, explicit=True)
        user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8,
                           allow_explicit=False)
        for r in Recommender(songs).recommend(user, k=5):
            assert not r.song.explicit

    def test_fallback_no_preferences(self):
        songs = make_catalog()
        user = UserProfile(favorite_genre="", favorite_mood="", target_energy=0.6)
        results = Recommender(songs).recommend(user, k=5)
        assert len(results) == 5

    def test_guardrails_single_genre_warning(self):
        songs = make_catalog()
        rec = Recommender(songs)
        fake = [
            Recommendation(song=s, score=3.0, confidence=60.0,
                           explanation="test", tag="safe match")
            for s in songs[:3]
            if s.genre_group == "ambient"
        ]
        # Force all to same group
        for r in fake:
            object.__setattr__(r.song, "genre_group", "ambient")
        warnings = rec.run_guardrails(fake)
        assert any("same genre group" in w for w in warnings)

    def test_guardrails_low_confidence_warning(self):
        songs = make_catalog()
        rec = Recommender(songs)
        fake = [Recommendation(song=songs[0], score=0.4, confidence=5.0,
                               explanation="weak", tag="partial match")]
        warnings = rec.run_guardrails(fake)
        assert any("low confidence" in w for w in warnings)
