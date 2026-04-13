# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**Flux 1.0**

---

## 2. Intended Use

Flux is designed to suggest songs from a small catalog based on a user's stated preferences for genre, mood, and energy level. It is built for classroom exploration only — not for real users or production music apps.

It assumes the user can describe their taste with a single genre, a single mood, and a target energy level between 0 and 1. It does not learn from listening history or change over time.

**Not intended for:** real music streaming, personalized listening at scale, or any context where fairness or diversity of results matters to actual people.

---

## 3. How the Model Works

Think of it like a judge at a talent show with a fixed scorecard.

Every song in the catalog gets evaluated against the user's stated preferences using three criteria:

1. **Genre** — If the song's genre matches what the user said they like, it gets 2 points. This is the biggest factor.
2. **Mood** — If the song's mood matches, it gets 1 point.
3. **Energy closeness** — The model measures how far the song's energy level is from the user's target. A perfect match gives 1 point; a large gap gives closer to 0. This is a sliding scale.

There is also a small bonus or penalty based on whether the user likes acoustic-sounding songs.

Once every song has a score, they are sorted from highest to lowest. The top 5 are returned as recommendations, along with a plain-language explanation of why each one scored the way it did.

---

## 4. Data

The catalog contains **10 songs** stored in `data/songs.csv`. Each song has the following attributes: title, artist, genre, mood, energy (0.0–1.0), tempo in BPM, valence, danceability, and acousticness.

**Genres represented:** pop, lofi, rock, ambient, jazz, synthwave, indie pop  
**Moods represented:** happy, chill, intense, relaxed, moody, focused

No songs were added or removed from the original starter file.

**Gaps in the data:** The catalog has no hip-hop, classical, country, R&B, metal, blues, or reggae. It also skews toward chill/lo-fi (3 of 10 songs are lofi). A user who prefers jazz or ambient has very few options and will likely see the same 1–2 songs at the top regardless of their other preferences.

---

## 5. Strengths

- Works well for users who prefer **pop or lofi**, since those genres have the most songs in the catalog.
- The energy similarity scoring is smooth — it does not just reward exact matches, so a song at 0.78 energy still scores well for a user targeting 0.8.
- Results are **fully explainable**: every recommendation includes a plain-language reason (e.g., "genre match (+2.0), mood match (+1.0), energy similarity (+0.96)"), which makes it easy to understand and audit.
- The logic is simple enough to trace by hand, which is a real strength for a teaching tool.

---

## 6. Limitations and Bias

**Genre dominates everything.** A +2.0 genre bonus is large enough that a song with the right genre but the wrong mood and wrong energy will still usually beat a song with a perfect mood and energy fit in a different genre. This means users who want cross-genre discoveries will almost never get them.

**Mood and genre are exact string matches.** "indie pop" and "pop" score zero genre overlap even though they are closely related. A user who types "pop" will never get "indie pop" results, and vice versa.

**Underrepresented genres are nearly invisible.** The catalog has one jazz song, one ambient song, one synthwave song. A user preferring any of these gets only one genre-matched result at best, and the rest of the top 5 will be filled with whatever scores best on mood and energy regardless of genre fit.

**Valence, danceability, and tempo are ignored.** These features are loaded but not used in scoring. A user who wants something danceable has no way to express that, and the model has no way to reward it.

**All users get the same scoring formula.** Someone who cares deeply about mood but less about genre gets no way to express that — the weights are fixed for everyone.

---

## 7. Evaluation

Three user profiles were tested to check behavior:

| Profile | Expected behavior | What happened |
|---|---|---|
| `pop / happy / energy 0.8` | Sunrise City first | ✅ Correct — score 3.98, all three criteria matched |
| `lofi / chill / energy 0.4` | Midnight Coding or Library Rain first | ✅ Correct — both lofi/chill songs ranked 1 and 2 |
| `jazz / relaxed / energy 0.4` | Coffee Shop Stories first, then a mixed bag | ✅ Jazz song ranked #1, but slots 2–5 were filled by energy-close songs from unrelated genres |

The pytest tests confirmed that the OOP `Recommender` class correctly ranks the pop/happy song above the lofi/chill song for a pop-preferring user, and that every explanation is a non-empty string.

The most surprising result: for the jazz profile, the system recommended "Spacewalk Thoughts" (ambient, chill) as #2 because it happened to have close energy and mood. This shows the model can produce plausible-sounding but genre-irrelevant results when the catalog is thin.

---

## 8. Future Work

1. **Softer genre matching** — Use a genre similarity table (e.g., "indie pop" is close to "pop") instead of exact string equality. This would fix the "indie pop vs. pop" blind spot.
2. **User-adjustable weights** — Let the user say "I care more about mood than genre" and adjust the point values accordingly. This would make the system serve different personality types instead of forcing everyone into the same formula.
3. **Diversity enforcement** — Right now the top 5 can be dominated by a single genre. A diversity pass after ranking that ensures at least 2–3 different genres appear in results would make recommendations feel less repetitive.

---

## 9. Personal Reflection

Building Flux made it clear how much of a recommendation is actually just arithmetic dressed up to look like understanding. The system has no idea what music sounds like — it is just comparing strings and doing subtraction on decimals. Yet when the output says "Recommended because it matches your favorite genre and energy level is very close to your target," it feels like the system "gets" you. That gap between what is actually happening and what it feels like is the most important thing I took from this project.

The biggest learning moment was seeing how much the +2.0 genre weight shapes everything. I expected mood to matter more in practice, but a single number in the scoring formula made genre dominate the results for almost every test profile. Real systems face the same problem at a much larger scale — a small design choice in how features are weighted can quietly push the entire system in a direction that affects millions of users.

AI tools were genuinely helpful for scaffolding the structure quickly, but I had to verify the logic carefully. The tools suggested correct-looking code, but the decision about *what the weights should be* and *what biases that introduces* required me to actually run the system and look at the outputs. The code being right does not mean the design is right.

If I extended this project, I would try replacing the fixed-weight formula with a nearest-neighbor approach — find the songs whose full feature vector (genre, mood, energy, valence, danceability) is most similar to the user's profile vector, using actual distance math. That would use all the features in the CSV instead of ignoring most of them.
