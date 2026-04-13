# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

This version loads a 10-song catalog from a CSV file, scores each song against a user's stated preferences (genre, mood, and energy level), and returns the top 5 recommendations ranked by score. Each result includes a plain-language explanation of why it was chosen. The system is fully explainable — there are no hidden layers, just arithmetic applied to song features.

---

## How The System Works

Each `Song` is represented by: **genre, mood, energy, tempo_bpm, valence, danceability, acousticness**.

The `UserProfile` stores: **favorite_genre, favorite_mood, target_energy, likes_acoustic**.

### Algorithm Recipe

Every song in the catalog is scored against the user profile using these rules:

| Rule | Points |
|---|---|
| Genre matches `favorite_genre` | +2.0 |
| Mood matches `favorite_mood` | +1.0 |
| Energy similarity: `1.0 - abs(song.energy - target_energy)` | 0.0 – 1.0 |
| User likes acoustic AND song acousticness ≥ 0.7 | +0.5 |
| User dislikes acoustic AND song acousticness ≥ 0.7 | −0.5 |

After every song is scored, they are sorted from highest to lowest. The top `k` songs are returned as recommendations along with a plain-language explanation of why each one was chosen.

**Data flow:**
```
Input (user_prefs dict)
  → Loop: score every song in songs.csv using score_song()
  → Sort all (song, score, explanation) tuples by score descending
  → Output: top k recommendations printed to terminal
```
### Sample Output

![Terminal output](output.png)


### Potential Biases

- **Genre dominates**: A genre match adds +2.0, so songs in the right genre will almost always outrank better energy/mood fits from other genres.
- **Categorical matching only**: Genre and mood are exact string matches — "indie pop" and "pop" are treated as completely different even though they are similar.
- **Energy is the only numeric feature used**: Valence, danceability, and tempo are loaded but not scored, so a danceable song a user would love could still rank low.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

**Experiment 1: Default pop/happy profile (energy 0.8)**
Running the system with `genre: pop, mood: happy, energy: 0.8` gave Sunrise City as #1 with a score of 3.98 — all three criteria matched. Gym Hero came in at #2 (score 2.87) despite being "intense" mood, purely because of the genre match bonus. This confirmed that genre dominates the ranking.

**Experiment 2: Jazz profile (genre: jazz, mood: relaxed, energy: 0.4)**
Coffee Shop Stories ranked #1 as expected. But slots #2–5 were filled by lofi and ambient songs that had no genre relationship — they just had similar energy levels. This revealed that with only one song per rare genre, the system fills the rest of the list with energy-close songs regardless of genre fit.

**Experiment 3: What if genre weight dropped from 2.0 to 0.5?**
Mentally tracing the math: Sunrise City would go from 3.98 to ~2.48. Rooftop Lights (indie pop, happy) would go from 1.96 to ~1.96 — unchanged. The gap narrows significantly, and mood + energy matches from other genres would start competing more. The top 5 would be more diverse but genre matches would feel less "intentional."

**Experiment 4: Lofi/chill profile (genre: lofi, mood: chill, energy: 0.4)**
Midnight Coding and Library Rain both scored above 3.0 and took the top two slots. The remaining three spots went to ambient and jazz songs (chill/relaxed mood, low energy). This was the profile that felt most "correct" — the recommendations actually matched the vibe.

---

## Limitations and Risks

- **Tiny catalog**: With only 10 songs, users preferring less-represented genres (jazz, ambient, synthwave) get at most 1–2 genre-matched results. The rest of their top 5 is filler.
- **Genre dominates**: The +2.0 genre weight means a song with the right genre but the wrong mood and energy will usually outrank a better overall match in a different genre.
- **Exact string matching**: "indie pop" and "pop" score zero overlap. A user who types "pop" will never see Rooftop Lights in their genre-matched results even though it fits the vibe.
- **Most features go unused**: Valence, danceability, and tempo are loaded but never factored into scores. A user who wants something danceable has no way to express that preference.
- **No learning or history**: The system treats every run as if the user has never listened to anything. It cannot avoid recommending the same songs repeatedly or adapt based on what the user skipped.
- **Fixed weights for everyone**: Someone who cares deeply about mood but little about genre gets the exact same scoring formula as everyone else. There is no way to personalize the importance of each factor.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Building this system made it clear that a recommendation is just arithmetic with a friendly label on top. The model has no idea what music sounds like — it compares strings and subtracts decimals. But when it prints "Recommended because it matches your favorite genre and energy level is very close to your target," it genuinely feels like the system understood something. That gap between what is actually happening under the hood and what it feels like to the user is the most important thing this project taught me about AI systems.

The bias piece was equally eye-opening. I did not set out to make genre dominate everything — I just chose +2.0 as a reasonable starting weight. But that single number quietly shapes every result the system produces. A user who loves jazz gets a worse experience than a user who loves pop, not because of any intentional decision, but because the catalog happened to have more pop songs and genre happened to get the highest weight. Real recommenders face the exact same problem at a scale of millions of users, which is why the weight choices baked into these systems are not just technical decisions — they are design decisions with real consequences for who gets served well and who does not.


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

