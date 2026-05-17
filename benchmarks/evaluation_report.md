# QueryPilot Evaluation Report

## Summary Metrics

| Metric | Value |
| --- | --- |
| Total questions | 30 |
| SQL execution success rate | 90.0% |
| Column accuracy (of successful) | 51.9% |
| Row count accuracy (of successful) | 81.5% |
| **Semantic accuracy (LLM-as-judge)** | **96.2%** (26 judged) |
| Average semantic score | 0.98 / 1.0 |
| Average execution time | 22.2ms |
| Average retries | 0.0 |

## By Difficulty

| Difficulty | Total | Success Rate |
| --- | --- | --- |
| easy | 10 | 100.0% |
| medium | 12 | 100.0% |
| hard | 8 | 62.5% |

## By Category

| Category | Total | Success Rate |
| --- | --- | --- |
| aggregation | 15 | 93.3% |
| filtering | 4 | 100.0% |
| ranking | 4 | 100.0% |
| complex | 7 | 71.4% |

## Per-Question Results

| ID | Difficulty | Category | Success | Rows | Cols OK | Time (ms) | Retries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | easy | aggregation | ✅ | 1 | ✅ | 11 | 0 |
| 2 | easy | aggregation | ✅ | 1 | ❌ | 25 | 0 |
| 3 | easy | aggregation | ✅ | 1 | ❌ | 4 | 0 |
| 4 | easy | aggregation | ✅ | 1 | ✅ | 14 | 0 |
| 5 | easy | aggregation | ✅ | 1 | ✅ | 22 | 0 |
| 6 | easy | filtering | ✅ | 1 | ✅ | 13 | 0 |
| 7 | easy | ranking | ✅ | 10 | ✅ | 30 | 0 |
| 8 | easy | ranking | ✅ | 1 | ❌ | 28 | 0 |
| 9 | easy | aggregation | ✅ | 1 | ✅ | 20 | 0 |
| 10 | easy | filtering | ✅ | 20 | ❌ | 14 | 0 |
| 11 | medium | aggregation | ✅ | 10 | ✅ | 20 | 0 |
| 12 | medium | aggregation | ✅ | 113 | ✅ | 22 | 0 |
| 13 | medium | aggregation | ✅ | 113 | ✅ | 23 | 0 |
| 14 | medium | ranking | ✅ | 10 | ✅ | 36 | 0 |
| 15 | medium | aggregation | ✅ | 113 | ✅ | 23 | 0 |
| 16 | medium | filtering | ✅ | 10 | ✅ | 17 | 0 |
| 17 | medium | aggregation | ✅ | 3 | ❌ | 25 | 0 |
| 18 | medium | filtering | ✅ | 20 | ✅ | 16 | 0 |
| 19 | medium | aggregation | ✅ | 20 | ❌ | 22 | 0 |
| 20 | medium | aggregation | ✅ | 113 | ❌ | 21 | 0 |
| 21 | medium | aggregation | ✅ | 113 | ❌ | 19 | 0 |
| 22 | medium | ranking | ✅ | 15 | ✅ | 36 | 0 |
| 23 | hard | complex | ✅ | 20 | ❌ | 37 | 0 |
| 24 | hard | complex | ✅ | 339 | ❌ | 34 | 0 |
| 25 | hard | complex | ✅ | 20 | ❌ | 20 | 0 |
| 26 | hard | complex | ✅ | 226 | ❌ | 23 | 0 |
| 27 | hard | complex | ✅ | 20 | ❌ | 24 | 0 |
| 28 | hard | aggregation | ❌ | — | — | 0 | 0 |
| 29 | hard | complex | ❌ | — | — | 0 | 0 |
| 30 | hard | complex | ❌ | — | — | 0 | 0 |

## Failure Analysis

### Q28: Which musical keys are associated with the highest average valence (happiness)?
- **Difficulty**: hard | **Category**: aggregation
- **Expected SQL**: `SELECT kn.key_name, ROUND(AVG(t.valence), 3) AS avg_valence, COUNT(*) AS track_count FROM tracks t J...`
- **Generated SQL**: `None...`
- **Error**: No LLM API key found (or all providers exhausted). Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY

### Q29: Find artists who appear in more than 3 different genres
- **Difficulty**: hard | **Category**: complex
- **Expected SQL**: `SELECT artist_name, COUNT(DISTINCT genre_name) AS genre_count, STRING_AGG(DISTINCT genre_name, ', ' ...`
- **Generated SQL**: `None...`
- **Error**: No LLM API key found (or all providers exhausted). Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY

### Q30: What is the audio fingerprint of the most popular genre? Show average values for all audio features.
- **Difficulty**: hard | **Category**: complex
- **Expected SQL**: `SELECT genre_name, ROUND(AVG(danceability), 3) AS avg_danceability, ROUND(AVG(energy), 3) AS avg_ene...`
- **Generated SQL**: `None...`
- **Error**: No LLM API key found (or all providers exhausted). Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY
