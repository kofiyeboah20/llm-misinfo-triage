# Results

Raw outputs behind every number in the main README. One JSON object per line, one line per model call.
Model: `openai/gpt-oss-20b` via Groq, temperature 0.

| File | Configuration | Claims | Used for |
|---|---|---|---|
| `groq_openai-gpt-oss-20b.jsonl` | v1: original prompt, default reasoning effort | 50 development | Baseline (notebook 2) |
| `groq_openai-gpt-oss-20b_v2.jsonl` | v2: corrected prompt, JSON mode, low effort | 50 development | Correction (notebook 2) |
| `groq_openai-gpt-oss-20b_holdout.jsonl` | v1 | 50 held-out | Held-out check (notebook 2) |
| `groq_openai-gpt-oss-20b_v2_holdout.jsonl` | v2 | 50 held-out | Held-out check (notebook 2) |
| `groq_openai-gpt-oss-20b_v2_default.jsonl` | Ablation A: v2 with default effort | 50 development | Ablation (notebook 3) |
| `groq_openai-gpt-oss-20b_v1_low.jsonl` | Ablation B: v1 with low effort | 400 (random sample) | Ablation + final run (notebook 3) |
| `groq_openai-gpt-oss-20b_v3.jsonl` | v3: v2 plus an "unverifiable" option | 124 | Ablation (notebook 3) |

Notes:
- **Development and held-out sets** are fixed 50-claim samples (seed 42), drawn without overlap.
- `v1_low` holds the final run: a 400-claim random sample. Its 50 development claims double as
  ablation B. Results are scored on the 337 claims that are in neither the development nor the held-out set.
- `v3` also holds 74 claims from a run that was started and then stopped. Notebook 3 analyses only its 50
  development claims.
- A claim retried after an error appears more than once. Analysis keeps the **last** record per claim.

## Fields

| Field | Meaning |
|---|---|
| `id`, `gold_label`, `statement` | LIAR claim ID, PolitiFact's rating and the claim text |
| `version` | Prompt and settings configuration (absent in the earliest v1 file) |
| `status` | `ok`, `refusal`, `invalid_json`, `schema_violation`, `empty_response`, `api_error`, `gave_up_after_retries`, `daily_limit` |
| `output` | The validated answer: `verdict`, `confidence`, `topic`, `entities`, `rationale` (+ `knowledge` in v2/v3) |
| `raw` | The model's text exactly as returned |
| `error`, `retries` | Final error message, and the history of retried errors |
| `latency_s`, `attempts`, `prompt_tokens`, `completion_tokens` | Cost and timing of the call |
