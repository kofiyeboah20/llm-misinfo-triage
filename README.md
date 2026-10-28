# LLM-Misinfo-Triage

An LLM pipeline that triages political claims, and an evaluation harness that measures where it fails.

Claims come from the [LIAR benchmark](https://aclanthology.org/P17-2067/) (Wang, 2017): 12.8K short
statements by US politicians, each rated by PolitiFact journalists on a six-point scale from
`pants-fire` to `true`. The model (`openai/gpt-oss-20b`, served by Groq through an OpenAI-compatible
API) returns a structured verdict, a confidence score, a topic, the entities named in the claim, and a rationale.

The project is about **evaluation**: diagnosing failure modes, fixing them, and measuring honestly
whether each fix helped, including when it didn't.

## Key findings

| | v1 baseline | v2 corrected |
|---|---|---|
| Empty replies (development + held-out sets) | 7 | **0** |
| Rationales citing events after 2016 | 14 | **4** |
| Median output tokens per call (held-out) | 466 | **173** |
| Share of verdicts "false" (labels: 17–31%) | 67–73% | 75–81% |
| Binary accuracy, held-out 50 claims (majority baseline 60–65%) | 56.2% | 53.2% |

1. **The baseline had six identifiable failure modes:** a strong bias toward "false"; overconfidence
   (13 of 45 answers wrong at 0.9+ confidence); judging 2007–2016 claims with later knowledge; treating
   "I can't recall this" as "false"; untraceable citations; and empty replies.
2. **Empty replies have a mechanical cause.** Every empty reply used exactly 2,048 output tokens: the
   model spent its whole output budget on hidden reasoning. Lowering reasoning effort removed them.
3. **The corrections fixed reliability but hurt accuracy.** Answer by answer, v2 fixed 2 claims and
   broke 11 across 93 paired claims (exact sign test, p ≈ 0.02).
4. **Without retrieved evidence, neither version beats a majority-class baseline on held-out claims.**
   This matches published work showing LLM fact-checkers need retrieved context
   ([Quelle & Bovet, 2023](https://arxiv.org/abs/2310.13549)).
5. **v2 changed three things at once** (prompt, JSON mode and reasoning effort), so notebook 3 ran
   one-factor-at-a-time ablations on the same 50 development claims.

## Ablation: what caused the accuracy drop?

| Configuration | Binary acc. (baseline 51%) | Macro-F1 | Calibration error | Failures | Median output tokens |
|---|---|---|---|---|---|
| v1 (default effort) | 60.0% | 0.58 | 0.28 | 5 empty | n/a |
| v1, low effort | 50.0% | 0.39 | 0.36 | 0 | 176 |
| v2 (low effort) | 45.8% | 0.39 | 0.50 | 2 | 175 |
| **v2, default effort** | **62.2%** | **0.62** | **0.24** | 5 invalid JSON | 491 |
| v3 (v2 + "unverifiable" option) | 50.0% | 0.50 | 0.43 | 0 (abstained on 4) | 192 |

- **Low reasoning effort caused the drop, not the prompt rewrite.** Raising effort under the v2 prompt
  fixed 8 claims and broke 2. Lowering it under the v1 prompt fixed 1 and broke 6. Changing only the
  prompt (both at low effort) fixed 3 and broke 6, which is no clear effect (p = 0.51). Each comparison
  alone is not significant at n = 50, but both point the same way.
- **There is a trade-off.** More reasoning gives better verdicts and better calibration, but costs about
  2.8 times the tokens and brings failures back. The failures look like the same output-budget
  exhaustion that caused the empty replies, but this is not confirmed: the provider returned no token counts for them.
- **Abstention did not help yet.** v3 used "unverifiable" on 4 of 50 claims, but those were not
  mainly the claims v2 got wrong.

## The pre-registered decision, and what it chose

Before seeing the ablation results, I fixed a rule: among runs with a failure rate of 5% or less, pick
the highest binary macro-F1, with ties going to the cheaper run. The rule **excluded the best run
(v2 at default effort, 10% failures)** and chose **v1 at low effort**. On 337 claims never used for
any decision (a random 400-claim sample, minus development and held-out claims):

| Metric | v1, low effort |
|---|---|
| Binary accuracy | 46.3% ± 5.3 (majority baseline 56.4%) |
| Verdicts "false" vs actual | 78% vs 21% |
| Recall on true-leaning claims | 16% |
| Calibration error (ECE) | 0.38 |
| Failures / median output tokens | 0 / 178 |

**This is the primary result, and it is below baseline.** The rule treated failures as disqualifying,
even though they are retryable and probably fixable with a larger output-token limit. The better
accuracy of default reasoning effort is an *exploratory* finding until it is confirmed on fresh claims.

## What's in the repo

| File | Purpose |
|---|---|
| `pipeline.py` | The pipeline: prompt versions, API calls, schema validation, retries, caching, run logic |
| `evaluate.py` | Metrics: accuracy, macro-F1, baselines, calibration (ECE), bias, anachronisms, paired sign tests |
| `01_run_and_analyse.ipynb` | Notebook 1: run and analyse a single version |
| `02_correction_story.ipynb` | Notebook 2: baseline → diagnosis → fixes → rerun → held-out check |
| `03_ablation_and_full_run.ipynb` | Notebook 3: ablations, a decision rule stated before the results, and the larger run |
| `results/` | Every raw model response behind the numbers in this README, one JSON line per claim |
| `requirements.txt`, `.env.example` | Dependencies, and a template for your API key (the real `.env` is never committed) |

## Engineering choices

- **Structured outputs:** every reply is validated against a Pydantic schema. Anything else is logged
  with a precise status: `refusal`, `invalid_json`, `schema_violation`, `empty_response`, `api_error`.
- **Built to survive real conditions:** backoff that honours the server's `Retry-After` hints; an
  immediate stop when the provider's daily quota runs out; resumable runs (finished claims are cached and
  skipped, and our own errors are retried on the next run); crash-safe writes, line by line.
- **Fair evaluation:** a 50-claim development set for diagnosis, a separate 50-claim held-out set for
  checking fixes, and a final scoring set that excludes both. Paired comparisons use an exact sign test.
- **Reproducibility:** temperature 0, fixed sampling seeds, and the model name set in `.env`, because
  hosted models get retired. That happened during this project.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # add your GROQ_API_KEY
# put LIAR's test.tsv in data/  (https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
python pipeline.py --provider groq --limit 50 --version v2
```

Or open a notebook and choose **Run All**.

## License and data

The code is released under the MIT License (see `LICENSE`). The LIAR dataset is not included; download
it from its authors. Files in `results/` contain model outputs only, alongside claim text and labels from LIAR.

## Limitations

- Small samples (50 claims per comparison) and one model, with a single run per configuration.
- LIAR claims (2007–2016) are probably in the model's training data, so accuracy may partly reflect memory.
- PolitiFact's middle ratings are judgement calls; human fact-checkers also disagree on them.
- The model had no access to sources, and its rationales were not checked against real documents.

## Next

- Confirm the exploratory finding: v2 prompt, default reasoning effort and a higher output-token limit,
  tested on claims not yet used.
- Add retrieval (web or document search) and compare.
- Build a set of recent Ghanaian claims to test whether performance drops on local, post-training claims.

---
Emmanuel Asiedu Yeboah · MSc Data Science, University of Ghana · [yeboahjnr.techiesans.cloud](https://yeboahjnr.techiesans.cloud)
