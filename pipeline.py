"""
LLM misinformation triage pipeline.

Reads political claims from the LIAR dataset, asks an LLM to triage each one as
structured JSON, and logs every outcome - including failures - for evaluation.

Two prompt versions exist so results can be compared:
  v1 - the original baseline prompt
  v2 - the corrected prompt (temporal framing, rating definitions, honest
       uncertainty, no invented sources) plus JSON mode and low reasoning effort

Usage:
    python pipeline.py --provider groq --limit 50                 # v1 baseline
    python pipeline.py --provider groq --limit 50 --version v2    # corrected
    python pipeline.py --provider groq --limit 50 --version v2 --holdout
"""
import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# Both providers expose an OpenAI-compatible API, so one client works for both.
# Model names change often - check each provider's docs and override in .env.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    },
}

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job", "state", "party",
    "barely_true_ct", "false_ct", "half_true_ct", "mostly_true_ct",
    "pants_fire_ct", "context",
]
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
Verdict = Literal["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

DEV_SAMPLE_SIZE = 50   # the first 50 claims we inspected and designed fixes on
DEV_SEED = 42


# ---------------------------------------------------------------- output schemas

class TriageV1(BaseModel):
    verdict: Verdict
    check_worthy: bool
    topic: str
    entities: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class TriageV2(BaseModel):
    verdict: Verdict
    knowledge: Literal["recalled", "inferred", "unknown"]
    topic: str
    entities: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


# ---------------------------------------------------------------- prompts

PROMPT_V1 = """You are a fact-checking triage assistant for a newsroom.
For the claim given, return ONLY a JSON object with these keys:
- "verdict": one of "pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"
- "check_worthy": true if a fact-checker should prioritise this claim
- "topic": a short topic label (e.g. "economy", "health care")
- "entities": list of people, organisations or places named in the claim
- "confidence": your confidence in the verdict, from 0 to 1
- "rationale": one or two sentences explaining the verdict
Use only your general knowledge. If unsure, still give your best verdict and a low confidence."""

PROMPT_V2 = """You are a fact-checking triage assistant. You rate US political claims on PolitiFact's scale.

TIMING: Each claim was made between 2007 and 2016. Judge it against what was true
WHEN IT WAS MADE, not against later events. Do not use facts from after 2016.

RATING SCALE (use exactly these values for "verdict"):
- "true": accurate; nothing significant is missing.
- "mostly-true": accurate but needs clarification or additional information.
- "half-true": partially accurate but leaves out important details or takes things out of context.
- "barely-true": contains an element of truth but ignores critical facts that would give a different impression.
- "false": not accurate.
- "pants-fire": not accurate and makes a ridiculous claim.
Many real claims fall in the middle of the scale. Use "false" only when the claim is clearly inaccurate.

UNCERTAINTY: If you cannot recall evidence about a claim, that does NOT make it false.
Set "knowledge" to "unknown", give your best estimate, and keep "confidence" at 0.5 or below.

SOURCES: Do not cite specific studies, reports, dates or figures unless you are confident
they exist. Otherwise reason in general terms.

Return ONLY a JSON object with these keys:
- "verdict": one of the six ratings above
- "knowledge": "recalled" (you remember specific facts about this claim), "inferred" (you reasoned
  from general knowledge), or "unknown" (you have no relevant knowledge)
- "topic": a short topic label (e.g. "economy", "health care")
- "entities": people, organisations or places named IN THE CLAIM TEXT (not the speaker)
- "confidence": probability, from 0 to 1, that your verdict is on the correct side of the scale
  (true-leaning vs false-leaning)
- "rationale": one or two sentences explaining the verdict"""

class TriageV3(TriageV2):
    verdict: Literal["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true", "unverifiable"]


PROMPT_V3 = PROMPT_V2.replace(
    """UNCERTAINTY: If you cannot recall evidence about a claim, that does NOT make it false.
Set "knowledge" to "unknown", give your best estimate, and keep "confidence" at 0.5 or below.""",
    """UNCERTAINTY: If you cannot recall evidence about a claim, that does NOT make it false.
If you cannot judge the claim from what you know, use the verdict "unverifiable" instead of guessing,
and set "knowledge" to "unknown". Use "unverifiable" only when you genuinely cannot judge.""",
).replace(
    '- "verdict": one of the six ratings above',
    '- "verdict": one of the six ratings above, or "unverifiable"',
)
assert PROMPT_V3 != PROMPT_V2

# Each ablation run differs from its parent in exactly ONE setting.
VERSIONS = {
    # baseline
    "v1": {"prompt": PROMPT_V1, "schema": TriageV1, "json_mode": False, "reasoning_effort": None,
           "with_context": False},
    # corrected prompt + JSON mode + low reasoning effort
    "v2": {"prompt": PROMPT_V2, "schema": TriageV2, "json_mode": True, "reasoning_effort": "low",
           "with_context": True},
    # ablation A: v2 with the default reasoning effort  (isolates reasoning effort, v2 prompt)
    "v2_default": {"prompt": PROMPT_V2, "schema": TriageV2, "json_mode": True, "reasoning_effort": None,
                   "with_context": True},
    # ablation B: v1 with low reasoning effort  (isolates reasoning effort, v1 prompt)
    "v1_low": {"prompt": PROMPT_V1, "schema": TriageV1, "json_mode": False, "reasoning_effort": "low",
               "with_context": False},
    # v3: v2 plus an explicit "unverifiable" option  (isolates abstention)
    "v3": {"prompt": PROMPT_V3, "schema": TriageV3, "json_mode": True, "reasoning_effort": "low",
           "with_context": True},
}

REFUSAL_MARKERS = ("i can't", "i cannot", "i'm unable", "i am unable", "i won't", "not able to")
# Outcomes that are real model behaviour, so they count as finished.
# Errors on our side (client_error, api_error, gave_up_after_retries) get retried on the next run.
STOP = threading.Event()  # set when the daily limit is hit, so no worker starts another call
FINAL_STATUSES = {"ok", "refusal", "invalid_json", "schema_violation", "empty_response"}


# ---------------------------------------------------------------- data

def load_claims(path: str, limit: int, holdout: bool = False, seed: int = DEV_SEED) -> pd.DataFrame:
    """Sample claims. holdout=True draws from claims NOT in the 50-claim dev sample,
    so fixes designed on the dev sample can be tested on unseen claims."""
    # dtype=str stops pandas turning the labels "true"/"false" into booleans.
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS, quoting=3,
                     dtype=str, keep_default_na=False)
    if holdout:
        dev_ids = set(df.sample(n=min(DEV_SAMPLE_SIZE, len(df)), random_state=DEV_SEED)["id"])
        df = df[~df["id"].isin(dev_ids)]
        seed = seed + 1
    if limit < len(df):
        df = df.sample(n=limit, random_state=seed)
    return df


def build_user_message(row: dict, with_context: bool) -> str:
    msg = f"Claim: {row['statement']}\nSpeaker: {row['speaker']}"
    if with_context and row.get("context"):
        msg += f"\nContext (where it was said): {row['context']}"
    return msg


# ---------------------------------------------------------------- model call

def classify_failure(raw: str) -> str:
    """Label a non-empty response that didn't produce valid JSON."""
    if any(m in raw.lower() for m in REFUSAL_MARKERS):
        return "refusal"
    return "invalid_json"


def extract_json(raw: str) -> dict:
    """Models often wrap JSON in ```json fences or add chatter; strip that."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    return json.loads(raw[start : end + 1])


def retry_wait(e: Exception, attempt: int) -> float:
    """Use the server's Retry-After hint when given, else exponential backoff with jitter."""
    try:
        hint = float(e.response.headers.get("retry-after"))
        return min(hint + random.random(), 60)
    except Exception:
        return min(2 ** attempt + random.random(), 60)


def triage_claim(client: OpenAI, model: str, row: dict, version: str = "v1", max_retries: int = 6) -> dict:
    cfg = VERSIONS[version]
    record = {
        "id": row["id"], "gold_label": row["label"], "statement": row["statement"],
        "model": model, "version": version, "status": None, "output": None, "raw": None,
        "error": None, "retries": [], "latency_s": None, "attempts": 0,
        "prompt_tokens": None, "completion_tokens": None,
    }
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": cfg["prompt"]},
            {"role": "user", "content": build_user_message(row, cfg["with_context"])},
        ],
        "temperature": 0,
        "timeout": 60,
    }
    if cfg["json_mode"]:
        kwargs["response_format"] = {"type": "json_object"}
    if cfg["reasoning_effort"]:
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]

    last_error = None
    for attempt in range(1, max_retries + 1):
        if STOP.is_set():
            record["status"] = "not_run"
            return record
        record["attempts"] = attempt
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            record["latency_s"] = round(time.time() - t0, 2)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                record["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
                record["completion_tokens"] = getattr(usage, "completion_tokens", None)
            raw = resp.choices[0].message.content or ""
            record["raw"] = raw
            if not raw.strip():
                # Reasoning models sometimes return no content. Retry once or twice.
                record["retries"].append("empty_response")
                if attempt < min(3, max_retries):
                    continue
                record["status"] = "empty_response"
                return record
            try:
                record["output"] = cfg["schema"](**extract_json(raw)).model_dump()
                record["status"] = "ok"
            except json.JSONDecodeError:
                record["status"] = classify_failure(raw)
            except ValidationError as e:
                record["status"] = "schema_violation"
                record["error"] = str(e)[:500]
            return record
        except RateLimitError as e:
            if "per day" in str(e).lower():
                # Daily quota used up: retrying is pointless until tomorrow. Stop fast.
                STOP.set()
                record["status"] = "daily_limit"
                record["error"] = str(e)[:500]
                return record
            record["retries"].append("RateLimitError")
            last_error = f"RateLimitError: {e}"[:500]
            time.sleep(retry_wait(e, attempt))
        except APITimeoutError as e:
            # Transient: wait, then retry. Kept as history, not as the final error.
            record["retries"].append(f"{type(e).__name__}")
            last_error = f"{type(e).__name__}: {e}"[:500]
            time.sleep(retry_wait(e, attempt))
        except APIError as e:
            msg = str(e)
            if "json_validate_failed" in msg:
                # JSON mode: the provider rejected the model's malformed JSON.
                record["status"] = "invalid_json"
            else:
                record["status"] = "api_error"
            record["error"] = msg[:500]
            return record
        except Exception as e:
            # Anything unexpected: log it for this claim and keep the batch running.
            record["status"] = "client_error"
            record["error"] = f"{type(e).__name__}: {e}"[:500]
            return record
    record["status"] = "gave_up_after_retries"
    record["error"] = last_error  # e.g. a daily token limit message, so the cause is visible
    return record


# ---------------------------------------------------------------- batch run

def results_path(provider: str, version: str = "v1", holdout: bool = False) -> Path:
    """Where results are saved. v1 keeps the original filename so earlier runs are reused."""
    model = PROVIDERS[provider]["model"].replace("/", "-")
    suffix = ("" if version == "v1" else f"_{version}") + ("_holdout" if holdout else "")
    return Path("results") / f"{provider}_{model}{suffix}.jsonl"


def already_done(out_path: Path) -> set:
    """Cache: claim IDs with a final result, so a crashed run can resume.
    Retried claims are appended again; evaluation keeps the last record per ID."""
    if not out_path.exists():
        return set()
    return {r["id"] for r in read_jsonl(out_path) if r.get("status") in FINAL_STATUSES}


def read_jsonl(path: Path) -> list:
    """Read results, skipping any line cut short by an interrupted run."""
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a half-written last line; that claim will simply be redone
    return records


def run(provider: str = "groq", data: str = "data/test.tsv", limit: int = 50, workers: int = 2,
        version: str = "v1", holdout: bool = False) -> Path:
    """Run the pipeline. Callable from the command line (main) or a notebook."""
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["key_env"])
    if not api_key:
        raise SystemExit(f"{cfg['key_env']} not found. Add it to your .env file.")
    # Ask for gzip only: some installs (e.g. Anaconda) ship an outdated brotli/zstd
    # decoder that crashes on compressed replies.
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key,
                    default_headers={"Accept-Encoding": "gzip"})
    out_path = results_path(provider, version, holdout)
    out_path.parent.mkdir(exist_ok=True)

    claims = load_claims(data, limit, holdout=holdout)
    done = already_done(out_path)
    todo = [r for r in claims.to_dict("records") if r["id"] not in done]
    print(f"[{version}{' holdout' if holdout else ''}] {len(done)} cached, "
          f"{len(todo)} to process with {cfg['model']}")

    STOP.clear()
    counts = {}
    with ThreadPoolExecutor(max_workers=workers) as pool, out_path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(triage_claim, client, cfg["model"], r, version) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if rec["status"] == "not_run":
                continue  # cancelled after the daily limit; nothing to save
            f.write(json.dumps(rec) + "\n")
            f.flush()  # write as we go, so nothing is lost on a crash
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {counts}")
            if rec["status"] == "daily_limit":
                for other in futures:
                    other.cancel()  # don't start any more claims
                print(f"  STOPPED at {i}/{len(todo)}: the provider's daily limit was reached.")
                print(f"  {rec['error'][:200]}")
                print("  Everything finished so far is saved. Run again tomorrow to continue.")
                break

    print(f"Done. Results in {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=PROVIDERS, default="groq")
    ap.add_argument("--data", default="data/test.tsv")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--version", choices=VERSIONS, default="v1")
    ap.add_argument("--holdout", action="store_true", help="sample claims outside the 50-claim dev set")
    args = ap.parse_args()
    run(args.provider, args.data, args.limit, args.workers, args.version, args.holdout)


if __name__ == "__main__":
    main()
