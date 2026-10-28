"""
Evaluation helpers for LIAR-Triage results. Pure pandas (no scikit-learn needed).
"""
import json
import re
from pathlib import Path

import pandas as pd

LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
FALSE_SIDE = {"pants-fire", "false", "barely-true"}
BIN_LABELS = ["false-leaning", "true-leaning"]
POST_2016_YEAR = re.compile(r"\b20(?:1[7-9]|2\d)\b")  # 2017-2029


def load_results(path) -> pd.DataFrame:
    """All records, keeping only the last record per claim (retries are appended)."""
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for l in f:
            if l.strip():
                try:
                    records.append(json.loads(l))
                except json.JSONDecodeError:
                    pass  # a line cut short by an interrupted run
    df = pd.DataFrame(records).drop_duplicates("id", keep="last").reset_index(drop=True)
    for col in ["retries", "prompt_tokens", "completion_tokens", "version"]:
        if col not in df:
            df[col] = None
    return df


def flatten_ok(df: pd.DataFrame) -> pd.DataFrame:
    """One row per valid answer, with the model's output fields as columns."""
    ok = df[df["status"] == "ok"].reset_index(drop=True)
    if ok.empty:
        raise ValueError("No usable answers in this results file - check the 'status' and 'error' columns.")
    out = pd.json_normalize(ok["output"].tolist())
    base = ok[["id", "gold_label", "statement", "latency_s"]]
    flat = pd.concat([base, out], axis=1)
    flat["gold_bin"] = to_binary(flat["gold_label"])
    flat["abstained"] = flat["verdict"] == "unverifiable"
    flat["pred_bin"] = to_binary(flat["verdict"]).where(~flat["abstained"], "abstain")
    flat["correct_6"] = flat["gold_label"] == flat["verdict"]
    flat["correct_bin"] = flat["gold_bin"] == flat["pred_bin"]
    flat["anachronism"] = flat["rationale"].fillna("").str.contains(POST_2016_YEAR)
    return flat


def answered(flat: pd.DataFrame) -> pd.DataFrame:
    """Rows where the model gave a verdict (not "unverifiable")."""
    return flat[~flat["abstained"]] if "abstained" in flat else flat


def to_binary(s: pd.Series) -> pd.Series:
    return s.map(lambda v: "false-leaning" if v in FALSE_SIDE else "true-leaning")


def per_class(y_true, y_pred, labels=LABELS) -> pd.DataFrame:
    rows = []
    for c in labels:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"label": c, "precision": p, "recall": r, "f1": f1, "support": int((y_true == c).sum())})
    return pd.DataFrame(rows).set_index("label")


def macro_f1(y_true, y_pred, labels=LABELS) -> float:
    present = [c for c in labels if (y_true == c).any() or (y_pred == c).any()]
    return per_class(y_true, y_pred, present)["f1"].mean() if present else float("nan")


def ece(flat: pd.DataFrame, bins=(0, 0.5, 0.7, 0.85, 1.0)) -> float:
    """Expected calibration error on binary correctness: the average gap between
    stated confidence and actual accuracy, weighted by how many answers fall in each bin.
    0 = perfectly calibrated. Abstentions are excluded."""
    flat = answered(flat)
    b = pd.cut(flat["confidence"], bins=list(bins), include_lowest=True)
    g = flat.groupby(b, observed=True).agg(n=("id", "count"), conf=("confidence", "mean"),
                                           acc=("correct_bin", "mean"))
    return float((g["n"] / g["n"].sum() * (g["conf"] - g["acc"]).abs()).sum())


def calibration_table(flat: pd.DataFrame, bins=(0, 0.5, 0.7, 0.85, 1.0)) -> pd.DataFrame:
    flat = answered(flat)
    b = pd.cut(flat["confidence"], bins=list(bins), include_lowest=True)
    return flat.groupby(b, observed=True).agg(answers=("id", "count"),
                                              mean_confidence=("confidence", "mean"),
                                              accuracy=("correct_bin", "mean")).round(2)


def health(df: pd.DataFrame) -> dict:
    """Pipeline reliability numbers."""
    n = len(df)
    retries = df["retries"].apply(lambda r: r if isinstance(r, list) else [])
    old_style = df["error"].fillna("").str.contains("429")  # v1 records stored retries in "error"
    rate_limited = retries.apply(lambda r: any("RateLimit" in x for x in r)) | old_style
    toks = (df["prompt_tokens"].fillna(0) + df["completion_tokens"].fillna(0))
    has_toks = toks.gt(0).any()
    return {
        "claims": n,
        "usable answers": int((df["status"] == "ok").sum()),
        "failure rate": round(float((df["status"] != "ok").mean()), 3),
        "empty replies": int(((df["status"].isin(["empty_response"])) |
                              ((df["status"] == "invalid_json") & (df["raw"].fillna("") == ""))).sum()),
        "rate-limited claims": int(rate_limited.sum()),
        "median latency (s)": float(df["latency_s"].median()),
        "mean tokens per call": round(float(toks[toks > 0].mean()), 0) if has_toks else None,
    }


def metrics(flat: pd.DataFrame) -> dict:
    """Quality numbers on the valid answers. Abstentions ("unverifiable") count as wrong in the
    accuracy and F1 figures; "selective accuracy" scores only the claims the model did answer."""
    ans = answered(flat)
    gold_majority_6 = flat["gold_label"].value_counts(normalize=True).max()
    gold_majority_bin = flat["gold_bin"].value_counts(normalize=True).max()
    return {
        "accuracy (6-class)": round(float(flat["correct_6"].mean()), 3),
        "macro-F1 (6-class)": round(float(macro_f1(flat["gold_label"], flat["verdict"])), 3),
        "baseline acc (6-class)": round(float(gold_majority_6), 3),
        "accuracy (binary)": round(float(flat["correct_bin"].mean()), 3),
        "macro-F1 (binary)": round(float(macro_f1(flat["gold_bin"], flat["pred_bin"], BIN_LABELS)), 3),
        "baseline acc (binary)": round(float(gold_majority_bin), 3),
        "share predicted 'false'": round(float((flat["verdict"] == "false").mean()), 3),
        "share actually 'false'": round(float((flat["gold_label"] == "false").mean()), 3),
        "calibration error (ECE)": round(ece(flat), 3),
        "abstention rate": round(float(flat["abstained"].mean()), 3),
        "selective accuracy (binary)": round(float(ans["correct_bin"].mean()), 3) if len(ans) else float("nan"),
        "confident (>=0.9) & wrong": int(((ans["confidence"] >= 0.9) & ~ans["correct_bin"]).sum()),
        "rationales citing post-2016 years": int(flat["anachronism"].sum()),
    }


def sign_test(fixed: int, broken: int) -> float:
    """Exact two-sided sign test on paired flips (McNemar's exact test)."""
    from math import comb
    n, k = fixed + broken, max(fixed, broken)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else 1.0


def paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Compare two flattened runs on the claims both answered (binary correctness)."""
    m = a[["id", "correct_bin"]].merge(b[["id", "correct_bin"]], on="id", suffixes=("_a", "_b"))
    fixed = int((~m["correct_bin_a"] & m["correct_bin_b"]).sum())
    broken = int((m["correct_bin_a"] & ~m["correct_bin_b"]).sum())
    return {"paired claims": len(m), "fixed": fixed, "broken": broken,
            "net": fixed - broken, "p (sign test)": round(sign_test(fixed, broken), 3)}
