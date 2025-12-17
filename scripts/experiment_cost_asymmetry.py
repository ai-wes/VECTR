"""
Experiment C — Cost-asymmetry micro-model (v2, coherent)
This script performs a parameterized value-of-information analysis.

Inputs:
- Measured verification overhead from aggregate.csv

Outputs:
- Break-even error reduction thresholds


This analysis does NOT use downstream outcome data.
It does NOT claim real-world cost savings.
"""

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def extract_total_tokens(rec: Dict[str, Any]) -> Optional[int]:
    """
    Tries to pull token usage from common autogen structures.
    Returns total_tokens if found, else None.
    """
    usage = rec.get("usage")

    # Case 1: usage is already a dict with total_tokens
    if isinstance(usage, dict):
        tt = usage.get("total_tokens")
        if isinstance(tt, int):
            return tt
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if isinstance(pt, int) and isinstance(ct, int):
            return pt + ct

    # Case 2: sometimes usage is embedded in assistant_response string (your logs show models_usage=RequestUsage(...))
    # We'll attempt a robust regex parse as fallback.
    text = rec.get("assistant_response", "") or ""
    m_pt = re.search(r"prompt_tokens\s*=\s*(\d+)", text)
    m_ct = re.search(r"completion_tokens\s*=\s*(\d+)", text)
    if m_pt and m_ct:
        return int(m_pt.group(1)) + int(m_ct.group(1))

    return None


def build_paired_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a paired dataframe keyed by task_id, comparing baseline vs vectr.
    Uses mean across repeats if multiple rows exist for same (task_id, condition).
    """
    rows = []
    for r in records:
        rows.append({
            "task_id": r.get("task_id"),
            "condition": r.get("condition"),
            "latency_s": r.get("latency_s"),
            "tool_call_count": r.get("tool_call_count"),
            "total_tokens": extract_total_tokens(r),
        })

    df = pd.DataFrame(rows)
    # Drop obviously broken rows without task_id/condition
    df = df.dropna(subset=["task_id", "condition"])

    # Average over repeats per task/condition
    g = df.groupby(["task_id", "condition"], as_index=False).agg({
        "latency_s": "mean",
        "tool_call_count": "mean",
        "total_tokens": "mean",  # may remain NaN if missing
    })

    # Pivot to baseline/vectr columns
    pivot = g.pivot(index="task_id", columns="condition")
    # Flatten columns
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    # Ensure both conditions exist
    needed = ["latency_s_baseline", "latency_s_vectr", "tool_call_count_baseline", "tool_call_count_vectr"]
    for col in needed:
        if col not in pivot.columns:
            raise RuntimeError(f"Missing required column after pivot: {col}")

    # Compute deltas
    pivot["delta_tool_calls"] = pivot["tool_call_count_vectr"] - pivot["tool_call_count_baseline"]
    pivot["delta_latency_s"] = pivot["latency_s_vectr"] - pivot["latency_s_baseline"]

    if "total_tokens_baseline" in pivot.columns and "total_tokens_vectr" in pivot.columns:
        pivot["delta_tokens"] = pivot["total_tokens_vectr"] - pivot["total_tokens_baseline"]
    else:
        pivot["delta_tokens"] = float("nan")

    return pivot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, required=True, help="Path to runs.jsonl")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument("--usd_per_tool_call", type=float, default=0.002, help="Assumed tool cost per call")
    ap.add_argument("--usd_per_1m_tokens", type=float, default=10.0, help="Assumed blended $ per 1M tokens")
    ap.add_argument("--usd_per_hour_time", type=float, default=0.0, help="Optional: monetize latency as $/hour (team/infra)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(Path(args.runs))
    paired = build_paired_df(records)

    # Empirical mean deltas
    mean_delta_tool_calls = float(paired["delta_tool_calls"].mean())
    mean_delta_latency_s = float(paired["delta_latency_s"].mean())

    # Tokens may be NaN if missing
    token_series = paired["delta_tokens"].dropna()
    mean_delta_tokens = float(token_series.mean()) if len(token_series) else None

    # Convert time cost
    usd_per_second_time = float(args.usd_per_hour_time) / 3600.0

    # Compute measured Cv components
    cv_tools = mean_delta_tool_calls * float(args.usd_per_tool_call)
    cv_time = mean_delta_latency_s * usd_per_second_time

    cv_tokens = 0.0
    if mean_delta_tokens is not None:
        usd_per_token = float(args.usd_per_1m_tokens) / 1_000_000.0
        cv_tokens = mean_delta_tokens * usd_per_token

    cv_measured_full = cv_tools + cv_tokens + cv_time

    # Build sensitivity table:
    # Rows = Ce tiers; Cols = Cv tiers (measured tools-only + measured full + optional larger tiers)
    ce_tiers = [
        (25_000.0, "repeat experiment batch"),
        (50_000.0, "~1 day delay (low)"),
        (100_000.0, "rework / delay"),
        (250_000.0, "legal/major rework"),
        (500_000.0, "diligence restart"),
        (1_000_000.0, "major program slip"),
        (100_000_000.0, "clinical-phase scale (anchor)"),
    ]

    # Cv tiers: include measured lower bound and a few scaled tiers
    cv_tiers = [
        ("Cv_tools_only_measured", cv_tools),
        ("Cv_full_measured", cv_measured_full),
        ("Cv_$0.10", 0.10),
        ("Cv_$1", 1.0),
        ("Cv_$10", 10.0),
    ]

    # Build grid: errors prevented per 1M packets = (Cv/Ce)*1e6
    grid_rows = []
    for ce, label in ce_tiers:
        row = {
            "Ce_usd": ce,
            "Ce_interpretation": label,
        }
        for name, cv in cv_tiers:
            row[name] = (cv / ce) * 1_000_000.0
        grid_rows.append(row)

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(outdir / "break_even_sensitivity.csv", index=False)
    (outdir / "break_even_sensitivity.md").write_text(grid.to_markdown(index=False), encoding="utf-8")

    # Write measured summary
    summary = {
        "n_tasks_paired": int(len(paired)),
        "usd_per_tool_call": float(args.usd_per_tool_call),
        "usd_per_1m_tokens": float(args.usd_per_1m_tokens),
        "usd_per_hour_time": float(args.usd_per_hour_time),
        "mean_delta_tool_calls": mean_delta_tool_calls,
        "mean_delta_latency_s": mean_delta_latency_s,
        "mean_delta_tokens": mean_delta_tokens,
        "Cv_tools_only_measured_usd": cv_tools,
        "Cv_time_component_usd": cv_time,
        "Cv_token_component_usd": cv_tokens,
        "Cv_full_measured_usd": cv_measured_full,
        "note": "Cv_tools_only is a conservative lower bound if token/time costs are set to 0 or missing.",
    }
    (outdir / "measured_overhead_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Experiment C (v2): Cost-asymmetry micro-model ===")
    print(f"Paired tasks: {len(paired)}")
    print(f"Measured Cv tools-only: ${cv_tools:.6f}")
    print(f"Measured Cv full:       ${cv_measured_full:.6f}")
    print("Wrote:")
    print(" -", outdir / "measured_overhead_summary.json")
    print(" -", outdir / "break_even_sensitivity.md")
    print(" -", outdir / "break_even_sensitivity.csv")


if __name__ == "__main__":
    main()
