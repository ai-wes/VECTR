"""
Aggregate adversarial citation runs into CSV + markdown tables.

Input: runs.jsonl produced by adversarial_citation_pipeline.py
Output:
- aggregate.csv (one row per run)
- summary_by_condition.csv
- table_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def parse_audit(rec: Dict[str, Any]) -> Dict[str, Any]:
    audit_ok = safe_get(rec, ["audit", "ok"], False)
    parsed = safe_get(rec, ["audit", "parsed"], None)
    if not audit_ok or not isinstance(parsed, dict):
        return None
    return parsed


def resolver_success_rate(resolver: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(resolver, dict):
        return None
    resolved = resolver.get("resolved")
    if not isinstance(resolved, dict):
        return None

    items = []
    for k in ("pmid", "doi", "arxiv", "nct"):
        arr = resolved.get(k, [])
        if isinstance(arr, list):
            items.extend(arr)

    if not items:
        return None

    ok = sum(1 for x in items if isinstance(x, dict) and x.get("ok") is True)
    return ok / len(items)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, required=True, help="Path to runs.jsonl")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    args = ap.parse_args()

    runs_path = Path(args.runs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(runs_path)

    rows = []
    for r in records:
        audit = parse_audit(r)
        rows.append({
            "task_id": r.get("task_id"),
            "condition": r.get("condition"),
            "model": safe_get(r, ["model", "name"]),
            "temperature": safe_get(r, ["model", "temperature"]),
            "latency_s": r.get("latency_s"),
            "tool_call_count": r.get("tool_call_count"),
            "citation_present": audit["citation_present"],
            "citation_resolvable": audit["citation_resolvable"],
            "fabricated_details": audit["fabricated_details"],
            "page_or_section_hallucinated": audit["page_or_section_hallucinated"],
            "overall_risk": audit["overall_risk"],
            "resolver_success_rate": resolver_success_rate(r.get("resolver_audit")),
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "aggregate.csv", index=False)

    def pct(series, value) -> float:
        s = series.dropna()
        if len(s) == 0:
            return 0.0
        return float((s == value).mean() * 100.0)

    summaries = []
    for cond, g in df.groupby("condition"):
        summaries.append({
            "condition": cond,
            "n_runs": len(g),
            "citations_present_pct": pct(g["citation_present"], "yes"),
            "citations_resolvable_pct": pct(g["citation_resolvable"], "yes"),
            "fabricated_details_pct": pct(g["fabricated_details"], "yes"),
            "page_or_section_hallucinated_pct": pct(g["page_or_section_hallucinated"], "yes"),
            "high_risk_pct": pct(g["overall_risk"], "high"),
            "mean_tool_calls": float(g["tool_call_count"].mean()) if len(g) else 0.0,
            "mean_latency_s": float(g["latency_s"].mean()) if len(g) else 0.0,
            "mean_resolver_success_rate": float(g["resolver_success_rate"].dropna().mean())
            if g["resolver_success_rate"].notna().any() else None,
        })

    sdf = pd.DataFrame(summaries).sort_values("condition")
    sdf.to_csv(outdir / "summary_by_condition.csv", index=False)

    md = []
    md.append("# Adversarial citation suite summary\n")
    md.append("## Summary by condition\n")
    md.append(sdf.to_markdown(index=False))
    md.append("\n\n## Notes\n")
    md.append("- `resolver_success_rate` is only populated if you ran with `--resolver-audit`.\n")
    md.append("- LLM audit is a strict screener; resolver audit checks existence of PMID/DOI/arXiv/NCT.\n")
    (outdir / "table_summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
