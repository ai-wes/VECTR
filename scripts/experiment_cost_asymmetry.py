"""
Experiment C — Cost-asymmetry micro-model (parameterized, quantitative)

Consumes aggregate.csv and produces:
- estimated incremental $ cost per run for VECTR condition vs baseline
- break-even curves: required absolute risk reduction delta_p = C_v / C_e

This is non-empirical about downstream outcomes, BUT empirical about:
- additional tool calls / latency / (optionally) tokens if you extend logging
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", type=str, required=True, help="Path to aggregate.csv")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for tables")
    ap.add_argument("--usd_per_tool_call", type=float, default=0.002, help="Assumed cost per tool call (proxy)")
    ap.add_argument("--usd_per_second_latency", type=float, default=0.0, help="Optional: attach infra $/sec (often set 0)")
    # Token cost parameters are included as placeholders if/when you log token counts reliably
    ap.add_argument("--usd_per_1m_tokens", type=float, default=5.0, help="Assumed blended $ per 1M tokens (placeholder)")
    ap.add_argument("--baseline_tokens_col", type=str, default="", help="Optional column name for token count")
    ap.add_argument("--vectr_tokens_col", type=str, default="", help="Optional column name for token count")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.aggregate)

    # Empirical overhead components available right now: tool_call_count + latency_s
    # Token counts depend on your logging; script supports it if you add cols later.
    df["tool_cost_usd"] = df["tool_call_count"] * float(args.usd_per_tool_call)
    df["latency_cost_usd"] = df["latency_s"] * float(args.usd_per_second_latency)

    df["token_cost_usd"] = 0.0
    if args.baseline_tokens_col and args.vectr_tokens_col:
        # Not used in default pipeline; kept as plug-in.
        pass

    df["total_cost_usd"] = df["tool_cost_usd"] + df["latency_cost_usd"] + df["token_cost_usd"]

    # Compare baseline vs vectr average cost
    baseline = df[df["condition"] == "baseline"].copy()
    vectr = df[df["condition"] == "vectr"].copy()

    if len(baseline) == 0 or len(vectr) == 0:
        raise RuntimeError("aggregate.csv must include both baseline and vectr conditions.")

    mean_cost_baseline = float(baseline["total_cost_usd"].mean())
    mean_cost_vectr = float(vectr["total_cost_usd"].mean())
    delta_cost = mean_cost_vectr - mean_cost_baseline

    # Break-even curve: delta_p = C_v / C_e
    # Provide scenario costs (user-editable) — these are *parameters*, not claims.
    scenarios: Dict[str, float] = {
        "1 day trial delay (50k)": 50_000.0,
        "1 day trial delay (100k)": 100_000.0,
        "repeat experiment batch (25k)": 25_000.0,
        "repeat experiment batch (100k)": 100_000.0,
        "legal rework cycle (250k)": 250_000.0,
        "single diligence rework (500k)": 500_000.0,
    }

    rows: List[Dict[str, float]] = []
    for name, Ce in scenarios.items():
        # absolute probability reduction required (e.g., 0.002 == 0.2%)
        delta_p = delta_cost / Ce if Ce > 0 else None
        rows.append({
            "scenario": name,
            "downstream_cost_Ce_usd": Ce,
            "incremental_verification_cost_Cv_usd_per_run": delta_cost,
            "break_even_delta_p_abs": delta_p,
            "break_even_delta_p_pct": (delta_p * 100.0) if delta_p is not None else None,
        })

    out = pd.DataFrame(rows)

    # Save artifacts
    (outdir / "cost_micro_model_inputs.md").write_text(
        "\n".join([
            "# Experiment C — Cost-asymmetry micro-model (inputs)",
            "",
            f"- usd_per_tool_call: {args.usd_per_tool_call}",
            f"- usd_per_second_latency: {args.usd_per_second_latency}",
            f"- mean_cost_baseline_usd: {mean_cost_baseline:.6f}",
            f"- mean_cost_vectr_usd: {mean_cost_vectr:.6f}",
            f"- incremental_Cv_usd_per_run: {delta_cost:.6f}",
            "",
            "Break-even condition: delta_p > C_v / C_e",
            "",
        ]),
        encoding="utf-8",
    )

    out.to_csv(outdir / "break_even_table.csv", index=False)
    (outdir / "break_even_table.md").write_text(out.to_markdown(index=False), encoding="utf-8")

    print("=== Experiment C: Cost micro-model ===")
    print(f"Mean baseline cost/run: ${mean_cost_baseline:.6f}")
    print(f"Mean VECTR cost/run:    ${mean_cost_vectr:.6f}")
    print(f"Incremental Cv/run:     ${delta_cost:.6f}")
    print("\nBreak-even table written to:")
    print(outdir / "break_even_table.md")


if __name__ == "__main__":
    main()
