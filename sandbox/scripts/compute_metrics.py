import json
import csv
from pathlib import Path

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"


def collect():
    rows = []
    for metrics_path in RUNS_ROOT.rglob("metrics.json"):
        run_dir = metrics_path.parent
        parts = run_dir.relative_to(RUNS_ROOT).parts
        exp, mode, seed = parts[0], parts[1], parts[2]
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        row = {"experiment": exp, "mode": mode, "seed": seed}
        row.update(data)
        rows.append(row)
    return rows


def write_csv(rows):
    if not rows:
        return
    headers = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    out = RUNS_ROOT / "aggregate.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


def write_md(rows):
    if not rows:
        return
    headers = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    out = RUNS_ROOT / "aggregate.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "---|" * len(headers) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n")
    print(f"wrote {out}")


def main():
    rows = collect()
    write_csv(rows)
    write_md(rows)


if __name__ == "__main__":
    main()
