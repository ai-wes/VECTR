## Reproducing the Experiments


```bash
python scripts/adversarial_citation_pipeline.py \
  --out sandbox/runs \
  --model gpt-4o-mini \
  --temperature 1.0 \
  --repeats 1

python scripts/aggregate_adversarial_results.py \
  --runs sandbox/runs/runs.jsonl \
  --outdir sandbox/analysis

python scripts/experiment_cost_asymmetry.py \
  --runs sandbox/runs/runs.jsonl \
  --outdir sandbox/analysis
```
Expected outputs (in sandbox/analysis/):

    aggregate.csv

    summary_by_condition.csv

    break_even_sensitivity.md

    break_even_sensitivity.csv

    measured_overhead_summary.json

    table_summary.md
