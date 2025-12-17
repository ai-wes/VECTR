## Reproducing the Experiments

```bash
python scripts/adversarial_citation_pipeline.py \
  --out sandbox/runs \
  --model gpt-4o-mini \
  --temperature 1.0 \
  --repeats 1

python scripts/aggregate_results.py \
  --runs sandbox/runs/runs.jsonl \
  --outdir sandbox/analysis

python scripts/cost_micro_model.py \
  --aggregate sandbox/analysis/aggregate.csv \
  --outdir sandbox/analysis
```
Expected outputs:

aggregate.csv

summary_by_condition.csv

break_even_table.md
