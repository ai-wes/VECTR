# Adversarial citation suite summary

## Summary by condition

| condition   |   n_runs |   citations_present_pct |   citations_resolvable_pct |   fabricated_details_pct |   page_or_section_hallucinated_pct |   high_risk_pct |   mean_tool_calls |   mean_latency_s |   mean_resolver_success_rate |
|:------------|---------:|------------------------:|---------------------------:|-------------------------:|-----------------------------------:|----------------:|------------------:|-----------------:|-----------------------------:|
| baseline    |       22 |                 42.8571 |                    19.0476 |                 28.5714  |                            9.52381 |        23.8095  |           1.13636 |          7.20896 |                          1   |
| vectr       |       22 |                 81.8182 |                    45.4545 |                  4.54545 |                           22.7273  |         9.09091 |           2.54545 |         12.0994  |                          0.5 |


## Notes

- `resolver_success_rate` is only populated if you ran with `--resolver-audit`.

- LLM audit is a strict screener; resolver audit checks existence of PMID/DOI/arXiv/NCT.
