# Adversarial citation suite summary

## Summary by condition

| condition   |   n_runs |   n_audited |   audit_parse_recovered |   audit_parse_failures |   citations_present_pct |   citations_resolvable_pct |   fabricated_details_pct |   page_or_section_hallucinated_pct |   high_risk_pct |   mean_tool_calls |   mean_latency_s |   mean_resolver_success_rate |
|:------------|---------:|------------:|------------------------:|-----------------------:|------------------------:|---------------------------:|-------------------------:|-----------------------------------:|----------------:|------------------:|-----------------:|-----------------------------:|
| baseline    |       22 |          22 |                       1 |                      0 |                 40.9091 |                    18.1818 |                 27.2727  |                            9.09091 |        22.7273  |           1.13636 |          7.20896 |                          1   |
| vectr       |       22 |          22 |                       0 |                      0 |                 81.8182 |                    45.4545 |                  4.54545 |                           22.7273  |         9.09091 |           2.54545 |         12.0994  |                          0.5 |


## Notes

- `resolver_success_rate` is only populated if you ran with `--resolver-audit`.

- LLM audit is a strict screener; resolver audit checks existence of PMID/DOI/arXiv/NCT.

- Audit parse recovery/failure counts are explicit; percentages exclude only failed parses.
