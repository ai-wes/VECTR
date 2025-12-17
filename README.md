## What this repository demonstrates

This repository provides a reproducible, diagnostic evaluation of
agentic AI reliability controls under adversarial prompting.

It empirically measures:
- Verification overhead (tool calls, latency, optional tokens)
- Citation resolvability vs presence
- Fabrication and high-risk failure rates

It does NOT estimate:
- Real-world prevalence
- Downstream financial outcomes
- Production ROI

Economic analysis in the paper is a parameterized value-of-information
calculation using these measured overheads.


## Run Record Schema

Each run record contains:

- task_id: adversarial prompt identifier
- condition: baseline | vectr
- latency_s: wall-clock latency
- tool_call_count: number of external tool invocations
- llm_audit.parsed: structured reliability labels

This schema enables independent re-analysis without access to model internals.


schema:
```json
{
  "task_id": "...",
  "condition": "baseline|vectr",
  "model": { "name": "...", "temperature": ... },
  "latency_s": ...,
  "tool_call_count": ...,
  "assistant_response": "...",
  "llm_audit": {
    "parsed": {
      "citation_present": "...",
      "citation_resolvable": "...",
      "fabricated_details": "...",
      "page_or_section_hallucinated": "...",
      "overall_risk": "..."
    }
  }
}
```

Downstream costs are scenario parameters, not measurements.
