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



## Future Work & Extensions

This repository contains the initial VECTR validation suite. Several extensions 
are planned or in progress:

### 1. Semantic Faithfulness Audit
Currently, the resolver audit checks whether citations **resolve** (PMID exists, 
DOI valid, metadata matches). Future versions will add **entailment-based 
faithfulness scoring** to verify that the cited evidence actually **supports** 
the claim (NLI model integration).

**Status**: Planned for v2.0  
**Estimated effort**: 2-3 days (add entailment checker to `aggregate_adversarial_results.py`)

### 2. Multi-Model Validation
Current suite tests GPT-4 family models. We plan to expand to:
- Claude 3.5 (Anthropic)
- Llama 3.1 70B (Meta)
- Mistral Large

**Status**: In progress  
**Estimated API cost**: ~$300-500 for full sweep

### 3. Expanded Task Suite
22 adversarial tasks cover citation hallucinations, precision traps, and time-gating. 
Future expansions:
- Compound property claims (hERG liability, blood-brain barrier)
- Regulatory pathway precedent (orphan designation, fast-track)
- Competitive landscape (overlapping patents, trial discontinuations)

**Status**: Planned  
**Target**: 50-100 task suite for comprehensive coverage


### 4. Longitudinal Deployment Study
Track VECTR in production drug-development workflows:
- Error rate over time
- Abstention frequency and reviewer override rates
- Calibrated trust (do reviewers learn to trust non-abstained outputs?)

**Status**: Pilot in progress at 
GlassBox Bio 
www.glassbox-bio.com/research) 
Contact wes@glassbox-bio.com


**Expected publication**: Q2 2026

---

**Contributions welcome!** If you'd like to help expand the suite, validate on 
additional models, or contribute human annotations, please open an issue or PR.

