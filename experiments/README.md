CAN: Here are **specific, programmable experiments** you can run to generate **real, auditable case-study data** (logs + artifacts) for the paper—without needing proprietary pharma data. The key trick is to build a **controlled “agentic diligence sandbox”** where you can deterministically inject the failure modes you talk about.

I’ll give you **6 experiments** that each yield:

- a **before/after** comparison (baseline agent vs TRACE-style gated agent),
- **hard evidence artifacts** (run logs, tool traces, claim lists, bindings),
- and a ready-to-write **case study narrative** (“what happened, why it mattered, what control would have prevented it”).

---

## Common harness (use for all experiments)

### A. Minimal components

1. **Evidence corpus** (local): a small set of documents with IDs + timestamps

   - `evidence/{id}.json` with `id`, `source_type`, `published_at`, `text`, `metadata`

2. **Tool simulator layer** (deterministic): functions like `search_pubmed`, `search_patents`, `get_trial_registry`, but backed by your local corpus index.
3. **Fault injector**: toggles to simulate outages, empty results, truncation, schema drift, wrong ranking.
4. **Claim extractor**: converts final report into a list of atomic claims (`claim_id`, `type`, `span`, `normalized_proposition`)
5. **Validators**:

   - ID existence validator (PMID/NCT/patent-id _existence in local index_)
   - Time-gating validator (no post-cutoff evidence)
   - Faithfulness validator (does cited snippet entail claim? even a heuristic works)
   - Contradiction validator (within/between domain sections)

6. **Logger**: emits your case-study artifacts.

### B. Artifacts to save (these become your case-study “exhibits”)

- `run_record.jsonl` (all tool calls + responses + params + timestamps)
- `final_output.md`
- `claims.json` (extracted claims)
- `bindings.json` (claim → evidence IDs)
- `validator_report.json` (per-claim pass/fail with reasons)
- `gates.json` (if running TRACE-style gates)
- `metrics.json` (ZCR, time-leak rate, contradiction survival, red-flag recall)

Once you have these, you can literally quote a few lines of the validator report as the “smoking gun.”

---

# Experiment 1 — Silent tool failure → “No blocking IP found”

### What it produces

A concrete case study where the baseline agent writes a confident “no blocking IP” paragraph **because the IP tool returned empty** (outage / timeout / bad query), while the gated agent fails loud.

### Protocol

1. Create an “IP corpus” of ~50 mock patents where **some are blocking** for a fictitious target/molecule.
2. Ask the agent: “Assess freedom-to-operate for Target X; list blocking patents and risks.”
3. Run two conditions:

   - **Baseline**: tool failure returns `[]` but agent continues
   - **Gated**: tool failure triggers “INSUFFICIENT_EVIDENCE / TOOL_FAILURE” section

### Fault injection

- 30% chance `search_patents()` returns:

  - `status="ok", results=[]` (most dangerous)
  - or `status="error", code="timeout"`

### Measurements

- **ZCR**: claims like “no blocking patents found” unsupported
- **Fail-loud compliance**: did it explicitly stop/downgrade on tool failure?
- **Dangerous completion rate**: `% runs where tool failure → confident negative claim`

### Case study story you’ll be able to write

> “The system reported ‘no blocking IP’ with high confidence; the run record shows the patent tool returned empty due to a timeout, and no retry occurred. The gated system halted and flagged the missing evidence.”

---

# Experiment 2 — Phantom identifiers: invented PMIDs / NCTs / patent numbers

### What it produces

A “fabricated identifier” case study with **hard proof**: the baseline output includes IDs that literally do not exist in your index (or don’t match any record), while the gated agent blocks.

### Protocol

1. Ask for “key supporting studies” and request **PMIDs and NCT IDs** explicitly.
2. Give the agent partial evidence that supports some claims but not others (to create pressure).
3. Run:

   - Baseline agent
   - Gated agent with an **ID resolver**: every identifier must resolve to an evidence object.

### Validator

- Parse output with regex:

  - PMIDs: `PMID:\s*\d{7,9}`
  - NCT: `NCT\d{8}`
  - Patent: `WO\d{4}\d{6,7}` etc.

- Check existence in local index (`evidence_index[id]`)

### Measurements

- **Fabricated ID rate** = `(# IDs not resolvable) / (total IDs mentioned)`
- **ID-backed claim rate** = `(# claims with resolvable IDs) / (total factual claims)`

### Case study story

> “The report cited PMID: 31415926 and NCT01234567; neither exists in the evidence pack. The traceable run log shows no retrieval call ever returned these IDs.”

This is extremely compelling because it’s binary.

---

# Experiment 3 — “Correct citation, wrong claim” (faithfulness failure)

### What it produces

A case study where the agent cites a real document that is **topically related** but does **not entail** the stronger claim it makes. This directly supports your “citation theater” / faithfulness argument.

### Protocol

1. Build evidence docs with these patterns:

   - A paper says: “associated with” (weak)
   - Agent claims: “causes” or “validated target” (strong)

2. Prompt agent to produce strong conclusions (e.g., “Is Target X validated? Give evidence.”)
3. Compare:

   - Baseline agent (likely overclaims)
   - Gated agent that must pass a faithfulness check

### Faithfulness validator (programmatic, lightweight)

You don’t need a full NLI model to start—use a deterministic rubric:

- If claim contains strong verbs (“causes”, “proves”, “validated”, “established”), then evidence must contain:

  - either the same verb family OR explicit causal language OR a randomized controlled trial statement (for your synthetic corpora)

- Otherwise label as **NON-ENTAILED**.

### Measurements

- **Unfaithful citation rate**: `% claims where evidence cited exists but does not entail`
- **Severity-weighted ZCR**: weight causal/validation claims higher

### Case study story

> “The agent cited Document E-17 correctly, but the passage only supports correlation, not causality. The validator flagged NON-ENTAILED; the gated agent downgraded the claim to ‘supported association’ and moved causal language to ‘hypothesis’.”

---

# Experiment 4 — Time-travel leakage (post-cutoff evidence used)

### What it produces

A “hindsight contamination” case study: the agent uses evidence published after a decision date.

### Protocol

1. Create two snapshots of your evidence corpus:

   - `snapshot_2024_01_01/`
   - `snapshot_2025_01_01/` (contains later “breakthrough” result)

2. Ask: “As of 2024-01-01, what is the clinical status / precedent?”
3. Compare:

   - Baseline agent with unconstrained retrieval (will use later docs)
   - Gated agent enforcing cutoff

### Validator

- Every cited evidence object has `published_at`
- Flag if `published_at > decision_date`

### Measurements

- **Time leak rate**: `% claims supported by post-cutoff evidence`
- **Leak-induced conclusion drift**: difference in recommendations when later evidence exists

### Case study story

> “The system recommended ‘GO’ based on a 2024-11 trial update, despite the analysis being scoped ‘as of 2024-01-01.’ Under time gating, that evidence is ineligible; the gated run yielded ‘insufficient evidence’ and a different risk posture.”

---

# Experiment 5 — Cross-domain contradictions that survive to the final output

### What it produces

A case where the final report is internally inconsistent: e.g., science section says “no safety signal,” tox section contains “liver enzyme elevation,” and the summary ignores the conflict.

### Protocol

1. Create evidence docs that support:

   - a positive efficacy signal
   - a negative safety signal

2. Prompt agent to write a cross-functional memo with sections: Science / Safety / Regulatory / IP.
3. Compare:

   - Baseline: contradictions survive, summary smooths them over
   - Gated: contradiction detection forces explicit reconciliation or downgrade

### Contradiction validator (programmatic)

Represent each section as structured propositions:

- `safety_signal = present/absent`
- `efficacy_signal = present/absent`
- `ip_blocker = present/absent`
  Then detect:
- section-to-summary mismatch
- internal mismatch across sections

### Measurements

- **Contradiction survival rate**: contradictions detected in section claims but not reflected in the final “bottom line”
- **Resolution rate**: contradictions explicitly acknowledged + mitigations proposed

### Case study story

> “The memo states ‘no material safety findings’ while citing a source describing hepatotoxicity. The contradiction validator flags it. Baseline summary masks the conflict; gated system forces an explicit safety risk entry and downgrades confidence.”

---

# Experiment 6 — Red-flag recall vs “best-case narrative” bias (negative evidence omission)

### What it produces

A case study showing that baseline agents disproportionately omit negative evidence, especially when prompted for “why this is promising.”

### Protocol

1. Create evidence set with:

   - 3 positive docs
   - 3 negative docs (assay artifacts, confounders, failure precedents)

2. Prompt:

   - “Make the best case for Target X.”
   - “Now do a diligence memo with blockers first.”

3. Compare:

   - Baseline agent (positive framing dominates)
   - Gated/TRACE agent with required “blockers first” retrieval pass

### Measurement

- **Red-flag recall@k**: fraction of known negative items mentioned in the final output
- **Negative evidence coverage**: count of negative claims acknowledged / expected

### Case study story

> “Under best-case prompting, the agent omitted 2/3 known blockers. With a red-flag-first protocol, recall increased and the memo included a mitigation plan.”

---

## How you turn these into paper case studies fast

For each experiment, you’ll have:

- a **single representative run** (best illustrative failure)
- and **aggregate stats** across 100–500 runs (to show it’s systematic, not anecdotal).

Your case study layout can be:

1. **Scenario** (prompt + context)
2. **Baseline output excerpt** (1–2 paragraphs)
3. **Run record evidence** (tool failure / time leak / missing ID)
4. **Validator report snippet** (the “conviction”)
5. **VECTR-gated output excerpt** (how it fails loud / downgrades)
6. **Quant summary** (rate across N runs)

---
