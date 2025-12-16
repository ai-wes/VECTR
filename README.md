# VECTR Reliability Harness

This repository provides a reproducible reference implementation for evaluating reliability controls for agentic AI systems supporting drug discovery and development.  It accompanies the VECTR framework described in the white‑paper and includes two major components:

* An **adversarial citation hallucination pipeline** that uses OpenAI models and [Autogen](https://github.com/microsoft/autogen) agents to probe the propensity of language models to fabricate or misattribute bibliographic references under adversarial prompting.
* A **deterministic sandbox harness** that runs a suite of offline experiments to demonstrate how reliability validators (identifier resolution, time gating, faithfulness checking, contradiction detection and red‑flag recall) and fail‑loud gating can be applied to tool‑calling agents under a declared context of use (COU).

The code is provided under the `sandbox/` and `experiments/` directories and is intended for researchers who wish to reproduce the adversarial evaluation or adapt the harness to their own pipelines.  It is not a drop‑in product; instead it aims to make reliability failure modes visible, auditable and measurable.

## Repository structure

The top‑level layout of this repository is organised as follows:

```
.
├── experiments/          # Text documentation describing the sandbox experiments and reliability metrics
├── sandbox/              # Python source code for the citation pipeline and sandbox experiments
│   ├── adapters/         # Adapters for external tools such as citation parsing
│   ├── agents/           # Autogen agent definitions and harness logic
│   ├── bert/             # Optional natural language inference models for faithfulness checking
│   ├── claims/           # Claim extraction utilities
│   ├── config.yaml       # Configuration for sandbox experiments (decision date, fault injection, etc.)
│   ├── evidence/         # Example evidence store snapshots used by the sandbox
│   ├── metrics/          # Metric definitions (e.g., zombie claim rate, unresolved identifier rate)
│   ├── runs/             # Example output logs from the sandbox harness
│   ├── scripts/          # Entry points to run the adversarial pipeline and experiments
│   ├── sim_tools/        # Simulated search tools (PubMed, patents, trials) for offline experiments
│   └── validators/       # Faithfulness, time gating, contradiction and red‑flag validators
└── README.md             # This file – overview and usage instructions
```

## Getting started

This project targets **Python 3.10+** and uses the [Autogen](https://github.com/microsoft/autogen) framework to orchestrate LLM‑driven agents.  To reproduce the experiments you will need access to an OpenAI API key and several Python packages.

### Installation

1. **Clone the repository** and change into the root directory:

   ```bash
   git clone https://github.com/yourusername/vectr-harness.git
   cd vectr-harness
   ```

2. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r sandbox/requirements.txt
   ```

   A `requirements.txt` file is not provided in this example; consult the import statements in the `sandbox/` package to assemble the necessary packages (for example `openai`, `autogen`, `transformers`, `duckduckgo_search`, `pandas`, etc.).

3. **Export your OpenAI API key** so that the Autogen clients can authenticate to the OpenAI API:

   ```bash
   export OPENAI_API_KEY=<your-key>
   ```

### Running the adversarial citation pipeline

The script `sandbox/scripts/adversarial_citation_pipeline.py` runs a suite of adversarial prompts against an OpenAI model to measure citation hallucination.  To execute it:

```bash
python sandbox/scripts/adversarial_citation_pipeline.py \
  --model gpt-4o \
  --temperature 0.1 \
  --output-dir results/citation_runs
```

This will iterate through a fixed set of prompts defined in the script, query the model via Autogen, log the full conversation transcript, and then run an LLM‑based auditor to classify citation presence, resolvability, fabricated details and hallucinated page numbers.  Aggregated statistics are reported at the end of the run.

### Running the deterministic sandbox experiments

The deterministic harness is configured via `sandbox/config.yaml` and includes six scenario families that deliberately trigger different reliability violations:

1. **Silent tool failure** – simulates missing or empty tool outputs.
2. **Phantom identifiers** – injects fabricated citation identifiers into the output.
3. **Faithfulness violations** – uses strong causal language unsupported by the retrieved evidence.
4. **Time‑gating violations** – draws on post‑cutoff evidence under a declared decision date.
5. **Cross‑domain contradictions** – emits mutually inconsistent summaries across efficacy, safety and IP domains.
6. **Red‑flag recall** – fails to surface known negative signals.

To run all experiments across seeds and modes:

```bash
python sandbox/scripts/run_experiments.py
```

Each run writes a reproducibility pack to `sandbox/runs/<experiment>/<mode>/<seed>/` containing:

- `run_record.jsonl` – minimal run record stub.
- `claims.json` – extracted claim candidates.
- `bindings.json` – citation identifiers.
- `validator_report.json` – validator outputs.
- `metrics.json` – computed rates (e.g., zombie claim rate, unresolved identifier rate, time‑leak rate, contradiction counts).
- `final_output.md` – the narrative output or a fail‑loud state.

A helper script in `sandbox/scripts/aggregate.py` can aggregate all metrics into a CSV and Markdown table for analysis.

### Understanding the metrics

Refer to the documentation in `experiments/README.md` and Appendix C of the accompanying white paper for formal definitions of the reliability metrics.  Briefly:

- **Zombie Claim Rate (ZCR)** – the fraction of claim candidates that are unsupported by the evidence packet.
- **Unresolved Identifier Rate (UIR)** – the fraction of cited identifiers that cannot be resolved in the allowed evidence object set.
- **Time‑Leak Rate** – the fraction of evidence objects that violate the decision‑date cutoff.
- **Contradiction counts** – the number of mutually conflicting assertions across domains.
- **Red‑flag recall** – the recall of expected negative‑signal evidence.

These metrics are illustrative; real systems should adapt claim extraction granularity, severity weighting and validation thresholds to their specific context of use.

## Contributing

Contributions are welcome.  If you extend the harness or find issues, feel free to open an issue or submit a pull request.  Please ensure that any new code is clearly documented and includes unit tests or example usage.

## License

This repository is provided for academic and research purposes.  See `LICENSE` for details (not included in this example).