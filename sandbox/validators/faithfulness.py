"""
Faithfulness checking with optional BERT NLI fallback.

If transformers + the configured model are available, we run an NLI entailment
score; otherwise we fall back to a lightweight verb-strength heuristic.
"""

import re
from typing import List, Optional

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline  # type: ignore
except Exception:  # transformers may not be installed or model unavailable
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None

_nli_pipe = None

STRONG_VERB_PATTERN = re.compile(r"\b(cause[sd]?|prove[sd]?|validated?|established?|conclusive)\b", re.IGNORECASE)


def _load_nli(model_name: str):
    global _nli_pipe
    if _nli_pipe is not None:
        return _nli_pipe
    if pipeline is None:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        _nli_pipe = pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True, truncation=True)
    except Exception:
        _nli_pipe = None
    return _nli_pipe


def is_strong_claim(text: str, strong_verbs: List[str]) -> bool:
    verbs = "|".join([re.escape(v) for v in strong_verbs]) or ""
    pattern = re.compile(rf"\b({verbs})\b", re.IGNORECASE) if verbs else STRONG_VERB_PATTERN
    return bool(pattern.search(text or ""))


def faithfulness_flag(claim_text: str, evidence_text: str, strong_verbs: List[str], *, model_name: str, threshold: float) -> bool:
    """
    Return True if claim is NON-ENTAILED (i.e., fails faithfulness).
    Uses NLI when available; otherwise falls back to heuristic.
    """
    if not claim_text:
        return False

    # Try NLI
    nli = _load_nli(model_name)
    if nli is not None and evidence_text:
        try:
            scores = nli({"text": evidence_text, "text_pair": claim_text})[0]
            entail_score = next((s["score"] for s in scores if s["label"].lower().startswith("entail")), 0.0)
            return entail_score < threshold
        except Exception:
            pass  # fall back to heuristic

    # Heuristic fallback
    strong = is_strong_claim(claim_text, strong_verbs)
    if not strong:
        return False
    return not is_strong_claim(evidence_text or "", strong_verbs)
