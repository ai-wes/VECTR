import re
from typing import List, Dict

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def extract_claims(text: str) -> List[Dict[str, str]]:
    claims = []
    if not text:
        return claims
    sentences = SENTENCE_SPLIT.split(text.strip())
    for idx, sent in enumerate(sentences):
        norm = sent.strip()
        if not norm:
            continue
        claims.append({
            "claim_id": f"C{idx:03d}",
            "span": norm,
            "type": infer_type(norm),
        })
    return claims


def infer_type(sent: str) -> str:
    sent_l = sent.lower()
    if any(w in sent_l for w in ["hypothesis", "suggest", "might", "could"]):
        return "hypothesis"
    if any(w in sent_l for w in ["may", "possible"]):
        return "inference"
    return "fact"
