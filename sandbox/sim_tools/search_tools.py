import random
from typing import List, Dict, Any, Optional

from .local_index import EvidenceStore


def search_patents(store: EvidenceStore, query: str, *, fault_config: Dict[str, float], rng: random.Random) -> Dict[str, Any]:
    # Explicit overrides
    if fault_config.get("patent_search_force_error"):
        return {"status": "error", "code": "forced", "results": []}
    if fault_config.get("patent_search_force_empty"):
        return {"status": "ok", "results": []}

    # Fault injection: empty results or timeout
    if rng.random() < fault_config.get("patent_search_timeout_prob", 0):
        return {"status": "error", "code": "timeout", "results": []}
    if rng.random() < fault_config.get("patent_search_empty_prob", 0):
        return {"status": "ok", "results": []}

    results = store.search(query=query, source_type="patent")
    # Optional ranking shuffle
    if rng.random() < fault_config.get("ranking_shuffle_prob", 0):
        rng.shuffle(results)

    return {"status": "ok", "results": results}


def search_pubmed(store: EvidenceStore, query: str, *, rng: random.Random) -> Dict[str, Any]:
    results = store.search(query=query, source_type="pubmed")
    return {"status": "ok", "results": results}


def get_trial_registry(store: EvidenceStore, trial_id: Optional[str] = None, *, rng: random.Random) -> Dict[str, Any]:
    if trial_id:
        rec = store.get(trial_id)
        return {"status": "ok", "results": [rec] if rec else []}
    results = store.search(source_type="trial")
    return {"status": "ok", "results": results}
