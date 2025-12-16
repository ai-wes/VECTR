import json
import os
from typing import Dict, List, Any

from sandbox.adapters.path_setup import EXTERNAL_APP_ROOT  # noqa: F401 side-effect


class EvidenceStore:
    """Simple in-memory evidence index backed by JSON files.

    Expected file shape:
      {
        "id": "PMID123",
        "source_type": "pubmed" | "patent" | "trial",
        "published_at": "2024-02-01",
        "text": "...",
        "metadata": {...}
      }
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.records: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if not os.path.isdir(self.root_dir):
            return
        for fname in os.listdir(self.root_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.root_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                rec_id = rec.get("id") or os.path.splitext(fname)[0]
                rec["id"] = rec_id
                self.records[rec_id] = rec
            except Exception as exc:
                print(f"[WARN] Failed to load {path}: {exc}")

    def search(self, query: str = "", source_type: str = None) -> List[Dict[str, Any]]:
        # naive filter; to be replaced with actual retrieval
        results = []
        for rec in self.records.values():
            if source_type and rec.get("source_type") != source_type:
                continue
            if query and query.lower() not in (rec.get("text") or "").lower():
                continue
            results.append(rec)
        return results

    def get(self, rec_id: str) -> Dict[str, Any]:
        return self.records.get(rec_id, {})
