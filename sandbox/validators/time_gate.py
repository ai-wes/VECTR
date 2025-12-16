from datetime import datetime
from typing import List, Dict


def parse_date(date_str: str) -> datetime:
    if isinstance(date_str, datetime):
        return date_str
    return datetime.fromisoformat(str(date_str))


def time_leaks(evidence_items: List[Dict[str, str]], decision_date: str) -> List[str]:
    cutoff = parse_date(decision_date)
    leaks = []
    for item in evidence_items:
        published = item.get("published_at") or item.get("published")
        if not published:
            continue
        try:
            ts = parse_date(published)
            if ts > cutoff:
                leaks.append(item.get("id") or "")
        except Exception:
            continue
    return leaks
