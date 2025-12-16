from typing import Dict, List


def zombie_claim_rate(unsupported_flags: List[bool]) -> float:
    if not unsupported_flags:
        return 0.0
    return sum(1 for f in unsupported_flags if f) / len(unsupported_flags)


def fabricated_id_rate(missing_ids: List[str], total_ids: int) -> float:
    if total_ids == 0:
        return 0.0
    return len(missing_ids) / total_ids
