from typing import List, Set


def recall(found: List[str], expected: List[str]) -> float:
    if not expected:
        return 1.0
    expected_set: Set[str] = set(expected)
    found_set: Set[str] = set(found)
    return len(found_set & expected_set) / len(expected_set)
