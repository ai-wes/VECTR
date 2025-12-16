from typing import Dict, List


def resolve_ids(ids: List[str], index_lookup: Dict[str, dict]) -> Dict[str, List[str]]:
    missing = [i for i in ids if i not in index_lookup]
    present = [i for i in ids if i in index_lookup]
    return {"present": present, "missing": missing}
