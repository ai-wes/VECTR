from typing import Dict, List, Tuple

# Very lightweight contradiction detector for structured section props.

def detect(section_props: Dict[str, str]) -> List[Tuple[str, str]]:
    """Return list of (field, reason) contradictions.
    Expected keys: safety_signal, efficacy_signal, ip_blocker (values present/absent/unknown)
    """
    contradictions = []
    if section_props.get("summary_says") == "no_safety" and section_props.get("safety_signal") == "present":
        contradictions.append(("safety", "summary omits safety signal"))
    if section_props.get("summary_says") == "clear_ip" and section_props.get("ip_blocker") == "present":
        contradictions.append(("ip", "summary omits IP blocker"))
    return contradictions
