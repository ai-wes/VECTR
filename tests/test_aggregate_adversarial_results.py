import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "aggregate_adversarial_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_adversarial_results", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ParseAuditTests(unittest.TestCase):
    def test_recovers_fenced_json_from_failed_audit(self):
        payload = {
            "citation_present": "no",
            "citation_resolvable": "no",
            "fabricated_details": "no",
            "page_or_section_hallucinated": "no",
            "overall_risk": "low",
        }
        record = {
            "audit": {
                "ok": False,
                "parsed": None,
                "raw": f"```json\n{json.dumps(payload)}\n```",
            }
        }

        parsed = MODULE.parse_audit(record)

        self.assertTrue(parsed["audit_parse_ok"])
        self.assertEqual(parsed["audit_parse_status"], "recovered_from_raw")
        self.assertEqual(parsed["overall_risk"], "low")

    def test_discloses_unparseable_audit_without_crashing(self):
        parsed = MODULE.parse_audit({"audit": {"ok": False, "raw": "not json"}})

        self.assertFalse(parsed["audit_parse_ok"])
        self.assertEqual(parsed["audit_parse_status"], "failed")
        self.assertIsNone(parsed["citation_present"])
        self.assertTrue(parsed["audit_parse_error"])


if __name__ == "__main__":
    unittest.main()
