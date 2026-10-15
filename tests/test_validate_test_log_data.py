"""The pure checks behind scripts/validate_test_log_data.py (the live layer runs them
against the real graph and server; these pin down what each one flags)."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ingestion"))
from ib_test_log_scanner import make_signature, normalize_text  # noqa: E402

_spec = importlib.util.spec_from_file_location("validate_test_log_data", ROOT / "scripts" / "validate_test_log_data.py")
vt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vt)


def _test(name, variant="primary", run_variant="", suite="relval"):
    return dict(variant=variant, run_variant=run_variant, suite=suite, name=name, tag="T")


def _payload(new=(), returned=(), gone=(), totals=None, imported=True):
    arch = dict(arch="el9_amd64_gcc14", new=list(new), returned=list(returned), gone=list(gone))
    counts = dict(new_problems=len(new), returned_problems=len(returned), gone_problems=len(gone), still_present=0)
    return dict(coverage=dict(to_imported=imported, runs=3 if imported else 0, message=None if imported else "no logs"),
                totals=totals or counts, architectures=[arch] if imported else [])


class SignatureLeakTests(unittest.TestCase):
    def test_normalized_signature_is_stable(self):
        sig = make_signature("fatal_exception", "ProductNotFound",
                             "Principal::getByToken at 12-Oct-2026 05:23:53 CEST in CMSSW_20_1_X_2026-10-12-1100")
        self.assertIsNone(vt.signature_leak(sig, normalize_text))

    def test_run_specific_text_in_signature_is_reported(self):
        sig = "ubsan|runtime error|member call on address 0x152ac7dfef98 in CMSSW_20_1_X_2026-10-12-1100"
        self.assertIn("<HEX>", vt.signature_leak(sig, normalize_text))

    def test_message_logger_signature_is_left_alone(self):
        self.assertIsNone(vt.signature_leak("message_logger|HGCalGeomT3|4|", normalize_text))


class JunkCategoryTests(unittest.TestCase):
    def test_exception_names_are_fine(self):
        for category in ("IndexError", "BrokenPipeError", "KeyboardInterrupt", "SystemExit", "InvalidCmd",
                         "Exception", "madgraph.MadGraph5Error"):
            self.assertFalse(vt.is_junk_category("python_traceback", category), category)

    def test_madgraph_subprocess_and_prose_are_junk(self):
        for category in ("P4_gq_qtamvl", "No valid web browser found. Please set in ./input/mg5_configuration.txt",
                         "Traceback (most recent call last)"):
            self.assertTrue(vt.is_junk_category("python_traceback", category), category)

    def test_only_tracebacks_are_judged(self):
        self.assertFalse(vt.is_junk_category("fatal_exception", "P4_gq_qtamvl"))


class ChangesPayloadTests(unittest.TestCase):
    def test_consistent_payload_passes(self):
        payload = _payload(new=[dict(signature_id="a", summary="X", tests=[_test("1.0:step1")])],
                           gone=[dict(signature_id="a", summary="X", tests=[_test("2.0:step1")])])
        self.assertEqual(vt.check_changes_payload(payload, "T"), [])

    def test_same_test_new_and_gone_is_a_contradiction(self):
        payload = _payload(returned=[dict(signature_id="a", summary="X", tests=[_test("1.0:step1")])],
                           gone=[dict(signature_id="a", summary="X", tests=[_test("1.0:step1")])])
        errors = vt.check_changes_payload(payload, "T")
        self.assertEqual(len(errors), 1)
        self.assertIn("both returned and gone", errors[0])

    def test_same_test_in_another_run_variant_is_not_a_contradiction(self):
        payload = _payload(new=[dict(signature_id="a", summary="X", tests=[_test("1.0:step1")])],
                           gone=[dict(signature_id="a", summary="X", tests=[_test("1.0:step1", run_variant="gpu/nvidia_t4")])])
        self.assertEqual(vt.check_changes_payload(payload, "T"), [])

    def test_totals_must_match_the_lists(self):
        payload = _payload(new=[dict(signature_id="a", summary="X", tests=[_test("1.0:step1")])],
                           totals=dict(new_problems=2, returned_problems=0, gone_problems=0, still_present=0))
        self.assertIn("totals.new_problems=2", vt.check_changes_payload(payload, "T")[0])

    def test_not_imported_needs_zero_totals_and_a_message(self):
        self.assertEqual(vt.check_changes_payload(_payload(imported=False), "T"), [])
        bad = _payload(imported=False, totals=dict(new_problems=1))
        bad["coverage"]["message"] = None
        errors = vt.check_changes_payload(bad, "T")
        self.assertEqual(len(errors), 2)


class NoisyNewTests(unittest.TestCase):
    def test_signature_new_in_many_windows_is_noise(self):
        flaky = dict(signature_id="idx", summary="Python IndexError", tests=[_test("516.0:step1")])
        once = dict(signature_id="asan", summary="AddressSanitizer", tests=[_test("t")])
        history = [_payload(returned=[flaky]) for _ in range(5)] + [_payload(new=[once])]
        self.assertEqual(vt.noisy_new_signatures(history, 3), [("Python IndexError", "idx", 5)])

    def test_signature_counted_once_per_window(self):
        flaky = dict(signature_id="idx", summary="E", tests=[_test("1")])
        history = [_payload(new=[flaky], returned=[flaky])]
        self.assertEqual(vt.noisy_new_signatures(history, 1), [("E", "idx", 1)])


if __name__ == "__main__":
    unittest.main()
