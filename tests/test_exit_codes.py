import unittest
from unittest.mock import patch

from graph.tools import get_exit_code_names
from server.api_server import _exit_code_meaning_answer, _format_exit_code

_TABLE = {16640: "OtherCMS", 17920: "ProductNotFound", 35584: "SIGSEGV"}


class ExitCodeLookupTests(unittest.TestCase):
    @patch("graph.tools.get_driver")
    def test_get_exit_code_names_returns_full_table_as_a_dict(self, get_driver):
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [
            {"code": 16640, "name": "OtherCMS"},
            {"code": 35584, "name": "SIGSEGV"},
        ]

        names = get_exit_code_names()

        self.assertEqual(names, {16640: "OtherCMS", 35584: "SIGSEGV"})
        get_driver.return_value.close.assert_called_once()

    def test_format_exit_code_includes_the_category_name_when_known(self):
        self.assertEqual(_format_exit_code(16640, {16640: "OtherCMS"}), "16640 (OtherCMS)")

    def test_format_exit_code_falls_back_to_the_bare_number_when_unknown(self):
        self.assertEqual(_format_exit_code(99999, {16640: "OtherCMS"}), "99999")

    @patch("graph.tools.get_driver")
    def test_get_exit_code_names_does_not_truncate_at_max_rows(self, get_driver):
        # The real table has ~160 entries, already over _run_query's MAX_ROWS=100
        # cap; silently dropping some would misreport known codes as unknown.
        session = get_driver.return_value.session.return_value.__enter__.return_value
        session.run.return_value = [{"code": code, "name": f"Code{code}"} for code in range(150)]

        names = get_exit_code_names()

        self.assertEqual(len(names), 150)

    # The imported table is keyed by the raw wait status RelVal records store
    # ((exit & 0xFF) << 8); shifters quote the process exit status (65) or the
    # edm::errors code (8001) just as often. Found live 2026-10-06: "exit code
    # 65" answered "not in the graph" although 16640 = 65 << 8 = OtherCMS.
    @patch("server.api_server.get_exit_code_names", return_value=_TABLE)
    def test_exit_code_meaning_resolves_all_three_forms(self, names):
        for question, code in (
            ("what does exit code 16640 mean", "16640"),
            ("what does exit code 65 mean in a relval", "65"),
            ("exit code 8001 meaning?", "8001"),
        ):
            answer = _exit_code_meaning_answer(question)
            self.assertIn("**OtherCMS**", answer, question)
            self.assertIn(f"`{code}`", answer)
        self.assertIn("**ProductNotFound**", _exit_code_meaning_answer("what does exit code 8006 mean"))
        self.assertIn("**SIGSEGV**", _exit_code_meaning_answer("what is exit status 139"))

    @patch("server.api_server.get_exit_code_names", return_value=_TABLE)
    def test_exit_code_meaning_is_honest_about_unknown_codes(self, names):
        answer = _exit_code_meaning_answer("what does exit code 4242 mean")
        self.assertIn("not in the imported CMSSW exit-code table", answer)
        self.assertNotIn("**", answer)

    @patch("server.api_server.get_exit_code_names", return_value=_TABLE)
    def test_exit_code_meaning_ignores_non_meaning_questions(self, names):
        self.assertIsNone(_exit_code_meaning_answer("which workflows failed with exit code 16640"))
        self.assertIsNone(_exit_code_meaning_answer("what is the latest IB"))


if __name__ == "__main__":
    unittest.main()
