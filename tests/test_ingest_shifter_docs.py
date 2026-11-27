import unittest

from ingestion.ingest_shifter_docs import split_into_chunks


class SplitIntoChunksTests(unittest.TestCase):
    def test_splits_on_markdown_headers(self):
        text = (
            "## ib-run-gpu-relvals\n"
            "Downstream of ib-run-relvals.\n"
            "## ib-run-relvals\n"
            "Runs RelVals in pieces.\n"
        )

        chunks = split_into_chunks(text)

        self.assertEqual([h for h, _ in chunks], ["ib-run-gpu-relvals", "ib-run-relvals"])
        self.assertIn("Downstream of ib-run-relvals.", chunks[0][1])
        self.assertIn("Runs RelVals in pieces.", chunks[1][1])

    def test_preamble_before_first_header_is_kept_as_its_own_chunk(self):
        text = "Intro paragraph with no heading yet.\n## First heading\nBody.\n"

        chunks = split_into_chunks(text)

        self.assertEqual(chunks[0], ("", "Intro paragraph with no heading yet."))
        self.assertEqual(chunks[1][0], "First heading")

    def test_text_with_no_headers_becomes_one_chunk(self):
        chunks = split_into_chunks("Just a plain paragraph, no markdown headers at all.")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], "")

    def test_blank_text_produces_no_chunks(self):
        self.assertEqual(split_into_chunks(""), [])
        self.assertEqual(split_into_chunks("   \n\n  "), [])

    def test_oversized_section_is_split_into_overlapping_windows(self):
        body = "x" * 10000
        text = f"## Big section\n{body}\n"

        chunks = split_into_chunks(text, max_section_chars=4000, window_chars=1500, overlap_chars=200)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(heading == "Big section" for heading, _ in chunks))
        self.assertTrue(all(len(c) <= 1500 for _, c in chunks))
        # consecutive windows overlap, so no content is silently dropped at a boundary
        reconstructed_tail = chunks[0][1][-200:]
        self.assertTrue(chunks[1][1].startswith(reconstructed_tail))


if __name__ == "__main__":
    unittest.main()
