import unittest

from gene2disease import candidates
from gene2disease.corpus import Block, Document, Mention, Sentence

TEXT = "AAA is here. BBB is there. CCC too."


def document():
    block = Block("d.b0", "Results", "paragraph", TEXT,
                  [Sentence("s0", 0, 12), Sentence("s1", 13, 26), Sentence("s2", 27, 35)])
    other = Block("d.b1", "Results", "paragraph", "DDD alone.", [Sentence("s3", 0, 10)])
    third = Block("d.b2", "Methods", "paragraph", "EEE alone.", [Sentence("s4", 0, 10)])
    return Document("d", [block, other, third])


MENTIONS = [
    Mention("gene:a", "gene", "d.b0", 0, 3, "AAA"),
    Mention("disease:b", "disease", "d.b0", 13, 16, "BBB"),
    Mention("disease:c", "disease", "d.b0", 27, 30, "CCC"),
    Mention("disease:d", "disease", "d.b1", 0, 3, "DDD"),
    Mention("disease:e", "disease", "d.b2", 0, 3, "EEE"),
]


def covered(level):
    return candidates.covered_pairs(candidates.build_candidates(document(), MENTIONS, level))


class WindowTests(unittest.TestCase):
    def test_levels_widen_coverage_by_hand(self):
        self.assertEqual(covered("sentence"), set())
        self.assertEqual(covered("adjacent"), {("gene:a", "disease:b")})
        self.assertEqual(covered("block"), {("gene:a", "disease:b"), ("gene:a", "disease:c")})
        self.assertEqual(len(covered("section")), 3)
        self.assertEqual(len(covered("document")), 4)

    def test_document_level_equals_all_pairs(self):
        self.assertEqual(covered("document"), candidates.all_pairs(MENTIONS))

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(ValueError):
            candidates.build_windows(document(), "chapter")

    def test_window_covers_overlapping_span_only(self):
        window = candidates.build_windows(document(), "sentence")[0]
        self.assertTrue(window.covers("d.b0", 5, 6))
        self.assertFalse(window.covers("d.b0", 13, 16))
        self.assertFalse(window.covers("d.b1", 0, 3))


if __name__ == "__main__":
    unittest.main()
