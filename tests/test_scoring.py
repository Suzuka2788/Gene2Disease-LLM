import unittest

from gene2disease import scoring
from gene2disease.scoring import Counts


class ScoringTests(unittest.TestCase):
    def test_hand_counts(self):
        counts = scoring.compare_sets({("g1", "d1"), ("g2", "d2")},
                                      {("g1", "d1"), ("g3", "d3"), ("g4", "d4")})
        self.assertEqual(counts, Counts(1, 2, 1))
        self.assertAlmostEqual(scoring.precision(counts), 1 / 3)
        self.assertAlmostEqual(scoring.recall(counts), 1 / 2)
        self.assertAlmostEqual(scoring.f1(counts), 0.4)

    def test_undefined_metrics_are_none_not_zero(self):
        empty = Counts(0, 0, 0)
        self.assertIsNone(scoring.precision(empty))
        self.assertIsNone(scoring.recall(empty))
        self.assertIsNone(scoring.f1(empty))

    def test_absent_paper_is_an_error(self):
        with self.assertRaises(ValueError):
            scoring.score_papers({"p1": set(), "p2": set()}, {"p1": set()}, "as_empty")

    def test_failed_paper_needs_a_policy(self):
        gold = {"p1": {"x"}, "p2": {"y"}}
        predicted = {"p1": {"x"}, "p2": scoring.FAILED}
        kept = scoring.score_papers(gold, predicted, "as_empty")
        self.assertEqual(kept["p2"], Counts(0, 0, 1))
        self.assertNotIn("p2", scoring.score_papers(gold, predicted, "exclude"))

    def test_micro_pools_counts(self):
        sample = [Counts(1, 0, 1), Counts(1, 2, 0)]
        self.assertAlmostEqual(scoring.micro(scoring.recall, sample), 2 / 3)


if __name__ == "__main__":
    unittest.main()
