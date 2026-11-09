import unittest

from gene2disease import audit, stats
from gene2disease.audit import (OPTIMISTIC, PESSIMISTIC, AuditedPaper, ItemStratum,
                                PaperStratum, Undecided)

SETTINGS = stats.BootstrapSettings(n_resamples=500, seed=3, level=0.95)
REQUIRED = frozenset({"unassessed"})
PLAIN = Undecided(False, False)


def design():
    first = AuditedPaper(ItemStratum(10, (True, True, False, True)),
                         {"unassessed": ItemStratum(100, (False, False, True, False))})
    second = AuditedPaper(ItemStratum(4, (True, True)),
                          {"unassessed": ItemStratum(20, (False, False))})
    third = AuditedPaper(ItemStratum(6, (False, True, True)),
                         {"unassessed": ItemStratum(0, ())})
    return {"research": PaperStratum(10, (first, second)), "review": PaperStratum(3, (third,))}


def parts():
    current = design()
    return ({name: value.sampled for name, value in current.items()},
            {name: value.population for name, value in current.items()})


class EstimatorTests(unittest.TestCase):
    """Hand values.

    research weight 10/2 = 5, review weight 3/1 = 3.
    emitted = 5*(10+4) + 3*6 = 88
    true    = 5*(10*3/4 + 4) + 3*(6*2/3) = 5*11.5 + 12 = 69.5
    missed  = 5*(100*1/4 + 0) + 0 = 125
    """

    def test_precision(self):
        self.assertAlmostEqual(audit.edge_precision(*parts(), PLAIN), 69.5 / 88)

    def test_missed_total_expands_to_population(self):
        self.assertAlmostEqual(audit.missed_total(*parts(), PLAIN), 125)

    def test_recall_combines_both(self):
        self.assertAlmostEqual(audit.conditional_recall(*parts(), PLAIN), 69.5 / 194.5)

    def test_missed_prevalence_is_not_recall(self):
        prevalence = 1 / 4
        self.assertNotAlmostEqual(audit.conditional_recall(*parts(), PLAIN), 1 - prevalence)

    def test_estimate_interval_contains_point(self):
        result = audit.estimate(design(), REQUIRED, SETTINGS, PLAIN)
        precision = result.edge_precision
        self.assertAlmostEqual(precision.point, 69.5 / 88)
        self.assertLessEqual(precision.low, precision.point)
        self.assertLessEqual(precision.point, precision.high)
        self.assertEqual(result.zero_positive_strata, {})
        self.assertIsNotNone(result.recall_interval)

    def test_zero_positive_stratum_withholds_intervals(self):
        paper = AuditedPaper(ItemStratum(10, (True,) * 10),
                             {"unassessed": ItemStratum(1000, (False,) * 10)})
        result = audit.estimate({"a": PaperStratum(5, (paper,))}, REQUIRED, SETTINGS, PLAIN)
        self.assertIsNone(result.missed_interval)
        self.assertIsNone(result.recall_interval)
        self.assertAlmostEqual(result.zero_positive_strata["unassessed"].prevalence_upper_bound,
                               0.3)

    def test_undecided_gives_both_bounds(self):
        paper = AuditedPaper(ItemStratum(4, (True, None)),
                             {"unassessed": ItemStratum(0, ())})
        strata, populations = {"a": (paper,)}, {"a": 1}
        self.assertAlmostEqual(audit.edge_precision(strata, populations, PESSIMISTIC), 0.5)
        self.assertAlmostEqual(audit.edge_precision(strata, populations, OPTIMISTIC), 1.0)

    def test_recall_bounds_move_edges_and_non_edges_in_opposite_directions(self):
        paper = AuditedPaper(ItemStratum(10, (True, None)),
                             {"unassessed": ItemStratum(10, (True, None))})
        strata, populations = {"a": (paper,)}, {"a": 1}
        self.assertAlmostEqual(audit.conditional_recall(strata, populations, PESSIMISTIC), 1 / 3)
        self.assertAlmostEqual(audit.conditional_recall(strata, populations, OPTIMISTIC), 2 / 3)

    def test_undeclared_stratum_is_rejected(self):
        paper = AuditedPaper(ItemStratum(1, (True,)), {})
        with self.assertRaises(ValueError):
            audit.estimate({"a": PaperStratum(1, (paper,))}, REQUIRED, SETTINGS, PLAIN)

    def test_paper_without_edges_is_allowed(self):
        empty = AuditedPaper(ItemStratum(0, ()), {})
        self.assertIsNone(audit.edge_precision({"a": (empty,)}, {"a": 5}, PLAIN))

    def test_allocation_is_proportional_capped_and_nonzero(self):
        self.assertEqual(audit.allocate(10, {"a": 90, "b": 9, "c": 1, "d": 0}),
                         {"a": 9, "b": 1, "c": 1, "d": 0})
        self.assertEqual(audit.allocate(100, {"a": 3, "b": 2}), {"a": 3, "b": 2})
        self.assertEqual(audit.allocate(10, {"a": 0}), {"a": 0})

    def test_invalid_designs_are_rejected(self):
        with self.assertRaises(ValueError):
            ItemStratum(1, (True, False))
        with self.assertRaises(ValueError):
            ItemStratum(3, ())
        with self.assertRaises(ValueError):
            PaperStratum(1, ())


if __name__ == "__main__":
    unittest.main()
