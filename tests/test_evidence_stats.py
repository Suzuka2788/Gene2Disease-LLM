"""Estimators of the evidence assessment (protocol §6)."""
import unittest

from gene2disease import evidence_stats as es


def item(item_id, stratum, runs, citations=None, sets=None):
    return {"item_id": item_id, "stratum": stratum, "runs": runs,
            "citations": citations or {}, "citation_sets": sets or {}}


class WilsonTests(unittest.TestCase):
    def test_whole_stratum_sampled_collapses_to_the_point(self):
        self.assertEqual(es.wilson(3, 4, 4), (0.75, 0.75))

    def test_finite_population_narrows_the_interval(self):
        wide = es.wilson(10, 20)
        narrow = es.wilson(10, 20, 30)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])
        self.assertAlmostEqual(sum(wide) / 2, 0.5, places=6)

    def test_newcombe_contains_the_difference(self):
        result = es.newcombe(15, 18, 300, 8, 16, 400)
        self.assertLess(result["low"], result["difference"])
        self.assertGreater(result["high"], result["difference"])


class RunRateTests(unittest.TestCase):
    def setUp(self):
        # S1: pairwise run only; S2: both runs.
        self.strata = {
            "S1": [item(f"a{i}", "S1", ["pw"]) for i in range(4)],
            "S2": [item(f"b{i}", "S2", ["pw", "jt"]) for i in range(4)] + [item("b9", "S2", ["jt"])]}
        self.q1 = {**{f"a{i}": "supported" if i < 1 else "no_relation" for i in range(4)},
                   **{f"b{i}": "supported" for i in range(4)}, "b9": "no_relation"}
        self.n_by_run = {"S1": {"pw": 100}, "S2": {"pw": 50, "jt": 200}}

    def test_domain_estimate_weights_by_run_counts(self):
        rates = es.run_rates(self.strata, self.q1, self.n_by_run, ["pw", "jt"])
        self.assertAlmostEqual(rates["pw"], (100 * 0.25 + 50 * 1.0) / 150)
        self.assertAlmostEqual(rates["jt"], 4 / 5)

    def test_too_few_items_of_a_run_is_not_estimated(self):
        strata = {"S1": self.strata["S1"][:2], "S2": self.strata["S2"]}
        self.assertIsNone(es.run_rates(strata, self.q1, self.n_by_run, ["pw"])["pw"])

    def test_bootstrap_shares_draws_and_reports_derived_values(self):
        out = es.bootstrap_runs(self.strata, self.q1, self.n_by_run, ["pw", "jt"],
                                lambda r: {"diff": None if None in (r["pw"], r["jt"]) else r["pw"] - r["jt"]},
                                n_resamples=200, seed=1)
        self.assertAlmostEqual(out["diff"]["point"], out["rate:pw"]["point"] - out["rate:jt"]["point"])
        low, high = out["rate:pw"]["percentile95"]
        self.assertLessEqual(low, high)


class CitationGapTests(unittest.TestCase):
    def test_any_and_all_rules(self):
        items = [item("e1", "S2", ["pw", "jt"], {"pw": [], "jt": []},
                      {"e1-A": {"runs": ["pw"]}, "e1-B": {"runs": ["pw", "jt"]}})]
        q1 = {"e1": "supported"}
        q2 = {"e1-A": "supported", "e1-B": "no_relation"}
        any_rule = es.citation_gap(items, q1, q2, "any")
        all_rule = es.citation_gap(items, q1, q2, "all")
        self.assertEqual(any_rule["by_run"]["pw"]["gap"], 0)    # one supporting set suffices
        self.assertEqual(any_rule["by_run"]["jt"]["gap"], 1)
        self.assertEqual(all_rule["by_run"]["pw"]["gap"], 1)
        self.assertEqual(any_rule["pooled_item_weighted"]["gap"], 0.5)

    def test_undecided_sets_and_items(self):
        items = [item("e1", "S2", ["pw"], {"pw": []},
                      {"e1-A": {"runs": ["pw"]}, "e1-B": {"runs": ["pw"]}}),
                 item("e2", "S2", ["pw"], {"pw": []}, {"e2-A": {"runs": ["pw"]}})]
        q2 = {"e1-A": "supported", "e1-B": "undecided", "e2-A": "no_relation"}
        q1 = {"e1": "supported", "e2": "undecided"}
        any_rule = es.citation_gap(items, q1, q2, "any")
        self.assertEqual(any_rule["by_run"]["pw"]["units"], 1)    # e1 settled by set A
        self.assertEqual(any_rule["left_out_undecided"], 1)       # e2: Q1 undecided, counted
        all_rule = es.citation_gap(items, q1, q2, "all")
        self.assertEqual(all_rule["left_out_undecided"], 2)       # e1 not settled under "all"

    def test_wilson_reference_value(self):
        low, high = es.wilson(2, 18)                               # standard Wilson score interval
        self.assertAlmostEqual(low, 0.0309, places=3)
        self.assertAlmostEqual(high, 0.3280, places=3)

    def test_items_without_passage_support_are_not_units(self):
        items = [item("e1", "S1", ["pw"], {"pw": []}, {"e1-A": {"runs": ["pw"]}})]
        result = es.citation_gap(items, {"e1": "no_relation"}, {"e1-A": "no_relation"}, "any")
        self.assertEqual(result["by_run"], {})


if __name__ == "__main__":
    unittest.main()
