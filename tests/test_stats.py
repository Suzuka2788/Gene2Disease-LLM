import unittest

from gene2disease import stats

SETTINGS = stats.BootstrapSettings(n_resamples=2000, seed=7, level=0.95)


def mean(sample):
    return sum(sample) / len(sample)


class QuantileTests(unittest.TestCase):
    def test_type7_hand_values(self):
        values = [1, 2, 3, 4]
        self.assertEqual(stats.quantile(values, 0), 1)
        self.assertEqual(stats.quantile(values, 1), 4)
        self.assertAlmostEqual(stats.quantile(values, 0.5), 2.5)
        self.assertAlmostEqual(stats.quantile(values, 0.25), 1.75)

    def test_interval_on_2000_values_uses_one_rule(self):
        low, high = stats.interval(list(range(1, 2001)), 0.95)
        self.assertAlmostEqual(low, 50.975)
        self.assertAlmostEqual(high, 1950.025)


class DecisionRuleTests(unittest.TestCase):
    def reading(self, low, high):
        return stats.classify_difference(stats.Estimate(0.0, low, high, 10, 10, 0), 0.05)

    def test_each_reading(self):
        self.assertEqual(self.reading(0.06, 0.2), "first_meaningfully_higher")
        self.assertEqual(self.reading(-0.2, -0.06), "second_meaningfully_higher")
        self.assertEqual(self.reading(-0.04, 0.05), "equivalent_within_delta")
        self.assertEqual(self.reading(0.01, 0.11), "first_higher_magnitude_unresolved")
        self.assertEqual(self.reading(-0.11, -0.01), "second_higher_magnitude_unresolved")
        self.assertEqual(self.reading(-0.1, 0.1), "inconclusive")
        self.assertEqual(self.reading(None, None), "inconclusive")

    def test_small_positive_interval_is_equivalence_not_advantage(self):
        self.assertEqual(self.reading(0.01, 0.04), "equivalent_within_delta")


class BootstrapTests(unittest.TestCase):
    def test_same_seed_same_interval(self):
        data = [0, 1, 1, 0, 1, 1, 1, 0]
        self.assertEqual(stats.bootstrap(data, mean, SETTINGS),
                         stats.bootstrap(data, mean, SETTINGS))

    def test_constant_data_has_zero_width(self):
        estimate = stats.bootstrap([2, 2, 2], mean, SETTINGS)
        self.assertEqual((estimate.point, estimate.low, estimate.high), (2, 2, 2))

    def test_undefined_draws_are_counted(self):
        def only_with_one(sample):
            return 1.0 if 1 in sample else None
        estimate = stats.bootstrap([0, 1], only_with_one, SETTINGS)
        self.assertGreater(estimate.n_undefined, 0)

    def test_paired_difference_uses_the_same_draws(self):
        estimate = stats.paired_difference(
            [(1, 0), (3, 2), (5, 4)],
            lambda s: sum(a for a, _ in s) / len(s),
            lambda s: sum(b for _, b in s) / len(s), SETTINGS)
        self.assertAlmostEqual(estimate.low, 1)
        self.assertAlmostEqual(estimate.high, 1)

    def test_stratified_keeps_stratum_sizes(self):
        sizes = set()

        def statistic(sample):
            sizes.add((len(sample["a"]), len(sample["b"])))
            return 0.0
        stats.stratified_bootstrap({"a": [1, 2], "b": [3]}, statistic, SETTINGS)
        self.assertEqual(sizes, {(2, 1)})

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            stats.bootstrap([], mean, SETTINGS)


if __name__ == "__main__":
    unittest.main()
