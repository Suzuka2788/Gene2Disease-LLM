"""F0 stage: window sampling, timing model and an end-to-end run on the fake server."""
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path

from gene2disease import f0, f0_report, io
from gene2disease.candidates import Candidate, Segment, Window

from test_runner import BIORED, FakeModel


def item(name, load, characters):
    window = Window(name, "block", (Segment(name, 0, characters),))
    genes = frozenset({f"g{i}" for i in range(load)})
    return f0.WindowItem("set", "p", Candidate(window, genes, frozenset({"d"})), characters, load)


class SelectionTests(unittest.TestCase):
    ITEMS = [item(f"w{i}", load, chars) for i, (load, chars) in enumerate(
        [(1, 10), (1, 500), (2, 20), (3, 900), (8, 30), (9, 800), (50, 40), (240, 100),
         (1, 5000), (2, 15)])]

    def test_sample_is_stable_and_keeps_extremes(self):
        first = f0.select_windows(self.ITEMS, [1, 4, 15, 40], 1, seed=3)
        second = f0.select_windows(self.ITEMS, [1, 4, 15, 40], 1, seed=3)
        self.assertEqual([x.window_id for x in first], [x.window_id for x in second])
        ids = {x.window_id for x in first}
        self.assertIn("w7", ids)       # heaviest
        self.assertIn("w8", ids)       # longest
        bands = {x.stratum.split("/")[0] for x in first}
        self.assertTrue({"<=1", "<=4", "<=15", ">40"} <= bands)

    def test_subset_keeps_every_stratum_and_is_seeded(self):
        chosen = f0.select_windows(self.ITEMS, [1, 4, 15, 40], 2, seed=3)
        small = f0.subset(chosen, 1, seed=5)
        self.assertEqual({x.stratum for x in small}, {x.stratum for x in chosen})
        self.assertEqual(small, f0.subset(chosen, 1, seed=5))
        self.assertEqual(f0.subset(chosen, None), chosen)

    def test_subset_does_not_always_take_the_lightest_window(self):
        items = [item(f"v{i}", 2 + i % 3, 10) for i in range(30)]
        chosen = f0.select_windows(items, [1, 40], 30, seed=1)
        picks = {f0.subset(chosen, 1, seed=s)[0].load for s in range(20)}
        self.assertGreater(len(picks), 1)

    def test_every_stratum_is_capped(self):
        chosen = f0.select_windows(self.ITEMS, [1, 4, 15, 40], 1, seed=3)
        strata = [x.stratum for x in chosen if not x.stratum.startswith("extreme")]
        self.assertEqual(len(strata), len(set(strata)))


WORKLOAD = {"set": {"levels": {"block": {
    "pairwise_calls": 12, "pairwise_passage_tokens_estimated": 100,
    "joint_calls": 2, "candidate_pairs": 10, "joint_passage_tokens_estimated": 20}}}}


def joint_call(pairs, characters, seconds):
    return {"pairs": pairs, "first_in_window": True, "window_characters": characters,
            "wall_seconds": seconds, "dataset": "a"}


class TimingModelTests(unittest.TestCase):
    def test_joint_fit_recovers_a_linear_cost(self):
        calls = [joint_call(p, c, 0.5 + 0.2 * p + 0.001 * c)
                 for p, c in [(1, 100), (2, 400), (5, 150), (9, 900), (3, 300)]]
        fit = f0.fit_seconds(calls)
        self.assertAlmostEqual(fit["intercept"], 0.5)
        self.assertAlmostEqual(fit["per_pairs"], 0.2)
        self.assertAlmostEqual(fit["per_new_window_characters"], 0.001)
        self.assertNotIn("per_first_call", fit)
        self.assertAlmostEqual(fit["predicted_over_measured_by_dataset"]["a"], 1.0)

    def test_joint_projection_counts_each_window_exposure(self):
        # 10 distinct pairs but 12 window exposures: joint calls list 12 pairs in total.
        fit = {"intercept": 1.0, "per_pairs": 0.5, "per_new_window_characters": 0.01}
        hours = f0.project_hours(fit, "joint", WORKLOAD, 4.0)
        self.assertAlmostEqual(hours["set"]["hours"], (2 * 1.0 + 12 * 0.5 + 0.01 * 80) / 3600)

    def test_pairwise_fit_charges_the_passage_to_the_first_call(self):
        calls = []
        for characters in (100, 300, 600):
            calls.append({"pairs": 1, "first_in_window": True, "window_characters": characters,
                          "wall_seconds": 1 + 0.5 + 0.002 * characters})
            for _ in range(2):
                calls.append({"pairs": 1, "first_in_window": False,
                              "window_characters": characters, "wall_seconds": 1.0})
        fit = f0.fit_seconds(calls)
        self.assertNotIn("per_pairs", fit)
        self.assertAlmostEqual(fit["intercept"], 1.0)
        self.assertAlmostEqual(fit["per_first_call"], 0.5)
        self.assertAlmostEqual(fit["per_new_window_characters"], 0.002)
        hours = f0.project_hours(fit, "pairwise", WORKLOAD, 4.0)
        self.assertAlmostEqual(hours["set"]["hours"], (12 * 1.0 + 2 * 0.5 + 0.002 * 80) / 3600)

    def test_cold_pairwise_charges_the_passage_to_every_call(self):
        calls = []
        for characters in (100, 300, 600):
            for index in range(3):
                calls.append({"pairs": 1, "first_in_window": index == 0,
                              "window_characters": characters,
                              "wall_seconds": 1 + 0.002 * characters})
        fit = f0.fit_seconds(calls, prefix_reuse=False)
        self.assertNotIn("per_first_call", fit)
        self.assertAlmostEqual(fit["per_new_window_characters"], 0.002)
        hours = f0.project_hours(fit, "pairwise", WORKLOAD, 4.0, prefix_reuse=False)
        self.assertAlmostEqual(hours["set"]["hours"], (12 * 1.0 + 0.002 * 400) / 3600)

    def test_retry_cost_counts_in_the_fit(self):
        calls = [joint_call(p, c, float(p)) for p, c in ((1, 50), (2, 300), (3, 100), (4, 700))]
        calls[1]["first_attempt"] = {"wall_seconds": 10.0}
        fit = f0.fit_seconds(calls)
        self.assertAlmostEqual(fit["predicted_over_measured_by_dataset"]["a"], 1.0)
        self.assertNotAlmostEqual(fit["intercept"], 0.0)

    def test_replayed_calls_keep_their_measurement(self):
        calls = [joint_call(p, c, float(p)) for p, c in ((1, 50), (2, 300), (3, 100), (4, 700))]
        calls[0]["cache_hit"] = True
        calls.append({"pairs": 9, "first_in_window": True, "window_characters": 10,
                      "cache_hit": True})
        self.assertEqual(f0.fit_seconds(calls)["calls"], 4)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        io.write_json(self.root / "biored" / "Dev.BioC.JSON", BIORED)
        io.write_json(self.root / "workload.json", {"chars_per_token_assumed": 4.0, "set": {
            "levels": {"block": {"pairwise_calls": 10, "pairwise_passage_tokens_estimated": 100,
                                 "joint_calls": 2, "candidate_pairs": 10,
                                 "joint_passage_tokens_estimated": 20}}}})
        (self.root / "key.txt").write_text("secret\n", encoding="utf-8")
        server = HTTPServer(("127.0.0.1", 0), FakeModel)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_port}"

    def config(self):
        conditions = [{"name": "json_pairwise", "form": "pairwise"},
                      {"name": "json_joint", "form": "joint"},
                      {"name": "lines_joint", "form": "joint", "output_format": "lines",
                       "per_stratum_limit": 1},
                      {"name": "cold_pairwise", "form": "pairwise", "cache_prompt": False}]
        return {"run": {"runs_dir": str(self.root / "runs")}, "f0": {
            "run_id": "f0_fake", "model_id": "fake", "base_url": self.url, "timeout_seconds": 5,
            "api_key_file": str(self.root / "key.txt"), "level": "block",
            "load_bands": [1, 4], "windows_per_stratum": 3, "seed": 7,
            "output_base_tokens": 200, "output_tokens_per_pair": 60,
            "workload_results": str(self.root / "workload.json"), "project_workloads": ["set"],
            "report_precision_for": ["biored_dev"],
            "datasets": [{"name": "biored_dev", "dataset": "biored",
                          "folder": str(self.root / "biored"), "splits": ["Dev"]}],
            "conditions": conditions}}

    def test_conditions_share_windows_and_report_aggregates(self):
        FakeModel.authorization.clear()
        result = f0.run(self.config())
        self.assertEqual(result["selected_windows"], 3)
        self.assertEqual(result["server"]["n_ctx"], 4096)
        by_name = {c["condition"]: c for c in result["conditions"]}
        self.assertEqual(by_name["json_pairwise"]["calls"], 6)
        self.assertEqual(by_name["json_joint"]["calls"], 3)
        self.assertEqual(by_name["lines_joint"]["windows"], 1)
        self.assertEqual(by_name["json_pairwise"]["parsed_call_rate"], 1.0)
        self.assertEqual(by_name["json_pairwise"]["answered_pair_rate"], 1.0)
        recall = "recall_of_cooccurring_document_gold"
        self.assertEqual(by_name["json_pairwise"]["quality"]["biored_dev"][recall], 1.0)
        self.assertEqual(by_name["json_joint"]["quality"]["biored_dev"][recall], 1.0)
        self.assertEqual(by_name["lines_joint"]["quality"]["biored_dev"]["pairs_no_assertion"], 1)
        self.assertAlmostEqual(
            by_name["json_pairwise"]["quality"]["biored_dev"]["edges_matching_document_gold"], 0.5)
        self.assertEqual(by_name["json_pairwise"]["completion_tokens"], 60)
        self.assertIn("set", by_name["json_joint"]["projected_hours_block"])
        self.assertTrue(FakeModel.authorization)
        self.assertEqual(set(FakeModel.authorization), {"Bearer secret"})
        calls = io.read_jsonl(self.root / "runs" / "f0_fake" / "json_joint" / "calls.jsonl")
        self.assertEqual({c["outcome"] for c in calls}, {"ok"})

    def test_rerun_uses_the_cache_and_changed_settings_are_refused(self):
        first = f0.run(self.config())
        again = f0.run(self.config())
        self.assertEqual(again["conditions"][0]["cache_hits_this_run"], 6)
        self.assertEqual(again["conditions"][0]["measured_calls"], 6)
        self.assertEqual(again["conditions"][0]["completion_tokens"],
                         first["conditions"][0]["completion_tokens"])
        history = io.read_jsonl(self.root / "runs" / "f0_fake" / "json_pairwise"
                                / "calls_history.jsonl")
        self.assertEqual(len(history), 12)
        memory = self.root / "runs" / "f0_fake" / "json_pairwise" / "memory.json"
        if memory.is_file():
            self.assertEqual(again["conditions"][0]["peak_total_gpu_memory_mib_including_desktop"],
                             io.read_json(memory)["peak_mib"])
        changed = self.config()
        changed["f0"]["output_tokens_per_pair"] = 80
        with self.assertRaises(ValueError):
            f0.run(changed)
        smaller = self.config()
        smaller["f0"]["conditions"][0]["per_stratum_limit"] = 1
        with self.assertRaises(ValueError):
            f0.run(smaller)

    def test_report_matches_windows_and_sums_the_schedule(self):
        result_path = self.root / "f0.json"
        io.write_json(result_path, f0.run(self.config()))
        plan = [{"item": "dev", "condition": "json_joint", "workload": "set", "multiplier": 2.0}]
        report = f0_report.report({"run": {"runs_dir": str(self.root / "runs")}, "f0_report": {
            "report_precision_for": ["biored_dev"], "margin": 0.2, "gate_hours": 72,
            "plan": plan, "not_costed": ["S1"],
            "once_plan": [{"item": "once", "condition": "json_pairwise", "workload": "set",
                           "multiplier": 1.0}],
            "models": [{"name": "fake", "results": [str(result_path)],
                        "comparisons": [["json_joint", "lines_joint", "f0_fake"]]}]}})
        comparison = report["models"]["fake"]["comparisons"][0]
        self.assertEqual(comparison["common_windows"], 1)
        self.assertEqual(comparison["by_dataset"]["biored_dev"]["pair_agreement"], 1.0)
        schedule = report["models"]["fake"]["schedule"]
        self.assertEqual([item["item"] for item in schedule["items"]], ["dev"])
        self.assertAlmostEqual(schedule["hours_with_margin"], schedule["hours"] * 1.2)
        self.assertEqual(report["run_once"]["charged_model"], "fake")
        self.assertEqual(report["not_costed"], ["S1"])
        # The fake pairwise windows are identical, so pairwise has no usable fit and the
        # schedule, hence the gate, is reported incomplete rather than met.
        once = report["run_once"]["by_model"]["fake"]
        self.assertEqual(report["schedule_complete"], schedule["complete"] and once["complete"])
        self.assertEqual(report["section15_baseline_gate_met"], report["schedule_complete"])
        self.assertIn("strict_recall_first",
                      comparison["by_dataset"]["biored_dev"])
        self.assertEqual(comparison["complete_windows_diagnostic"]["windows"], 1)

    def test_cross_run_comparison_and_validation(self):
        first_path, second_path = self.root / "a.json", self.root / "b.json"
        io.write_json(first_path, f0.run(self.config()))
        repeat = self.config()
        repeat["f0"]["run_id"] = "f0_fake_repeat"
        io.write_json(second_path, f0.run(repeat))
        report = f0_report.report({"run": {"runs_dir": str(self.root / "runs")}, "f0_report": {
            "margin": 0.2, "gate_hours": 72, "plan": [],
            "models": [{"name": "fake", "results": [str(first_path)],
                        "comparisons": [["f0_fake", "json_joint", "f0_fake_repeat", "json_joint"]],
                        "validations": [["json_joint", "f0_fake_repeat", "json_joint"]]}]}})
        model = report["models"]["fake"]
        comparison = model["comparisons"][0]
        self.assertEqual(comparison["second"], "f0_fake_repeat/json_joint")
        self.assertEqual(comparison["by_dataset"]["biored_dev"]["pair_agreement"], 1.0)
        self.assertEqual(model["validations"][0]["calls"], 3)

    def test_schedule_correction_scales_by_the_independent_ratio(self):
        conditions = {"c": {"projected_hours_block": {"w": {"hours": 2.0}}}}
        plan = [{"item": "x", "condition": "c", "workload": "w", "multiplier": 1.0}]
        result = f0_report.schedule_hours(conditions, plan, 0.2, {"c": 0.8})
        self.assertAlmostEqual(result["hours_with_margin"], 2.4)
        self.assertAlmostEqual(result["corrected_by_independent_sample"]["hours_with_margin"],
                               2.0 / 0.8 * 1.2)

    def test_independent_sample_excludes_earlier_windows(self):
        f0.run(self.config())
        config = self.config()
        config["f0"]["run_id"] = "f0_fake_validation"
        config["f0"]["exclude_selections"] = [str(self.root / "runs" / "f0_fake" / "selection.jsonl")]
        config["f0"]["windows_per_stratum"] = 1
        result = f0.run(config)
        self.assertEqual(result["selected_windows"], 0)

    def test_random_sample_option(self):
        config = self.config()
        config["f0"]["run_id"] = "f0_fake_random"
        config["f0"]["random_windows_per_dataset"] = 2
        config["f0"]["conditions"] = [{"name": "json_joint", "form": "joint"}]
        result = f0.run(config)
        self.assertEqual(result["selected_windows"], 2)
        self.assertEqual(result["selected_by_load_band"], {"random": 2})

    def test_results_hold_no_text_or_ids(self):
        result = f0.run(self.config())
        text = str(result)
        for marker in ("AAA", "BBB", "D1", "'7'", "block0"):
            self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
