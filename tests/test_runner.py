"""End-to-end judgment against a fake model server on this machine."""
import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from gene2disease import adjudicate, experiments, io, runner

BIORED = {"documents": [{"id": str(number), "relations": [
    {"infons": {"entity1": "7", "entity2": "D1"}}],
    "passages": [{"offset": 0, "text": "AAA causes BBB. CCC is here.", "annotations": [
        {"text": "AAA", "infons": {"type": "GeneOrGeneProduct", "identifier": "7"},
         "locations": [{"offset": 0, "length": 3}]},
        {"text": "BBB", "infons": {"type": "DiseaseOrPhenotypicFeature", "identifier": "D1"},
         "locations": [{"offset": 11, "length": 3}]},
        {"text": "CCC", "infons": {"type": "DiseaseOrPhenotypicFeature", "identifier": "D2"},
         "locations": [{"offset": 16, "length": 3}]}]}]} for number in (1, 2, 3)]}


class FakeModel(BaseHTTPRequestHandler):
    """Asserts the first listed pair only, so joint calls drop the second pair's edge."""
    authorization: list = []

    def do_GET(self):
        data = json.dumps({"default_generation_settings": {"n_ctx": 4096},
                           "total_slots": 1}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        if body["model"] == "toolong":
            FakeModel.authorization.append(self.headers.get("Authorization"))
            data = json.dumps({"error": {"code": 400, "type": "exceed_context_size_error",
                                         "message": "request exceeds context"}}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        prompt = body["messages"][0]["content"]
        pairs = re.findall(r'- gene "([^"]+)" \(.*?\), disease "([^"]+)"', prompt)
        sentence = re.search(r"\[([^\]]+\.s\d+)\]", prompt).group(1)
        entries = [{"gene": gene, "disease": disease, "assertions": [] if index else [{
            "polarity": "positive", "certainty": "stated", "relation_class": "association",
            "provenance": "this_study", "sentence_ids": [sentence]}]}
            for index, (gene, disease) in enumerate(pairs)]
        if body["model"] == "omitter" and len(entries) > 1:
            entries = entries[1:]
        FakeModel.authorization.append(self.headers.get("Authorization"))
        if "<gene id>|<disease id>|none" in prompt:
            content = "\n".join(
                f"{e['gene']}|{e['disease']}|pos|stated|assoc|this|{sentence}" if e["assertions"]
                else f"{e['gene']}|{e['disease']}|none" for e in entries)
        else:
            content = json.dumps({"pairs": entries})
        answer = {"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                  "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 10 * len(entries)},
                  "timings": {"cache_n": 0, "prompt_n": len(prompt) // 4, "prompt_ms": 5.0,
                              "predicted_ms": 20.0 * len(entries)}}
        data = json.dumps(answer).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *arguments):
        pass


class PipelineTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        io.write_json(self.root / "biored" / "Dev.BioC.JSON", BIORED)
        server = HTTPServer(("127.0.0.1", 0), FakeModel)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_port}"

    def config(self, form):
        return {"run": {"runs_dir": str(self.root / "runs")},
                "data": {"dataset": "biored", "folder": str(self.root / "biored"),
                         "splits": ["Dev"]},
                "judge": {"run_id": form, "form": form, "level": "block", "base_url": self.url,
                          "model_id": "fake", "timeout_seconds": 5,
                          "output_base_tokens": 200, "output_tokens_per_pair": 60}}

    def test_judge_and_compare_end_to_end(self):
        pairwise = runner.judge(self.config("pairwise"))
        joint = runner.judge(self.config("joint"))
        self.assertEqual(pairwise["logical_requests"], 6)
        self.assertEqual(pairwise["backend_attempts"], 6)
        self.assertEqual(joint["logical_requests"], 3)
        self.assertEqual(pairwise["dispositions"], {"edge": 6})
        self.assertEqual(joint["dispositions"], {"edge": 3, "no_assertion": 3})
        config = self.config("pairwise")
        config["stats"] = {"n_resamples": 50, "seed": 1, "level": 0.95}
        config["compare"] = {"first": "pairwise", "second": "joint", "view": "main",
                             "delta": 0.05, "report_precision": True, "load_bands": [1, 5]}
        result = runner.compare(config)
        self.assertEqual(result["papers"], 3)
        self.assertEqual(result["recall"]["first"], 1.0)
        self.assertEqual(result["recall"]["second"], 1.0)
        self.assertEqual(result["recall"]["reading"], "equivalent_within_delta")
        self.assertAlmostEqual(result["precision"]["first"], 0.5)
        self.assertEqual(result["precision"]["second"], 1.0)
        self.assertEqual(result["precision"]["reading"], "second_meaningfully_higher")

    def compare_config(self, **overrides):
        config = self.config("pairwise")
        config["stats"] = {"n_resamples": 50, "seed": 1, "level": 0.95}
        config["compare"] = {"first": "pairwise", "second": "joint", "view": "main",
                             "delta": 0.05, "report_precision": False,
                             "load_bands": [1, 5], **overrides}
        return config

    def test_gold_positive_omission_counts_against_the_form(self):
        pairwise, joint = self.config("pairwise"), self.config("joint")
        pairwise["judge"]["model_id"] = joint["judge"]["model_id"] = "omitter"
        runner.judge(pairwise)
        result = runner.judge(joint)
        self.assertEqual(result["dispositions"], {"omitted_by_model": 3, "no_assertion": 3})
        comparison = runner.compare(self.compare_config())
        self.assertEqual(comparison["recall"]["first"], 1.0)
        self.assertEqual(comparison["recall"]["second"], 0.0)
        self.assertEqual(comparison["recall"]["reading"], "first_meaningfully_higher")
        window = comparison["window_level"]
        self.assertEqual(window["gold_exposures_by_window_load"],
                         {"<=5": {"gold_exposures": 3, "found_first": 3, "found_second": 0,
                                  "found_both": 0, "only_first": 3, "only_second": 0}})
        self.assertEqual(window["gold_exposures_by_request_size"],
                         {"first": {"<=1": {"gold_exposures": 3, "found": 3}},
                          "second": {"<=5": {"gold_exposures": 3, "found": 0}}})

    def test_failed_run_scores_as_missing_predictions(self):
        runner.judge(self.config("pairwise"))
        joint = self.config("joint")
        joint["judge"]["base_url"] = "http://127.0.0.1:9"
        runner.judge(joint)
        result = runner.compare(self.compare_config())
        self.assertEqual(result["recall"]["second"], 0.0)
        self.assertEqual(result["strict_reading"]["recall"]["second"], 0.0)
        diagnostic = result["complete_windows_diagnostic"]
        self.assertEqual((diagnostic["windows"], diagnostic["windows_complete_in_both"]), (3, 0))
        self.assertNotIn("recall_first", diagnostic)

    def test_strict_reading_and_complete_windows_are_reported(self):
        runner.judge(self.config("pairwise"))
        runner.judge(self.config("joint"))
        result = runner.compare(self.compare_config())
        self.assertEqual(result["strict_reading"]["recall"]["first"], result["recall"]["first"])
        diagnostic = result["complete_windows_diagnostic"]
        self.assertEqual(diagnostic["windows_complete_in_both"], 3)
        self.assertEqual((diagnostic["recall_first"], diagnostic["recall_second"]), (1.0, 1.0))

    def test_strict_reading_differs_when_canonicalization_mattered(self):
        runner.judge(self.config("pairwise"))
        runner.judge(self.config("joint"))
        for path in (self.root / "runs" / "joint").glob("*/calls.jsonl"):
            calls = io.read_jsonl(path)
            for call in calls:
                call["raw_pair_outcomes"] = {k: "omitted_by_model" for k in call["raw_pair_outcomes"]}
            io.write_jsonl(path, calls)
        result = runner.compare(self.compare_config())
        self.assertEqual(result["recall"]["second"], 1.0)
        self.assertEqual(result["strict_reading"]["recall"]["second"], 0.0)

    def test_instruction_set_is_recorded_and_must_be_allowed_to_differ(self):
        runner.judge(self.config("pairwise"))
        guided = self.config("pairwise")
        guided["judge"].update(run_id="guided", instructions="biored_guideline")
        runner.judge(guided)
        self.assertNotIn("instructions", io.read_json(self.root / "runs" / "pairwise" / "run.json"))
        self.assertEqual(io.read_json(self.root / "runs" / "guided" / "run.json")["instructions"], "biored_guideline")
        with self.assertRaisesRegex(ValueError, "differ in instructions"):
            runner.compare(self.compare_config(second="guided"))
        result = runner.compare(self.compare_config(second="guided", allow_differing=["instructions"]))
        self.assertEqual(result["allowed_to_differ"], ["instructions"])
        self.assertIsNotNone(result["window_level"])
        with self.assertRaisesRegex(ValueError, "cannot differ"):
            runner.compare(self.compare_config(second="guided", allow_differing=["model_id"]))

    def test_input_scope_comparison_skips_window_matched_diagnostics(self):
        runner.judge(self.config("pairwise"))
        other = self.config("pairwise")
        other["judge"]["run_id"] = "scoped"
        runner.judge(other)
        path = self.root / "runs" / "scoped" / "run.json"
        io.write_json(path, {**io.read_json(path), "input_scope": "abstract"})
        with self.assertRaisesRegex(ValueError, "differ in input_scope"):
            runner.compare(self.compare_config(second="scoped"))
        result = runner.compare(self.compare_config(second="scoped",
                                                    allow_differing=["input_scope"]))
        self.assertIsNone(result["window_level"])
        self.assertIsNone(result["complete_windows_diagnostic"])
        self.assertEqual(result["recall"]["first"], result["recall"]["second"])

    def test_abstract_scope_needs_abstract_blocks(self):
        config = self.config("pairwise")
        config["judge"]["input_scope"] = "abstract"
        with self.assertRaisesRegex(ValueError, "no title or abstract blocks"):
            runner.judge(config)
        config["judge"]["input_scope"] = "methods"
        with self.assertRaisesRegex(ValueError, "unknown input scope"):
            runner.judge(config)

    def test_truncated_pair_file_is_refused(self):
        runner.judge(self.config("pairwise"))
        runner.judge(self.config("joint"))
        io.write_jsonl(self.root / "runs" / "joint" / "BioRED1" / "pairs.jsonl", [])
        with self.assertRaisesRegex(ValueError, "incomplete pair records"):
            runner.compare(self.compare_config())

    def test_run_id_cannot_be_reused_with_other_settings(self):
        config = self.config("joint")
        runner.judge(config)
        config["judge"]["model_id"] = "other"
        with self.assertRaisesRegex(ValueError, "different settings"):
            runner.judge(config)
        config = self.config("joint")
        config["judge"]["max_papers"] = 1
        with self.assertRaisesRegex(ValueError, "different settings"):
            runner.judge(config)

    def test_runs_from_different_models_are_not_compared(self):
        runner.judge(self.config("pairwise"))
        other = self.config("joint")
        other["judge"]["model_id"] = "other"
        runner.judge(other)
        with self.assertRaisesRegex(ValueError, "model_id"):
            runner.compare(self.compare_config())

    def test_biotriplex_precision_is_refused(self):
        config = self.compare_config(report_precision=True)
        config["data"]["dataset"] = "biotriplex"
        with self.assertRaisesRegex(ValueError, "recall only"):
            runner.compare(config)

    def test_second_run_uses_cache_and_reports_no_inference(self):
        config = self.config("joint")
        runner.judge(config)
        config["judge"]["base_url"] = "http://127.0.0.1:9"
        repeated = runner.judge(config)
        self.assertEqual(repeated["failed_calls"], 0)
        self.assertEqual((repeated["backend_attempts"], repeated["cache_hits"]), (0, 3))

    def test_server_down_is_recorded_not_fatal(self):
        config = self.config("joint")
        config["judge"]["base_url"] = "http://127.0.0.1:9"
        result = runner.judge(config)
        self.assertEqual(result["failed_calls"], 3)
        self.assertEqual(result["dispositions"], {"failed": 6})

    def test_context_refusal_is_terminal_and_replayed_without_a_call(self):
        config = self.config("joint")
        config["judge"]["model_id"] = "toolong"
        first = runner.judge(config)
        self.assertEqual(first["context_refused_calls"], 3)
        self.assertEqual(first["dispositions"], {"failed": 6})
        calls = len(FakeModel.authorization)
        second = runner.judge(config)                     # resume: replayed from the cache
        self.assertEqual(len(FakeModel.authorization), calls)
        self.assertEqual(second["context_refused_calls"], 3)
        self.assertEqual(second["cache_hits"], 3)

    def test_server_failure_stops_the_run_before_recording_the_paper(self):
        config = self.config("pairwise")
        config["judge"]["base_url"] = "http://127.0.0.1:9"   # nothing listens there
        config["judge"]["stop_on_operational_failure"] = True
        with self.assertRaises(RuntimeError):
            runner.judge(config)
        self.assertFalse(list((self.root / "runs" / "pairwise").glob("*/pairs.jsonl")))

    def test_compare_refuses_a_missing_run(self):
        with self.assertRaises(io.MissingInputError):
            runner.compare(self.compare_config(first="absent"))

    def test_locked_split_refused_at_stage_level(self):
        config = self.config("joint")
        config["data"]["splits"] = ["Test"]
        with self.assertRaises(runner.corpus.LockedSplitError):
            runner.judge(config)

    def test_stages_are_registered(self):
        self.assertEqual(set(experiments.STAGES),
                         {"e1", "workload", "judge", "compare", "f0", "f0_report", "freeze"})
        self.assertIs(adjudicate.CallFailure.__base__, Exception)


class CountTypeTests(unittest.TestCase):
    def test_single_exposure_band_counts_are_integers(self):
        # One gold exposure in a band: an empty Counter would store the booleans as given.
        with tempfile.TemporaryDirectory() as folder:
            runs = []
            for name, outcome in (("one", "edge"), ("two", "no_assertion")):
                run = Path(folder) / name
                io.write_jsonl(run / "P1" / "calls.jsonl", [{
                    "window_id": "P1.block0", "window_pairs": 1, "pairs": 1, "outcome": "ok",
                    "pair_outcomes": {"gene:1|disease:D1": outcome}}])
                runs.append(run)
            papers = {"P1": (None, [], {("gene:1", "disease:D1")})}
            level = runner._window_level(runs[0], runs[1], papers, ["P1"], [1, 4])
            complete = runner._complete_windows(runs[0], runs[1], papers, ["P1"])
        band = level["gold_exposures_by_window_load"]["<=1"]
        self.assertEqual(band, {"gold_exposures": 1, "found_first": 1, "found_second": 0,
                                "found_both": 0, "only_first": 1, "only_second": 0})
        self.assertTrue(all(type(value) is int for value in band.values()))
        for side in level["gold_exposures_by_request_size"].values():
            self.assertTrue(all(type(v) is int for c in side.values() for v in c.values()))
        self.assertIs(type(complete["found_second"]), int)


if __name__ == "__main__":
    unittest.main()
