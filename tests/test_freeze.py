"""The frozen protocol: source digest, locked-split detection and the unlock path."""
import subprocess
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

from gene2disease import corpus, experiments, freeze, io, runner
from test_runner import BIORED, FakeModel


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


class DigestTests(unittest.TestCase):
    def test_code_digest_tracks_every_source_file_including_nested(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            (package / "a.py").write_text("x = 1\n", encoding="utf-8")
            first = freeze.code_digest(package)
            self.assertEqual(first, freeze.code_digest(package))
            (package / "a.py").write_text("x = 2\n", encoding="utf-8")
            second = freeze.code_digest(package)
            self.assertNotEqual(first, second)
            (package / "sub").mkdir()
            (package / "sub" / "b.py").write_text("", encoding="utf-8")
            self.assertNotEqual(second, freeze.code_digest(package))

    def test_locked_splits_per_dataset(self):
        self.assertEqual(freeze.locked_splits({"dataset": "biored", "splits": ["Dev", "Test"]}),
                         {"Test"})
        self.assertEqual(freeze.locked_splits({"dataset": "biotriplex",
                                               "splits": ["train", "val", "test"]}),
                         {"val", "test"})
        self.assertEqual(freeze.locked_splits({"dataset": "biored", "splits": ["Dev"]}), set())

    def test_unlocked_config_needs_no_manifest(self):
        config = {"run": {"stage": "judge"}, "data": {"dataset": "biored", "splits": ["Dev"]}}
        self.assertIsNone(freeze.authorize("unused.toml", config))

    def test_other_stages_cannot_read_locked_splits(self):
        config = {"run": {"stage": "e1"}, "data": {"dataset": "biored", "splits": ["Test"]}}
        with self.assertRaises(corpus.LockedSplitError):
            freeze.authorize("unused.toml", config)


class UnlockTests(unittest.TestCase):
    """A temporary git repository with a Test split, two frozen judge runs and a
    frozen comparison."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        git(self.root, "init", "-q")
        git(self.root, "config", "user.email", "test@example.invalid")
        git(self.root, "config", "user.name", "test")
        io.write_json(self.root / "biored" / "Test.BioC.JSON", BIORED)
        server = HTTPServer(("127.0.0.1", 0), FakeModel)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_port}"
        self.manifest = (self.root / "frozen" / "manifest.json").as_posix()
        self.judges = {form: self.write_judge(form) for form in ("pairwise", "joint")}
        self.compare_path = self.root / "e4_compare.toml"
        self.compare_path.write_text("\n".join(self.head("compare") + [
            "[stats]", "n_resamples = 20", "seed = 1", "level = 0.95",
            "[compare]", 'first = "e4_pairwise"', 'second = "e4_joint"', 'view = "main"',
            "delta = 0.05", "report_precision = true", "load_bands = [1, 5]"]) + "\n",
            encoding="utf-8")
        self.freeze_path = self.root / "freeze.toml"
        listed = ", ".join(f'"{path.as_posix()}"' for path in
                           [*self.judges.values(), self.compare_path])
        self.freeze_path.write_text("\n".join([
            "[run]", 'stage = "freeze"', f'output = "{self.manifest}"',
            "[freeze]", f"configs = [{listed}]"]) + "\n", encoding="utf-8")

    def head(self, stage, name=None):
        return ["[run]", f'stage = "{stage}"',
                f'output = "{(self.root / f"{name or stage}.json").as_posix()}"',
                f'runs_dir = "{(self.root / "runs").as_posix()}"',
                f'freeze_manifest = "{self.manifest}"',
                "[data]", 'dataset = "biored"',
                f'folder = "{(self.root / "biored").as_posix()}"', 'splits = ["Test"]']

    def judge_lines(self, form):
        return self.head("judge", form) + [
            "[judge]", f'run_id = "e4_{form}"', f'form = "{form}"', 'level = "block"',
            f'base_url = "{self.url}"', 'model_id = "fake"', "timeout_seconds = 5",
            "output_base_tokens = 200", "output_tokens_per_pair = 60"]

    def write_judge(self, form, lines=None):
        path = self.root / f"e4_{form}.toml"
        path.write_text("\n".join(lines or self.judge_lines(form)) + "\n", encoding="utf-8")
        return path

    def freeze_and_commit(self):
        experiments.run(self.freeze_path)
        git(self.root, "add", "frozen/manifest.json")
        git(self.root, "commit", "-q", "-m", "freeze")

    def test_frozen_runs_and_comparison_succeed(self):
        with self.assertRaises(corpus.LockedSplitError):
            experiments.run(self.judges["pairwise"])           # not frozen yet
        self.freeze_and_commit()
        result = io.read_json(experiments.run(self.judges["pairwise"]))
        self.assertEqual(result["logical_requests"], 6)
        self.assertEqual(result["manifest"]["freeze_manifest"], self.manifest)
        experiments.run(self.judges["joint"])
        journal = io.read_json(self.root / "runs" / "e4_joint" / runner.LOCKED_JOURNAL)
        self.assertEqual(len(journal["starts"]), 1)
        experiments.run(self.judges["joint"])                   # resume: all from cache
        journal = io.read_json(self.root / "runs" / "e4_joint" / runner.LOCKED_JOURNAL)
        self.assertEqual(len(journal["starts"]), 2)
        compared = io.read_json(experiments.run(self.compare_path))
        self.assertEqual(compared["papers"], 3)
        with self.assertRaises(FileExistsError):
            experiments.run(self.freeze_path)                   # freezing twice

    def test_uncommitted_manifest_is_refused(self):
        experiments.run(self.freeze_path)
        with self.assertRaises(corpus.LockedSplitError):
            experiments.run(self.judges["pairwise"])

    def test_edited_config_is_refused(self):
        self.freeze_and_commit()
        self.write_judge("pairwise", self.judge_lines("pairwise") + ["# edited"])
        with self.assertRaises(corpus.LockedSplitError):
            experiments.run(self.judges["pairwise"])

    def test_changed_data_is_refused(self):
        self.freeze_and_commit()
        changed = dict(BIORED, documents=BIORED["documents"][:2])
        io.write_json(self.root / "biored" / "Test.BioC.JSON", changed)
        with self.assertRaises(corpus.LockedSplitError):
            experiments.run(self.judges["pairwise"])

    def test_source_change_is_refused(self):
        self.freeze_and_commit()
        with mock.patch.object(freeze, "code_digest", return_value="0" * 64):
            with self.assertRaises(corpus.LockedSplitError):
                experiments.run(self.judges["pairwise"])

    def test_foreign_run_directory_is_refused(self):
        self.freeze_and_commit()
        (self.root / "runs" / "e4_pairwise" / "cache").mkdir(parents=True)
        with self.assertRaises(corpus.LockedSplitError):
            experiments.run(self.judges["pairwise"])

    def test_comparison_of_unfrozen_runs_is_refused(self):
        self.freeze_and_commit()
        experiments.run(self.judges["pairwise"])
        with self.assertRaises(corpus.LockedSplitError):
            experiments.run(self.compare_path)                  # e4_joint never started

    def test_config_cannot_unlock_itself(self):
        lines = [line for line in self.judge_lines("pairwise")
                 if not line.startswith("freeze_manifest")]
        lines.insert(1, "locked_authorized = true")
        self.write_judge("pairwise", lines)
        with self.assertRaises(corpus.LockedSplitError):
            experiments.run(self.judges["pairwise"])

    def test_direct_runner_call_stays_refused(self):
        config = io.read_toml(self.judges["pairwise"])
        config["run"]["locked_authorized"] = True
        with self.assertRaises(corpus.LockedSplitError):
            runner.judge(config)

    def test_freeze_requires_configs_to_name_the_manifest(self):
        self.write_judge("pairwise", [line for line in self.judge_lines("pairwise")
                                      if not line.startswith("freeze_manifest")])
        with self.assertRaises(ValueError):
            experiments.run(self.freeze_path)

    def test_duplicate_run_ids_are_refused(self):
        lines = self.judge_lines("joint")
        lines[lines.index('run_id = "e4_joint"')] = 'run_id = "e4_pairwise"'
        self.write_judge("joint", lines)
        with self.assertRaises(ValueError):
            experiments.run(self.freeze_path)

    def test_same_condition_under_another_run_id_is_refused(self):
        lines = self.judge_lines("pairwise")
        lines[lines.index('run_id = "e4_pairwise"')] = 'run_id = "e4_joint"'
        self.write_judge("joint", lines)
        with self.assertRaises(ValueError):
            experiments.run(self.freeze_path)

    def test_second_concurrent_start_fails_before_any_call(self):
        self.freeze_and_commit()
        (self.root / "runs").mkdir()
        (self.root / "runs" / "e4_pairwise.lock").write_text("active", encoding="utf-8")
        calls = len(FakeModel.authorization)
        with self.assertRaises(corpus.LockedSplitError):
            experiments.run(self.judges["pairwise"])
        self.assertEqual(len(FakeModel.authorization), calls)
        self.assertFalse((self.root / "runs" / "e4_pairwise").exists())

    def test_lock_is_released_after_a_run(self):
        self.freeze_and_commit()
        experiments.run(self.judges["pairwise"])
        self.assertFalse((self.root / "runs" / "e4_pairwise.lock").exists())


if __name__ == "__main__":
    unittest.main()
