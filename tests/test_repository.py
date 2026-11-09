"""Rules that protect against the old project's failures."""
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "src" / "gene2disease").glob("*.py"))


class RepositoryTests(unittest.TestCase):
    def test_only_stats_module_resamples(self):
        for path in SOURCES:
            text = path.read_text(encoding="utf-8")
            if path.name != "stats.py":
                self.assertNotRegex(text, r"^\s*(import|from)\s+random", path.name)

    def test_no_seed_or_resample_literals_outside_configuration(self):
        for path in SOURCES:
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"seed\s*=\s*\d", path.name)
            self.assertNotRegex(text, r"n_resamples\s*=\s*\d", path.name)

    def test_single_entry_point(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(list(project["project"]["scripts"]), ["g2d"])
        mains = [path.name for path in SOURCES if "__main__" in path.read_text(encoding="utf-8")]
        self.assertEqual(mains, ["cli.py"])

    def test_no_sys_path_edits(self):
        for path in list(SOURCES) + sorted((ROOT / "tests").glob("*.py")):
            if path.name != Path(__file__).name:
                self.assertNotIn("sys.path", path.read_text(encoding="utf-8"), path.name)

    def test_restricted_locations_are_ignored(self):
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        for required in ("data/", "runs/", "deliverables/", "vendor/", "annotation/sheets/"):
            self.assertIn(required, lines)

    def test_sources_hold_no_data_paths(self):
        for path in SOURCES:
            self.assertNotIn("archive_v1", path.read_text(encoding="utf-8"), path.name)

    def test_configs_do_not_enable_locked_splits(self):
        for path in sorted((ROOT / "configs").glob("*.toml")):
            config = tomllib.loads(path.read_text(encoding="utf-8"))
            for section in config.values():
                self.assertFalse(set(section.get("splits", [])) & {"val", "test", "Test"}, path.name)


if __name__ == "__main__":
    unittest.main()
