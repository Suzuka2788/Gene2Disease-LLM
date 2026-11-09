import tempfile
import unittest
from pathlib import Path

from gene2disease import io


class IoTests(unittest.TestCase):
    def test_missing_file_raises_instead_of_returning_empty(self):
        for reader in (io.read_json, io.read_jsonl, io.read_text, io.read_toml):
            with self.assertRaises(io.MissingInputError):
                reader("does/not/exist.file")

    def test_jsonl_round_trip_and_explicit_empty_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "rows.jsonl"
            io.write_jsonl(path, [{"a": 1}, {"b": "基因"}])
            self.assertEqual(io.read_jsonl(path), [{"a": 1}, {"b": "基因"}])
            io.write_jsonl(path, [])
            self.assertEqual(io.read_jsonl(path), [])

    def test_bad_jsonl_line_reports_position(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ":2:"):
                io.read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
