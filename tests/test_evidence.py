"""Evidence assessment: checks A and B, frame and strata, sampling, and the sheets."""
import csv
import tempfile
import unittest
from pathlib import Path

from gene2disease import evidence, io
from gene2disease.corpus import Block, Document, GoldLink, Mention, Sentence

TEXT = "AAA causes BBB. CCC is unrelated. AAA and DDD."
BLOCK = Block("P1.b0", "Results", "paragraph", TEXT,
              [Sentence("P1.b0.s0", 0, 15), Sentence("P1.b0.s1", 16, 33), Sentence("P1.b0.s2", 34, 46)])
OTHER = Block("P1.b1", "Discussion", "paragraph", "It is linked to EEE.",
              [Sentence("P1.b1.s0", 0, 20)])
TITLE = Block("P1.t", "title", "paragraph", "A title", [Sentence("P1.t.s0", 0, 7)])
MENTIONS = [Mention("gene:a", "gene", "P1.b0", 0, 3, "AAA"),
            Mention("disease:b", "disease", "P1.b0", 11, 14, "BBB"),
            Mention("disease:c", "disease", "P1.b0", 16, 19, "CCC"),
            Mention("gene:a", "gene", "P1.b0", 34, 37, "AAA"),
            Mention("disease:d", "disease", "P1.b0", 42, 45, "DDD"),
            Mention("disease:e", "disease", "P1.b1", 16, 19, "EEE")]
# The trigger "causes" sits at block-local 4..10 of P1.b0, inside sentence s0 (0..15).
PAPER = evidence.Paper(Document("P1", [TITLE, BLOCK, OTHER]), MENTIONS, {("gene:a", "disease:b")},
                       [GoldLink("gene:a", "disease:b", "P1.b0", 4, 10, "causes", "result")])


def edge(gene, disease, *cited):
    return {"gene": gene, "disease": disease, "disposition": "edge", "reason": "",
            "assertions": [{"polarity": "positive", "certainty": "stated",
                            "relation_class": "association", "provenance": "this_study",
                            "sentence_ids": list(ids)} for ids in cited]}


def fill(path, **values):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(values)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        # pairwise: a-b (gold; two assertions, citing s0 and s1), a-c (cites s1, no gene there),
        # a-d (cites s2), a-e (cites the other passage, where the gene is only "It")
        self.write_run("pw", [edge("gene:a", "disease:b", ["P1.b0.s0"], ["P1.b0.s1"]),
                              edge("gene:a", "disease:c", ["P1.b0.s1"]),
                              edge("gene:a", "disease:d", ["P1.b0.s2"]),
                              edge("gene:a", "disease:e", ["P1.b1.s0"])])
        # joint: a-b citing s1 only (misses the trigger) and a-d
        self.write_run("jt", [edge("gene:a", "disease:b", ["P1.b0.s1"]),
                              edge("gene:a", "disease:d", ["P1.b0.s2"])])
        self.runs = {"pw": (self.root / "pw", "pairwise"), "jt": (self.root / "jt", "joint")}

    def write_run(self, name, rows):
        io.write_jsonl(self.root / name / "P1" / "pairs.jsonl", rows)

    def sheet(self):
        frame = evidence.build_frame({"P1": PAPER}, self.runs)
        items, _ = evidence.draw(frame, self.runs, {"S1": 9, "S2": 9, "G": 9}, 7, "biored")
        folder = self.root / "sheet"
        evidence.write_stage1(items, {"biored": {"P1": PAPER}}, folder, 7)
        return folder

    def test_block_of_sentence_ids(self):
        self.assertEqual(evidence.block_of("BioRED123.b1.s9"), "BioRED123.b1")
        self.assertEqual(evidence.block_of("BTX5.x2.p210.s11"), "BTX5.x2.p210")

    def test_check_a_uses_the_union_of_one_assertions_citations(self):
        self.write_run("cross", [edge("gene:a", "disease:c", ["P1.b0.s0", "P1.b0.s1"])])
        self.assertEqual(evidence.check_a(self.root / "cross", {"P1": PAPER})
                         ["both_entities_in_cited_sentences"], 1)    # gene in s0, disease in s1
        result = evidence.check_a(self.root / "pw", {"P1": PAPER})
        # a-b s0 yes, a-b s1 no, a-c no, a-d yes, a-e no (coreference is not resolved)
        self.assertEqual((result["positive_assertions"], result["both_entities_in_cited_sentences"]),
                         (5, 2))

    def test_check_b_trigger_overlap(self):
        self.assertEqual(evidence.check_b(self.root / "pw", {"P1": PAPER})["overlapping"], 1)
        joint = evidence.check_b(self.root / "jt", {"P1": PAPER})
        self.assertEqual((joint["gold_edges_with_trigger"], joint["overlapping"]), (1, 0))

    def test_check_b_boundary_and_missing_trigger(self):
        touching = evidence.Paper(PAPER.document, MENTIONS, PAPER.gold,
                                  [GoldLink("gene:a", "disease:b", "P1.b0", 15, 16, "x", "")])
        self.assertEqual(evidence.check_b(self.root / "jt", {"P1": touching})["overlapping"], 0)
        unplaced = evidence.Paper(PAPER.document, MENTIONS, PAPER.gold,
                                  [GoldLink("gene:a", "disease:b", None, 0, 0, "x", "")])
        result = evidence.check_b(self.root / "pw", {"P1": unplaced})
        self.assertEqual((result["gold_edges_with_trigger"],
                          result["gold_edges_without_trigger_position"]), (0, 1))

    def test_strata_and_multiple_citation_sets(self):
        frame = evidence.build_frame({"P1": PAPER}, self.runs)
        strata = {key[2]: evidence.stratum(item, self.runs) for key, item in frame.items()}
        self.assertEqual(strata, {"disease:b": "G", "disease:c": "S1", "disease:d": "S2",
                                  "disease:e": "S1"})
        self.assertEqual(len(frame[("P1", "gene:a", "disease:b")].passages["P1.b0"]["pw"]), 2)

    def test_assertion_spanning_two_passages_is_refused(self):
        self.write_run("bad", [edge("gene:a", "disease:b", ["P1.b0.s0", "P1.b1.s0"])])
        with self.assertRaises(ValueError):
            evidence.build_frame({"P1": PAPER}, {"bad": (self.root / "bad", "pairwise")})

    def test_draw_is_deterministic_and_records_run_counts(self):
        frame = evidence.build_frame({"P1": PAPER}, self.runs)
        plan = {"S1": 5, "S2": 5, "G": 5}
        first = evidence.draw(frame, self.runs, plan, 7, "biored")
        self.assertEqual(first, evidence.draw(frame, self.runs, plan, 7, "biored"))
        items, design = first
        self.assertEqual(len(items), 4)                       # small strata are taken whole
        self.assertEqual(design["strata"]["S2"]["N_by_run"], {"jt": 1, "pw": 1})

    def test_size_rule(self):
        plans = {"biored": {"S1": 18, "S2": 16, "G": 6}, "biotriplex": {"S1": 16, "S2": 14, "G": 6}}
        same, minutes = evidence.scaled_plans(plans, 5.0)
        self.assertEqual((same, minutes), (plans, 380.0))
        scaled, minutes = evidence.scaled_plans(plans, 6.5)
        self.assertLessEqual(minutes, 480)                    # rounding down keeps the budget
        self.assertEqual(scaled["biored"], {"S1": 17, "S2": 15, "G": 5})
        fitted, minutes = evidence.scaled_plans(plans, 16.0)
        self.assertLessEqual(minutes, 480)                    # the minimum of 4 is accounted for
        self.assertGreater(sum(sum(p.values()) for p in fitted.values()), 24)   # above the minimum
        floor, minutes = evidence.scaled_plans(plans, 30.0)
        self.assertEqual(floor["biored"], {"S1": 4, "S2": 4, "G": 4})
        self.assertGreater(minutes, 480)                      # minimum design, time reported
        for bad in (0.0, -1.0, float("inf")):
            with self.assertRaises(ValueError):
                evidence.scaled_plans(plans, bad)

    def test_stage1_hides_the_source(self):
        page = (self.sheet() / "stage1.html").read_text(encoding="utf-8")
        for hidden in ("pairwise", "joint", ">S1<", ">S2<", "stated", "this_study", "P1.b0.s"):
            self.assertNotIn(hidden, page)
        self.assertIn('<span class=\\"gene\\">AAA<\\/span>', page)   # embedded as JSON
        self.assertIn('stage1_labels.csv', page)

    def test_stage2_needs_every_item_once_with_valid_fields(self):
        folder = self.sheet()
        papers = {"biored": {"P1": PAPER}}
        with self.assertRaises(ValueError):                    # labels still empty
            evidence.write_stage2(folder, papers, 7)
        rows = fill(folder / "stage1_labels.csv", q1="supported", fulltext_consulted="no",
                    minutes="0")
        with self.assertRaises(ValueError):                    # zero minutes
            evidence.write_stage2(folder, papers, 7)
        fill(folder / "stage1_labels.csv", minutes="3")
        with open(folder / "stage1_labels.csv", "a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([rows[0]["item_id"], "supported", "", "", "no", "3", ""])
        with self.assertRaises(ValueError):                    # duplicate row
            evidence.write_stage2(folder, papers, 7)

    def test_stage1_is_locked_and_timing_covers_both_stages(self):
        folder = self.sheet()
        papers = {"biored": {"P1": PAPER}}
        fill(folder / "stage1_labels.csv", q1="supported", fulltext_consulted="no", minutes="3")
        evidence.write_stage2(folder, papers, 7)
        self.assertIn('class=\\"cited\\"', (folder / "stage2.html").read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            evidence.write_stage2(folder, papers, 7)
        fill(folder / "stage2_labels.csv", q2="supported")
        with self.assertRaises(ValueError):                    # stage-2 minutes missing
            evidence.median_item_minutes(folder)
        key = io.read_json(folder / "_key_DO_NOT_OPEN.json")
        first = {next(iter(item["citation_sets"])) for item in key["items"]}
        path = folder / "stage2_labels.csv"
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["minutes"] = "1" if row["set_id"] in first else ""
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual(evidence.median_item_minutes(folder), 4.0)   # 3 + 1 per item
        fill(folder / "stage1_labels.csv", q1="negative")      # edit after the lock
        with self.assertRaises(ValueError):
            evidence.verify_stage1_lock(folder)


if __name__ == "__main__":
    unittest.main()
