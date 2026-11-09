import json
import tempfile
import unittest
from pathlib import Path

from gene2disease import entities
from gene2disease.corpus import Block, Document
from gene2disease.schema import EntityStatus

PASSAGE = "TP53  was mutated in glioma."
PAYLOAD = {"PubTator3": [{"passages": [
    {"offset": 100, "text": PASSAGE, "infons": {"section_type": "RESULTS"}, "annotations": [
        {"text": "TP53", "infons": {"type": "Gene", "identifier": "7157"},
         "locations": [{"offset": 100, "length": 4}]},
        {"text": "glioma", "infons": {"type": "Disease", "identifier": "MESH:D005910"},
         "locations": [{"offset": 121, "length": 6}]},
        {"text": "mutated", "infons": {"type": "Chemical", "identifier": "x"},
         "locations": [{"offset": 110, "length": 7}]}]},
    {"offset": 500, "text": "MYC in lymphoma", "infons": {"section_type": "REF"}, "annotations": [
        {"text": "MYC", "infons": {"type": "Gene", "identifier": "4609"},
         "locations": [{"offset": 500, "length": 3}]}]},
    {"offset": 700, "text": "EGFR lung cancer", "infons": {"section_type": "TABLE"}, "annotations": [
        {"text": "EGFR", "infons": {"type": "Gene", "identifier": "1956"},
         "locations": [{"offset": 700, "length": 4}]}]}]}]}


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "1.json"
        path.write_text(json.dumps(PAYLOAD), encoding="utf-8")
        self.annotations = entities.read_pubtator(path)
        self.document = Document("d", [
            Block("d.b0", "Results", "paragraph", "Intro. TP53 was mutated in glioma."),
            Block("d.b1", "Results", "table_cell", "EGFR | lung cancer")])

    def test_reader_keeps_only_genes_and_diseases_outside_references(self):
        self.assertEqual({item.text for item in self.annotations}, {"TP53", "glioma", "EGFR"})

    def test_alignment_survives_whitespace_and_reports_counts(self):
        mentions, report = entities.align(self.document, self.annotations)
        self.assertEqual(report.aligned, 3)
        self.assertEqual(report.unaligned, 0)
        for mention in mentions:
            block = next(b for b in self.document.blocks if b.block_id == mention.block_id)
            self.assertEqual(block.text[mention.start:mention.end], mention.text)
        spans = {m.text: (m.block_id, m.start, m.end) for m in mentions}
        self.assertEqual(spans["TP53"], ("d.b0", 7, 11))
        self.assertEqual(spans["glioma"], ("d.b0", 27, 33))
        self.assertEqual(spans["EGFR"], ("d.b1", 0, 4))

    def test_unplaced_annotation_is_counted_not_dropped_silently(self):
        document = Document("d", [Block("d.b0", "R", "paragraph", "Nothing relevant.")])
        mentions, report = entities.align(document, self.annotations)
        self.assertEqual(mentions, [])
        self.assertEqual(report.unaligned, 3)
        self.assertEqual(report.unaligned_by_section["RESULTS"], 2)

    def test_citation_markers_and_dashes_do_not_block_alignment(self):
        passage = "The anti-TP53 signal in glioma was strong and clear today."
        annotation = entities.RawAnnotation("gene", "7157", "TP53", "RESULTS", passage, 9)
        block = Block("d.b0", "R", "paragraph",
                      "The anti\u2010TP53 signal [12] in glioma was strong and clear today.")
        mentions, report = entities.align(Document("d", [block]), [annotation])
        self.assertEqual(report.aligned, 1)
        self.assertEqual((mentions[0].start, mentions[0].end), (9, 13))

    def test_second_occurrence_maps_to_second_occurrence(self):
        passage = "TP53 and more words here then TP53 again in this sentence."
        annotation = entities.RawAnnotation("gene", "7157", "TP53", "RESULTS", passage, 30)
        block = Block("d.b0", "R", "paragraph", passage)
        mentions, _ = entities.align(Document("d", [block]), [annotation])
        self.assertEqual(mentions[0].start, 30)

    def test_word_order_settles_a_tie(self):
        passage = "MYC activates TP53 strongly in these cells."
        first = Block("d.b0", "R", "paragraph", "TP53 activates MYC strongly in these cells.")
        second = Block("d.b1", "R", "paragraph", passage)
        annotation = entities.RawAnnotation("gene", "7157", "TP53", "RESULTS", passage, 14)
        mentions, _ = entities.align(Document("d", [first, second]), [annotation])
        self.assertEqual(mentions[0].block_id, "d.b1")

    def test_identical_blocks_are_not_guessed(self):
        passage = "TP53 is altered in these cells today."
        blocks = [Block(f"d.b{i}", "R", "paragraph", passage) for i in range(2)]
        annotation = entities.RawAnnotation("gene", "7157", "TP53", "RESULTS", passage, 0)
        mentions, report = entities.align(Document("d", blocks), [annotation])
        self.assertEqual(mentions, [])
        self.assertEqual(report.unaligned, 1)

    def test_unavailable_occurrence_is_unaligned_not_clamped(self):
        passage = "TP53 then more words here and TP53 again"
        block = Block("d.b0", "R", "paragraph", "TP53 then more words here and again")
        annotation = entities.RawAnnotation("gene", "7157", "TP53", "RESULTS", passage, 30)
        mentions, report = entities.align(Document("d", [block]), [annotation])
        self.assertEqual((mentions, report.unaligned), ([], 1))

    def test_table_fallback_stays_inside_the_matched_table(self):
        passage = "Gene Disease TP53 glioma KRAS colon"
        rows = [Block("d.b0", "R", "table_cell", "Gene | Disease", group="table0"),
                Block("d.b1", "R", "table_cell", "TP53 | glioma", group="table0"),
                Block("d.b2", "R", "table_cell", "KRAS | colon", group="table0"),
                Block("d.b3", "R", "table_cell", "TP53 | unrelated other values", group="table1")]
        annotation = entities.RawAnnotation("gene", "7157", "TP53", "TABLE", passage, 13)
        mentions, _ = entities.align(Document("d", rows), [annotation])
        self.assertEqual([m.block_id for m in mentions], ["d.b1"])

    def test_ambiguous_identifier_is_ineligible(self):
        annotation = entities.RawAnnotation("gene", "1;2", "X1", "RESULTS", "X1 here", 0)
        self.assertEqual(entities.ineligible_entities([annotation]), {"gene:1;2"})

    def test_unknown_section_label_is_not_copied_into_results(self):
        report = entities.AlignmentReport()
        report.miss("free text label")
        self.assertEqual(report.unaligned_by_section, {"other": 1})

    def test_exact_passage_offset_beats_first_occurrence(self):
        passage = "TP53 binds MYC."
        block = Block("d.b0", "R", "paragraph", "TP53 summary. TP53 binds MYC.")
        annotation = entities.RawAnnotation("gene", "7157", "TP53", "RESULTS", passage, 0)
        mentions, _ = entities.align(Document("d", [block]), [annotation])
        self.assertEqual(mentions[0].start, 14)

    def test_context_chooses_among_occurrences_when_offsets_do_not_transfer(self):
        passage = "Early TP53 summary line and then later TP53 binds MYC strongly."
        block = Block("d.b0", "R", "paragraph",
                      "Early TP53 summary line [4] and then later TP53 binds MYC strongly.")
        annotation = entities.RawAnnotation("gene", "7157", "TP53", "RESULTS", passage, 39)
        mentions, _ = entities.align(Document("d", [block]), [annotation])
        self.assertEqual(mentions[0].start, block.text.rindex("TP53"))

    def test_status(self):
        self.assertIs(entities.entity_status("7157"), EntityStatus.NORMALIZED)
        self.assertIs(entities.entity_status("-"), EntityStatus.UNNORMALIZED)
        self.assertIs(entities.entity_status("1;2"), EntityStatus.AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
