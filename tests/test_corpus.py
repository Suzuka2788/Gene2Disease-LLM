import json
import tempfile
import unittest
from pathlib import Path

from gene2disease import corpus

JATS = """<article><front><article-meta><title-group><article-title>BRCA1 and cancer
</article-title></title-group><abstract><p>Abstract text. Second one.</p></abstract>
</article-meta></front><body><sec><title>Results</title>
<p>TP53 was mutated in glioma (Fig. 1). This was seen twice.
<fig><label>Figure 1</label><caption><p>KRAS in colon cancer.</p></caption></fig></p>
<table-wrap><caption><p>Table of genes</p></caption><table><thead><tr><th>Gene</th><th>Disease</th></tr></thead>
<tbody><tr><td>EGFR</td><td>lung cancer</td></tr></tbody></table></table-wrap>
</sec></body><back><ref-list><ref><mixed-citation>MYC in lymphoma.</mixed-citation></ref></ref-list></back></article>"""


def biotriplex_paper(split="train"):
    text = "Duchenne muscular dystrophy (DMD) is caused by DYS loss.\nSecond line."
    return {"split": split, "section_data": {"Abstract": {
        "text": text,
        "genes": {"G0": {"spans": "47~50", "text": "DYS", "location": "result"}},
        "diseases": {"D0": {"spans": "0~27", "text": "Duchenne muscular dystrophy",
                            "location": "result"},
                     "D1": {"spans": "29~32", "text": "DMD", "location": "result"}},
        "relations": {"R0": {"spans": "37~43", "text": "caused", "location": "result",
                             "relation": "causative mutation"}},
        "entity_linkings": {"E0": {"geneID": "G0", "geneText": "DYS", "diseaseID": "D1",
                                   "diseaseText": "DMD", "relationID": "R0",
                                   "relationText": "caused"}}}}}


class SentenceTests(unittest.TestCase):
    def test_lowercase_gene_can_start_a_sentence(self):
        text = "Glioma was diagnosed. p53 expression was measured."
        self.assertEqual(len(corpus.split_sentences(text)), 2)

    def test_initial_does_not_end_a_sentence(self):
        self.assertEqual(len(corpus.split_sentences("In S. cerevisiae cells this holds.")), 1)

    def test_splits_and_protects_abbreviations(self):
        text = "Smith et al. found it. See Fig. 2 for more. Done"
        pieces = [text[a:b] for a, b in corpus.split_sentences(text)]
        self.assertEqual(pieces, ["Smith et al. found it.", "See Fig. 2 for more.", "Done"])


class JatsTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "PMC1.xml"
        self.path.write_text(JATS, encoding="utf-8")
        self.document = corpus.parse_jats(self.path)

    def test_reference_list_is_excluded(self):
        self.assertFalse(any("MYC" in block.text for block in self.document.blocks))

    def test_source_types(self):
        types = {block.source_type for block in self.document.blocks}
        self.assertEqual(types, {"title", "paragraph", "figure_caption",
                                 "table_caption", "table_cell"})

    def test_caption_is_not_duplicated_in_paragraph(self):
        paragraph = next(b for b in self.document.blocks if "TP53" in b.text)
        self.assertNotIn("KRAS", paragraph.text)

    def test_table_row_keeps_header_context(self):
        row = next(b for b in self.document.blocks if "EGFR" in b.text)
        self.assertEqual(row.context, "Gene | Disease")

    def parse(self, xml):
        path = self.path.with_name("PMC2.xml")
        path.write_text(xml, encoding="utf-8")
        return corpus.parse_jats(path)

    def test_nested_list_paragraph_appears_once(self):
        document = self.parse("<article><body><p>Intro<list><list-item><p>TP53 in glioma."
                              "</p></list-item></list></p></body></article>")
        self.assertEqual([block.text for block in document.blocks], ["Intro", "TP53 in glioma."])

    def test_direct_abstract_text_is_kept(self):
        document = self.parse("<article><front><abstract>TP53 is linked.</abstract></front>"
                              "</article>")
        self.assertEqual([block.section for block in document.blocks], ["Abstract"])

    def test_table_footnote_is_kept_with_its_table(self):
        document = self.parse("<article><body><table-wrap><table><tr><td>A</td></tr></table>"
                              "<table-wrap-foot><fn><p>TP53 denotes a gene.</p></fn>"
                              "</table-wrap-foot></table-wrap></body></article>")
        note = next(b for b in document.blocks if "TP53" in b.text)
        self.assertEqual(note.group, "table0")
        self.assertEqual(sum("TP53" in b.text for b in document.blocks), 1)

    def test_table_cell_paragraphs_are_kept(self):
        document = self.parse("<article><body><table-wrap><table><tr><td><p>TP53</p></td>"
                              "<td><p>glioma</p></td></tr></table></table-wrap></body></article>")
        self.assertEqual([block.text for block in document.blocks], ["TP53 | glioma"])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            corpus.parse_jats("missing.xml")


class BioTriplexTests(unittest.TestCase):
    def write(self, papers):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "splits.json"
        path.write_text(json.dumps(papers), encoding="utf-8")
        return path

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            corpus.load_biotriplex(self.write({}), {"TEST"})

    def test_locked_split_is_refused(self):
        path = self.write({"1": biotriplex_paper("test")})
        with self.assertRaises(corpus.LockedSplitError):
            corpus.load_biotriplex(path, {"train", "test"})

    def test_locked_paper_is_not_loaded_with_train(self):
        path = self.write({"1": biotriplex_paper("train"), "2": biotriplex_paper("val")})
        self.assertEqual(set(corpus.load_biotriplex(path, {"train"})), {"BTX1"})

    def test_abbreviation_merges_and_link_resolves(self):
        path = self.write({"1": biotriplex_paper()})
        paper = corpus.load_biotriplex(path, {"train"})["BTX1"]
        diseases = {m.entity for m in paper.mentions if m.category == "disease"}
        self.assertEqual(len(diseases), 1)
        self.assertEqual(paper.unplaced_spans, 0)
        block = paper.document.blocks[0]
        for mention in paper.mentions:
            self.assertEqual(block.text[mention.start:mention.end], mention.text)
        self.assertEqual(paper.gold_pairs(), {("gene:dys", next(iter(diseases)))})
        self.assertEqual(paper.links[0].location, "result")

    def test_parenthetical_without_abbreviation_is_not_merged(self):
        self.assertFalse(corpus.is_abbreviation("glioma", "cancer"))
        self.assertFalse(corpus.is_abbreviation("glioma", "glioblastoma"))
        self.assertTrue(corpus.is_abbreviation("DMD", "Duchenne muscular dystrophy"))

    def test_discontinuous_span_keeps_components(self):
        self.assertEqual(corpus.parse_spans("0~4,20~24"), [(0, 4), (20, 24)])

    def test_unplaceable_trigger_keeps_the_gold_pair(self):
        paper = biotriplex_paper()
        paper["section_data"]["Abstract"]["relations"]["R0"]["spans"] = "30~60"
        loaded = corpus.load_biotriplex(self.write({"1": paper}), {"train"})["BTX1"]
        self.assertEqual(len(loaded.gold_pairs()), 1)
        self.assertIsNone(loaded.links[0].block_id)

    def test_no_relation_goes_to_negatives(self):
        paper = biotriplex_paper()
        relation = paper["section_data"]["Abstract"]["relations"]["R0"]
        relation["relation"] = corpus.NO_RELATION
        loaded = corpus.load_biotriplex(self.write({"1": paper}), {"train"})["BTX1"]
        self.assertEqual(loaded.links, [])
        self.assertEqual(len(loaded.negatives), 1)


class BioRedTests(unittest.TestCase):
    def test_split_name_in_other_case_is_rejected(self):
        with self.assertRaises(ValueError):
            corpus.load_biored("no/such/folder", {"test"})

    def test_locked_split_is_refused_before_any_read(self):
        with self.assertRaises(corpus.LockedSplitError):
            corpus.load_biored("no/such/folder", {"Test"})

    def test_pairs_use_identifiers_and_skip_other_types(self):
        item = {"id": "1", "relations": [
            {"infons": {"entity1": "D1", "entity2": "7"}},
            {"infons": {"entity1": "7", "entity2": "C9"}}],
            "passages": [{"offset": 0, "text": "TP53 in glioma with drug",
                          "infons": {"type": "title"}, "annotations": [
                {"text": "TP53", "infons": {"type": "GeneOrGeneProduct", "identifier": "7"},
                 "locations": [{"offset": 0, "length": 4}]},
                {"text": "glioma",
                 "infons": {"type": "DiseaseOrPhenotypicFeature", "identifier": "D1"},
                 "locations": [{"offset": 8, "length": 6}]},
                {"text": "drug", "infons": {"type": "ChemicalEntity", "identifier": "C9"},
                 "locations": [{"offset": 20, "length": 4}]}]}]}
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        Path(folder.name, "Dev.BioC.JSON").write_text(json.dumps({"documents": [item]}),
                                                     encoding="utf-8")
        _, mentions, pairs = corpus.load_biored(folder.name, {"Dev"})["BioRED1"]
        self.assertEqual(pairs, {("gene:7", "disease:D1")})
        self.assertEqual(len(mentions), 2)


if __name__ == "__main__":
    unittest.main()
