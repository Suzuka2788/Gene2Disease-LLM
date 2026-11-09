"""End-to-end aggregates on synthetic inputs."""
import json
import tempfile
import unittest
from pathlib import Path

from gene2disease import experiments, io

TEXT = "AAA causes BBB here. CCC is listed elsewhere today."
PAPER = {"split": "train", "section_data": {"Results": {
    "text": TEXT,
    "genes": {"G0": {"spans": "0~3", "text": "AAA", "location": "result"}},
    "diseases": {"D0": {"spans": "11~14", "text": "BBB", "location": "result"},
                 "D1": {"spans": "21~24", "text": "CCC", "location": "result"}},
    "relations": {"R0": {"spans": "4~10", "text": "causes", "location": "result",
                         "relation": "causative mutation"}},
    "entity_linkings": {"E0": {"geneID": "G0", "geneText": "AAA", "diseaseID": "D0",
                               "diseaseText": "BBB", "relationID": "R0",
                               "relationText": "causes"}}}}}
BIORED = {"documents": [{"id": "1", "relations": [
    {"infons": {"entity1": "7", "entity2": "D1"}}],
    "passages": [{"offset": 0, "text": "AAA here. BBB there.", "annotations": [
        {"text": "AAA", "infons": {"type": "GeneOrGeneProduct", "identifier": "7"},
         "locations": [{"offset": 0, "length": 3}]},
        {"text": "BBB", "infons": {"type": "DiseaseOrPhenotypicFeature", "identifier": "D1"},
         "locations": [{"offset": 10, "length": 3}]}]}]}]}
PUBTATOR = {"PubTator3": [{"passages": [
    {"offset": 0, "text": "AAA causes BBB here and more words.", "infons": {"section_type": "RESULTS"},
     "annotations": [
         {"text": "AAA", "infons": {"type": "Gene", "identifier": "1"},
          "locations": [{"offset": 0, "length": 3}]},
         {"text": "BBB", "infons": {"type": "Disease", "identifier": "MESH:D1"},
          "locations": [{"offset": 11, "length": 3}]},
         {"text": "CCC", "infons": {"type": "Gene", "identifier": "2;3"},
          "locations": [{"offset": 0, "length": 0}]}]}]}]}
JATS = ('<article><front><article-meta><article-id pub-id-type="pmid">9</article-id>'
        "</article-meta></front><body><p>AAA causes BBB here and more words.</p></body></article>")


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        io.write_json(self.root / "release.json", {"1": PAPER})
        io.write_json(self.root / "pmids.json", {"1": 5})
        io.write_json(self.root / "pubtator" / "5.json", PUBTATOR)
        io.write_json(self.root / "biored" / "Dev.BioC.JSON", BIORED)
        io.write_json(self.root / "task_annotations" / "9.json", PUBTATOR)
        (self.root / "xml").mkdir()
        (self.root / "xml" / "PMC9.xml").write_text(JATS, encoding="utf-8")
        self.config = {
            "stats": {"n_resamples": 50, "seed": 1, "level": 0.95},
            "workload": {"chars_per_token": 4.0, "window_cap_per_pair": 3},
            "biotriplex": {"release": str(self.root / "release.json"),
                           "pmid_map": str(self.root / "pmids.json"),
                           "pubtator_dir": str(self.root / "pubtator"), "splits": ["train"]},
            "biored": {"folder": str(self.root / "biored"), "splits": ["Dev"]},
            "task_papers": {"xml_dir": str(self.root / "xml"),
                            "pubtator_dir": str(self.root / "task_annotations")},
        }

    def test_e1_hand_values(self):
        result = experiments.e1_candidate_ceiling(self.config)
        sentence = result["levels"]["sentence"]
        self.assertEqual(result["annotated_pairs"], 1)
        self.assertEqual(result["all_pairs"], 2)
        self.assertEqual(sentence["counts"], {"tp": 1, "fp": 0, "fn": 0})
        self.assertEqual(sentence["anchored_link_coverage"], 1.0)
        self.assertEqual(result["levels"]["block"]["counts"], {"tp": 1, "fp": 1, "fn": 0})
        biored = result["biored_development"]["levels"]
        self.assertEqual(biored["sentence"]["counts"]["fn"], 1)
        self.assertEqual(biored["adjacent"]["counts"]["tp"], 1)

    def test_workload_excludes_ambiguous_entities_and_counts_papers(self):
        result = experiments.workload(self.config)
        alignment = result["pubtator_alignment"]
        self.assertEqual(alignment["expected_papers"], 1)
        self.assertTrue(alignment["complete"])
        self.assertEqual(alignment["ineligible_entities"], 1)
        self.assertEqual(result["task_papers_pubtator_entities"]["all_pairs"], 1)

    def test_paper_without_annotation_marks_workload_incomplete(self):
        (self.root / "xml" / "PMC8.xml").write_text(JATS.replace(">9<", ">8<"), encoding="utf-8")
        alignment = experiments.workload(self.config)["pubtator_alignment"]
        self.assertEqual(alignment["excluded_without_annotation"], 1)
        self.assertFalse(alignment["complete"])

    def test_missing_task_directory_raises(self):
        self.config["task_papers"]["xml_dir"] = str(self.root / "absent")
        with self.assertRaises(io.MissingInputError):
            experiments.workload(self.config)

    def test_results_hold_no_paper_text(self):
        text = json.dumps(experiments.e1_candidate_ceiling(self.config))
        self.assertNotIn("causes", text)
        self.assertNotIn("BTX", text)


if __name__ == "__main__":
    unittest.main()
