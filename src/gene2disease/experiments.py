"""Zero-cost experiments. Output is aggregate only: no paper text, no per-paper rows."""
from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import asdict
from pathlib import Path

from . import candidates, corpus, entities, f0, f0_report, freeze, io, runner, scoring, stats
from .candidates import LEVELS
from .scoring import Counts


def bootstrap_settings(config: dict) -> stats.BootstrapSettings:
    section = config["stats"]
    return stats.BootstrapSettings(section["n_resamples"], section["seed"], section["level"])


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _estimate(metric, per_paper: list[Counts], settings: stats.BootstrapSettings) -> dict:
    return asdict(stats.bootstrap(per_paper, lambda sample: scoring.micro(metric, sample),
                                  settings))


def _anchored(paper: corpus.AnnotatedDocument, level: str) -> tuple[int, int]:
    """Links whose gene, disease and relation trigger all fall in one window."""
    anchored = [link for link in paper.links if link.block_id is not None]
    windows = candidates.build_candidates(paper.document, paper.mentions, level)
    hits = sum(any(link.gene in item.genes and link.disease in item.diseases
                   and item.window.contains(link.block_id, link.trigger_start, link.trigger_end)
                   for item in windows) for link in anchored)
    return hits, len(anchored)


def e1_candidate_ceiling(config: dict) -> dict:
    """Candidate coverage of annotated pairs on the development split."""
    settings = bootstrap_settings(config)
    data = config["biotriplex"]
    papers = corpus.load_biotriplex(data["release"], set(data["splits"]))
    ordered = [papers[key] for key in sorted(papers)]
    result: dict = {
        "papers": len(ordered),
        "papers_with_links": sum(1 for paper in ordered if paper.links),
        "annotated_pairs": sum(len(paper.gold_pairs()) for paper in ordered),
        "annotated_links": sum(len(paper.links) for paper in ordered),
        "unplaced_spans": sum(paper.unplaced_spans for paper in ordered),
        "links_without_trigger_position":
            sum(link.block_id is None for paper in ordered for link in paper.links),
        "negative_links": sum(len(paper.negatives) for paper in ordered),
        "all_pairs": sum(len(candidates.all_pairs(paper.mentions)) for paper in ordered),
        "levels": {},
    }
    for level in LEVELS:
        per_paper, anchored_hits, anchored_total = [], 0, 0
        covered_by_location = {"result": [0, 0], "background": [0, 0]}
        for paper in ordered:
            covered = candidates.covered_pairs(
                candidates.build_candidates(paper.document, paper.mentions, level))
            per_paper.append(scoring.compare_sets(paper.gold_pairs(), covered))
            hits, total_links = _anchored(paper, level)
            anchored_hits += hits
            anchored_total += total_links
            for link in paper.links:
                if link.location in covered_by_location:
                    covered_by_location[link.location][1] += 1
                    covered_by_location[link.location][0] += (link.gene, link.disease) in covered
        pooled = scoring.total(per_paper)
        result["levels"][level] = {
            "counts": asdict(pooled),
            "pair_coverage": _estimate(scoring.recall, per_paper, settings),
            "cooccurrence_annotated_match_rate": scoring.precision(pooled),
            "anchored_link_coverage": anchored_hits / anchored_total if anchored_total else None,
            "link_coverage_by_location": covered_by_location,
        }
    result["pubtator_surface_coverage"] = _pubtator_coverage(ordered, config, settings)
    result["biored_development"] = _biored_coverage(config, settings)
    return result


def _biored_coverage(config: dict, settings: stats.BootstrapSettings) -> dict:
    """BioRED pairs are document-level, unlike BioTriplex's trigger-anchored links."""
    data = config["biored"]
    papers = corpus.load_biored(data["folder"], set(data["splits"]))
    ordered = [papers[key] for key in sorted(papers)]
    result: dict = {"papers": len(ordered),
                    "annotated_pairs": sum(len(pairs) for _, _, pairs in ordered),
                    "levels": {}}
    for level in LEVELS:
        per_paper = [scoring.compare_sets(pairs, candidates.covered_pairs(
            candidates.build_candidates(document, mentions, level)))
            for document, mentions, pairs in ordered]
        result["levels"][level] = {
            "counts": asdict(scoring.total(per_paper)),
            "pair_coverage": _estimate(scoring.recall, per_paper, settings),
            "cooccurrence_annotated_match_rate": scoring.micro(scoring.precision, per_paper),
        }
    return result


def _pubtator_coverage(ordered: list[corpus.AnnotatedDocument], config: dict,
                       settings: stats.BootstrapSettings) -> dict:
    """Text-level proxy: BioTriplex has no identifiers, so surfaces are compared."""
    data = config["biotriplex"]
    pmids = io.read_json(data["pmid_map"])
    folder = Path(data["pubtator_dir"])
    per_paper: list[Counts] = []
    gene_hits = gene_total = disease_hits = disease_total = 0
    extra_genes = extra_diseases = tool_genes = tool_diseases = 0
    without_annotation = 0
    for paper in ordered:
        pmid = pmids.get(paper.document.doc_id.removeprefix("BTX"))
        path = folder / f"{pmid}.json"
        if pmid is None or not path.is_file():
            without_annotation += 1
            continue
        annotations = entities.read_pubtator(path)
        tool = {"gene": entities.surfaces(annotations, "gene"),
                "disease": entities.surfaces(annotations, "disease")}
        clusters: dict[str, set[str]] = {}
        for mention in paper.mentions:
            clusters.setdefault(mention.entity, set()).add(corpus.canonical_surface(mention.text))
        found = {entity: bool(forms & tool[entity.split(":", 1)[0]])
                 for entity, forms in clusters.items()}
        gene_hits += sum(v for k, v in found.items() if k.startswith("gene:"))
        gene_total += sum(1 for k in found if k.startswith("gene:"))
        disease_hits += sum(v for k, v in found.items() if k.startswith("disease:"))
        disease_total += sum(1 for k in found if k.startswith("disease:"))
        gold_forms = {category: set().union(*(forms for k, forms in clusters.items()
                                              if k.startswith(category + ":")), set())
                      for category in ("gene", "disease")}
        text = corpus.canonical_surface(" ".join(block.text for block in paper.document.blocks))
        in_text = {category: {form for form in forms if _occurs(form, text)}
                   for category, forms in tool.items()}
        extra_genes += len(in_text["gene"] - gold_forms["gene"])
        extra_diseases += len(in_text["disease"] - gold_forms["disease"])
        tool_genes += len(in_text["gene"])
        tool_diseases += len(in_text["disease"])
        gold = paper.gold_pairs()
        kept = {pair for pair in gold if found.get(pair[0]) and found.get(pair[1])}
        per_paper.append(Counts(len(kept), 0, len(gold) - len(kept)))
    return {
        "papers_with_annotation": len(per_paper),
        "papers_without_annotation": without_annotation,
        "gene_cluster_coverage": gene_hits / gene_total if gene_total else None,
        "disease_cluster_coverage": disease_hits / disease_total if disease_total else None,
        "pair_coverage_both_entities": _estimate(scoring.recall, per_paper, settings)
        if per_paper else None,
        "tool_gene_surfaces_in_text_absent_from_gold": extra_genes / tool_genes if tool_genes else None,
        "tool_disease_surfaces_in_text_absent_from_gold":
            extra_diseases / tool_diseases if tool_diseases else None,
    }


def _occurs(form: str, canonical_text: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(form) + r"(?![a-z0-9])"
    return re.search(pattern, canonical_text) is not None


def _quantiles(values: list[float]) -> dict:
    return {name: stats.quantile(values, probability) for name, probability in
            (("min", 0), ("p50", 0.5), ("p90", 0.9), ("max", 1))} if values else {}


def _load_task_papers(config: dict) -> tuple[list, dict]:
    data = config["task_papers"]
    paths = sorted(Path(data["xml_dir"]).glob("*.xml"))
    if not paths:
        raise io.MissingInputError(f"no task papers in {data['xml_dir']}")
    annotations_dir = Path(data["pubtator_dir"])
    loaded, totals = [], {"expected_papers": len(paths), "excluded_without_annotation": 0,
                          "aligned": 0, "unaligned": 0, "unaligned_by_section": {},
                          "ineligible_entities": 0}
    for path in paths:
        document = corpus.parse_jats(path)
        pmid = corpus.read_pmid(path)
        annotation_path = annotations_dir / f"{pmid}.json"
        if pmid is None or not annotation_path.is_file():
            totals["excluded_without_annotation"] += 1
            continue
        annotations = entities.read_pubtator(annotation_path)
        mentions, report = entities.align(document, annotations)
        ineligible = entities.ineligible_entities(annotations)
        totals["ineligible_entities"] += len(ineligible)
        mentions = [mention for mention in mentions if mention.entity not in ineligible]
        totals["aligned"] += report.aligned
        totals["unaligned"] += report.unaligned
        for section, count in report.unaligned_by_section.items():
            sections = totals["unaligned_by_section"]
            sections[section] = sections.get(section, 0) + count
        loaded.append((document, mentions))
    totals["complete"] = totals["excluded_without_annotation"] == 0
    return loaded, totals


def _workload(loaded: list, chars_per_token: float, window_cap: int) -> dict:
    paper_chars = [sum(len(b.text) for b in document.blocks) for document, _ in loaded]
    pair_counts = [len(candidates.all_pairs(mentions)) for _, mentions in loaded]
    result: dict = {
        "papers": len(loaded),
        "paper_passage_tokens_estimated": _quantiles([c / chars_per_token for c in paper_chars]),
        "pairs_per_paper": _quantiles(pair_counts),
        "all_pairs": sum(pair_counts),
        "papers_without_pairs": sum(1 for count in pair_counts if count == 0),
        "levels": {},
    }
    for level in LEVELS:
        joint_calls = pairwise_calls = capped_calls = covered_total = 0
        joint_chars = pairwise_chars = capped_chars = 0
        window_tokens: list[float] = []
        window_pairs: list[int] = []
        for document, mentions in loaded:
            seen: dict[tuple[str, str], int] = {}
            pair_set: set[tuple[str, str]] = set()
            for item in candidates.build_candidates(document, mentions, level):
                size = candidates.window_characters(document, item.window)
                pairs = item.pairs()
                window_pairs.append(len(pairs))
                pair_set |= pairs
                joint_calls += 1
                joint_chars += size
                window_tokens.append(size / chars_per_token)
                pairwise_calls += len(pairs)
                pairwise_chars += size * len(pairs)
                for pair in pairs:
                    seen[pair] = seen.get(pair, 0) + 1
                    if seen[pair] <= window_cap:
                        capped_calls += 1
                        capped_chars += size
            covered_total += len(pair_set)
        result["levels"][level] = {
            "candidate_pairs": covered_total,
            "joint_calls": joint_calls,
            "joint_passage_tokens_estimated": joint_chars / chars_per_token,
            "pairwise_calls": pairwise_calls,
            "pairwise_passage_tokens_estimated": pairwise_chars / chars_per_token,
            "pairwise_capped_calls": capped_calls,
            "pairwise_capped_passage_tokens_estimated": capped_chars / chars_per_token,
            "window_passage_tokens_estimated": _quantiles(window_tokens),
            "pairs_per_window": _quantiles(window_pairs),
        }
    return result


def workload(config: dict) -> dict:
    """Projected call counts and input size per judgment form."""
    settings = config["workload"]
    ratio, cap = settings["chars_per_token"], settings["window_cap_per_pair"]
    data = config["biotriplex"]
    papers = corpus.load_biotriplex(data["release"], set(data["splits"]))
    development = [(paper.document, paper.mentions) for _, paper in sorted(papers.items())]
    task_papers, alignment = _load_task_papers(config)
    biored = corpus.load_biored(config["biored"]["folder"], set(config["biored"]["splits"]))
    biored_loaded = [(document, mentions) for _, (document, mentions, _) in sorted(biored.items())]
    return {
        "biored_development_gold_entities": _workload(biored_loaded, ratio, cap),
        "chars_per_token_assumed": ratio,
        "window_cap_per_pair": cap,
        "biotriplex_development_gold_entities": _workload(development, ratio, cap),
        "task_papers_pubtator_entities": _workload(task_papers, ratio, cap),
        "pubtator_alignment": alignment,
    }


STAGES = {"e1": e1_candidate_ceiling, "workload": workload, "judge": runner.judge,
          "compare": runner.compare, "f0": f0.run, "f0_report": f0_report.report,
          "freeze": freeze.write_manifest}


def run(config_path: str | Path) -> Path:
    config = io.read_toml(config_path)
    stage = config["run"]["stage"]
    output = Path(config["run"]["output"])
    if stage == "freeze" and output.exists():
        raise FileExistsError(f"the protocol is already frozen: {output}")
    # Checked before any data is read. The authorization is passed as an argument, so
    # nothing in a configuration file can unlock a split.
    locked = freeze.authorize(config_path, config)
    result = STAGES[stage](config, locked) if locked is not None else STAGES[stage](config)
    inputs = config["run"].get("hashed_inputs", {})
    result["manifest"] = {
        "stage": stage,
        "config": Path(config_path).name,
        "config_sha256": file_digest(config_path),
        "inputs_sha256": {role: file_digest(path) for role, path in inputs.items()},
        "python": platform.python_version(),
        "code_commit": "none: no local repository yet",
        "code_sha256": freeze.code_digest(),
    }
    if locked is not None:
        result["manifest"]["freeze_manifest"] = locked.manifest
        result["manifest"]["freeze_manifest_sha256"] = locked.manifest_sha256
    if stage == "freeze":
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "x", encoding="utf-8") as handle:   # exclusive creation
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return output
    io.write_json(output, result)
    return output
