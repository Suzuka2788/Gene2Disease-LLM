"""Evidence assessment, as fixed in annotation/EVIDENCE_PROTOCOL.md.

A: programmatic checks over every positive assertion of a run.
B: trigger proxy on BioTriplex gold-matched edges.
C: the sampling frame, the stratified sample, and the two-stage annotation sheets.

This runs after E4, on stored outputs only; it reads the locked test splits on purpose
(recorded in E4_PROVENANCE.md) and calls no model. Sheets contain article text and go
under annotation/sheets/, which is never committed.
"""
from __future__ import annotations

import csv
import hashlib
import html
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from . import annotator, corpus, io

Pair = tuple[str, str]
Key = tuple[str, str, str]          # (paper, gene, disease)
Q_CODES = ("supported", "speculative", "negative", "no_relation", "undecided")
NO_RELATION_SUBCODES = ("co_mention", "prognostic", "symptom", "wrong_entity", "other")


def block_of(sentence_id: str) -> str:
    """Sentence ids are "<block id>.s<index>"."""
    return sentence_id.rsplit(".s", 1)[0]


@dataclass
class Paper:
    document: corpus.Document
    mentions: list[corpus.Mention]
    gold: set[Pair]
    links: list = field(default_factory=list)      # BioTriplex GoldLink records


def load_papers(data: dict) -> dict[str, Paper]:
    """The papers of a data section, locked splits included (post-E4 analysis)."""
    if data["dataset"] == "biored":
        loaded = corpus.load_biored(data["folder"], set(data["splits"]), allow_locked=True)
        return {key: Paper(doc, mentions, gold) for key, (doc, mentions, gold) in loaded.items()}
    loaded = corpus.load_biotriplex(data["release"], set(data["splits"]), allow_locked=True)
    return {key: Paper(p.document, p.mentions, p.gold_pairs(), p.links) for key, p in loaded.items()}


def positive_assertions(run_dir: Path, paper_id: str) -> dict[Pair, list[tuple[str, ...]]]:
    """Edges of one paper and the cited sentence ids of each positive assertion."""
    edges: dict[Pair, list[tuple[str, ...]]] = {}
    for row in io.read_jsonl(run_dir / paper_id / "pairs.jsonl"):
        if row["disposition"] != "edge":
            continue
        cited = [tuple(item["sentence_ids"]) for item in row["assertions"]
                 if item["polarity"] == "positive"]
        edges[(row["gene"], row["disease"])] = cited
    return edges


def _sentence_spans(document: corpus.Document) -> dict[str, tuple[str, int, int]]:
    return {sentence.sentence_id: (block.block_id, sentence.start, sentence.end)
            for block in document.blocks for sentence in block.sentences}


def _mentioned(entity: str, cited: tuple[str, ...], spans, mentions) -> bool:
    for sentence_id in cited:
        if sentence_id not in spans:
            continue
        block_id, start, end = spans[sentence_id]
        if any(m.entity == entity and m.block_id == block_id and m.start < end and m.end > start
               for m in mentions):
            return True
    return False


def check_a(run_dir: Path, papers: dict[str, Paper]) -> dict:
    """Share of positive assertions whose cited sentences, taken together, contain a
    given mention of the gene and of the disease. Rejected assertions come from the
    run summary (the validator drops them before they are stored)."""
    total = both = 0
    for paper_id, paper in papers.items():
        spans = _sentence_spans(paper.document)
        for (gene, disease), assertions in positive_assertions(run_dir, paper_id).items():
            for cited in assertions:
                total += 1
                both += (_mentioned(gene, cited, spans, paper.mentions)
                         and _mentioned(disease, cited, spans, paper.mentions))
    return {"positive_assertions": total, "both_entities_in_cited_sentences": both,
            "share": both / total if total else None}


def check_b(run_dir: Path, papers: dict[str, Paper]) -> dict:
    """BioTriplex gold-matched edges: does any cited sentence of a positive assertion
    intersect a gold trigger of that pair in the same passage? Edges whose gold links
    have no in-passage trigger position are counted apart, outside the denominator."""
    denominator = overlapping = no_trigger = 0
    for paper_id, paper in papers.items():
        spans = _sentence_spans(paper.document)
        for pair, assertions in positive_assertions(run_dir, paper_id).items():
            if pair not in paper.gold:
                continue
            triggers = [link for link in paper.links if (link.gene, link.disease) == pair
                        and link.block_id is not None]
            if not triggers:
                no_trigger += 1
                continue
            denominator += 1
            overlapping += any(
                sentence_id in spans and spans[sentence_id][0] == link.block_id
                and spans[sentence_id][1] < link.trigger_end and spans[sentence_id][2] > link.trigger_start
                for cited in assertions for sentence_id in cited for link in triggers)
    return {"gold_edges_with_trigger": denominator, "overlapping": overlapping,
            "share": overlapping / denominator if denominator else None,
            "gold_edges_without_trigger_position": no_trigger}


@dataclass
class FrameEdge:
    gold: bool
    runs: set[str] = field(default_factory=set)
    # passage -> run -> distinct cited-sentence sets of that run's positive assertions
    passages: dict[str, dict[str, set[tuple[str, ...]]]] = field(default_factory=dict)


def build_frame(papers: dict[str, Paper], runs: dict[str, tuple[Path, str]]) -> dict[Key, FrameEdge]:
    """`runs` maps a run name to (run directory, "pairwise" or "joint")."""
    frame: dict[Key, FrameEdge] = {}
    for name, (run_dir, _) in runs.items():
        for paper_id, paper in papers.items():
            for (gene, disease), assertions in positive_assertions(run_dir, paper_id).items():
                edge = frame.setdefault((paper_id, gene, disease),
                                        FrameEdge((gene, disease) in paper.gold))
                edge.runs.add(name)
                for cited in assertions:
                    blocks = {block_of(sentence_id) for sentence_id in cited}
                    if len(blocks) != 1:
                        # E4 windows are single passages, so this would mean damaged output.
                        raise ValueError(f"{name} {paper_id}: an assertion cites {len(blocks)} passages")
                    edge.passages.setdefault(blocks.pop(), {}).setdefault(
                        name, set()).add(tuple(sorted(cited)))
    return frame


def stratum(edge: FrameEdge, runs: dict[str, tuple[Path, str]]) -> str:
    if edge.gold:
        return "G"
    if any(runs[name][1] == "joint" for name in edge.runs):
        return "S2"
    return "S1"


def draw(frame: dict[Key, FrameEdge], runs: dict[str, tuple[Path, str]],
         plan: dict[str, int], seed: int, corpus_name: str) -> tuple[list[dict], dict]:
    """Equal-probability sampling without replacement within each stratum, then one
    displayed passage per item, all from one seeded generator. A stratum smaller than
    its plan is taken whole; the shortfall is not moved elsewhere."""
    generator = random.Random(seed)
    by_stratum: dict[str, list[Key]] = {"S1": [], "S2": [], "G": []}
    for key in sorted(frame):
        by_stratum[stratum(frame[key], runs)].append(key)
    items, design = [], {"corpus": corpus_name, "seed": seed, "strata": {}}
    for name in ("S1", "S2", "G"):
        keys = by_stratum[name]
        chosen = generator.sample(keys, min(plan[name], len(keys)))
        design["strata"][name] = {
            "N": len(keys), "n": len(chosen),
            "N_by_run": {run: sum(run in frame[key].runs for key in keys) for run in sorted(runs)}}
        for key in chosen:
            passage = generator.choice(sorted(frame[key].passages))
            items.append({"corpus": corpus_name, "stratum": name, "paper": key[0],
                          "gene": key[1], "disease": key[2], "runs": sorted(frame[key].runs),
                          "passage": passage,
                          "citations": {run: sorted(map(list, sets)) for run, sets
                                        in sorted(frame[key].passages[passage].items())}})
    return items, design


def scaled_plans(plans: dict[str, dict[str, int]], median_minutes: float,
                 hours: float = 8.0) -> tuple[dict[str, dict[str, int]], float]:
    """The size rule (protocol revision of 2026-11-08), applied once before the test
    sample is drawn: one factor for every stratum of every corpus, rounded down, with
    at least 4 per stratum. Returns the plans and the predicted minutes; if the floor of
    4 still exceeds the budget, that minimum design is used and its time is reported."""
    if not (median_minutes > 0 and median_minutes < float("inf")):
        raise ValueError(f"median minutes per item must be positive and finite: {median_minutes}")
    budget = hours * 60

    def at(factor: float) -> dict[str, dict[str, int]]:
        return {corpus: {name: max(4, int(size * factor)) for name, size in plan.items()}
                for corpus, plan in plans.items()}

    def minutes(scaled: dict[str, dict[str, int]]) -> float:
        return sum(sum(plan.values()) for plan in scaled.values()) * median_minutes

    if minutes(plans) <= budget:
        return {corpus: dict(plan) for corpus, plan in plans.items()}, minutes(plans)
    # The largest shared factor whose design, minimum of 4 included, fits the budget.
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if minutes(at(middle)) <= budget else (low, middle)
    scaled = at(low)
    return scaled, minutes(scaled)


# ---- sheets ---------------------------------------------------------------------------

def _names(mentions: list[corpus.Mention]) -> dict[str, str]:
    names: dict[str, str] = {}
    for mention in mentions:
        names.setdefault(mention.entity, mention.text)
    return names


def _title(document: corpus.Document) -> str:
    """BioRED's first passage is the title (its BioC type field reads "passage");
    the BioTriplex release has no title section."""
    for block in document.blocks:
        if block.section.strip().lower() == "title":
            return block.text
    if document.doc_id.startswith("BioRED") and document.blocks:
        return document.blocks[0].text
    return "(no title in the corpus release)"


def _section(document: corpus.Document, block: corpus.Block) -> str:
    if document.doc_id.startswith("BioRED") and block.section == "passage":
        return "Title" if block is document.blocks[0] else "Abstract"
    return block.section


def _highlight(block: corpus.Block, mentions, gene: str, disease: str,
               cited: set[str] = frozenset()) -> str:
    """The passage with the given gene and disease mentions marked, sentence by
    sentence; cited sentences are shaded in stage 2."""
    marks = sorted((m.start, m.end, "gene" if m.entity == gene else "disease")
                   for m in mentions if m.block_id == block.block_id and m.entity in (gene, disease))
    parts = []
    for sentence in block.sentences or [corpus.Sentence(f"{block.block_id}.s0", 0, len(block.text))]:
        cursor, out = sentence.start, []
        for start, end, kind in marks:
            if start < sentence.start or end > sentence.end or start < cursor:
                continue
            out.append(html.escape(block.text[cursor:start]))
            out.append(f'<span class="{kind}">{html.escape(block.text[start:end])}</span>')
            cursor = end
        out.append(html.escape(block.text[cursor:sentence.end]))
        text = "".join(out)
        parts.append(f'<span class="cited">{text}</span>' if sentence.sentence_id in cited else text)
    return " ".join(parts)


def _block(paper: Paper, block_id: str) -> corpus.Block:
    return next(block for block in paper.document.blocks if block.block_id == block_id)


def _header(item: dict, paper: Paper, block: corpus.Block) -> str:
    names = _names(paper.mentions)
    return (f'<p class="meta"><b>Title:</b> {html.escape(_title(paper.document))}'
            f'<br><b>Section:</b> {html.escape(_section(paper.document, block))}<br><b>Gene:</b> '
            f'<span class="gene">{html.escape(names.get(item["gene"], item["gene"]))}</span> '
            f'&nbsp; <b>Disease:</b> <span class="disease">'
            f'{html.escape(names.get(item["disease"], item["disease"]))}</span></p>')


def _store_key(folder: Path, stage: int, shown: list[dict]) -> str:
    """Browser storage key, distinct per sheet, so the pilot and the test never mix."""
    ids = ",".join(item["item_id"] for item in shown)
    return f"g2d-evidence-{folder.name}-stage{stage}-" + hashlib.sha256(ids.encode()).hexdigest()[:12]


def write_stage1(items: list[dict], papers: dict[str, dict[str, Paper]], folder: Path,
                 seed: int) -> None:
    """Shuffle all items together, give them neutral ids, and write the stage-1 sheet,
    the label template and the key (the only file that links items to their source)."""
    folder.mkdir(parents=True, exist_ok=True)
    order = list(items)
    random.Random(seed + 1).shuffle(order)
    shown, rows = [], []
    for number, item in enumerate(order, 1):
        item["item_id"] = f"E{number:03d}"
        paper = papers[item["corpus"]][item["paper"]]
        block = _block(paper, item["passage"])
        shown.append({"item_id": item["item_id"], "header": _header(item, paper, block),
                      "passage": _highlight(block, paper.mentions, item["gene"], item["disease"])})
        rows.append({"item_id": item["item_id"], "q1": "", "subcode": "", "undecided_reason": "",
                     "fulltext_consulted": "", "minutes": "", "notes": ""})
    io.write_text(folder / "stage1.html",
                  annotator.page(f"证据核验 第一阶段（{folder.name}）", 1, shown,
                                 _store_key(folder, 1, shown), "stage1_labels.csv"))
    _write_csv(folder / "stage1_labels.csv", rows)
    io.write_json(folder / "_key_DO_NOT_OPEN.json", {"items": order})


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _minutes(value: str, where: str) -> float:
    try:
        minutes = float(value)
    except ValueError as error:
        raise ValueError(f"{where}: minutes must be a number, got {value!r}") from error
    if not (0 < minutes < float("inf")):
        raise ValueError(f"{where}: minutes must be positive and finite")
    return minutes


def read_labels(path: Path, question: str, expected: list[str], stage1: bool) -> list[dict]:
    """Exactly one complete row per expected id (item ids in stage 1, set ids in stage 2)."""
    id_field = "item_id" if stage1 else "set_id"
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row[id_field] for row in rows]
    if sorted(ids) != sorted(expected) or len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: rows must be exactly the {len(expected)} expected ids, once each")
    for row in rows:
        where, code = row[id_field], row[question].strip()
        if code not in Q_CODES:
            raise ValueError(f"{where}: {question} must be one of {Q_CODES}, got {code!r}")
        if code == "no_relation" and row["subcode"].strip() not in NO_RELATION_SUBCODES:
            raise ValueError(f"{where}: no_relation needs a subcode")
        if code == "undecided" and not row["undecided_reason"].strip():
            raise ValueError(f"{where}: undecided needs a reason")
        if stage1:
            if row["fulltext_consulted"].strip() not in ("yes", "no"):
                raise ValueError(f"{where}: fulltext_consulted must be yes or no")
            _minutes(row["minutes"], where)
    return rows


def write_stage2(folder: Path, papers: dict[str, dict[str, Paper]], seed: int) -> None:
    """Only after every stage-1 label is complete: keep a locked copy of the stage-1
    labels and its digest, then show each item's distinct citation sets in random order
    without their source."""
    lock = folder / "stage1_lock.json"
    if lock.exists():
        raise FileExistsError("stage 2 was already generated; stage 1 stays locked")
    key = io.read_json(folder / "_key_DO_NOT_OPEN.json")
    labels = folder / "stage1_labels.csv"
    read_labels(labels, "q1", [item["item_id"] for item in key["items"]], stage1=True)
    locked = folder / "stage1_labels.locked.csv"
    locked.write_bytes(labels.read_bytes())
    io.write_json(lock, {"stage1_labels_sha256": hashlib.sha256(locked.read_bytes()).hexdigest()})
    generator = random.Random(seed + 2)
    shown, rows = [], []
    for item in key["items"]:
        paper = papers[item["corpus"]][item["paper"]]
        block = _block(paper, item["passage"])
        sets: dict[tuple[str, ...], list[str]] = {}
        for run, cited_sets in item["citations"].items():
            for cited in cited_sets:
                sets.setdefault(tuple(cited), []).append(run)
        distinct = sorted(sets)
        generator.shuffle(distinct)
        item["citation_sets"] = {}
        shown_sets = []
        for index, cited in enumerate(distinct):
            set_id = f"{item['item_id']}-{chr(65 + index)}"
            item["citation_sets"][set_id] = {"sentences": list(cited), "runs": sorted(sets[cited])}
            shown_sets.append({"set_id": set_id, "passage": _highlight(
                block, paper.mentions, item["gene"], item["disease"], set(cited))})
            rows.append({"item_id": item["item_id"], "set_id": set_id, "q2": "", "subcode": "",
                         "undecided_reason": "", "minutes": "", "notes": ""})
        shown.append({"item_id": item["item_id"], "header": _header(item, paper, block),
                      "sets": shown_sets})
    io.write_text(folder / "stage2.html",
                  annotator.page(f"证据核验 第二阶段（{folder.name}）", 2, shown,
                                 _store_key(folder, 2, shown), "stage2_labels.csv"))
    _write_csv(folder / "stage2_labels.csv", rows)
    io.write_json(folder / "_key_DO_NOT_OPEN.json", key)


def verify_stage1_lock(folder: Path) -> None:
    """For the analysis: the stage-1 labels are the ones locked when stage 2 was made."""
    digest = io.read_json(folder / "stage1_lock.json")["stage1_labels_sha256"]
    for name in ("stage1_labels.locked.csv", "stage1_labels.csv"):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name} differs from the locked stage-1 labels")


def median_item_minutes(folder: Path) -> float:
    """Median minutes per item over both stages (stage 1 plus the stage-2 minutes
    recorded on the item's first set), from a completed pilot."""
    verify_stage1_lock(folder)
    key = io.read_json(folder / "_key_DO_NOT_OPEN.json")
    stage1 = read_labels(folder / "stage1_labels.csv", "q1",
                         [item["item_id"] for item in key["items"]], stage1=True)
    set_ids = [set_id for item in key["items"] for set_id in item["citation_sets"]]
    stage2 = read_labels(folder / "stage2_labels.csv", "q2", set_ids, stage1=False)
    total = {row["item_id"]: _minutes(row["minutes"], row["item_id"]) for row in stage1}
    first_sets = {item["item_id"]: next(iter(item["citation_sets"])) for item in key["items"]}
    for row in stage2:
        if row["set_id"] == first_sets[row["item_id"]]:
            # Convention: the item's stage-2 minutes are recorded on its first set.
            total[row["item_id"]] += _minutes(row["minutes"], row["set_id"])
        elif row["minutes"].strip():
            raise ValueError(f"{row['set_id']}: record stage-2 minutes on the item's first set only")
    return statistics.median(total.values())
