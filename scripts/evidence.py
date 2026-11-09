"""Evidence assessment (annotation/EVIDENCE_PROTOCOL.md) on stored outputs.

Usage (from the project root, PYTHONPATH=src):
  python scripts/evidence.py checks        A and B over the E4 full-text runs -> results/evidence_checks.json
  python scripts/evidence.py pilot         10 pilot items from development outputs -> annotation/sheets/pilot/
  python scripts/evidence.py test          the test sample, sized by the pilot's median minutes -> annotation/sheets/test/
  python scripts/evidence.py stage2 DIR    stage-2 sheet for DIR once its stage-1 labels are complete

Calls no model. Sheets hold article text and stay under annotation/sheets/ (gitignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gene2disease import evidence, io  # noqa: E402

RUNS = ROOT / "runs"
SHEETS = ROOT / "annotation" / "sheets"
MODELS = ("qwen", "gemma")
FULL_TEXT_FORMS = {"qwen": ("pairwise", "chunked", "joint"), "gemma": ("pairwise", "joint")}
TEST_PLAN = {"biored": {"S1": 18, "S2": 16, "G": 6}, "biotriplex": {"S1": 16, "S2": 14, "G": 6}}
PILOT_PLAN = {"S1": 2, "S2": 2, "G": 1}
TEST_SEED, PILOT_SEED = 20261108, 20261107


def data(prefix: str, corpus: str, splits: list[str] | None = None) -> dict:
    """The data section of the frozen (e4) or development (ef_dev) configuration."""
    folder = "e4" if prefix == "e4" else "ef_dev"
    section = io.read_toml(ROOT / "configs" / folder / f"{prefix}_{corpus}_qwen_pairwise.toml")["data"]
    return {**section, "splits": splits or section["splits"]}


def frame_runs(prefix: str, corpus: str) -> dict:
    return {f"{prefix}_{corpus}_{model}_{form}": (RUNS / f"{prefix}_{corpus}_{model}_{form}", form)
            for model in MODELS for form in ("pairwise", "joint")}


def checks() -> None:
    result = {"protocol": "annotation/EVIDENCE_PROTOCOL.md", "A": {}, "B": {}}
    for corpus in ("biored", "biotriplex"):
        papers = evidence.load_papers(data("e4", corpus))
        for model in MODELS:
            for form in FULL_TEXT_FORMS[model]:
                name = f"e4_{corpus}_{model}_{form}"
                summary = io.read_json(ROOT / "results" / f"{name}.json")
                result["A"][name] = {**evidence.check_a(RUNS / name, papers),
                                     "rejected_assertions": summary["rejected_assertions"]}
        if corpus == "biotriplex":
            test_papers = evidence.load_papers(data("e4", corpus, ["test"]))
            for model in MODELS:
                for form in FULL_TEXT_FORMS[model]:
                    name = f"e4_{corpus}_{model}_{form}"
                    result["B"][name] = evidence.check_b(RUNS / name, test_papers)
    io.write_json(ROOT / "results" / "evidence_checks.json", result)
    print("wrote results/evidence_checks.json")


def sample(prefix: str, plans: dict, seed: int, splits: dict, folder: Path) -> None:
    if (folder / "_key_DO_NOT_OPEN.json").exists():
        raise SystemExit(f"{folder} already holds a sample; it is drawn once")
    items, designs, papers = [], {}, {}
    for corpus in ("biored", "biotriplex"):
        runs = frame_runs(prefix, corpus)
        papers[corpus] = evidence.load_papers(data(prefix, corpus, splits[corpus]))
        frame = evidence.build_frame(papers[corpus], runs)
        drawn, designs[corpus] = evidence.draw(frame, runs, plans[corpus], seed, corpus)
        items += drawn
    evidence.write_stage1(items, papers, folder, seed)
    io.write_json(folder / "design.json", {"seed": seed, "plans": plans, "corpora": designs,
                                           "items": len(items)})
    print(f"wrote {len(items)} items to {folder}")


def estimate() -> None:
    """Protocol §6 on the locked stage-1 labels and the stage-2 labels of the test sheet."""
    from gene2disease import evidence_stats as es
    folder = SHEETS / "test"
    evidence.verify_stage1_lock(folder)
    key = io.read_json(folder / "_key_DO_NOT_OPEN.json")
    design = io.read_json(folder / "design.json")
    items = key["items"]
    q1 = {row["item_id"]: row["q1"] for row in evidence.read_labels(
        folder / "stage1_labels.csv", "q1", [item["item_id"] for item in items], stage1=True)}
    set_ids = [set_id for item in items for set_id in item["citation_sets"]]
    q2 = {row["set_id"]: row["q2"] for row in evidence.read_labels(
        folder / "stage2_labels.csv", "q2", set_ids, stage1=False)}
    result = {"protocol": "annotation/EVIDENCE_PROTOCOL.md",
              "implementation_notes": es.IMPLEMENTATION_NOTES, "corpora": {}}
    for corpus in ("biored", "biotriplex"):
        mine = [item for item in items if item["corpus"] == corpus]
        strata = design["corpora"][corpus]["strata"]
        by_stratum = {name: [item for item in mine if item["stratum"] == name] for name in ("S1", "S2", "G")}
        section = {"strata": {name: es.stratum_proportions(by_stratum[name], q1, strata[name]["N"])
                              for name in ("S1", "S2", "G")}}
        s1, s2 = section["strata"]["S1"], section["strata"]["S2"]
        section["H_E1_S1_minus_S2"] = es.newcombe(s1["supported"], s1["n"], s1["N"],
                                                   s2["supported"], s2["n"], s2["N"])
        section["H_E1_S1_minus_S2_without_fpc"] = es.newcombe(s1["supported"], s1["n"], None,
                                                               s2["supported"], s2["n"], None)
        runs = sorted(frame_runs("e4", corpus))
        n_by_run = {name: strata[name]["N_by_run"] for name in ("S1", "S2")}
        counts = {}
        if corpus == "biored":
            papers = evidence.load_papers(data("e4", corpus))
            for run in runs:
                tp = fp = 0
                for paper_id, paper in papers.items():
                    for pair in evidence.positive_assertions(RUNS / run, paper_id):
                        tp, fp = (tp + 1, fp) if pair in paper.gold else (tp, fp + 1)
                counts[run] = {"TP": tp, "FP": fp}

        def extra(rates, counts=counts, runs=runs):
            out = {}
            for model in MODELS:
                pw, jt = (f"e4_{corpus}_{model}_pairwise", f"e4_{corpus}_{model}_joint")
                out[f"difference:{model} pairwise minus joint"] = (
                    None if rates[pw] is None or rates[jt] is None else rates[pw] - rates[jt])
            for run, c in counts.items():
                out[f"adjusted_precision:{run}"] = (
                    None if rates[run] is None else (c["TP"] + c["FP"] * rates[run]) / (c["TP"] + c["FP"]))
            return out
        section["runs"] = es.bootstrap_runs({k: v for k, v in by_stratum.items() if k != "G"},
                                            q1, n_by_run, runs, extra)
        section["run_sample_counts"] = {run: {name: sum(run in item["runs"] for item in by_stratum[name])
                                              for name in ("S1", "S2")} for run in runs}
        if counts:
            section["BioRED_TP_FP"] = counts
        section["H_E2_any"] = es.citation_gap(mine, q1, q2, "any")
        section["H_E2_all"] = es.citation_gap(mine, q1, q2, "all")
        result["corpora"][corpus] = section
    result["H_E2_pooled_both_corpora_any"] = es.citation_gap(items, q1, q2, "any")["pooled_item_weighted"]
    io.write_json(ROOT / "results" / "evidence_estimates.json", result)
    print("wrote results/evidence_estimates.json")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "checks":
        checks()
    elif command == "estimate":
        estimate()
    elif command == "pilot":
        sample("ef_dev", {"biored": PILOT_PLAN, "biotriplex": PILOT_PLAN}, PILOT_SEED,
               {"biored": ["Dev"], "biotriplex": ["train"]}, SHEETS / "pilot")
    elif command == "test":
        # Needs the pilot's stage 1 and stage 2 both completed (minutes over both stages).
        minutes = evidence.median_item_minutes(SHEETS / "pilot")
        plans, predicted = evidence.scaled_plans(TEST_PLAN, minutes)
        print(f"pilot median {minutes} min per item; plans {plans}; predicted {predicted / 60:.1f} h")
        sample("e4", plans, TEST_SEED, {"biored": ["Test"], "biotriplex": ["test"]}, SHEETS / "test")
    elif command == "stage2":
        folder = Path(sys.argv[2])
        prefix, splits = (("e4", {"biored": ["Test"], "biotriplex": ["test"]}) if folder.name == "test"
                          else ("ef_dev", {"biored": ["Dev"], "biotriplex": ["train"]}))
        papers = {corpus: evidence.load_papers(data(prefix, corpus, splits[corpus]))
                  for corpus in ("biored", "biotriplex")}
        seed = TEST_SEED if folder.name == "test" else PILOT_SEED
        evidence.write_stage2(folder, papers, seed)
        print(f"wrote {folder / 'stage2.html'}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
