"""Write paper/run_failures.tex: every development and test run with its failure counts,
read directly from results/ef_dev_*.json and results/e4_*.json (no hand-copied numbers).

Usage (from the project root):  python scripts/make_run_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = {"qwen": "Qwen3.5-9B", "gemma": "Gemma-4-12B"}


def label(run_id: str) -> tuple[str, str, str, str]:
    stage, rest = ("Test", run_id[3:]) if run_id.startswith("e4_") else ("Dev", run_id[7:])
    corpus, model, form, *extra = rest.split("_")
    variant = {"lines": "line format", "guideline": "guideline rules", "abstract": "abstract only"}
    form = form + (f", {variant[extra[0]]}" if extra else "")
    return stage, "BioRED" if corpus == "biored" else "BioTriplex", MODEL[model], form


def main() -> None:
    rows = []
    for path in sorted((ROOT / "results").glob("ef_dev_*.json")) + sorted((ROOT / "results").glob("e4_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        dispositions = result["dispositions"]
        rows.append((*label(result["run_id"]), result["logical_requests"], result["failed_calls"],
                     result["truncated_calls"], result.get("context_refused_calls", 0),
                     result["rejected_assertions"], dispositions.get("omitted_by_model", 0),
                     dispositions.get("failed", 0)))
    lines = [r"\begin{table}[p]", r"\centering",
             r"\caption{\textbf{Failure accounting for every run.} Calls: logical requests. "
             r"Failed: failed calls, including truncations. No context refusals occurred. "
             r"Rejected: assertions rejected by the validator. Omitted and failed pairs: candidate "
             r"pairs whose final disposition in the paper is an omission by the model or a failure "
             r"(a failed call or wholly rejected assertions). Generated from the run summaries.}",
             r"\label{tab:run-failures}", r"\scriptsize",
             r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lllllrrrrrr@{}}",
             r"\toprule",
             r"Data & Corpus & Model & Form & & Calls & Failed & Trunc. & Rejected & Omitted & Failed pairs \\",
             r"\midrule"]
    for stage, corpus, model, form, calls, failed, truncated, refused, rejected, omitted, failed_pairs in rows:
        assert refused == 0, "context refusals must be added to the table"
        lines.append(f"{stage} & {corpus} & {model} & {form} & & {calls:,} & {failed} & {truncated} & "
                     f"{rejected:,} & {omitted:,} & {failed_pairs:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (ROOT / "paper" / "run_failures.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote paper/run_failures.tex with {len(rows)} runs")


if __name__ == "__main__":
    main()
