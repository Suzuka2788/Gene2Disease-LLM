"""Write the E4 configurations (frozen test evaluation) and the freeze configuration.

Usage (from the project root):  python scripts/make_e4_configs.py

Every judge setting is the development setting of the same condition (configs/ef_dev),
with only the run id, the output path, the split and the freeze manifest changed.
The script refuses to overwrite configs/e4/ once the manifest exists.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "e4"
MANIFEST = "configs/e4/frozen_manifest.json"

DATA = {
    "biored": ['dataset = "biored"', 'folder = "../../archive_v1/experiments/biored/data/BioRED"',
               'splits = ["Test"]'],
    "biotriplex": ['dataset = "biotriplex"',
                   'release = "../../archive_v1/experiments/biotriplex/data/splits.json"',
                   'splits = ["val", "test"]'],
}
MODELS = {"qwen": "qwen3.5-9b", "gemma": "gemma-4-12b"}
# (corpus, model, form, input scope)
RUNS = [(corpus, model, form, "full") for corpus in ("biored", "biotriplex")
        for model, forms in (("qwen", ("pairwise", "chunked", "joint")),
                             ("gemma", ("pairwise", "joint")))
        for form in forms]
RUNS += [("biotriplex", model, "pairwise", "abstract") for model in ("qwen", "gemma")]

# Primary family: pairwise minus joint, recall and (BioRED) precision, per model and
# corpus: 6 contrasts, Bonferroni level 1 - 0.05 / 6. The rest is secondary at 0.95.
PRIMARY_LEVEL = 1 - 0.05 / 6
COMPARES = [(corpus, model, "pairwise", "joint", "full", PRIMARY_LEVEL)
            for corpus in ("biored", "biotriplex") for model in ("qwen", "gemma")]
COMPARES += [(corpus, "qwen", "pairwise", "chunked", "full", 0.95) for corpus in ("biored", "biotriplex")]
COMPARES += [("biotriplex", model, "pairwise", "pairwise", "abstract", 0.95) for model in ("qwen", "gemma")]


def run_id(corpus: str, model: str, form: str, scope: str) -> str:
    return f"e4_{corpus}_{model}_{form}" + ("_abstract" if scope == "abstract" else "")


def judge_config(corpus: str, model: str, form: str, scope: str) -> str:
    name = run_id(corpus, model, form, scope)
    lines = [
        f"# E4 frozen test evaluation. Same settings as ef_dev_{corpus}_{model}_{form}"
        + ("_abstract" if scope == "abstract" else "") + " (development).",
        "[run]", 'stage = "judge"', f'output = "results/{name}.json"', 'runs_dir = "runs"',
        f'freeze_manifest = "{MANIFEST}"', "", "[data]", *DATA[corpus], "", "[judge]",
        f'run_id = "{name}"', f'form = "{form}"']
    if form == "chunked":
        lines.append("chunk_size = 5")
    lines += ['level = "block"', 'base_url = "http://127.0.0.1:8081"',
              'api_key_file = "%LOCALAPPDATA%/gene2disease/api_key.txt"',
              f'model_id = "{MODELS[model]}"', "timeout_seconds = 900", 'output_format = "json"',
              "thinking = false", "cache_prompt = false"]
    if scope == "abstract":
        lines.append('input_scope = "abstract"')
    lines += ["output_base_tokens = 1000", "output_tokens_per_pair = 150",
              "# Execution control, not part of the condition: stop at a server failure.",
              "stop_on_operational_failure = true"]
    return "\n".join(lines) + "\n"


def compare_name(corpus, model, first, second, scope) -> str:
    if scope == "abstract":
        return f"compare_e4_{corpus}_{model}_full_vs_abstract"
    return f"compare_e4_{corpus}_{model}_{first}_vs_{second}"


def compare_config(corpus, model, first, second, scope, level) -> str:
    name = compare_name(corpus, model, first, second, scope)
    role = "primary family (Bonferroni, 6 contrasts)" if level != 0.95 else "secondary"
    lines = [
        f"# E4 comparison, {role}.", "[run]", 'stage = "compare"',
        f'output = "results/{name}.json"', 'runs_dir = "runs"',
        f'freeze_manifest = "{MANIFEST}"', "", "[data]", *DATA[corpus], "", "[stats]",
        "n_resamples = 10000", "seed = 20261107", f"level = {level!r}", "", "[compare]",
        f'first = "{run_id(corpus, model, first, "full")}"',
        f'second = "{run_id(corpus, model, second, scope)}"', 'view = "main"', "delta = 0.05",
        f"report_precision = {'true' if corpus == 'biored' else 'false'}",
        "load_bands = [1, 4, 15, 40]"]
    if scope == "abstract":
        lines.append('allow_differing = ["input_scope"]')
    return "\n".join(lines) + "\n"


FREEZE = '''# The E4 freeze. Running it writes configs/e4/frozen_manifest.json once; commit that
# file before any E4 run. Usage: PYTHONPATH=src python -m gene2disease.cli run --config configs/e4/freeze.toml
[run]
stage = "freeze"
output = "{manifest}"

[freeze]
configs = [
{configs}
]
weights = [
  "%LOCALAPPDATA%/gene2disease/downloads/Qwen3.5-9B-Q4_K_M.gguf",
  "%LOCALAPPDATA%/gene2disease/downloads/gemma-4-12b-it-qat-q4_0.gguf",
]
scripts = ["scripts/run_ef_dev.py", "scripts/serve_local.ps1"]
notes = "Protocol fixed on development data (EF_DEV_RESULTS.md, RESEARCH_PLAN.md D14-D18)."

[freeze.protocol]
estimand = "end-to-end relation-judgment performance under this protocol, output failures included (D18)"
server_build = "b12182-fd27b2034"
server = "llama.cpp llama-server, --ctx-size 32768 --parallel 1 --n-gpu-layers 99 --flash-attn on --reasoning off"
models = ["qwen3.5-9b: Qwen3.5-9B-Q4_K_M.gguf", "gemma-4-12b: gemma-4-12b-it-qat-q4_0.gguf"]
decoding = "temperature 0, thinking off, prompt cache off, one user message"
output = "JSON for every form and both models; minimal instruction; paragraph-level passages"
output_cap = "1000 + 150 x listed pairs"
failures = "truncation, parse failure and context refusal (server error type exceed_context_size_error) are terminal outcomes of a request: no prediction, stored and replayed on resume; no repair or re-query"
operational_failures = "server or transport errors stop the run before the affected paper is recorded; the run is resumed from the answer cache after the server is fixed, and every start is journaled"
provider_block_retry = "exception to no re-query: the code retries a provider refusal once, identically; a second refusal is a terminal no-prediction outcome; the local server refused no request in development; any retry is reported"
readings = "primary: canonical spellings; strict reading reported alongside as a descriptive sensitivity analysis"
metrics = "micro recall on both corpora; precision and F1 on BioRED only; paper-level paired bootstrap, 10000 resamples, seed 20261107; delta 0.05 four-way reading on the interval of each comparison"
primary_family = "canonical reading, pairwise minus joint: recall (BioRED, BioTriplex) and precision (BioRED), per model: 6 contrasts, Bonferroni interval level 1 - 0.05/6 = 0.99167"
secondary = "descriptive: F1 at the level of its comparison config (0.99167 in pairwise-versus-joint configs); Qwen pairwise minus chunked and full text minus abstract (pairwise) at 0.95; strict-reading results; failure counts"
no_changes_after_e4 = "no change of context size, chunking, format or any setting after E4 outcomes are seen"
context_limit_check = "verified 2026-11-08 with synthetic prompts on this build and Qwen3.5-9B: a 34768-token prompt is refused with error type exceed_context_size_error; a 31868-token prompt stops generation at 32768 total tokens with finish_reason length (truncation), without context shifting"
'''


def main() -> None:
    if (ROOT / MANIFEST).exists():
        raise SystemExit("the protocol is already frozen; configs/e4 must not change")
    OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for item in RUNS:
        name = run_id(*item)
        (OUT / f"{name}.toml").write_text(judge_config(*item), encoding="utf-8")
        names.append(name)
    for item in COMPARES:
        name = compare_name(*item[:5])
        (OUT / f"{name}.toml").write_text(compare_config(*item), encoding="utf-8")
        names.append(name)
    listed = ",\n".join(f'  "configs/e4/{name}.toml"' for name in names)
    (OUT / "freeze.toml").write_text(FREEZE.format(manifest=MANIFEST, configs=listed),
                                     encoding="utf-8")
    print(f"wrote {len(names)} configurations and configs/e4/freeze.toml")


if __name__ == "__main__":
    main()
