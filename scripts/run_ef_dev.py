"""Run a batch unattended: development EF (main, extra) or the frozen test evaluation (e4).

Usage (from the project root):  python scripts/run_ef_dev.py [main|extra|e4]

The e4 batch also checks the weight files against the frozen manifest before each
server start and the server build after it, and counts a request the server refuses as
exceeding the context as a failed call (frozen protocol) without stopping.

Starts each model's server once (scripts/serve_local.ps1), checks that the served model
and context are the expected ones, runs that model's judging configs in a fixed order,
then stops the server. After every run it records elapsed time and call outcomes in
runs/ef_dev_progress.jsonl. It stops at the first operational failure: a calls record
with outcome "failed" (HTTP, connection or envelope errors), a non-zero exit, or a
wrong model. Truncations and parse failures are results and do not stop it. A judging
run that was interrupted can be resumed by running this again: answered calls come back
from the run's answer cache.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gene2disease import f0, io  # noqa: E402

BASE_URL = "http://127.0.0.1:8081"
KEY_FILE = "%LOCALAPPDATA%/gene2disease/api_key.txt"
CONTEXT = 32768
PLAN = [
    ("qwen3.5-9b", ["ef_dev_biored_qwen_pairwise", "ef_dev_biored_qwen_joint", "ef_dev_biored_qwen_chunked",
                    "ef_dev_biotriplex_qwen_pairwise", "ef_dev_biotriplex_qwen_joint",
                    "ef_dev_biotriplex_qwen_chunked"]),
    ("gemma-4-12b", ["ef_dev_biored_gemma_pairwise", "ef_dev_biored_gemma_joint",
                     "ef_dev_biotriplex_gemma_pairwise", "ef_dev_biotriplex_gemma_joint"]),
]
COMPARES = ["compare_ef_dev_biored_qwen_pairwise_vs_joint", "compare_ef_dev_biored_qwen_pairwise_vs_chunked",
            "compare_ef_dev_biotriplex_qwen_pairwise_vs_joint", "compare_ef_dev_biotriplex_qwen_pairwise_vs_chunked",
            "compare_ef_dev_biored_gemma_pairwise_vs_joint", "compare_ef_dev_biotriplex_gemma_pairwise_vs_joint"]
# Second batch (TWO_WEEK_PLAN.md): D14 lines runs, the exploratory BioRED-guideline instruction runs and the
# matched abstract-only runs, then their comparisons. Run after the main batch.
EXTRA_PLAN = [
    ("qwen3.5-9b", ["ef_dev_biotriplex_qwen_pairwise_lines", "ef_dev_biotriplex_qwen_joint_lines",
                    "ef_dev_biored_qwen_pairwise_guideline", "ef_dev_biotriplex_qwen_pairwise_abstract"]),
    ("gemma-4-12b", ["ef_dev_biotriplex_gemma_pairwise_lines", "ef_dev_biotriplex_gemma_joint_lines",
                     "ef_dev_biored_gemma_pairwise_guideline", "ef_dev_biotriplex_gemma_pairwise_abstract"]),
]
EXTRA_COMPARES = [f"compare_ef_dev_{name}" for model in ("qwen", "gemma") for name in (
    f"biotriplex_{model}_pairwise_vs_joint_lines", f"biotriplex_{model}_pairwise_json_vs_lines",
    f"biotriplex_{model}_joint_json_vs_lines", f"biored_{model}_minimal_vs_guideline",
    f"biotriplex_{model}_full_vs_abstract")]
# E4, the frozen test evaluation (configs/e4, written by scripts/make_e4_configs.py).
E4_PLAN = [
    ("qwen3.5-9b", [f"e4_{corpus}_qwen_{form}" for corpus in ("biored", "biotriplex")
                    for form in ("pairwise", "chunked", "joint")] + ["e4_biotriplex_qwen_pairwise_abstract"]),
    ("gemma-4-12b", [f"e4_{corpus}_gemma_{form}" for corpus in ("biored", "biotriplex")
                     for form in ("pairwise", "joint")] + ["e4_biotriplex_gemma_pairwise_abstract"]),
]
E4_COMPARES = [f"compare_e4_{corpus}_{model}_pairwise_vs_joint" for corpus in ("biored", "biotriplex")
               for model in ("qwen", "gemma")]
E4_COMPARES += [f"compare_e4_{corpus}_qwen_pairwise_vs_chunked" for corpus in ("biored", "biotriplex")]
E4_COMPARES += [f"compare_e4_biotriplex_{model}_full_vs_abstract" for model in ("qwen", "gemma")]
# batch: (plan, compares, config folder, progress log)
BATCHES = {"main": (PLAN, COMPARES, "configs/ef_dev", "ef_dev_progress.jsonl"),
           "extra": (EXTRA_PLAN, EXTRA_COMPARES, "configs/ef_dev", "ef_dev_progress.jsonl"),
           "e4": (E4_PLAN, E4_COMPARES, "configs/e4", "e4_progress.jsonl")}
E4_MANIFEST = ROOT / "configs" / "e4" / "frozen_manifest.json"
WEIGHTS = {"qwen3.5-9b": "Qwen3.5-9B-Q4_K_M.gguf", "gemma-4-12b": "gemma-4-12b-it-qat-q4_0.gguf"}
PROGRESS = ROOT / "runs" / "ef_dev_progress.jsonl"
CONFIG_FOLDER = "configs/ef_dev"


def check_weights(model: str) -> None:
    """E4 only: the weight file must have the digest recorded at the freeze."""
    from gene2disease import freeze
    expected = io.read_json(E4_MANIFEST)["weights_sha256"][WEIGHTS[model]]
    path = Path(os.path.expandvars("%LOCALAPPDATA%/gene2disease/downloads")) / WEIGHTS[model]
    actual = freeze.file_digest(path)
    log({"event": "weights_checked", "model": model, "matches": actual == expected})
    if actual != expected:
        raise SystemExit(f"{WEIGHTS[model]} differs from the frozen weights")


def log(record: dict) -> None:
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    print(json.dumps(record), flush=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def start_server(model: str) -> subprocess.Popen:
    if healthy():
        raise SystemExit("a server is already listening on port 8081; stop it first")
    server_log = open(ROOT / "runs" / f"server_{model}_ef_dev.log", "a", encoding="utf-8")
    process = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(ROOT / "scripts" / "serve_local.ps1"), "-Model", model],
        stdout=server_log, stderr=subprocess.STDOUT)
    for _ in range(180):
        if healthy():
            break
        if process.poll() is not None:
            raise SystemExit(f"server for {model} exited early")
        time.sleep(2)
    else:
        stop_server(process)
        raise SystemExit(f"server for {model} did not become healthy")
    props = f0.server_properties({"base_url": BASE_URL, "api_key_file": KEY_FILE})
    log({"event": "server_started", "model": model, **props})
    if props.get("model_alias") != model or props.get("n_ctx") != CONTEXT:
        stop_server(process)
        raise SystemExit(f"unexpected server: {props}")
    if CONFIG_FOLDER == "configs/e4":
        build = io.read_json(E4_MANIFEST)["protocol"]["server_build"]
        if props.get("build_info") != build or props.get("total_slots") != 1:
            stop_server(process)
            raise SystemExit(f"server differs from the frozen protocol: {props}")
    return process


def stop_server(process: subprocess.Popen) -> None:
    # Only the process tree this driver started; other llama.cpp servers are left alone.
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    log({"event": "server_stopped"})


def run_config(name: str) -> None:
    started = time.monotonic()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-m", "gene2disease.cli", "run", "--config",
                             f"{CONFIG_FOLDER}/{name}.toml"], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    elapsed = round(time.monotonic() - started, 1)
    if result.returncode != 0:
        log({"event": "run_error", "config": name, "elapsed_seconds": elapsed,
             "stderr_tail": result.stderr[-1500:]})
        raise SystemExit(f"{name} exited with {result.returncode}")
    record = {"event": "run_done", "config": name, "elapsed_seconds": elapsed}
    run_dir = ROOT / "runs" / name
    if (run_dir / "run.json").is_file() and not name.startswith("compare_"):
        outcomes: Counter = Counter()
        cache_hits = 0
        details: Counter = Counter()
        for path in sorted(run_dir.glob("*/calls.jsonl")):
            for call in io.read_jsonl(path):
                outcomes[call["outcome"]] += 1
                cache_hits += bool(call.get("cache_hit"))
                if call["outcome"] == "failed":
                    details[call.get("detail", "")[:120]] += 1
        record.update(outcomes=dict(outcomes), cache_hits=cache_hits)
        if details:
            record["failed_details"] = dict(details.most_common(5))
        log(record)
        # Truncations, parse failures, context refusals and final provider blocks are
        # results under the protocol; only "failed" (server or transport) stops the batch.
        if outcomes.get("failed"):
            raise SystemExit(f"{name}: {outcomes['failed']} operational failures; see the progress log")
    else:
        log(record)


def main() -> None:
    batch = sys.argv[1] if len(sys.argv) > 1 else "main"
    if batch not in BATCHES:
        raise SystemExit(f"unknown batch {batch}; choose from {sorted(BATCHES)}")
    global PROGRESS, CONFIG_FOLDER
    plan, compares, CONFIG_FOLDER, progress = BATCHES[batch]
    PROGRESS = ROOT / "runs" / progress
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    if batch == "e4":
        if not E4_MANIFEST.is_file():
            raise SystemExit("E4 needs the frozen manifest; run configs/e4/freeze.toml and commit it")
        from gene2disease import freeze
        frozen = io.read_json(E4_MANIFEST).get("scripts_sha256", {})
        for script in ("scripts/run_ef_dev.py", "scripts/serve_local.ps1"):
            if frozen.get(script) != freeze.file_digest(ROOT / script):
                raise SystemExit(f"{script} differs from the frozen version")
    log({"event": "driver_started", "batch": batch})
    for model, configs in plan:
        if batch == "e4":
            check_weights(model)
        process = start_server(model)
        try:
            for name in configs:
                log({"event": "run_started", "config": name})
                run_config(name)
        finally:
            stop_server(process)
    for name in compares:
        run_config(name)
    log({"event": "driver_finished", "batch": batch})


if __name__ == "__main__":
    main()
