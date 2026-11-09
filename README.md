# Gene2Disease-LLM

**Request design, input scope, and evaluation choices in gene–disease relation judgment with local large language models.**

Project page: <https://suzuka2788.github.io/Gene2Disease-LLM/>

This repository is an empirical study, not a new extraction method. It asks how three choices change the conclusions drawn about LLM judgment of gene–disease relations when the entities are given:

- **Request form:** one entity pair per request (pairwise), up to five pairs (chunked), or all pairs of a passage (joint).
- **Input scope:** abstract passages versus full-text passages for the same papers, entities, and gold relations.
- **Evaluation choices:** how model output is read (strict versus canonical), the output format, failure accounting, and whether cited sentences actually support a prediction.

Two local models (Qwen3.5-9B and Gemma-4-12B, through llama.cpp) were compared under a protocol fixed on development data, then evaluated once on BioRED Test and on the BioTriplex validation and test splits. A source-hidden, single-annotator assessment checked whether the passage and the cited sentences support 76 sampled predictions.

It succeeds the earlier project [GeneDiseaseTrace-LLM](https://github.com/Suzuka2788/GeneDiseaseTrace-LLM). None of the earlier numerical results are reused here; the pipeline was rewritten and every reported result was regenerated.

## Main results (frozen test evaluation)

| Question | Result |
|---|---|
| Request form, BioRED (abstracts) | Pairwise recall higher than joint by 0.214 (Qwen3.5-9B) and 0.122 (Gemma-4-12B); joint precision higher by 0.102 and 0.207 (99.17% intervals exclude zero). |
| Request form, BioTriplex (full text, 30 articles) | Recall difference 0.039 in both models, inconclusive. |
| Input scope | Full-text recall 0.968 against 0.422 and 0.448 from abstracts; abstract passages cover only 73 of 154 gold pairs. |
| Evidence assessment, BioRED | Predictions outside the gold set made only by pairwise requests were supported by their displayed passage in 2 of 18 sampled cases, against 7 of 16 for joint requests. |
| Citations | Every retained citation identifier resolves; among passage-supported items, the cited sentences did not support the relation in 10.7% (BioRED) and 14.9% (BioTriplex). |
| Cost | Joint requests saved 81–89% of prompt tokens but only 25–40% of local compute time. |

These are single runs of two 9–12B local models; see the limitations in each results document.

## What is here

| Path | Contents |
|---|---|
| `src/gene2disease/` | The pipeline: corpus readers, passages and candidate pairs, request forms, validation, scoring, paired bootstrap, the frozen-protocol unlock path, the evidence assessment. Standard library only. |
| `tests/` | 218 unit and end-to-end tests (`PYTHONPATH=src python -m unittest discover -s tests`). |
| `configs/` | Every configuration: development runs (`ef_dev/`), frozen test runs (`e4/`, with `frozen_manifest.json`), smoke tests. |
| `scripts/` | Run driver, local server launcher, configuration generator, evidence-assessment tool, table generator. |
| `results/` | Aggregate results of every run and comparison (no per-paper output). |
| `annotation/` | The evidence-assessment protocol (with its dated revision log) and the annotation criteria. |
| `EF_DEV_RESULTS.md`, `E4_RESULTS.md`, `EVIDENCE_RESULTS.md` | Development, test, and evidence-assessment results with their interpretation (in Chinese). |
| `E4_PROVENANCE.md` | Append-only audit record of the freeze and the test evaluation. |
| `F0_REPORT.md`, `EF_FORMAT_SENSITIVITY.md` | Feasibility run and the output-format sensitivity plan. |
| `docs/` | The project page. |

**Not included:** the manuscript (in preparation), corpus texts, gold annotations, per-paper model outputs, annotation sheets, model weights, and internal planning and review documents. BioRED and BioTriplex must be obtained from their publishers under their own terms; the configurations point to local copies.

## Frozen protocol and audit trail

The test splits can only be read through `configs/e4/frozen_manifest.json`, which records the digests of the code, configurations, data files, model weights, and execution scripts at the freeze. This public repository starts from a clean history; the freeze and test runs were made in the private development repository (freeze commit `5ac1447`, code commit `8a64732`), and `E4_PROVENANCE.md` records the chronology. The manifest is included unchanged, so its file digests refer to the files as frozen; the code in this repository may have changed since (for example, the evidence-assessment tool was added after the freeze).

## Reproducing a run

1. Obtain BioRED and BioTriplex, and adjust the data paths in `configs/`.
2. Download llama.cpp (release b12182) and the two GGUF models (`unsloth/Qwen3.5-9B-GGUF`, Q4_K_M; `google/gemma-4-12B-it-qat-q4_0-gguf`), then start the server with `scripts/serve_local.ps1`.
3. Run a configuration: `PYTHONPATH=src python -m gene2disease.cli run --config configs/ef_dev/ef_dev_biored_qwen_pairwise.toml`, or a whole batch with `python scripts/run_ef_dev.py main`.

Greedy decoding on a local server keeps runs close to deterministic, but outputs are not guaranteed to be identical across hardware or builds.
