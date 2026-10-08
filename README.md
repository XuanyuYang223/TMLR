# Correct relational geometry and unseen composition

This repository contains the code, fixed protocols, and results for **When Does Correct Relational Geometry Support Unseen Composition?**

The revised manuscript follows relation-specific functional gains, predicted readout-null information transfer, and the interpretation after adequate affine fitting. Its claims are conditional on relation-prepared representations.

## Paper and results

- [Revised manuscript PDF](paper/manuscript.pdf) and [LaTeX source](paper/manuscript.tex).
- [Related-work comparison: 12 primary studies](paper/literature_review.csv).
- [Final five-source results](results/final_mechanism_confirmation/results.json) and [report](results/final_mechanism_confirmation/report.html).
- [Downloadable experiment archives](https://github.com/XuanyuYang223/ICML/releases/tag/experiment-snapshot-2026-10-07).
- [Historical experiment documentation](docs/README_history.md).

Final collision AB accuracy is 20.31% / 8.44% for GELU correct / answer-matched incorrect pairing, versus 8.44% / 9.45% for budget-linear operators. Correct predicted-null transfer raises the linear recipient to 17.19% while preserving first-step logits. Same-loss converged affine maps reach 19.92% / 8.36%; an additional functional expressivity advantage was not confirmed. Numerical conditioning and precision sensitivity remain limitations.

## Repository and archives

All project implementations, fixed configurations, and tests are in `experiments/`, `configs/`, and `tests/`. Readable JSON/CSV results, reports, original audit manifests, and generated figures from the experiment history are tracked in Git.

| Release archive | Contents | Download size |
|---|---|---:|
| `mechanism-confirmations.tar.gz` | Final five-source confirmation, preceding three-source activation confirmation, reused-source mechanism diagnostics, and historical test shards needed by the final split audit | 131 MB |
| `directional-loss-confirmation.tar.gz` | Full/null-only correct/incorrect confirmation in matrix and permutation domains | 1.94 GB |
| `initial-cross-domain.tar.gz` | Initial joint affine matrix and polynomial experiments | 249 MB |
| `historical-inputs.tar.gz` | Inputs referenced by historical dataset aliases; completes the cross-domain matrix inputs | 212 MB |
| `reviewer-controls.tar.gz` | Frozen-model reviewer controls, new held-out inputs, probes, and independent numerical replay | 3.9 MB |

Each archive has a per-file SHA-256 inventory. Absolute workspace symlinks are stored as regular files for portability. Early exploratory bulk caches and legacy permutation transformer weights from the 133 GB local workspace are not included in these assets; their code, protocols, and readable results remain available. Downloaded third-party full-text papers are excluded.

Reviewer-requested frozen-model controls are in `results/null_space_review_controls`, with the readable [report](results/null_space_review_reporting/report.html) and independent reconstruction in `results/null_space_review_audit`. They are post-confirmation diagnostics. Fresh correct-null versus incorrect-null transfer replicates (15.31% versus 7.19%), but answer probes and state-distribution controls do not establish natural relation-specific encoding. Oracle error-calibrated controls use true intermediate states and are not learned predictors.

## Install and reproduce

Use Python 3.11 or newer. Archived final training used Python 3.13 and PyTorch 2.11. GPU or library changes may affect exact replay.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
git clone https://github.com/XuanyuYang223/neurips.git external/neurips
git -C external/neurips checkout 74f0de2017115f06f11b7285366b66e7e8431d30
python -m pytest -q
```

The pinned upstream checkout supplies PermWorld definitions and transformer code. It is not vendored or modified.

To restore the archived groups and independently replay the final confirmation:

```bash
gh release download experiment-snapshot-2026-10-07 --repo XuanyuYang223/ICML --dir downloads
python -m experiments.restore_public_artifacts --directory downloads
python -m experiments.public_reproduction --replay-final
python -m experiments.null_space_review_verify
```

Restore verifies archive/file checksums and refuses to overwrite a differing local file. Replay runs in a temporary copy, preserving sealed timestamps and manifests. Historical absolute manifest paths are resolved relative to the checkout. Before downloading data, `python -m experiments.public_reproduction --allow-missing` explicitly inventories missing binaries.

Fresh-training stages are implemented in `experiments/final_mechanism_train.py`: `prepare`, `sources`, `fits`, `evaluate`. Use the separate-output wrapper to preserve the sealed snapshot:

```bash
python -m experiments.retrain_final_confirmation --output results/my_reproduction --stage prepare
python -m experiments.retrain_final_confirmation --output results/my_reproduction --stage run
```

Restore the historical test shards first. The wrapper keeps the archived configuration and seeds, changing only the output location. Source preparation and fitting require CUDA; evaluation follows completion of all registered fits.

## Rebuild the revised paper

```bash
python -m experiments.manuscript_revision
python -m experiments.null_space_review_report
cd paper
tectonic --untrusted --keep-logs manuscript.tex
```

Generated tables read immutable completed results. Editorial sections and the new bibliography are separate from the sealed original drafts. The PDF was compiled with Tectonic 0.17.0.

## Evidence boundaries

Five final sources are independent initializations with shared source corpus/test data. Supports and endpoints are not extra independent replications. Bootstrap intervals are descriptive; zero crossing does not establish equivalence. Predicted-null swaps preserve one fixed linear readout, not every possible answer code. True-state repairs are diagnostics. Polynomial results also have primitive-generalization failures. Protocol differences and exploratory versus confirmation status are documented in the manuscript.

Original training experiments are closed. Reviewer controls add frozen-model interventions, independent label probes, and a small newly held-out test, without retraining encoders or relation operators. Their post-confirmation status and oracle information use are explicit.
