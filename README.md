# Correct relational geometry and unseen composition

Code, fixed protocols, results, and manuscript for **When Does Correct Relational Geometry Support Unseen Composition?** Findings concern supplied correspondences in relation-prepared representations.

## Paper and results

- [Revised manuscript PDF](paper/manuscript.pdf) and [LaTeX source](paper/manuscript.tex).
- [New-field confirmation report](results/reviewer_fresh_reporting_v2/report.html), [all 17 units](results/reviewer_fresh_confirmation/results.json), [frozen protocol](results/reviewer_fresh_confirmation/protocol.json), and [independent audit](results/reviewer_fresh_audit/verification.json).
- [Expanded old-model diagnostic report](results/reviewer_revision_reporting/report.html): larger tests, natural-state swaps, full ridge path, and secondary TOST.
- [Earlier five-initialization results](results/final_mechanism_confirmation/results.json) and [frozen reviewer controls](results/null_space_review_reporting/report.html).
- [Related work: 13 primary studies](paper/literature_review.csv) and [historical experiment documentation](docs/README_history.md).
- [New archives](https://github.com/XuanyuYang223/2026TMLR/releases/tag/reviewer-revision-2026-10-09) and [earlier archives](https://github.com/XuanyuYang223/2026TMLR/releases/tag/experiment-snapshot-2026-10-07).

The frozen F17 confirmation uses **17 independently sampled datasets and source initializations**, width 512 (four times the old width), and 2,000 collision pairs plus 512 IID inputs per unit. All registered fits finish before composite testing. GELU correct/answer-matched incorrect compound accuracy is **11.52% / 6.36%**, a **5.157pp** source-level difference (marginal 95% CI **[1.924, 8.391]**, Holm p **0.003807**); every unit's difference is positive. The second-step noise candidate is selected using known single-step validation only. Original second operators are also reported.

Both stronger gates **fail**: compound accuracy remains below the declared 40%, and same-future-answer natural null swaps change **28.10%** of predictions versus a 5% threshold. Distinct-future null donors are followed in 52.02% of cases, versus 2.70% for row donors; row swaps are a localization diagnostic and change first logits. Null swaps preserve all first logits. These outcomes support downstream answer-information transport, not natural mathematics-specific causal encoding. The selected robustness intervention does not improve mean compound accuracy over the original second operator (11.52% versus 11.82%).

In the old five-initialization study, adequate unregularized affine fitting largely recovers GELU's functional benefit. Expanded old-world tests narrow the difference, but validation-selected ridge fits lose much of the benefit. **Additional expressivity advantage remains unconfirmed in that old unregularized diagnostic; equivalence is secondary and posthoc.** F17 has no converged-affine comparison. Different fields, widths, coverage, and robustness procedures cannot be compared as a width-only intervention.

## Repository and archives

Implementations, fixed configurations, and tests are in `experiments/`, `configs/`, and `tests/`. Readable results, reports, original manifests, and figures are tracked in Git. Checkpoints, arrays, and full data are in checksum-verified release archives.

| Archive | Contents | Release |
|---|---|---|
| `mechanism-confirmations.tar.gz` | Five-source confirmation, preceding activation confirmation, mechanisms, historical test shards | 2026-10-07 |
| `directional-loss-confirmation.tar.gz` | Full/null-only correct/incorrect confirmation in matrix and permutation domains | 2026-10-07 |
| `initial-cross-domain.tar.gz` | Initial joint affine matrix and polynomial experiments | 2026-10-07 |
| `historical-inputs.tar.gz` | Portable inputs referenced by historical dataset aliases | 2026-10-07 |
| `reviewer-controls.tar.gz` | Frozen reviewer controls, held-out inputs, probes, numerical replay | 2026-10-07 |
| `reviewer-revision-diagnostics.tar.gz` | Enlarged old-model census, balanced natural donors, ridge fits, expanded collision pairing, independent audit | 2026-10-09 |
| `fresh-confirmation-f17.tar.gz` | All 17 new source datasets/checkpoints, 68 first predictors, 51 second candidates, predictions, audits, reports | 2026-10-09 |

Per-file SHA-256 inventories and [asset metadata](docs/artifact_inventory.json) specify all contents. Historical workspace aliases are dereferenced as regular files. Early exploratory bulk caches and legacy permutation transformer weights from the 133 GB local workspace are excluded; their code, protocols, and readable results remain available. Third-party full-text papers are excluded.

[Publication verification](docs/reviewer_revision_publication_audit.json) records the clean GitHub checkout, seven-archive restore, 1,686 manifest entries, 181 passing tests, and independent replays. All five new public downloads and the public manuscript match their local SHA-256 checksums. New-world replay reconstructs 2,161,059 entries; expanded old-world replay reconstructs 468,150. These are computation checks, not independent statistical repetitions.

## Install and reproduce

Use Python 3.11 or newer. Current runs use Python 3.13 and PyTorch 2.11; hardware/library changes can affect exact replay.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
git clone https://github.com/XuanyuYang223/neurips.git external/neurips
git -C external/neurips checkout 74f0de2017115f06f11b7285366b66e7e8431d30
python -m pytest -q
```

The pinned upstream supplies PermWorld definitions and transformer code, without modification. Download both releases into one directory:

```bash
gh release download experiment-snapshot-2026-10-07 --repo XuanyuYang223/2026TMLR --dir downloads --pattern '*.tar.gz' --pattern '*.inventory.json'
gh release download reviewer-revision-2026-10-09 --repo XuanyuYang223/2026TMLR --dir downloads --pattern '*.tar.gz' --pattern '*.inventory.json'
python -m experiments.restore_public_artifacts --directory downloads
python -m experiments.public_reproduction --replay-final
python -m experiments.reviewer_fresh_verify --output results/my_fresh_audit
```

Restore checks archives and individual files and refuses to overwrite differing local files. Historical absolute manifest paths are resolved relative to the checkout. `public_reproduction --allow-missing` inventories omitted binaries explicitly. The new-field verifier reconstructs labels, orbit splits, matched pairing, every compound prediction, numerical swaps, single-step noise selection, and the two primary corrected tests. It requires CUDA for encoder replay and writes to a new audit directory, preserving archived audits.

The expanded old-model diagnostic verifier also writes to a separate audit directory:

```bash
python -m experiments.reviewer_revision_verify --output results/my_old_audit
```

For cold training, use separate-output wrappers rather than rerunning against a sealed directory:

```bash
python -m experiments.retrain_final_confirmation --output results/my_old_training --stage prepare
python -m experiments.retrain_final_confirmation --output results/my_old_training --stage run
python -m experiments.retrain_fresh_confirmation --output results/my_f17_training --stage register
python -m experiments.retrain_fresh_confirmation --output results/my_f17_training --stage run
```

The F17 wrapper retains all scientific settings and seeds, changing only the output root. All 17 units and every declared fit complete before composite evaluation. These are reproductions of existing protocols, not new prospective confirmations. Source preparation and fitting require CUDA. Restore historical shards before reproducing the old study.

## Rebuild the paper

```bash
python -m experiments.manuscript_revision
python -m experiments.null_space_review_report
python -m experiments.reviewer_revision_report
python -m experiments.reviewer_fresh_report
cd paper
tectonic --untrusted --keep-logs manuscript.tex
```

The last reporting stage sets the current abstract, introduction, discussion and new-field tables. It uses completed records and performs no new model selection. Compiler intermediates are excluded; Tectonic 0.17.0 built the PDF. Initial protocol/interface records and the presentation-only correction are retained.

## Evidence boundaries

The old five sources share corpus and tests; their bootstraps and expanded TOST are secondary or descriptive. The new 17 units vary both data and initialization in **one** previously untrained field; cross-unit overlap is permitted. The legacy joint-preparation batch-index schedule is shared, while ordinary source training, initialization and predictor-fitting random streams vary. Primary t tests use the dataset/source unit, with Holm correction for two contrasts and marginal 95% intervals.

Answer-matched hidden targets preserve visible gold answers, not necessarily identical teacher probability distributions. Every arm uses the same correct soft supervision. Null components are invisible to one fixed linear readout, not all answer information. True-state repairs and oracle-calibrated random donors are diagnostics. Natural components do not guarantee natural hybrids. Every failed source or gate is retained, and no candidate is added after composite testing. Polynomial primitive failures remain reported. No spontaneous algebra-discovery, universal causal-mechanism, or prospective affine-equivalence claim is made.
