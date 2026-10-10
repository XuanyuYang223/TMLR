PermWorld existing-checkpoint inventory and exploratory geometry diagnostics
2026-10-09

Main report: report.html
Full scientific tables: metric_results.csv, source_metric_summary.csv,
relation_metric_summary.csv, direction_agreement.csv, leave_one_family.csv.
Protocol/identity: metric_protocol.json, protocol_freeze.json, audit.json,
training_inventory.csv, checkpoint_dedup.csv, publication_manifest.json.

This is a separate exploratory PermWorld result, not an addition to the F17
functional-composition confirmation. No new source training, paused run resume,
or encoding batch was performed. The original scientific audit's published=false
records its status before the later user request to upload. It is retained exactly.
Existing F17 files and releases are unchanged.

Main findings and limitations:
- The two original inverse task pairs have positive Delta at all 21 lengths
  and all three original source seeds across CKA, kNN and held-out Procrustes.
  Most CKA Delta comes from decreased wrong-pair similarity, not correct-pair
  convergence. PermWorld original initialization identity remains unverified
  (constructor reconstruction C), not an exact saved step0 reference.
- Expanded relations are dependent exploratory comparisons. Some weak trained
  preferences vary with metric/debiased version. F5 is mixed across metrics.
- F5 rank4 uses45 ordinary sources with all625 inputs seen in source training.
  Analysis-map fit/test holdout does not imply unseen-source-input generalization.
  Strict full-answer strata are undefined. Do not interpret residual geometry
  as proof of mathematical mechanism or extra information beyond outputs.
- New source training required for this scope:0. Future candidate protocols
  remain drafts; the768-model encoding plan is not scheduled.

Binary artifacts are in a NEW release, leaving the F17 latest release unchanged:
https://github.com/XuanyuYang223/2026TMLR/releases/tag/permworld-existing-checkpoints-2026-10-09
All164 new feature archives, evaluation inputs, original CKA/k replay dependencies,
original k activation caches, and F5 ordinary checkpoint/activation dependencies
are supplied. Archive/file SHA-256 manifests authenticate all bytes. Large arrays
are intentionally excluded from Git history. Historical absolute paths in
scientific inventory tables are provenance and are preserved unchanged.

Restore (repository root, Python3.11+, install requirements-dev.txt):
  gh release download permworld-existing-checkpoints-2026-10-09 \
    --repo XuanyuYang223/2026TMLR --dir downloads/permworld \
    --pattern '*.tar' --pattern '*.inventory.json'
  python diagnostics/permworld_scope_inventory_20261009/restore_cached_artifacts.py \
    --archives downloads/permworld --destination .

Clone the pinned mathematical definitions before metric replay:
  git clone https://github.com/XuanyuYang223/neurips.git external/neurips
  git -C external/neurips checkout 74f0de2017115f06f11b7285366b66e7e8431d30
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python \
    diagnostics/permworld_scope_inventory_20261009/replay_cached_metrics.py \
    --output results/my_permworld_cached_replay --stage all

Replay uses the unchanged hash-frozen evaluator and metric implementation with
portable repository-path bindings. It writes a fresh output, compares all metric
rows with the published scientific tables, and performs no source updates.
PermWorld/k metric replay needs no source weights; F5 rechecks restored ordinary
sources. Full forward/checkpoint identity replay additionally needs historical
PermWorld source checkpoints from the listed upstream release/local inventory;
this release does not duplicate the large legacy PermWorld weight collection.

The old full-checkpoint verifier is an original-workspace audit, including
original untracked protection records; it is not a portable fresh-checkout gate.
Use restore_cached_artifacts.py and replay_cached_metrics.py for public replay.

Publication validation: restore verified551 files in a fresh staged-tree checkout.
All62,007 PermWorld/F5/k metric rows replayed with maximum numeric difference0;
see publication_replay_audit.json. This validation performed no source training.
