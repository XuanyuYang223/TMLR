# Algebraic structure in neural representations: PermWorld experiments

This workspace contains an audit of the actual PermWorld/Property32 study,
finite-field transfer experiments, and PermWorld studies of transformation
geometry and withheld-relation prediction. The current follow-up report is
`results/until_10_followup/report.html`. The upstream repository
is cloned into `external/neurips` and is left unchanged. No upstream release
datasets or checkpoints are needed for this initial stage.

## Reproduce

Use the existing `.venv`, or create an environment with the dependencies in
`requirements.txt`. Run commands from this directory:

```bash
git clone https://github.com/XuanyuYang223/neurips.git external/neurips
git -C external/neurips checkout 74f0de2017115f06f11b7285366b66e7e8431d30
python -m experiments.permutation_audit
python -m unittest discover -s tests -v
python -m experiments.run --config configs/pilot.json --output results/pilot
```

The source-learning-only calibration can be reproduced with:

```bash
python -m experiments.run --config configs/calibration.json \
  --output results/calibration_v2 --pretrain-only
```

Outputs preserve exact configuration/code fingerprints. Completed runs resume
only when their fingerprint matches. Use a fresh output directory for changed
configurations or training code. GPU computations use deterministic algorithms;
CPU execution is supported. Reproduction across PyTorch versions or GPU models
is not guaranteed to be bitwise identical.

## What the real PermWorld audit establishes

The original `r=0,1,2,4` conditions count direct correspondences **between two
disjoint four-task sets**. They do not count edges inside one four-task set.
The eight registered pair relations have sixteen unique endpoints, so this
registered graph is a matching. Chain/star/cycle comparisons cannot be obtained
just by rearranging these eight edges.

`results/permutation_audit/` contains:

- `summary.json`: upstream commit, configuration hash, six proved joint
  constraints and the eight-transformation composition table;
- `affine_candidates.csv`: exhaustive candidates for
  `target(T(pi)) = sign * source(pi) + b*n + c`;
- `existing_bundles.csv`: properties of the twenty existing source bundles;
- `existing_cross_bundle_relations.csv`: inverse/complement relation counts and
  existing final-layer CKA, averaged within each three-seed layout;
- `candidate_bundles.csv`: all 35,960 four-property subsets;
- `matched_candidates.csv`: candidate three-task/four-task dependency comparisons
  selected using algebra and fresh synthetic labels, without consulting CKA.

Four primitive joint identities follow from counting position types, cycle
types, and interior comparison patterns. Two additional minimal supports follow
by subtracting primitive identities. They are validated against all 5,912
permutations of lengths 2–7. An exact mod-prime rank-30 witness, together with
four independent universal identities, certifies that these four span all
same-input rational linear identities among the 32 values, length `n`, and
constant `1` for `n>=2`. Searching supports of size at most four gives exactly
six minimal circuits. The certificate does not cover nonlinear or transformed
joint dependencies. None of the twenty existing bundles contains these circuits.
The certificate and witness rows are saved in `linear_rank_certificate.npz`.

The affine search uses identity, complement, reversal, inversion, and their
eight distinct compositions. Exhaustive candidates are not mathematical proofs
for arbitrary permutation lengths. New candidate selection uses 200 independently
generated permutations at each length 10–30, seed 20261004; published test shards
are never read. Matching average entropy/correlation is approximate and does
not establish equal task difficulty or identical full output distributions.

## Finite-field intervention

Inputs range over all of F5^4, represented by categorical one-hot coordinates.
Each random invertible input basis defines three uniform, independent latent
variables `u,v,w` and one nuisance dimension. Every source model has four tasks:

- family A: `u, v, w, a*u+b*v`;
- family B: `u, v, w, a*u+b*v+c*w`, with nonzero `a,b,c`.

There are four coefficient variants per family. All variants have rank three,
uniform marginal outputs, pairwise-independent labels and joint entropy
`3*log(5)`. A has a minimum three-task circuit; B has a four-task circuit.
The eight source sets are distinct even after quotienting out scalar output
relabelings. The full joint distributions deliberately differ in their
higher-order dependence.

The pilot crosses eight source combinations with input-basis seeds 17, 42, 101
and one initialization seed, training 24 source MLPs. All source combinations
use identical input exposure, architecture, updates, head permutations within
each basis, and paired initialization. Separate single-source models estimate
learning difficulty through their learning curves. The 98% minimum per-source
training-accuracy gate is checked after fixed training; no run receives extra
updates. This gate measures fit to exposed source labels, not generalization
to new inputs.

Ten target directions are chosen before training using algebra only, excluding
every scalar-equivalent source task across all eight combinations. The target
pool and balanced, nested 10/25/50-label support sets are shared across source
conditions. Each target has 125 disjoint, balanced test labels. Source training
has exposed **all inputs**, including target-test inputs, but never their target
labels. This is **transductive new-task transfer**, not new-entity or OOD
generalization.

Both frozen linear probes and full encoder fine-tuning receive the same label
budgets. Matched random initialization is evaluated using exactly the same
support/test sets, heads, update counts and adaptation learning rates. Each
target adaptation owns an independent encoder copy; batched computation does
not share parameters between targets.

## Prediction and interpretation

The primary endpoint, fixed in `configs/pilot.json`, is full fine-tuning gain
over random initialization at 25 target labels. A ridge baseline uses source
learning-curve AUC, final source accuracy, mean single-source AUC, and target/input
basis fixed effects. Structural features add minimum circuit size and the
minimum number of source tasks needed to span a target. Number of tasks, source
rank, exposure and pairwise label MI are constant by design.

Additional design-only predictions use target/input-basis effects and structure
without observing the held-out source model's learning measurements. The
learning-controlled comparison asks for structure's incremental information
conditional on measured source fitting. Both are retained, with separate
analysis-code hashes. Figures can be regenerated without retraining with
`python -m experiments.analysis --output results/pilot`.

Leave-one-combination-out prediction removes **all** basis, initialization and
target rows for each held-out source combination. Features are standardized
inside training folds and the ridge penalty is fixed before transfer evaluation.
A separate leave-one-input-basis-out check measures encoding sensitivity.
These holdouts test coefficient variants of known templates, not a previously
unseen abstract relation family. Target identities are present in both training
and held-out combination rows; this is not prediction of new target identities.

CKA compares different source encoders on identical ordered input rows and also
records similarity to random initialization. It is an auxiliary measurement,
not a behavioral outcome. Plot error bars are sample SD across three input
bases. Targets, budgets and variants are correlated observations, not independent
replications. With three bases, one initialization seed, one MLP architecture
and one finite-field domain, results support an initial feasibility assessment
only. No confirmatory significance claim is made.

The same protocol was then replicated using initialization seeds 42 and 101,
training 48 additional source models. This replication keeps all hyperparameters,
target sets, input bases and support/test indices fixed; it was prompted by the
weak initial result. The aggregate has 72 source models and 4,320 target
endpoints, with three initialization seeds crossed with the original three
input bases. It is not a replication on new coefficient combinations or new
input bases. Reproduce and aggregate with:

```bash
python -m experiments.run --config configs/replication.json --output results/replication
python -m experiments.diagnostics --output results/pilot
python -m experiments.diagnostics --output results/replication
python -m experiments.aggregate --inputs results/pilot results/replication --output results/combined
```

The post-hoc symbolic oracle decodes pretrained `u/v/w` heads and applies the
externally supplied target formula, on the same target-test indices. Its perfect
accuracy establishes retained information on exposed inputs; it does not show
spontaneously learned composition. This is explicitly separate from ordinary
probe/fine-tuning endpoints. No extra target learner was fitted for this oracle.

Read `results/combined/report.html` for all-seed measured outcomes, or
`results/pilot/report.html` for the initial stage, with raw endpoints in
`metrics.csv`, prediction folds in `prediction_scores.json`, exact algebra in
`algebra_audit.json`, and provenance in `metadata.json`. Subsequent sections
describe new PermWorld combinations and an independent finite-field input-basis
replication. Existing CKA outcomes alone cannot establish that high-order
dependencies explain the earlier non-monotonicity.

## Related primary studies

- [Convergent World Representations and Divergent Tasks](https://proceedings.mlr.press/v306/park26d.html): multi-task representation convergence and harmful new-entity adaptation.
- [Representational Homomorphism Error Predicts Compositional Generalization](https://proceedings.mlr.press/v282/an26a.html): a learned-representation diagnostic of compositional generalization.
- [Intrinsic Task Symmetry Drives Generalization in Algorithmic Tasks](https://proceedings.mlr.press/v306/hwang26d.html): task symmetry and generalization in algorithmic domains.

This pilot examines pretraining task-set structure as a prospective predictor;
it does not establish priority or novelty over this literature.

## Continuation: verify the original CKA effect and test readout capacity

`results/cka_review/report.html` recomputes the original contrasts directly from
the upstream raw CKA CSVs. Correct inversion/complementation increases final-layer
CKA versus identity in 24/24 relation-by-seed cells and versus the wrong transform
in 24/24 cells. Averaging seeds within each mathematical relation preserves both
effects in 8/8 relations. This uses the same trained weights while changing
input pairing. In contrast, the four-task training-bundle comparison has positive
`r4-r0` in 7/12 cells and monotonic curves in 1/12 cells. Universal positivity
also appears at blocks 3 and 4; earlier layers do not all show it. The finite-field
transfer results do not contradict or replicate this Transformer alignment test.

The exploratory readout continuation freezes the existing 72 finite-field
encoders and retains all original target/support/test splits. It compares linear
and 128-hidden-unit MLP heads with matched random features, at identical 200-update
training budgets and both learning rates 0.01 and 0.03. The primary comparison
is fixed at 25 labels and learning rate 0.01. Both rates are reported; no target
test outcome is used for selection or tuning. Individual target heads own
independent parameters. New encoders are not trained.

Three additional diagnostics use the four decoded source outputs:

- **Learned field fit:** use target support labels to solve unknown affine
  coefficients over F5. This does not receive the true target formula, but it
  supplies a privileged finite-field affine hypothesis class. It cannot establish
  spontaneous neural composition.
- **Source-state lookup:** reuse the target label for exactly matching decoded
  source tuples in the support set, otherwise predict the support majority class.
  This quantifies repeated latent states across support/test in this transductive
  world.
- **Shuffled field fit:** permute complete source-code rows, preserving their
  full joint distribution while destroying their input/target correspondence.
  Inconsistent systems fall back to the same majority-class rule. Inconsistency
  rates and source-space identification ranks are retained.

The continuation was designed after observing negative initial transfer; it is
an exploratory readout diagnosis, not an independently confirmed new relation
effect. The finite-field hypothesis class and scalar residue encoding are
additional structural assumptions. Code fitting and lookup receive support
labels only; target-test labels are used exclusively for evaluation.

```bash
python -m experiments.cka_review
python -m unittest discover -s tests -v
python -m experiments.readout_followup --config configs/readout_followup.json \
  --output results/readout_followup
python -m experiments.verify_followup
```

Read `results/readout_followup/report.html` for measured continuation outcomes.
The frozen-head endpoint table, code-diagnostic endpoint table, metadata and
figures are retained separately from the initial experiment artifacts.

All 72 source models completed the continuation, producing 8,640 neural and
6,480 code-diagnostic endpoints. At the primary 25-label, 0.01-learning-rate
setting, frozen MLP accuracy is 19.54% for family A and 21.97% for family B,
versus 25.05% for matched random features. These are negative transfer gains
of 5.51 and 3.08 percentage points. Increasing head capacity in this comparison
does not establish a positive transfer benefit.

The explicit F5 affine learner reaches 100% with 10, 25 and 50 support labels
in all evaluated model/target cells; shuffled source tuples reach 20% after
inconsistent-fit fallback. Exact source-state lookup reaches 32.40% at 25
labels, with 15.71% of test tuples present in support. Thus the field learner
also succeeds on source tuples absent from the target support set, while the
source-pretraining inputs remain fully exposed. This supplies evidence of
recoverable target information under an explicit mathematical prior. It does
not establish that ordinary neural readout learns that structure, that CKA
causes transfer, or that either relation family gives greater symbolic accuracy.

`verification.json` checks endpoint uniqueness, matched controls, preserved
training fingerprints and support/test splits, agreement with original CKA
values, and report resources. All 2,160 linear-head endpoints at learning rate
0.03 exactly reproduce the corresponding initial frozen-probe endpoints.

## New PermWorld task combinations

`configs/permworld_combinations.json` fixes eight four-task groups with three
paired initialization seeds. Position and cycle counting each compare a
three-task identity, a four-task identity, and a group with no proved identity
of the audited rational linear form. Interior pattern counts compare a
four-task identity with a control. Controls minimize distance in label entropy
and within-length correlation while preserving the expanded affine pair count.
Selection uses the prior independent design sample and never neural outcomes.
The interior control is a weaker statistical match. Actual source-label
statistics are rechecked separately on the new training inputs.

This pilot reuses upstream exact property functions, Passage vocabulary and
causal Transformer code. It uses a smaller 96-wide, two-layer model, no dropout,
and answer-token-only loss for 1,200 updates. Every group uses the same input
sampling schedule per seed, with exactly 38,400 exposures per source task.
This is not a reproduction of the original four-layer, 256-wide, 20,000-update
protocol. Source train/validation accuracy, learning curves and a train-label
length-majority validation baseline are reported; source fitting is measured
rather than strictly matched. There are no single-task difficulty controls.

The five fresh input splits contain 6,930 globally distinct permutations of
lengths 10–30: 4,200 source training, 630 source validation, 420 representation,
1,050 target-support pool, and 630 target test inputs. This differs from the
earlier finite-field transductive experiment: target-test inputs are absent from
source training. It still does not test new permutation lengths. Four common
target tasks are excluded from every source group: descents, recoils, lis_length
and longest_increasing_run. Uniformly sampled 64/256-input target supports are
nested and shared, without enforcing balanced label classes. Frozen linear
probes and full-model fine-tuning both have paired random-model controls.

CKA is measured at the task-free ONE_END landmark on the same ordered
representation inputs, reporting both raw and within-length-mean-subtracted
features. Cross-seed consistency is saved separately. This does not repeat the
original correct-versus-wrong transformation experiment.

```bash
python -m unittest discover -s tests -v
python -m experiments.permworld_combinations
python -m experiments.permworld_combinations_report
```

Read `results/permworld_combinations/report.html` for measured outcomes,
`metadata.json` for exact groups and protocol, and `verification.json` for
input disjointness, exposure, support and checkpoint checks. All combinations,
targets, budgets and both adaptation modes are retained. Three initialization
seeds share one data world; target/seed rows are correlated observations.
Weak source learning prevents a causal conclusion about which identity size
improves representation sharing or transfer.

The completed pilot has 24 source models, 384 pretrained transfer endpoints and
48 random-control endpoints. At the preselected primary 256-label fine-tuning
endpoint, all group means are below the random baseline: 31.27%–32.91% versus
34.44%. Relative to their corresponding no-proved-identity controls, identity
groups differ by -0.25 to +0.91 percentage points. Secondary frozen probes at
64 labels give positive gains for some groups: +4.29 pp for interior_four and
+3.40 pp for position_four, compared with +0.79 and +2.58 pp for their controls.
All other groups and both budgets are reported; these examples are exploratory.

Source validation exceeds the length-majority baseline by -1.27 to +6.75 pp
across groups, indicating limited source-task learning. Some large raw CKA
values drop sharply after subtracting within-length means; for example,
position_three versus position_none averages 0.569 raw and 0.0038 after this
centering. The necessary next step is stronger source learning with measured
difficulty controls before assessing whether the identity structure explains
representation or transfer differences. These observations do not negate the
original correct-input-transformation CKA result.

## Authorized six-hour follow-up

The active session runs from 2026-10-05 11:51:45 to 17:51:45 UTC. Its durable
protocol is `configs/six_hour_session.json`; snapshots and revisions in
`results/six_hour_session` retain the initial plan and changes made before any
target validation or testing. The original pilot artifacts and upstream
repository remain unchanged.

Nine architecture-calibration models completed. Source validation alone chose
the 256-wide, four-layer, eight-head Transformer for the 24 planned multi-task
models and 54 optional single-task controls. Each source task receives 640,000
input exposures over 20,000 updates, with paired initialization and sampling.
Source validation, source-training accuracy and a separate fresh-input source
audit measure what each model actually learned. Equivalent SDPA attention uses
the same state keys and causal/padding semantics, checked against the upstream
implementation. Long-run conditions all use BF16 training and FP32 evaluation;
their numerical trajectories need not reproduce the small FP32 pilot bitwise.

Seven disjoint splits contain 52,290 fresh permutations of lengths 10–30:
42,000 source training, 2,100 source validation, 840 representation inputs,
2,100 target-support pool, 1,050 target validation, 2,100 target test and 2,100
source audit. The 64/256-label supports remain nested and shared. Every source
condition and random initialization receives an equally sized target-validation
search at initialization seed 17, averaged across all four targets and both
budgets. Task-free linear, query linear, task-free MLP and whole-model adaptation
select their own settings. Frozen settings apply to all three initialization
seeds; shared random-selected fine-tuning settings are also evaluated. Feature
standardization uses only target support inputs. The initial plan reserved the
final 90 minutes for validation, evaluation and analysis. Measured single-task
update timing prompted a handoff starting at 15:21:45 UTC, finishing the active
optional model and preserving approximately the final 150 minutes for the
unchanged target grid, testing and analysis. `target_handoff_state.json` records
this scheduling; `time_allocation_revision.json` is written at the actual handoff,
before target validation or testing. Unfinished optional source controls are
saved and distinguished from completed models.

```bash
python -m experiments.six_hour_marathon --wait-for-calibration
OPENBLAS_NUM_THREADS=4 python -m experiments.six_hour_report
python -m experiments.verify_six_hour
```

The report at `results/six_hour_session/report.html` can be regenerated during
training. It excludes incomplete models from endpoint summaries and only
computes transfer gains when the same seed, target, budget and readout has a
completed random baseline. Raw CKA, within-length-centered CKA and paired
cross-seed initial/trained CKA are retained at 5k, 10k and 20k source updates.
One data world is shared across initialization seeds. Source fitting and label
statistics remain imperfectly matched, so group differences alone cannot
establish a causal benefit of a particular identity size.

Two supplementary finite-field analyses narrow the proposed explanation.
`experiments.field_mechanism` audits the 72 prior rank-three source encoders.
It separates within-state nuisance from task-dependent Fourier energy and
tests a generic categorical subset decoder on target-heldout latent states.
This decoder sees decoded source answers and support labels, without algebraic
coefficients. The mathematical minimum source subset is an external predictor.
The diagnostic is transductive and specific to its lookup rule; it does not
replace the negative ordinary neural-readout results.

The prospectively registered `configs/field_symmetry.json` defines 45 new
rank-four models on F5^4. Five four-task sets have exactly identical uniform
marginals, zero pairwise MI and the full uniform joint output distribution.
Under the same latent-coordinate swap, their ideal source-answer categorical
kernels have exact CKA 1, 0.75 or 0.5. The hidden-layer ordering is a hypothesis,
not a theorem about trained networks. `configs/field_symmetry_transfer.json`
adds eight common heldout tasks, paired random encoders, fixed linear/MLP
readouts and separately labeled categorical subset/full-tuple lookups. Target
composition sizes vary between two, three and four. Both protocols were
registered before this source cohort was trained. All 625 inputs are exposed
during source pretraining, so this tests transductive new-task behavior.

```bash
python -m experiments.field_mechanism_report
python -m experiments.field_symmetry_report --verify
python -m experiments.field_symmetry_transfer_report
```

All 24 long-run PermWorld multitask models completed. Source audit accuracy
varies substantially: position_three averages 99.61%, interior_four 47.22%,
and position_none 36.38%. All groups exceed their training-label length-majority
audit baselines, but their fitting is far from matched. Completed single-task
controls and their paired source accuracy differences are recorded as they
finish; they share one task's exposure and updates, while the optimized loss
differs from the four-task average.

`results/six_hour_session/landmarks/report.html` records an exploratory
extraction-position analysis added after preliminary ONE_END measurements.
After subtracting within-length means, cross-seed CKA increases in 0/24 pairs
at ONE_END, 3/24 at a common heldout query, 5/24 for prefix pooling, and 19/24
for the concatenated trained-task query features. Those concatenations are
compared only for the same task set. This does not negate the original
correct-input-transformation effect. Three seed pairs per group are correlated.
Adding an exact source-answer transformation kernel improves whole-group-heldout
geometric prediction in all eight extraction/centering settings, but several
raw-feature R² scores remain negative. The analysis retains every setting and
does not designate the strongest position as a prospective primary endpoint.

The 45 rank-four finite-field models and their 3,168 behavior endpoints completed.
Every source task reaches 100% accuracy on the exposed 625 inputs. The registered
full-hidden CKA ordering holds in only 3/9 paired settings. A second cohort
adds 27 models on independently selected bases, with every source function
having all four nonzero physical coefficients; its hidden ordering holds in
4/9 settings. It has 2,016 behavior endpoints. The individual functions are
equivalent up to per-coordinate category relabeling, although multitask
optimization trajectories are not identical.

`results/field_forecast/report.html` checks 6,480 forecasts saved at
2026-10-05 14:39:02 UTC, before the independent target-readout process. The three
replication source formulas also occur in the fit cohort; only the input bases
and model runs are new. Composition order improves categorical-subset accuracy
prediction, with independent-basis R² 0.786/0.914 at 25/50 labels versus baseline
0.438/0.287. It worsens the frozen neural-gain predictions for both budgets and
both linear/MLP heads. Independent-basis mean ordinary neural transfer gains
range from -1.47 to -3.21 percentage points. Categorical dependency order is
not specific evidence of algebra learning, and lookup behavior cannot substitute
for a neural-transfer finding.

Two later diagnostics are explicitly post hoc:

- `results/field_readout_geometry/report.html` separates the source-class-contrast
  weight row space from its null space. It also rescales hidden coordinates
  while inversely rescaling source weights, preserving every source decision.
  Eight fixed equivalent scalings yield an average within-model CKA range of
  0.174 across the 72 new finite-field encoders. Support-based per-coordinate
  standardization cancels positive scales when the variance floor is inactive.
  Raw hidden CKA changes therefore need not imply readout behavior changes.
  Perfect source probabilities reproduce the ideal answer kernel as a direct
  consequence of source fitting, rather than an additional mechanism discovery.
- `results/six_hour_session/source_consistency/report.html` evaluates source
  predictions under all 15 available same-input identity-by-seed tests and
  39 proved-action-by-seed tests. Identity satisfaction averages 70.73%, while
  correctness of every answer in the identity averages 38.10%. Training-label
  length-majority predictions are reported alongside consistency to expose
  trivial agreement. FP32 CPU inference leaves the GPU queue uninterrupted.

`experiments.factor_kernel_readout` additionally fixes cumulative factor
interaction kernels of degrees one through four and ridge penalty 1. Learned
source-head probabilities, initial source-head probabilities, and direct physical
input categorical factors use the same target supports/tests. All degrees are
retained without choosing a test winner. This supplies a generic factor
interaction readout class; it is separate from the registered ordinary hidden
linear/MLP readouts and does not establish spontaneous algebraic composition.

```bash
OPENBLAS_NUM_THREADS=4 python -m experiments.longrun_geometry_report
OPENBLAS_NUM_THREADS=4 python -m experiments.verify_native_geometry
OPENBLAS_NUM_THREADS=4 python -m experiments.field_predictions
OPENBLAS_NUM_THREADS=4 python -m experiments.matched_field_report
OPENBLAS_NUM_THREADS=4 python -m experiments.field_forecast_report
OPENBLAS_NUM_THREADS=4 python -m experiments.field_readout_geometry
OPENBLAS_NUM_THREADS=4 python -m experiments.native_output_consistency
OPENBLAS_NUM_THREADS=4 python -m experiments.factor_kernel_readout
```

`field_forecast` is run between the independent source phase and its target
readouts. It deliberately refuses first-time forecasting if independent behavior
metadata already exists. Existing saved forecasts can be scored without fitting
again. Future changed source/target protocols require a fresh output directory.

An additional target-only followup retains the same 27 independent-basis source
encoders and nine paired random encoders. `controlled_target_followup` selects
eight new projective target directions using mathematical criteria alone:
physical support is at least three in each basis, every direction is absent
from the source sets and the original target list, and the minimum source
subset size varies across source sets. This was motivated by the observed
large losses on easy physical-coordinate targets in the original pool, so it
is a conditional, exploratory followup; the original negative results remain.
The old forecast weights are reused without fitting again, with unseen target
fixed effects set to their old-target average. All 6,480 predictions were
saved at 15:29:30 UTC before the new CPU readout behavior metadata appeared.
Its 2,016 readout endpoints give mean ordinary neural gains of 0.04–1.10 pp,
but ordinary accuracy remains below 20% chance. Composition order continues
to predict generic categorical-subset behavior better than neural gains.

`native_layer_diagnostic` is an explicitly post hoc CPU check of the actual
source answer-loss graph and every prefix layer, from embedding through all
four blocks and final normalization. It distinguishes activation derivatives
from gradients of shared parameters: final-layer prefix outputs have zero
derivative for loss at the final query, while preceding-layer prefix outputs
participate through the final attention block. This does not mean the prefix
parameters are untrained, and it is not a causal intervention on geometry.
All layer/site/centering results are retained without choosing a test winner.
Once target validation and all 27 target evaluations finish, the remaining GPU
time resumes optional single-task controls in their original task/seed order,
using unchanged source parameters and stopping before the analysis reserve.

```bash
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python -m experiments.controlled_target_followup
OPENBLAS_NUM_THREADS=4 python -m experiments.controlled_target_report
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python -m experiments.native_layer_diagnostic
```

`native_transfer_prediction_protocol.json` fixes a further exploratory native
analysis before any long-run target-test record exists. The rules were saved
at 16:08:06 UTC and the feature code at 16:11:02 UTC. It leaves out an entire
source group, including all its seeds and target tasks, and predicts paired
random-adjusted gains. A source-fitting and label-similarity baseline is compared
with mathematical descriptors, measured geometry, and both. All four variants,
five readouts and two label budgets use fixed ridge penalty 1, with training-fold
feature standardization. It waits for all 27 target evaluations before fitting.
Previous pilot outcomes and current source geometry were already known, so
this remains exploratory. Target/source conditional MI uses true task labels
on source-training inputs as an analytical similarity reference; these target
labels were not supplied to source optimization. Eight groups from a single
data world cannot establish a causal or cross-domain result.

```bash
OPENBLAS_NUM_THREADS=4 python -m experiments.native_transfer_prediction --wait
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python -m experiments.verify_supplementary_artifacts
```

`results/field_kernel_identity/report.html` gives a post hoc derivation of the
ideal categorical-answer CKA identity. On a complete uniform prime-field grid,
the centered kernel decomposes into orthogonal nontrivial characters along
task projective directions. CKA is the cosine between direction multiplicities;
with four distinct directions, it is the fraction of shared directions. The
P/M1/M4/M2/M3 examples therefore have exact CKA 1/.75/.75/.5/.5 under the
coordinate swap. All source vectors nevertheless remain linearly closed because
each source set spans all four latent coordinates. Five cohort cases and 90
additional examples at p=3/5/7 verify the identity by direct enumeration. It
does not predict arbitrary hidden vectors and is not claimed as a novel theorem.

The frozen native predictor has one documented implementation deviation: a
separate no-known-circuit indicator was constant zero because the source catalog
uses sentinel 0 rather than None. The size coding 3/4/5 was correct. This was
found from source features after partial native testing but before prediction
fitting. `native_prediction_sensitivity` retains the original frozen code and
reports every intended-indicator correction alongside the original, with no
score-based choice of version. See `transfer_prediction/indicator_correction.json`.

`native_length_baseline` adds a practical, explicitly post hoc control after
partial testing. It only sees the same target support labels and input lengths,
never source-training target labels. Global majority, exact-length majority
with global fallback, and Gaussian-smoothed length category counts are all
reported. One common bandwidth is selected on mean target-validation accuracy
over four tasks and both budgets. The smoothed baseline yields 31.02%/33.74%
test accuracy at 64/256 labels. It is deterministic across initialization seeds;
these are not three independent baseline repetitions. The native main report
retains both aggregate and per-target, exactly paired transfer results.

The first native target evaluation completed all 27 conditions and 1,080
endpoints. At 256 labels, condition-selected fine-tuning gives interior_none
34.79% mean accuracy (+2.52 pp against paired random, positive in all three
seed means), while most other source groups lose accuracy. Whole-group-heldout
predictive improvements are readout/budget dependent: mathematical features
help shared-policy fine-tuning at 256 labels (original R² -0.289 to 0.274;
indicator-corrected sensitivity 0.364) but fail at 64 labels and do not provide
a consistent benefit for condition-selected fine-tuning or frozen heads.
Every score remains reported; the successful cell was not the primary endpoint.

`native_target_repeat` then freezes every original predictor and both corrected
mathematical variants on all first-pool gains, before generating 5,250 fresh
target inputs globally disjoint from the original 52,290 inputs. It reuses
the same source models, tasks and target policies, so this tests new target
support/test sampling, not unseen task sets or a new source world. The GPU
queue first finishes all 18 seed-17 single-task controls, temporarily pauses
the owned optional-control process, runs every repeat condition, then resumes
the original source queue. Paused source-job elapsed wall time includes that
window; update counts and sampling do not change. All 5,760 forecasts and their
60 fitted weight sets are retained, without selecting the best version.

Both native target pools are now complete: 27 conditions and 1,080 endpoints
per pool. With 256 labels, interior_none condition-selected fine-tuning gains
+2.52/+2.47 pp over paired random initialization, with positive seed means
3/3 and 2/3. Its second-pool accuracy is 33.85%, below the frozen length-only
baseline of 35.56%. The LIS target has gains +3.60/+3.21 pp over random and
+5.06/+1.57 pp over length; these per-target findings are post hoc and do not
imply all targets benefit. Other groups and all readouts are retained in
`results/native_target_repeat/comparison.html` and the two comparison CSVs.
On the fresh pool, shared-policy 256-label gain prediction improves only
slightly from baseline R² 0.267 to 0.305 (original mathematical features) or
0.307 (indicator correction). The predictors fitted all same-named groups in
the first pool, so this is a target-sampling check, not another whole-group
heldout or independent-source test.

```bash
OPENBLAS_NUM_THREADS=4 python -m experiments.native_target_repeat --phase prepare
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 python -m experiments.native_target_repeat --phase evaluate
```

`native_source_subspace` is a post hoc native diagnostic. For each source model,
the centered output-weight row space for numeric classes 0–30 has rank 30.
Projection preserves those numeric logit differences; its complement can still
affect other vocabulary decisions and future tasks. Within-length source-query
CKA increases in 23/24 pairs in the numeric contrast space, 16/24 in its
orthogonal complement, and 23/24 for centered numeric logits. Numeric contrast
space accounts for 87.5% of source-query variance on average. ONE_END CKA does
not increase in either subspace. This identifies measurement dependence,
not a causal explanation or proof that query convergence is solely output coding.

```bash
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python -m experiments.native_source_subspace
OPENBLAS_NUM_THREADS=4 python -m experiments.verify_native_repeat
```

The fixed-candidate follow-up is configured in `configs/native_confirmation.json`.
It freezes `interior_none → lis_length` before a fresh source-data world and
source-model seeds 1009/2027/3037/4051/5059. The candidate and all four of its
single-task controls are retrained at every seed; the other seven source groups
use the first three new seeds, yielding 26 multi-task and 20 single-task models.
The 54,390 new permutations exclude all 64,470 previously used native inputs.
All source models retain the 20,000-step schedule and 640,000 exposures per task.

The primary LIS/256-label comparison uses the same 0.0003/100-update adaptation
policy for candidate, singles and random models. The old shared 0.0001/100 policy
and old condition-specific policies remain secondary checks. All four targets
and both label budgets are evaluated, with 4,200 common target-test inputs.
Source-seed bootstrap intervals use five paired gains rather than treating the
1,224 planned endpoints as independent repetitions. They are conditional on
one new data world and do not quantify population-level world variation.

Four prediction variants compare source-learning measures alone, adding
mathematical relations, adding label statistics, and adding both. Old target-pool
gains are averaged within group/seed/target before fitting. All 16 ridge weight
sets freeze before new data generation; numerical forecasts freeze after source
learning but before new adaptation. The balanced predictive test uses only
three specified seeds per task group. It tests new source inputs and models for
the same eight task sets, not unseen task combinations. See
`results/native_confirmation/protocol.json`, `weights.json`, and `forecasts.json`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 python -m experiments.native_confirmation --phase run
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python -m experiments.native_confirmation_analysis --phase analyze
OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python -m experiments.native_confirmation_analysis --phase verify
```

`native_confirmation_identity_control` is a separate supplementary check
registered after the first five source models completed, before any new target
adaptation or outcome. Because the new world retains the same eight task sets,
it forecasts gains from old group/target means and from source learning plus
task-group indicators. The original four variants remain the primary analysis.
Its forecasts freeze when all source records are available, before adaptation,
and its results are reported separately in `identity_control.html`. This control
distinguishes incremental prediction over learning measures from recognition
of a previously observed task set; neither tests unseen task-set generalization.

`results/native_confirmation/progress.html` provides a read-only live view of
source completion and worker stage. It does not use partial target outcomes to
modify the plan.

This follow-up completed on 2026-10-06 at 00:46 UTC (2026-10-05 in Los Angeles).
The frozen primary LIS/256-label comparison gives `interior_none` a mean
2.96 percentage-point gain over the same-policy random model, with positive
gains at all five new source seeds and a conditional source-seed bootstrap
interval of [2.11, 3.80] percentage points. The candidate also beats every one
of the four matched single-task models at every seed: mean differences are
3.83 points over peaks, 5.27 over valleys, 3.71 over left-to-right maxima, and
3.86 over right-to-left maxima. Its gain over the support-only length baseline
is 4.03 points, also positive at all five seeds. This supports the selected
combination's local LIS transfer benefit; it does not isolate a causal effect
of algebraic relations or establish that four-task combinations are necessary.

In the primary prediction check (old shared policy, 256 labels), adding
relations changes new-world R² from 0.394 to 0.445; adding relations after
label-statistic controls changes it from 0.416 to 0.451. Both group-level
descriptive error-improvement intervals include zero, with lower error in only
four of eight groups. The supplementary old group/target mean reference has
R² 0.596, and source learning plus task-group indicators has R² 0.448. Under
the condition-specific policy at 256 labels, relations instead reduce R²
from 0.334 to 0.295. These results do not establish reliable incremental
prediction from algebraic structure or generalization to unseen task sets.

All 46 source models and 1,224 target endpoints completed. Verification
recomputed every target accuracy from the saved 4,200-input prediction arrays,
all 1,740,480 exact labels, all primary and supplementary forecasts, and all
24 length-baseline prediction arrays. The 67 regression tests and one separate
identity-control test passed. See `results/native_confirmation/report.html`,
`identity_control.html`, `verification.json`, `supplementary_verification.json`,
and `source_sampling_verification.json` for results and provenance.

The next prospective LIS study uses `configs/native_ablation.json` and
`experiments/native_ablation.py`. It freezes three paired source seeds
(1009/2027/3037), the same 42,000-input source pool, and new globally disjoint
support/test pools (2,100/4,200 inputs). Forty-two new source models cover four
leave-one-out triples, supervision-matched controls, and four previously
untrained four-task sets; fifteen source checkpoints and three random models
are reused as references. All source models use 20,000 optimizer updates.

The primary supervision budget is 1.92 million labels: four tasks each use
24 examples/update, each leave-one-out triple uses 32, and each single task
uses 96. A second comparison gives four tasks eight examples/update versus
existing single-task models using 32, matching 640,000 total labels. Original
four-task models using 32 remain the per-task-exposure reference. The unchanged
32-index source sampler supplies common lengths and core inputs; smaller
batches use its prefix, larger batches append independently sampled inputs.
Total labels and updates are matched, while input count per update and
per-task exposure differ; actual input coverage is recorded.

Four unseen task sets test the old frozen prediction weights without refitting:
all four directional record counts; peaks/valleys with directional minima;
peaks with three record counts; valleys with three record counts. Generic
operator rules reproduce all eight existing task-set certificates and extend
to these new sets, with exact checks against permutation enumeration. Early
forecasts use pooled old source-learning grades and freeze before new training
and target generation. Later numerical forecasts use actual new source grades
and freeze before any target adaptation. Primary ablations use the old
candidate-shared 0.0003/100 policy; primary unseen-set prediction uses the
old shared 0.0001/100 policy. Both label budgets and both policies are retained.

The four new tests plus the previous regression suite passed (72 tests).
Run `python -m experiments.native_ablation --phase run`; protocol, forecasts,
checkpoints and the completed report are under `results/native_ablation`.
This is a follow-up selected from earlier outcomes, with three seeds in a
reused source-data world and four deliberately chosen new task sets.

This supplementary LIS ablation was paused on 2026-10-06 at 05:43 UTC after
the user redirected the main research question to algebraic relations and
representation geometry. Thirteen new source models completed; the fourteenth
retains its 3,000-update checkpoint. No new target adaptation or evaluation
ran. See `results/native_ablation/pause_record.json`. The completed
`native_confirmation` results remain the local-transfer evidence; the ablation
is unfinished and its early forecasts must not be reported as outcomes.

The algebra/geometry follow-up is now under `results/algebra_structure`.
It reuses the 24 completed confirmation checkpoints (eight four-task groups,
three source seeds), with three matched random initializations. The 588 fresh
eight-action permutation orbits contain 4,704 states; complete orbits are
split 336/84/168 for fit/validation/test. Exact composition of complement,
reverse and inverse is checked by permutation enumeration. Generators are
fitted independently; derived words never enter regression targets.

An important diagnostic correction is recorded explicitly: orbit-mean
subtraction can manufacture sign actions from otherwise unpredictable pairs.
After early auxiliary outputs, a separate direct protocol was registered:
predict raw `h(Tx)` from raw `h(x)`, subtract only fit-derived length means,
and normalize errors by the real action displacement. Its full-space
identity baseline is one. Low-rank probes and an all-groups full-width
sensitivity are reported, together with task-code and numeric-readout-null
controls. The original orbit-centered results remain auxiliary evidence.

The old groups do not show uniformly better complete group structure than
random models. A specific repeatable result is `position_four` inverse
prediction at the source-query landmark: mean displacement NMSE 0.0471,
versus 0.4926 at random initialization, over three source seeds. The complete
width sensitivity gives 0.0376. Other operations and longer compositions
can fail; small consistency errors alone cannot certify a representation.

Three completed source models from the paused ablation were then reused for
new structural predictions, without resuming training or target adaptation.
Before their hidden features were extracted, the all-directional-record set
was predicted to have lower generator and inverse errors than the minima-pair
and peak-mixed sets, and to distinguish correct CI from wrong IC. All three
predictions matched the one-seed pilot. Source-query generator/composite NMSE
was 0.1414/0.2138 for records, 0.3308/0.4897 for the minima pair, and
0.3190/0.3964 for peak-mixed. Records' complete-width errors are 0.1110/0.1820.
Its prefix-only landmark does not beat random initialization. This is a
query-specific pilot, with unmatched source grades and joint label statistics,
not an independently replicated universal group representation.

The cross-system check reuses 27 statistical-control models on F5^4. It
tests C4 rotations, D4 rotations/reflections, and an irreversible coordinate
projection satisfying P²=P. Group probes withhold complete orbits; projection
probes withhold retained-coordinate fibers, including their image points.
These models saw all 625 inputs during source training, so only the probes
have input holdouts. Projection prediction improves over the random models,
but idempotence alone also admits collapsed predictions. Differentiation and
degree truncation of polynomials are further non-group candidates; they have
not been trained in this follow-up.

Run `.venv/bin/python -m experiments.algebra_structure_run` to reproduce or
resume the pipeline. The runner normalizes the frozen native program's
tuple/list metadata comparison without editing its registered source code.
Frozen protocols, probe matrices, all outcomes and
PNG/PDF figures accompany `report.html`. Both verification modules recompute
the direct errors from saved matrices and check the source/data provenance.
The 80-test suite passed. The prior source/transfer code remains unchanged;
the LIS ablation is still paused.

The next independent-source check uses
`configs/algebra_structure_replication.json`. It freezes three new source
seeds (2027/3037/4051) for the records, minima-pair, and peak-mixed task sets,
with four tasks and identical 20,000 updates / 2.56 million source labels.
Its 588 new complete probe orbits exclude 129,864 earlier inputs, including
every image in the first structural assay. The original seed 1009 is retested
separately and is not counted among the blind source-seed confirmations.

The comparison explicitly reports identity prediction (displacement NMSE 1),
shuffled-pair fits, independent-generator composite products, wrong CI/IC
order, and full-space recovery after two inversions. Task-query vectors,
numeric readout contrasts, their orthogonal null space, numeric logits and
task-free prefix states are reported separately. Only generator maps are fit;
no algebraic relations or composite maps are imposed during regression.
The new source training is isolated in `results/algebra_structure_replication`;
the older LIS study remains paused.

Terminology correction: the projection erases a coordinate's value by setting
it to zero in a fixed ambient space, so P²=P. Actually deleting a coordinate
changes dimension and generally cannot be described by the same equation.
`experiments/algebra_noninvertible.py` checks fixed-degree polynomial operators
exactly: D P_k = P_(k−1) D, P_j P_k = P_min(j,k), and D^(d+1)=0 on degree≤d.
The 375 integer-matrix checks cover degrees 0 through 8; no polynomial neural
training has yet been performed. The four added tests and previous suite pass
(84 tests). Run the replication module, then its analysis and verification
modules, for the completed protocol and report.

The structural source-seed replication is complete. Three new source
initializations for each of three four-task groups produced nine models,
with matched within-seed initialization, sampled input sequence, and 2.56
million source labels per model. On new complete held-out permutation orbits,
the records group has mean generator/composite displacement NMSE
0.1232/0.1623, compared with matched random 0.3615/0.5001, shuffled-fit
0.6479/0.6140, and identity prediction 1. Each of the three new seeds beats
both task-group controls on generator and unfitted-composite prediction.
The four pre-specified mean predictions are supported; the known 1009 pilot
is reported separately. Two inverse operations reconstruct the centered
query representation with mean NMSE 0.0583 (a different denominator from
single-action displacement errors).

Numeric-readout-null query structure remains weaker but detectable
(generator/composite 0.2728/0.3331, random 0.3611/0.4996). Task-free prefix
states do not improve overall on random (0.3986/0.6433 versus
0.3599/0.4999). These outcomes concern four task-conditioned query states
concatenated together. Exact task codes also have closed action relations;
source grades and joint label statistics remain unmatched. Thus this is a
replicated local structural signal, with neither a whole-network faithful
group-representation claim nor validation on unseen task combinations.

A supplemental D P1 = P0 D assay reuses the existing F5^4 encoders, with a
preserved context coordinate and disjoint probe contexts. P does not beat
both mixed groups in the registered PCA-32 primary analysis. All 625 inputs
were previously seen in source training, and all polynomial coefficient
triples occur in fitting contexts, so this is not new polynomial training
or independent-domain/OOD confirmation. Its complete outcomes, including
full-width sensitivity and exact-code controls, are in
`results/algebra_structure_replication/polynomial/report.html`.

The main verifier replayed all nine source sampling streams and recomputed
2,016 matrix/law values from 168 saved map archives. The polynomial verifier
recomputed 5,859 scalar values from 189 archives. Both passed, and the local
86-test suite passed. Use `.venv/bin/python -m pytest tests -q` in this
workspace; unscoped pytest also collects the external reference repository,
whose relative config/data tests require its own setup. The LIS branch is
still paused. Main report: `results/algebra_structure_replication/report.html`.

The answer-control and true relation-holdout follow-up is complete in
`results/algebra_hidden_relations/report.html`. A privileged baseline using
only four true answers and their task prompts reproduces the old records
group's generator/composite geometry with displacement NMSE approximately
1.0e-7/4.1e-7, and also puts this group ahead of the two nonclosed task sets.
No permutation input or neural training is used by that baseline. Therefore
the earlier geometry alone does not identify a mechanism beyond answer
encoding. Removing a fitted additive linear answer component raises the
records group's residual generator/composite errors to 0.4140/0.4766; this
does not remove all nonlinear answer information.

The new assay trains nine fresh models: ordinary task training, correct
generator-relation supervision, and shuffled relation supervision, each with
three source seeds (5081/6091/7103), matched initialization and input streams,
20,000 updates and 1.92 million source labels per model. Source training and
validation expose only e, C and I in each complete eight-state permutation
orbit. C complements values and I takes the inverse. The remaining five
states and their edges, including CI and ICI, are excluded from source inputs
and labels. Generator maps are either learned with known relation supervision
or fitted afterward using only known-state probe pairs. Compound maps are
products of these maps, with no compound fitting or hidden-label tuning;
compound answer predictions use only the base hidden vector and the fixed
numeric readout. The composition rule itself is supplied by the framework.

The conditional test contains 1,344 permutation pairs with identical length
and three known scalar answers, but different missing RL-max answers. Any
deterministic predictor using only those known answers, length and task
identity has accuracy at most 50% and cannot answer both members correctly.
Correct relation supervision achieves mean CI/ICI accuracy 50.69%, versus
27.15% for shuffled relation supervision and 24.93% for ordinary models with
posthoc correct generator probes. Its mean hidden displacement NMSE is
0.2332, versus 0.5349 and 0.5103 respectively; approximately 23.69% of pairs
are answered correctly in both members, versus 0.12% and 0.52%.

The mean gates saved before hidden evaluation all hold descriptively, but
this is not a significance result. Correct-supervision accuracy averaged
within each source seed is 54.59%, 46.35% and 51.13%; one seed is below the
answer-only ceiling. An explicitly post-evaluation, exploratory source-seed
95% t interval is 40.42%–60.97%, conditional on the shared source world and
test set. CI and ICI predict the same missing property and are not independent
replicates. These results support relation-specific, composable structure
under explicit known-relation supervision, with information beyond the three
known scalar answers. They do not establish stable above-ceiling inference,
spontaneous discovery of composition rules in ordinary task training, or
generalization across mathematical systems. The unfiltered iid results and
all negative controls are reported separately.

Independent verification checks all nine sampling traces, source/holdout
disjointness, 63 prediction archives, 1,620 numerical prediction values,
216 answer-control values and nine independently reconstructed linear-answer
residuals. Verification passed, and the project suite passed all 93 tests.
The original replication artifacts remain unchanged and the LIS branch
remains paused. Run the source, evaluation, analysis and verification modules
(`hidden_relation_train`, `hidden_relation_evaluate`,
`hidden_relation_analysis`, `hidden_relation_verify`) in that order; then run
`.venv/bin/python -m experiments.hidden_relation_finalize` to append the
clearly labeled uncertainty interpretation without changing the frozen
models, maps, metrics or mean gates. `completion.json` records artifact hashes.

## Relation follow-up, 2026-10-06

The follow-up runs through 10:00 America/Los_Angeles (17:00 UTC). It adds
36 source models with 20,000 updates each: 18 new same-world relation sources,
nine sources in three disjoint new data worlds, and nine ordinary four-task
sources. The ordinary sources have nine seed-matched untrained controls.
Direct source-label exposure totals 74.88 million; relation-loss edges are
additional supervision, accounted for separately. All new source models
finished before the deadline. Verification and reporting continue through
the deadline, with the final state in `results/until_10_followup/completion.json`.

The six new correct-relation seeds achieve collision-test CI/ICI accuracy
41.34%/39.94%; all six are below the general 50% scalar-answer ceiling on
both endpoints. The three new data worlds average 47.40%/45.93%. Visible
e/C/I grades, unfiltered iid scores and per-seed uncertainty are separate.
On identical frozen additional-seed backbones, true versus shuffled full-source
generator fitting yields mean CI/ICI accuracy 45.73% versus 27.32%.
These controls support relation specificity but do not establish stable
above-ceiling hidden-answer inference. Starting from an extra true C-state
forward pass improves the new-seed next-I score to 74.37%; this is generator
generalization with additional known-state input, not discovery of a new
relation type or autonomous composition from the base hidden vector.

Ordinary records-task geometry replicates across three new source seeds:
the numeric-readout-null composite pair-target errors are 0.9127, 0.9089,
0.9126, versus mean same-seed initialization error 1.0104. Pairs match all six
tasks' true answers across all eight orbit states, cancelling any deterministic
true-answer/task/length code. Same-selected-pair comparisons also survive
conditioning on equal or entirely correct discrete model answers. Raw-query
and task-free prefix conclusions are weaker, and output confidence remains
uncontrolled by those discrete-answer checks.
The new records sources rank the correct action first for three, two and
three of four composite words respectively. CI and RI rank first in all
three seeds; RC does not rank first in any. This is partial relation
specificity rather than consistent identification of every composite action.

A further explicitly exploratory nuisance regression uses only calibration
FIT/VAL inputs to predict the null hidden vectors from numeric logits,
softmax probabilities, entropy, maximum probability and length. It uses no
true task labels or conditional-test vectors for fitting or parameter selection.
The records residual retains only 13%–18% of its original pair energy, and
composite target error becomes about 1.278 versus 1.148 for initialization.
The geometric advantage therefore does not survive this declared confidence
association control. Subtraction can also remove genuine computational
features correlated with confidence, so this is not proof that confidence is
the sole cause. Current evidence supports replicable task-output-related
geometry, without confirming an independent algebraic computation mechanism.

An additional exploratory output proxy reconstructs the original null hidden
vectors from the same output/confidence features. At prediction time it uses
only current outputs and length, but its decoder was fitted to calibration
hidden vectors and adds fitting capacity. Against the same original targets
and denominator, records-task composite errors are 0.8672, 0.8566, 0.8471
(mean 0.8570), compared with 0.9114 for the original hidden-state predictor.
This provides a concrete output-associated alternative explanation of the
geometric signal. It is not a capacity-matched causal identification of the
sole mechanism, and uses no conditional-test hidden targets for fitting.

CI and ICI are correlated endpoints predicting the same missing statistic;
ordinary composite words are averaged within a source before seed summaries.
Source task difficulty, joint label statistics and final grades are not exactly
matched across ordinary task groups. Their comparisons do not isolate closure
as the sole cause. Ordinary probes use complete calibration orbits and supplied
composition rules, unlike the e/C/I-only relation-holdout branch. Confidence
regression, return-route readout calibration and selected-answer comparisons
are exploratory diagnostics, reported with their scopes and all controls.
The earlier polynomial boundary result remains negative, and the LIS branch
remains paused.

The report links frozen protocols, source-level statistics, independent
verifications and exportable PNG/PDF figures. Deadline-bounded scientific
controllers retain their original dates and code hashes. A future training
campaign needs a fresh output/protocol rather than editing this run in place.
Saved results can be reviewed with the `until_10_report`, `until_10_figures`
and `until_10_*_verify` modules. The final manifest records the local 107-test
suite and the precise scope of each independent numerical check.
The 107-test command is `.venv/bin/python -m pytest -q tests` from this project.
The original repository's tests pass separately from `external/neurips`.
Mixed root-level discovery produced 31 failures, including relative fixture
paths and a shared vocabulary-state assertion; that log is retained alongside
the successful scoped runs in `tests_scopes.json` and is not represented as a
successful combined suite.


## Output-information follow-up, 2026-10-06

The finite follow-up reuses all saved sources and adds no model training.
The report is `results/output_information_followup/report.html`.
A primary matched-budget comparison gives the hidden branch a fixed random
277-coordinate input projection and retains all 277 output-code features
in the output branch. Both fit the same 278-by-1024 affine reconstruction,
1024-by-64 FIT-PCA probe and three 64-by-64 generator maps, using identical
2688 FIT / 672 VAL rows and four ridge candidates per fit. Each prediction
retains only its own reconstructed residual. Full original numeric-null
hidden targets and test pairs are shared. Fitted input-PCA277 is a separately
reported sensitivity, with extra hidden input loadings; effective ranks and
feature families are not identical causal controls.

On the three new records sources, the primary hidden-input composite errors
average 0.8931 and the output-input errors average 0.8570. The output proxy
remains better in each source and reproduces the earlier full-feature proxy.
Extra reconstruction capacity alone therefore does not resolve the output
explanation under this declared comparison. This does not identify output
confidence as the unique internal cause.

Collision-pair statistics also distinguish partial hidden-answer information
from stable aggregate inference. New-six native CI has 15.15% both-correct
and 32.48% neither-correct pairs; ICI has 13.58% and 33.69%. Accuracy equals
0.5 + 0.5 * (both-correct fraction - neither-correct fraction), giving the
previous 41.34% / 39.94%. Swapped pair predictions yield only 2.29% / 2.16%
both-correct, so a nonzero double-correct rate is supplemented by a direction
control rather than interpreted alone. On identical frozen new-six sources,
correct versus shuffled relation fitting averages 20.46% / 1.07% double
correctness and 17.80 / 0.27 percentage points of original-minus-swapped
orientation excess across CI/ICI. The words are averaged within source;
old seeds, new same-world seeds and new data worlds remain separate.
These results support partial target-specific readout under supplied relation
rules, not spontaneous discovery or stable above-ceiling hidden inference.

All 18 trained/initialization budget cases have independently refitted
decoders, generator coefficients and parameter selection, checked FIT-PCA
subspaces and explicitly replayed full predictions. All 504 pair endpoints
are independently recomputed from the original saved answers. The scoped
project test command `.venv/bin/python -m pytest -q tests` passes 113 tests.
Prior completion artifacts and all 36 prior new source checkpoints were
hash-checked and preserved; the new completion manifest lists new artifacts.
Run `budget_matched_geometry`, `collision_pair_followup`,
`output_information_verify budget`, `output_information_verify pairs`, and
`output_information_finalize` to reproduce the follow-up in its existing
frozen output locations. New scientific choices require a fresh protocol.

## Original specialist CKA controls, 2026-10-06

The new [specialist report](results/specialist_cka_controls/report.html) returns
to the original eight cross-model relations and 48 existing specialist
checkpoints in `/home/yangx/neurips`. No source model was retrained. All 2,688
new test anchors and their complete eight-state orbits were excluded against
the original 16M parent inputs and previous local input archives. Core rules
were registered before new data and inference; the initial compact-JSON reader
failure and its pre-inference re-registration are retained.

The pooled final-prefix correct-minus-wrong CKA contrast is positive in all
24 relation/seed cells, including common length/answer strata. However,
computing CKA separately at each length changes the initialization comparison:
only the six cells belonging to the two inverse relations exceed either
initialization control. All 18 complement cells are lower than initialization.
Invariant-input Gram regression and PSD projection preserve this distinction.
That supplemental input-control protocol was saved after the core results,
before its own adjusted outcomes; it is explicitly exploratory. Raw pooled
CKA, answer matching, per-length aggregation and input-adjusted CKA have
different interpretations. These outcomes do not establish a uniform learned
algebraic mechanism or independence from model outputs.

The frozen operator ablation uses three existing relation-supervised sources,
visible-state-only tuning and a separate fresh 1,344-pair collision test.
Roundtrip regularization adds about 0.04 percentage points over equally
budgeted unconstrained refitting; norm and combined constraints perform worse.
All five methods, CI/ICI and paired double correctness are reported.

Reproduce within the saved protocol and exclusion catalog:

```bash
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.specialist_cka_resume evaluate
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.specialist_input_adjustment
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.frozen_operator_constraint_ablation evaluate
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.specialist_cka_verify
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.specialist_input_adjustment_verify
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.frozen_operator_constraint_verify
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.specialist_cka_report
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.specialist_cka_finalize
```

`specialist_cka_resume` checks the stored source and archive hashes without
adding datasets produced by later independent studies to an already generated
cohort. This preserves the original exclusion catalog. The report includes
every relation/seed, original model learning grades, initialization controls,
all layers, input-only baselines, fitted-confidence sensitivity, supplemental
per-length results, standalone PNG/PDF figures and separate verification.

## Inverse functional alignment assay, 2026-10-06

The [functional report](results/inverse_functional_alignment/report.html)
tests the supplied identity `recoils(x) = descents(inverse(x))`. Three frozen
descents teachers passed a separate 1,280-input reliability gate at
99.84–99.92% accuracy. Six paired target/support replicates train five
conditions from the same random initialization, with identical batches and
256 target labels: 192 for gradients and 64 for selecting among three
checkpoints. All thirty fits and selections preceded opening a new 2,560-input
test. Full eight-state orbits are excluded against all 16M original inputs,
previous local data, and every other new cohort.

Mean validation-selected target accuracy is 23.26% for ordinary training,
22.36% for correct geometry, 22.88% for answer-matched wrong geometry,
23.31% for output distillation, and 22.68% for distillation plus geometry.
Correct-minus-ordinary is -0.89 percentage points (positive in one of six
replicates); adding geometry to distillation gives -0.63 points (positive in
none). The fixed-final-step distillation contrast is +0.10 points with mixed
signs, so this does not establish that geometry invariably harms performance.
All three predeclared comparisons, every replicate, lengths, final-step
sensitivity and conditional/three-teacher descriptive intervals are retained.

Predeclared fresh-test correct CKA is only 0.0313 under correct alignment,
versus 0.0292 under ordinary training. An explicitly
[post-test diagnostic](results/inverse_functional_alignment/diagnostics/report.html)
controls CKA sample counts to twelve rows per length: correct geometry has
training CKA 0.9691 but fresh-test CKA 0.2148, versus ordinary fresh-test CKA
0.2174. This indicates fitted alignment without fresh-input generalization in
this assay; it does not test a setting with strong generalizing alignment.
The intervention supplies the inverse correspondence, so no spontaneous
discovery claim follows. The negative result is specific to the fixed losses,
label budget and randomly initialized target models.

Independent checks rescan all 16M original inputs, recreate all six target
initializations and batch schedules, replay thirty validation selections and
sixty test endpoints using the original model forward, and verify 1,650
predeclared plus 1,650 exploratory CKA scores in Gram and covariance forms.
All 128 project tests pass. Previous completed studies and source weights are
preserved; new completion records authenticate the current assay artifacts.

```bash
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_alignment teachers
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_alignment train
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_alignment evaluate
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_auxiliary run
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_verify
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_report
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_diagnostics
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_functional_finalize
```

The primary protocol and auxiliary/statistical registrations must be retained.
They already fix the exclusion catalog and all training/evaluation choices.
The fit-generalization diagnostic is registered after test outcomes and does
not alter them. Do not use the current test to retune these conditions.

## Unlabeled inverse-alignment coverage, 2026-10-06

The [coverage report](results/inverse_alignment_coverage/report.html) keeps the
same six support sets and target initializations, 192 gradient labels and 64
validation labels, and adds 4,096 new unlabeled permutations per repeat. All
six registered conditions forward the same 32 labeled plus 32 unlabeled
examples for 1,200 updates. Unlabeled archives contain no true answers;
mismatch groups use frozen teacher predictions, and output distillation
provides additional source knowledge. New pools and the 2,560-example test
are separated by their complete eight-state orbits from original 16M inputs
and all prior local data.

The primary endpoint is the fixed final update, with validation-selected
checkpoints as sensitivity. Fresh CKA uses 128 examples separately at each
length, unlike the previous twelve-example exploratory diagnostic. Correct
coverage gives CKA 0.0353 versus ordinary exposure 0.0270, but predicted-answer
matched mismatch gives 0.0417 and exact target initialization gives 0.0430.
Correct coverage accuracy is 24.10%, ordinary exposure 23.83%, matched
mismatch 24.35%, output distillation 27.81%, and distillation plus geometry
27.74%. These outcomes do not confirm a relation-specific geometry advantage
or an additional functional gain over distillation.

Six extra [equal-row-count controls](results/inverse_alignment_coverage/size_control/report.html)
were registered from code inspection after base-test computation began but
before the agent inspected its numerical results. They preserve the 64-example
forward budget and use exactly the eligible labeled geometry row count each
step, applied instead to unlabeled inputs. Fresh CKA is 0.0292, accuracy
23.77%; the CKA increment over support-only alignment is 0.0016 with a
teacher-cluster descriptive interval crossing zero. These supplements are
separate from the original six-condition registration.

An explicitly [post-test length-prior diagnostic](results/inverse_alignment_coverage/length_priors/report.html)
fits per-length majority answers using only the 192 true training labels
(26.23% accuracy) or the same 4,096 teacher predictions (29.40%). Neither
lookup uses permutation content or test labels to fit. The latter exceeds
every neural condition's mean, so a gain over the weak ordinary neural
baseline alone does not demonstrate input-dependent mathematical learning.

All 42 neural fits are complete. Independent checks replay 126 validation
checkpoints and 84 test endpoints, recheck 2,340 main plus 60 supplementary
Gram scores, reconstruct 96 bootstrap bounds, match all 7,200 supplementary
geometry row counts, rescan all 16M parent inputs, and check 72 length-prior
fit/accuracy calculations. All 133 project tests pass. The previous assay's
artifacts, models and code are hash-preserved. Registrations, all repeats and
the distinction between primary, pre-inspection supplement and post-test
diagnostic are retained in the report and completion manifest.

```bash
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_alignment_coverage train
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_alignment_coverage evaluate
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_size_control train
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_size_control evaluate
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_analysis analyze
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_verify
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_size_analysis
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_priors
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_analysis report
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m experiments.inverse_coverage_finalize
```

## Overnight inverse-function research, October 6–7

The autonomous session runs until **October 7, 2026, 10:00 AM
America/Los_Angeles (17:00 UTC)**. Its [local progress report](results/overnight_inverse_functional/progress.html)
refreshes while fitting. All previous coverage artifacts and the six frozen
source/target specialist checkpoints are preserved.

The [existing-prediction diagnostic](results/overnight_inverse_functional/existing_diagnostic/report.html)
uses training-fitted length modes to separate modal and nonmodal answers.
Teacher accuracy on the latter is 99.85%; distillation reaches 20.93%
versus 19.43% for ordinary training. A [new 5,120-input confirmation](results/overnight_inverse_functional/confirmation/diagnostic/report.html)
keeps those mappings fixed: teacher nonmodal accuracy is 99.87%, and
correct geometry and matched mismatch have essentially identical overall
accuracy, approximately 23.92%. These are frozen old models, with no new
source training or independent source-world replication.

The next [registered functional assay](configs/overnight_inverse_functional.json)
starts from existing competent **recoils** specialists. Six fresh paired
supports contain 192 training and 64 validation labels each; each repeat
also has 16,384 independent unlabeled inputs. Six matched conditions run
40,000 updates with the same data, initialization, RNG, and optimizer.
The fixed final update is primary; all registered earlier doses are
reported without test selection. Existing target pretraining supplies
substantial prior supervision, so these are few-label *fine-tuning*
comparisons, not learning from scratch with 256 total labels.

The still-unstarted tail groups `n4` and `n5` also run through separately
recorded concurrent execution wrappers that reuse the immutable main
training routine. Worker completion markers are isolated. The `n4` worker
returns control when the serial trainer enters the last `n3` condition;
the `n5` worker returns when the serial trainer reaches `n4`. Both guards
act before their job scopes can overlap.
Completed fits are skipped and partial fits resume with their saved
optimizer and CPU/CUDA RNG state. This changes execution order, with the
same registered numerical settings and endpoints. A main-only budget
audit independently regenerates all six schedules and checks all 36 fit
records before supplementary training has to finish.

A [predeclared supplementary intervention](configs/overnight_inverse_residual.json)
compares distillation alone, raw correct geometry, within-answer residual
correct geometry, and within-answer residual mismatch. The residual loss
subtracts teacher-predicted answer group means on the same 128 unlabeled
rows. It removes linear group means, not all answer information. All 24
conditions proceed through matched 5,000/10,000/20,000-update rounds;
10,000 is primary, and incomplete higher-dose comparisons are omitted.
Its independent 5,120-input test is separate from both main tests. The
supports, unlabeled pools and pretrained model pairs are shared with the
main assay, so this is not an independent source/data replication.

An [auxiliary readout probe](results/overnight_inverse_functional/readout_probe/protocol.json)
fits 30 fixed-penalty linear heads to teacher pseudo-answers before the
main test opens. It checks answer decodability at the exact ONE_END
landmark being aligned, alongside the teacher's reliable query output.
No true unlabeled answers or source neural-weight updates are used.

A separately registered [frozen-target readout diagnostic](results/overnight_inverse_functional/frozen_target_probe/protocol.json)
uses the same pseudo-answers to fit fixed linear heads on native target
ONE_END features. This exploratory extension was proposed after inspecting
teacher-readout validation monitoring but before opening the main test.
Its length-specific decoder and optimization differ from the neural
interventions, so it measures decodable information rather than a matched
causal baseline.

An [extra-length measurement](results/overnight_inverse_functional/heldout_lengths/protocol.json)
evaluates all fixed 40,000-update models on 4,096 fresh inputs of lengths
12, 18, 24 and 28. These lengths were absent from the current fine-tuning
intervention but present in the original specialist pretraining. Another
2,048 unlabeled calibration inputs determine teacher length modes before
this test opens. No student fitting or true calibration answers are used.
All eight input orbit states are excluded from the parent training data
and prior local cohorts.

The [additional independent audit](results/overnight_inverse_functional/addendum_verification/protocol.json)
checks both sets of linear heads, original model forward caches, all
subgroup counts and extra-length Gram geometry and paired intervals.
The 60 fitted heads have already passed independently reconstructed
normal equations; supplementary final-test audits run after the registered
main fits and evaluation finish. This audit does not modify any training
or select a model from test results.

Independent verification reconstructs all eight orbit states, rescans
the original 16M parent inputs, regenerates paired schedules and mismatch
masks, replays original unaccelerated model forwards, and checks subgroup
accuracy and Gram CKA calculations. The current project suite has **153
passing tests**. Final results are published only after matched fitting;
registrations do not claim spontaneous discovery or an output-independent
algebraic mechanism.

After the 10AM campaign audit, the queued report assembler writes
`results/overnight_inverse_functional/final_report.html` and
`final_results.json`. It gathers both registered interventions, readout
diagnostics and extra-length results, and explicitly lists any missing
independent supplementary checks. Individual registrations and raw reports
remain available alongside the combined report.

## Two-step relation correctness factorial

[The new registered assay](configs/two_step_relation_factorial.json) learns
complement and inverse operators on three frozen ordinary LR-max backbones.
Its four conditions are both correct, only complement correct, only inverse
correct, and both wrong. Correct single-step output distillation, initial
operators, frozen numeric readouts and exact whole-epoch input exposures are
shared; only geometric correspondence changes. Wrong pairs preserve length
and all three visible answers. Strata smaller than three are removed in
every condition, and neither discrete wrong-pair composition has a fixed
point. Learned numerical cancellation is still measured separately.

The primary missing-answer operation is **complement then inverse**, `ci`.
For this statistic, **inverse then complement**, `ic`, has exactly the
visible complement answer and is therefore only a secondary diagnostic.
There are 640 fresh collision pairs with equal length and equal e/C/I
answers but different ci answers. Any deterministic predictor using only
these visible answer codes has collision accuracy at most50% and pair-both-
correct0. This ceiling does not apply to the full probability vectors.

Six fresh fit pools reuse three frozen sources. Twenty-four hidden-space
fits are primary; twenty-four output-space fits diagnose whether output
information can reproduce composition, with different parameter counts.
All48 fits finish before the new compound feature evaluation. Predictions
take only a base vector, learned operators and fixed readout. True
intermediate inputs are restricted to explicitly labeled diagnostics.
Source models were trained from scratch on the local observed-state
dataset; new full orbits are excluded against prior local archives. This
assay does not claim original16M-corpus exclusion or spontaneous discovery.

Run `.venv/bin/python -m experiments.two_step_relation_factorial run`.
Artifacts are under `results/two_step_relation_factorial/`; the source
snapshot and implementation amendment preserve the initial registration
and the pre-training integer-type audit correction. Fitting choices and
generated data were retained. The final report is `report.html` after
evaluation and independent verification.

The subsequent joint-encoder assay is under
`results/two_step_relation_joint/`. All24 fixed-budget fits completed
before compound evaluation; original numeric readouts and tied token
embeddings were preserved. Native observed-state labels and both output
distillation losses are common across the four conditions. Only geometric
pairing changes. The fixed single-generator competence gate is reported
alongside compound performance, so weak known generators are not presented
as evidence against composition itself. Both completed factorials passed
independent original-model replay, accuracy/Gram reconstruction and exact
schedule/exposure audits. The project suite has **155 passing tests**.

`results/two_step_relation_review/report.html` consolidates both assays and
their interpretation limits. A separately registered visible-only pilot
increases unique known-input coverage16-fold; it evaluates only e/C/I
states. The prospective coverage confirmation is conditional on that
pilot passing its fixed native and generator validation thresholds.
If the pilot fails, confirmation fitting and compound prediction do not
start. If it passes, all24 fresh four-way fits use the already registered
10-epoch endpoint before a new compound test opens. Six fitting pools
reuse three source initializations, not six independently trained sources.

The10-epoch coverage factorial completed all24 fits, original-model replay,
exact supervision budgets, and independent statistical verification before
its final report was frozen. The historical primary report and its source
remain unchanged. A registered six-fit output-only baseline uses the same
sources, supports, correct output supervision and10-epoch budgets, with
geometry weight0. Saved-prediction recounts and source-bootstrap contrasts
are independently verified under `results/two_step_relation_output_only/`.
This baseline is matched to10 epochs only, not to the20-epoch extension.

The registered20-epoch sensitivity analysis extends each of the24 original
models and optimizers, preserving its first10 epochs. It reuses the same
supports and sources, so is not an independent replication. A separate
fresh compound test and a replay of both doses on that same test separate
budget changes from test-sampling changes. The combined final report is
`results/two_step_relation_dose_review/report.html`; it includes source and
whole-collision-pair uncertainty, relation-type effects, a same-CI-truth
order control, and the output-only baseline. Registrations, the original
primary report, and incomplete higher doses are retained separately.

After the original timed campaign finishes its independent audits,
`results/inverse_relation_project/report.html` provides one overview of
the single-step functional assay, its within-answer residual supplement,
and the two-step correctness factorial. It keeps prefix CKA and query
CKA separate and does not interpret geometric similarity as automatic
functional or compositional success. The reporting module performs no
new fitting or checkpoint selection.
