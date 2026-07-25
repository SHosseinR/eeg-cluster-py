# Static stimulation and advanced TD-BRAIN classifiers

## Direct-plus-one-hop activation with original plasticity

The hybrid model requested after the direct-edge static experiment is
selected with:

```toml
[optimization]
stimulation_model = "adjacency_activation"
stimulation_activation_amount_bounds = [-3.0, 3.0]
adjacency_activation_neighbor_scale = 1.0
```

For selected node `k`, the stimulation stage computes:

```text
delta_x = amount * (e_k + neighbor_scale * A_norm @ e_k)
final_activation = baseline_activation + delta_x
```

`A_norm` is the same spectrally normalized, zero-diagonal adjacency used by
the state-space model. With the current matrix convention, `A_norm @ e_k` is
column `k`. For symmetric coherence matrices this is simply the selected
node's connectivity profile.

The selected node therefore changes by exactly `amount`; each directly
connected node changes by
`amount * neighbor_scale * A_norm[i, k]`. A node connected only through a
two-hop path does not change. There are no `A_norm^2` or higher propagation
terms, no duration, no leak, and no trajectory. `amount` is the signed
direct-node activation change, not the L1 sum over all changed nodes.

Everything after activation is retained from the original model:

```text
raw_ratio_i = final_activation_i / baseline_activation_i
ratio_i = clip(raw_ratio_i, 0.1, 10)
A_new[i,j] = A_original[i,j] * (ratio_i * ratio_j)^plasticity_scaling
```

The optimizer still rejects candidates whose *raw* ratios fall outside the
configured feasibility bounds. In classifier-probability mode, plasticity
updates preserve natural coherence scale instead of applying candidate-wise
min-max normalization.

The complete isolated logistic profiles are:

- `tdbrain_coherence_adjacency_activation_signed_no_rejection_logistic.toml`
- `first_paper_coherence_adjacency_activation_signed_no_rejection_logistic.toml`

They use delta, alpha, and beta coherence classifiers, retain every patient,
use neighbor scale `1.0`, signed direct-node amount bounds `[-3, 3]`,
population 100, 50 generations, and four optimization workers.

Run the complete two-dataset pipeline:

```powershell
Set-Location D:\university\projects\worktree\eeg-static-stim-graph-classifiers

.\run_coherence_probability_pipeline.ps1 `
  -DatasetConfigs @(
    "tdbrain_coherence_adjacency_activation_signed_no_rejection_logistic.toml",
    "first_paper_coherence_adjacency_activation_signed_no_rejection_logistic.toml"
  ) `
  -RunOutputDir "results-adjacency-activation-signed-no-rejection-logistic"
```

## Dynamics-free stimulation model

Set `optimization.stimulation_model = "static_adjacency"` in a dataset TOML
profile. For a selected node, the model changes its configured incident,
incoming, or outgoing adjacency entries in proportion to their original
absolute weights:

```text
delta_ij = total_change * sign(A_ij) * abs(A_ij) / sum_selected(abs(A))
```

The L1 norm of the realized edge updates therefore equals
`abs(total_change)` (apart from the exact zero case). Positive values
strengthen existing magnitudes and negative values weaken them. The diagonal,
matrix shape, channel order, and unselected entries are preserved.

For fixed-band NSGA-II this model has two decision variables:

1. stimulation node;
2. signed total adjacency change.

Duration and leak are absent because there is no state-space trajectory.
Saved solutions use `stimulation_total_change`; `stimulation_amplitude`
contains the same value only as a compatibility alias for older plot loaders.

The supplied logistic profiles run complete, isolated analysis and
classification pipelines for TD-BRAIN and First Paper. Their static
optimization stages skip raw-EEG baseline loading, retain every patient, use
natural-scale coherence, and consume the artifacts created earlier in the
same pipeline:

- `tdbrain_coherence_static_signed_no_rejection_logistic.toml`
- `first_paper_coherence_static_signed_no_rejection_logistic.toml`

The TD-BRAIN RBF alternative reuses the matching prior analysis and classifier
artifacts read-only:

- `tdbrain_coherence_static_signed_no_rejection_rbf.toml`

All static profiles use:

- bands: delta, alpha, beta (one optimization per band);
- method: coherence (`coh`);
- objective: minimize the matching held-out-validated classifier's
  `P(Patient)`;
- total-change bounds: `[-3.0, 3.0]`;
- edge scope: `incident`;
- patient rejection: 0%;
- NSGA-II: population 100, 50 generations;
- workers: 4.

Run the complete two-dataset logistic experiment from this worktree (not run
during development):

```powershell
Set-Location D:\university\projects\worktree\eeg-static-stim-graph-classifiers

.\run_coherence_probability_pipeline.ps1 `
  -DatasetConfigs @(
    "tdbrain_coherence_static_signed_no_rejection_logistic.toml",
    "first_paper_coherence_static_signed_no_rejection_logistic.toml"
  ) `
  -RunOutputDir "results-static-signed-no-rejection-logistic"
```

TD-BRAIN RBF optimization-only alternative:

```powershell
$env:EEG_DATASET_CONFIG = "tdbrain_coherence_static_signed_no_rejection_rbf.toml"
D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe run_optimization.py
```

### One-subject smoke result

Only a small validation run was performed: alpha-band logistic optimization
for TD-BRAIN patient `sub-88064657`, population 20, 5 generations, seed 17.
The selected candidate targeted T7 with total change `-2.8893`. The exact
realized L1 change was `2.8893` across 50 symmetric matrix entries; duration
and leak were `None`; aggregate constraint violation was zero. The classifier
probability changed only from `0.9999978` to `0.9999933`, so this smoke run
validates the mechanics and artifact contract, not efficacy.

The local smoke artifact is:

```text
results-static-signed-no-rejection-logistic/
  results-TDBRAIN-restEC-coherence/optimization/smoke/
  sub-88064657_alpha_static_smoke.npy
```

## Advanced classifier comparison

Two optional PyTorch model families are registered in the same classifier
configuration used by the main pipeline:

- `gcn`: a spectral graph convolution model with shared electrode embeddings;
- `brainnetcnn`: an edge-to-edge, edge-to-node, node-to-graph convolution
  model specialized to connectivity matrices.

The standalone benchmark also supports `gcn_3band` and
`brainnetcnn_3band`, which fuse delta, alpha, and beta graphs. All evaluation
is subject-level. Each outer training fold owns its imputation, neural
training, stratified early-stopping split, class weighting, and temperature
scaling. No test-fold subject is used by those steps.

The completed TD-BRAIN benchmark used 327 subjects (176 Healthy, 151
Patient), 5-fold stratified cross-validation repeated 5 times, natural-scale
coherence, and CPU PyTorch. Runtime was 9.9 minutes.

| Band | Model | ROC AUC | Balanced accuracy | Brier |
|---|---|---:|---:|---:|
| delta | logistic L2 | 0.798 | 0.714 | 0.185 |
| delta | RBF SVM | 0.753 | 0.684 | 0.204 |
| delta | BrainNetCNN | 0.713 | 0.678 | 0.217 |
| delta | GCN | 0.538 | 0.500 | 0.249 |
| alpha | logistic L2 | 0.826 | 0.784 | 0.170 |
| alpha | RBF SVM | 0.794 | 0.710 | 0.185 |
| alpha | BrainNetCNN | 0.777 | 0.715 | 0.195 |
| alpha | GCN | 0.519 | 0.500 | 0.249 |
| beta | logistic L2 | **0.872** | **0.817** | **0.141** |
| beta | RBF SVM | 0.851 | 0.788 | 0.156 |
| beta | BrainNetCNN | 0.795 | 0.760 | 0.190 |
| beta | GCN | 0.556 | 0.500 | 0.249 |
| delta+alpha+beta | BrainNetCNN 3-band | 0.778 | 0.725 | 0.197 |
| delta+alpha+beta | GCN 3-band | 0.528 | 0.500 | 0.249 |

### Neural-training completion audit

All 200 planned neural outer-fold fits completed: 150 single-band fits and 50
three-band fits. The saved OOF tables contain exactly 9,810 single-band and
3,270 fused prediction rows, all 327 subjects appear once in every one of the
five repeats for every model/band, and there are no missing probabilities.
Training uses early stopping, so "complete" means reaching the declared
stopping rule rather than forcing all 160 maximum epochs.

A separate representative beta-fold audit also reconstructed each model's
seeded initial weights and compared them with the trained state:

- GCN: stopped at epoch 66/160; parameter L2 change 1.245; validation loss
  0.693.
- BrainNetCNN: stopped at epoch 42/160; parameter L2 change 2.666; validation
  loss 0.495.

Thus both networks really updated their weights. The poor GCN result is a
model collapse/generalization problem, not a skipped fit: its averaged OOF
probabilities had only 0.0016-0.0040 standard deviation and stayed close to
the class prior. Dense, all-positive coherence graphs become nearly
homogeneous after GCN degree normalization and global pooling, so this
architecture discards much of the electrode-pair-specific signal.
BrainNetCNN did learn nontrivial predictions (probability standard deviation
0.163-0.196) but generalized less consistently than logistic regression.

The advanced neural models therefore did not improve on the regularized
beta-band logistic classifier in this experiment. This does not establish
that every possible neural architecture must lose. It shows that with 327
subjects, 325 explicitly aligned edge features, and these predeclared neural
architectures/hyperparameters, the lower-variance L2 logistic model
generalizes better. A pretrained graph model was not used because the
repository provides no checkpoint with the same 26-channel order, coherence
estimator, band definitions, and cohort contract; transferring an unrelated
graph checkpoint would not preserve this experiment's feature semantics.

Reproduce the complete benchmark:

```powershell
Set-Location D:\university\projects\worktree\eeg-static-stim-graph-classifiers

D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe `
  -m classification_score.advanced_graph_benchmark `
  D:/university/projects/worktree/eeg-coherence-band-classifier/results-signed-no-rejection-logistic/results-TDBRAIN-restEC-coherence/data/connectivity_matrices.npy `
  results-advanced-classifiers/benchmark `
  --mode quick --outer-splits 5 --repeats 5 --inner-splits 3 --n-jobs 1 `
  --baseline-summary D:/university/projects/worktree/eeg-coherence-band-classifier/results-signed-no-rejection-logistic/results-TDBRAIN-restEC-coherence/data/connectivity_classifiers/classification_summary_by_band_connectivity.csv `
  --baseline-summary D:/university/projects/worktree/eeg-coherence-band-classifier/results-signed-no-rejection-rbf/results-TDBRAIN-restEC-coherence/data/connectivity_classifiers/classification_summary_by_band_connectivity.csv
```

Generated outputs include the combined metrics CSV, all neural out-of-fold
predictions, a PNG comparison, and a generated Markdown report below
`results-advanced-classifiers/benchmark/`.

## Verification

- 30 final-state focused tests passed for static stimulation, the two
  full-pipeline profile contracts, NSGA variable layout, classifier
  constraints, graph-model fit/predict/serialization, three-band fusion, and
  saved-result plotting.
- An earlier broader focused run passed 35 tests, and all four suites in
  `test_optimization.py` passed.
- Python compilation and `git diff --check` passed.
- The real one-subject static artifact generated inspected edge-change,
  edge-profile, and before/after adjacency figures; the state-space branch of
  the same subject plotting command was also regression-tested.
- The discoverable suite has one unrelated existing failure in
  `test_plot_weighted_selection_target_3d`: its bipolar-reference position
  assertion fails identically in the untouched source worktree.
- The full multi-subject optimization was intentionally not run. Both static
  profiles passed their artifact/classifier preflight checks, and the
  one-subject smoke run described above exercised the complete static NSGA
  path.

These are research classifiers, not diagnostic models. Neither classifier
probability changes nor simulated target selection demonstrate clinical
stimulation effectiveness.
