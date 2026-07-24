# Static stimulation and advanced TD-BRAIN classifiers

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

The supplied TD-BRAIN profiles reuse the matching prior analysis and
classifier artifacts read-only, skip raw-EEG baseline loading, retain every
patient, use natural-scale coherence, and write new results locally:

- `tdbrain_coherence_static_signed_no_rejection_logistic.toml`
- `tdbrain_coherence_static_signed_no_rejection_rbf.toml`

Both profiles use:

- bands: delta, alpha, beta (one optimization per band);
- method: coherence (`coh`);
- objective: minimize the matching held-out-validated classifier's
  `P(Patient)`;
- total-change bounds: `[-3.0, 3.0]`;
- edge scope: `incident`;
- patient rejection: 0%;
- NSGA-II: population 100, 50 generations;
- workers: 4.

Run one of the full experiments from this worktree (not run during
development):

```powershell
Set-Location D:\university\projects\worktree\eeg-static-stim-graph-classifiers

$env:EEG_DATASET_CONFIG = "tdbrain_coherence_static_signed_no_rejection_logistic.toml"
D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe run_optimization.py
```

RBF alternative:

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

The advanced neural models did not improve on the regularized beta-band
logistic classifier. That negative comparison is important: 327 subjects are
modest for deep graph learning, the channel-aligned edge representation is
already well suited to regularization, and added model capacity did not
generalize better. A pretrained graph model was not used because the
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

- 21 final-state focused tests passed for static stimulation, NSGA variable
  layout, classifier constraints, graph-model fit/predict/serialization,
  three-band fusion, and result plotting.
- The broader focused run passed 35 tests, and all four suites in
  `test_optimization.py` passed.
- Python compilation and `git diff --check` passed.
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
