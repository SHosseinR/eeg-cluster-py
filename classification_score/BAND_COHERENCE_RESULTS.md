# Per-band coherence classifier and probability-objective results

Run date: 2026-07-17

## Experiment

The pipeline now trains and validates one classifier for each configured
frequency band. Each model receives only the natural-scale upper-triangle
coherence edges from its own band; bands are never concatenated. Healthy is
label 0 and Patient is label 1.

Model-family selection used leakage-safe nested stratified cross-validation.
The selected family was then evaluated with five repeated five-fold outer CV
runs and three-fold inner tuning. Selection favors calibration among model
families whose ROC AUC is within 0.02 of the best family, because the deployed
quantity is a probability rather than only a class label.

## Held-out classification

The table reports pooled repeated out-of-fold metrics. `AUC SD` and `BA SD`
are the standard deviations across the five complete CV repeats.

| Dataset | Band | Selected model | Subjects | Edges | ROC AUC | AUC SD | Balanced accuracy | BA SD | Accuracy | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TD-BRAIN | delta | L2 logistic | 327 | 325 | 0.807 | 0.021 | 0.715 | 0.022 | 0.716 | 0.179 | 0.041 |
| TD-BRAIN | alpha | L2 logistic | 327 | 325 | 0.820 | 0.009 | 0.758 | 0.016 | 0.758 | 0.173 | 0.058 |
| TD-BRAIN | beta | L2 logistic | 327 | 325 | 0.883 | 0.013 | 0.816 | 0.021 | 0.817 | 0.135 | 0.038 |
| First paper | delta | RBF SVM | 58 | 190 | 0.848 | 0.062 | 0.827 | 0.074 | 0.828 | 0.162 | 0.130 |
| First paper | alpha | RBF SVM | 58 | 190 | 0.821 | 0.047 | 0.723 | 0.054 | 0.724 | 0.174 | 0.132 |
| First paper | beta | RBF SVM | 58 | 190 | 0.788 | 0.030 | 0.758 | 0.046 | 0.759 | 0.197 | 0.167 |

All six bands pass the configured deployment evidence gate: AUC at least
0.75, balanced accuracy at least 0.70, and Brier score at most 0.20. The
first-paper estimates are less precise and less well calibrated because they
contain only 58 subjects. The TD-BRAIN models are the stronger evidence.

For context, the earlier TD-BRAIN coherence graph-measure classification
accuracies were 0.587 (delta), 0.535 (alpha), and 0.639 (beta). The per-band
edge classifiers improve accuracy by approximately 0.129, 0.223, and 0.177,
respectively.

Exact CSV results and fitted bundles are stored under each profile's
`data/connectivity_classifiers` directory. Bundles include the exact channel
order, feature mapping, CV evidence, fitted hyperparameters, orientation
checks, and out-of-distribution thresholds.

## Classifier-probability optimization

The optimizer no longer uses `optimization_measures_by_band` as its objective
in these two profiles. Its sole objective is to minimize the matching band's
`P(Patient)`; equivalently, it maximizes `P(Healthy)`. The connectivity matrix
is still required by the state-space and plasticity simulation.

Before optimization, each band independently retains the 50% of patients with
the highest repeated out-of-fold patient probability. Candidate matrices must
also satisfy activation, coherence-range, global feature-distribution,
nearest-observed-subject, and patient-local change constraints. Signed
stimulation amplitudes from -3 to 3 are enabled explicitly in these new
profiles.

| Dataset | Band | Retained | Median initial P(Patient) | Median optimized P(Patient) | Median reduction | Improved | Median amplitude | Most frequent rank-1 node |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TD-BRAIN | delta | 75 | 0.928 | 0.023 | 0.866 | 75/75 | -0.807 | Fp1 (51/75) |
| TD-BRAIN | alpha | 75 | 0.926 | 0.002 | 0.914 | 75/75 | -0.811 | Fp1 (36/75) |
| TD-BRAIN | beta | 75 | 0.952 | 0.036 | 0.883 | 75/75 | -0.649 | P7 (46/75) |
| First paper | delta | 15 | 0.971 | 0.208 | 0.763 | 15/15 | -0.445 | P4-LE (6/15) |
| First paper | alpha | 15 | 0.884 | 0.500 | 0.374 | 15/15 | +0.349 | T4-LE (5/15) |
| First paper | beta | 15 | 0.837 | 0.598 | 0.239 | 15/15 | -0.378 | C4-LE (3/15) |

All saved rank-1 solutions have finite objectives and constraints, zero
aggregate constraint violation, the correct classifier band and channel
contract, and the one-dimensional `patient_probability` objective. The local
trust-region constraint is active or nearly active for many solutions.

TD-BRAIN's conservative cross-band comparison used the 32 retained patients
common to all three independently selected cohorts. It selected alpha
(Friedman p = 1.16e-7) because alpha had the strongest conservative
improvement bound and beat both alternatives. Only four first-paper patients
were common to all three bands, so its band winner is correctly reported as
inconclusive despite Friedman p = 0.018.

## Interpretation limits

The large TD-BRAIN probability shifts and concentrated Fp1/P7 selections show
that the optimizer can strongly exploit the learned classifier through the
current state-space/plasticity model. They do not establish a biological or
clinical stimulation effect. The frequent solutions at the patient-local
trust boundary make the result a useful computational hypothesis and an
important sensitivity-analysis target, not a treatment recommendation.

External-cohort validation, probability recalibration, perturbation tests for
the dynamics/plasticity assumptions, and stability across seeds and trust
radii are needed before interpreting electrode preferences scientifically.
