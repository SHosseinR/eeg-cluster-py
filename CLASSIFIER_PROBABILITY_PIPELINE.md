# Band-specific coherence probability pipeline

The opt-in profiles `tdbrain_coherence.toml` and
`first_paper_coherence.toml` implement signed stimulation with 50% independent
per-band patient rejection. The separate
`*_coherence_enhancement_no_rejection.toml` profiles use strictly positive
amplitudes `[0.1, 3.0]`, retain every patient, and write to isolated output
trees. Both variants are distinct from the legacy graph-measure objective.

For each of delta, alpha, and beta, the analysis pipeline extracts only that
band's natural-scale upper-triangle coherence edges. It compares probabilistic
model families with nested cross-validation, repeats validation for the
selected family, fits one deployable model per band, and saves held-out
probabilities and patient rankings. Healthy is class 0 and Patient is class 1.
Model families within 0.02 ROC AUC of the screen leader are treated as
practically tied; Brier score and calibration error then choose the model
because optimization depends on probability magnitude.

In `classifier_patient_probability` optimization mode, the matching band model
receives the simulated post-plasticity connectivity matrix and NSGA-II directly
minimizes `P(Patient)`. The historical `optimization_measures_by_band` entries
remain in the TOML files for legacy compatibility but are not objectives in
this mode.

Candidate matrices are not min-max normalized. They must satisfy activation
ratio bounds, natural coherence bounds `[0, 1]`, a global OOD threshold,
proximity to an observed training subject, and a patient-local change radius
derived from training nearest-neighbor variation. The opt-in profiles use signed amplitude bounds `[-3, 3]` and
include a zero-stimulation anchor. These safeguards make the classifier score
computationally consistent; they do not prove that an updated matrix is a
physiologically realizable post-stimulation coherence matrix or that a target
is clinically effective.

Run both configured experiments from PowerShell:

```powershell
./run_coherence_probability_pipeline.ps1
```

Run TD-BRAIN only:

```powershell
./run_coherence_probability_pipeline.ps1 -DatasetConfigs tdbrain_coherence.toml
```

After the signed/rejected experiment is complete, run both enhancement-only,
zero-rejection experiments with:

```powershell
./run_coherence_probability_pipeline.ps1 `
  -DatasetConfigs @(
    "tdbrain_coherence_enhancement_no_rejection.toml",
    "first_paper_coherence_enhancement_no_rejection.toml"
  ) `
  -ComparisonOutputDir "results-coherence-classifier-comparison-enhancement-no-rejection"
```

The runner now creates fixed baseline-fitted PCA cohort/shift figures, replaces
the one-objective Pareto plot with convergence and trust-boundary diagnostics,
and creates three per-band plus one all-band validity-weighted 2D target map.
The same PCA scaler/projection fitted to original subjects transforms optimized
matrices; it is never refitted after optimization.

If optimization results already exist and only downstream figures/audits need
to be resumed, use `-FiguresOnly`. This mode does not run EEG analysis,
classification, or optimization:

```powershell
./run_coherence_probability_pipeline.ps1 `
  -DatasetConfigs tdbrain_coherence_enhancement_no_rejection.toml `
  -FiguresOnly
```

The enhancement/no-rejection profiles deliberately use the short optimization
subdirectory `opt-clfprob-trust-enh-norej` to remain below Windows `MAX_PATH`.
Each run also writes `optimization_subject_completeness.csv`; future
optimizations additionally write exact per-subject error manifests in each
band's subject-results directory.

## Signed stimulation without rejection

The profiles `tdbrain_coherence_signed_no_rejection.toml` and
`first_paper_coherence_signed_no_rejection.toml` retain every patient and use
signed amplitude bounds `[-3, 3]`. Unlike the older flat experiment outputs,
all artifacts are grouped under one parent while retaining the familiar
`results-*` folder names:

```text
results-signed-no-rejection/
  results-TDBRAIN-restEC-coherence/
    data/
    figures/
    reports/
    optimization/
  results-first-paper-coherence/
    data/
    figures/
    reports/
    optimization/
  results-coherence-classifier-comparison/
  pipeline_timings_<timestamp>.csv
```

Run the complete two-dataset experiment with:

```powershell
./run_coherence_probability_pipeline.ps1 `
  -DatasetConfigs @(
    "tdbrain_coherence_signed_no_rejection.toml",
    "first_paper_coherence_signed_no_rejection.toml"
  ) `
  -RunOutputDir "results-signed-no-rejection"
```

Timing outputs are written at three levels:

- `pipeline_timings_<timestamp>.csv` in the repository root for each invoked command;
- `<output>/reports/analysis_stage_timings.csv`, including connectivity and classification;
- `<optimization-output>/optimization_stage_timings.csv`, including each optimization band.

The main classifier artifacts are written under
`<output>/data/connectivity_classifiers/`. Optimization results and figures are
written to
`<output>/optimization-nsga-classifier-probability-coherence-trust-region/`.
