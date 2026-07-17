# Band-specific coherence probability pipeline

The opt-in profiles `tdbrain_coherence.toml` and
`first_paper_coherence.toml` implement a distinct experiment from the legacy
graph-measure objective.

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

The main classifier artifacts are written under
`<output>/data/connectivity_classifiers/`. Optimization results and figures are
written to
`<output>/optimization-nsga-classifier-probability-coherence-trust-region/`.
