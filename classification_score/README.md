# Standalone probabilistic EEG classification study

This directory is isolated from the analysis and optimization pipelines. It reads their stable saved artifacts, writes only below `classification_score/`, and never changes the experiment configuration.

The main finding is that the configured graph-measure space is the wrong representation for classification in these data. Spatial covariance geometry from band-filtered EEG is much more discriminative. See [RESULTS.md](RESULTS.md) for the measured results and limitations.

## What is compared

Feature families:

- 60 global graph metrics across delta/alpha/beta (the current baseline);
- every directed, off-diagonal GC edge, without symmetrizing;
- GC node strengths and singular-value topology summaries;
- channel and regional band power, relative power, epoch variability, and Hjorth mobility/complexity;
- regularized spatial correlation matrices mapped with a symmetric matrix logarithm and norm-preserving upper-triangle vectorization;
- a portable 19-electrode covariance schema shared by both datasets;
- EEG-only, connectivity-only, and all-feature fusions.

Model families:

- prior-only dummy baseline;
- L2 and elastic-net logistic regression;
- shrinkage LDA;
- calibrated linear SVM;
- probability-calibrated RBF SVM;
- K-nearest neighbors;
- Gaussian naive Bayes;
- random forest and extremely randomized trees;
- histogram gradient boosting.

Every preprocessing, univariate feature selection, scaling, and hyperparameter choice is fitted inside the training portion of nested subject-level cross-validation. No epoch is treated as an independent sample. Primary metrics include ROC AUC, balanced accuracy, Brier score, log loss, and expected calibration error.

## Reproduce the comparisons

Use the established project interpreter from the repository root:

```powershell
D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe classification_score/benchmark.py `
  --profiles first_paper tdbrain --mode quick --n-jobs 2
```

This full screen is deliberately broad and can take a long time. A focused confirmation of the selected families is:

```powershell
D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe classification_score/benchmark.py `
  --profiles first_paper tdbrain `
  --feature-sets covariance_logcorr `
  --models logistic_l2 rbf_svm `
  --mode quick --repeats 5 --inner-splits 4 --n-jobs 2 `
  --run-name repeated_confirmation --resume
```

The cache is versioned and stored in `classification_score/cache/`. Use `--force-features` after changing feature extraction. Results and out-of-fold probabilities are written to `classification_score/results/` after every model/feature combination, so `--resume` is safe.

TD-BRAIN confound sensitivity and direct cross-dataset tests:

```powershell
D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe classification_score/confound_checks.py `
  --model rbf_svm --mode quick --repeats 3 --n-jobs 2

D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe classification_score/cross_dataset_test.py `
  --feature-set eeg_portable_fused --model rbf_svm --mode quick --n-jobs 2
```

`audit_participant_workbook.mjs` is the read-only, artifact-tool-based utility used to extract the local TD-BRAIN age/sex audit table. It is optional for the classification screen.

## Fit and use the selected models

Train the two dataset-specific development models:

```powershell
D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe classification_score/selected_model.py train `
  --profile tdbrain --mode quick --n-jobs 2

D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe classification_score/selected_model.py train `
  --profile first_paper --mode quick --n-jobs 2
```

Score a saved filtered-epoch subject file:

```powershell
D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe classification_score/selected_model.py score-epochs `
  --model classification_score/models/tdbrain__covariance_logcorr__rbf_svm.joblib `
  --epochs results-TDBRAIN-restEC/data/filtered_epochs/Patient/sub-88005805.npy
```

Python API:

```python
from selected_model import load_scoring_model

model = load_scoring_model("classification_score/models/tdbrain__covariance_logcorr__rbf_svm.joblib")
patient_probability = model.predict_band_epochs(filtered_epochs_by_band, channel_names)
```

The artifact enforces exact feature names and channel order. `Patient=1`; `predict_proba(...)[1]` is the reported Patient probability.

## Files

- `data_features.py`: artifact loading and all feature families;
- `modeling.py`: model registry, nested CV, probability metrics, and final tuning;
- `benchmark.py`: resumable comparison CLI;
- `confound_checks.py`: age/sex baseline and matched-cohort validation;
- `cross_dataset_test.py`: strict train-one-dataset/test-the-other experiment;
- `selected_model.py`: final fit/load/score API and CLI;
- `test_classification_score.py`: focused synthetic contract tests;
- `results/`: CSV summaries and out-of-fold predictions;
- `models/`: fitted joblib artifacts plus JSON provenance.

## Scientific boundary

These probabilities are research scores, not diagnoses. The cross-dataset transfer failure shows that acquisition/reference/site shift remains substantial. A covariance score can only become an optimization objective after simulated trajectories are demonstrated to reproduce the real-EEG covariance feature distribution and the intervention-induced score change is externally validated. Until then, minimizing Patient probability is not evidence of a therapeutic effect.

