# Results (2026-07-16)

## Primary conclusion

The original graph-measure representation does not separate TD-BRAIN MDD from Healthy subjects: the best regularized graph-only screen remained near chance (AUC 0.517; balanced accuracy 0.524). Directed GC edges improved only to AUC 0.642. In contrast, spatial covariance geometry from band-filtered EEG produced stable within-dataset discrimination.

The primary values below are means over five complete repeated nested-CV runs; the standard deviations describe split sensitivity, not population confidence intervals.

| Dataset | Selected EEG representation | Classifier | ROC AUC | Balanced accuracy | Brier |
|---|---|---:|---:|---:|---:|
| TD-BRAIN (176 Healthy, 151 MDD) | Full 26-channel log-correlation covariance, 3 bands | RBF SVM + internal sigmoid probability fit | 0.910 ± 0.004 | 0.852 ± 0.014 | 0.116 ± 0.004 |
| First-paper (28 Healthy, 30 MDD) | Regional spectral/Hjorth + portable 19-channel log-correlation covariance | RBF SVM + internal sigmoid probability fit | 0.934 ± 0.012 | 0.904 ± 0.025 | 0.089 ± 0.012 |

The probabilities averaged over the five out-of-fold repeats gave AUC 0.914 / balanced accuracy 0.867 / Brier 0.112 on TD-BRAIN and AUC 0.945 / balanced accuracy 0.931 / Brier 0.077 on the first-paper dataset. These averaged values are useful summaries but are less conservative than the repeat means above.

## TD-BRAIN confound sensitivity

The local participant workbook shows a measured age imbalance: Healthy mean 40.38 years and MDD mean 46.22 years; sex counts were similar. Age+sex alone achieved only AUC 0.602. One-to-one matching within sex retained all 151 MDD subjects and 151 controls (median within-pair age difference 1.41 years). On this matched cohort, the covariance RBF model retained mean AUC 0.890, balanced accuracy 0.830, aggregate Brier 0.123, and aggregate log loss 0.398 across three repeated nested runs. This argues against age/sex fully explaining the covariance result, but it does not exclude medication, severity, preprocessing, recording-year, or other unmeasured confounding.

## Representation comparison on TD-BRAIN

| Representation/model (quick nested screen) | AUC | Balanced accuracy | Brier |
|---|---:|---:|---:|
| Graph metrics / logistic | 0.517 | 0.524 | 0.268 |
| Directed GC edges / logistic | 0.642 | 0.598 | 0.308 |
| Regional spectral/Hjorth / calibrated linear SVM | 0.676 | 0.590 | 0.228 |
| Channel spectral/Hjorth / calibrated linear SVM | 0.721 | 0.674 | 0.218 |
| Full covariance / logistic | 0.884 | 0.832 | 0.132 |
| Full covariance / calibrated linear SVM | 0.886 | 0.810 | 0.148 |
| Full covariance / extra trees | 0.873 | 0.812 | 0.173 |
| Full covariance / RBF SVM | **0.909** | **0.847** | **0.117** |

The portable 19-channel covariance RBF model retained AUC 0.882 on TD-BRAIN and 0.887 on the first-paper dataset when fitted and tested within each dataset.

## External transfer test

Direct cross-dataset deployment was not successful. With the identical portable covariance schema, first-paper→TD-BRAIN AUC was 0.485 and TD-BRAIN→first-paper AUC was 0.383. The portable EEG fusion improved ranking in one direction (TD-BRAIN→first-paper AUC 0.844), but its threshold behavior was unusable: every first-paper subject was classified Patient, giving balanced accuracy 0.500 and Brier 0.286. In the reverse direction its AUC was 0.569 with balanced accuracy 0.498.

Therefore the saved models are dataset-specific. Their probabilities must not be compared across datasets without an independent harmonization/calibration cohort.

## Why these methods were included

- Spatial covariance matrices retain multichannel structure that scalar graph summaries discard; Riemannian covariance methods are established EEG descriptors ([Barachant et al., 2012](https://hal.science/hal-00681328v1/document)). This study uses a dependency-free log-Euclidean mapping rather than claiming an exact reproduction of that algorithm.
- MDD EEG studies support testing power, connectivity, asymmetry, complexity, and periodic/aperiodic spectral structure, but their reported results vary materially by cohort and validation design ([large systematic validation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8699348/); [periodic/aperiodic MDD study](https://pubmed.ncbi.nlm.nih.gov/39338848/)).
- TDBRAIN is a heterogeneous clinical lifespan resource, so demographic and dataset-shift checks are essential ([TDBRAIN data descriptor](https://www.nature.com/articles/s41597-022-01409-z)).
- Hyperparameter and feature selection can overfit the CV criterion itself; all per-model choices here are nested, and the remaining across-family selection risk is reported explicitly ([Cawley & Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)).

## What this means for optimization

The classification problem is improved, but the optimization-objective problem is not solved yet.

The winning TD-BRAIN score uses covariance of real band-limited EEG. The current stimulation model generates a short state-space trajectory and an updated GC-like adjacency matrix. Although the code can extract a covariance score from a complete simulated band-epoch payload, that number is not valid unless:

1. unstimulated simulated trajectories reproduce each real subject's covariance feature distribution;
2. simulated and real features use identical bands, duration, scaling, channel order, and preprocessing;
3. the classifier is calibrated on a held-out external cohort;
4. score changes are shown not to exploit out-of-distribution directions;
5. intervention-related score movement is associated with an independently meaningful outcome.

Until those checks pass, the classifier is a stronger separation experiment—not a validated stimulation target or treatment recommendation.
