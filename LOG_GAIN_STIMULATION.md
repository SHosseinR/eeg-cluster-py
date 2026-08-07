# TD-BRAIN log-gain stimulation experiment

The profile
`dataset_configs/tdbrain_coherence_adjacency_activation_log_gain_signed_no_rejection_logistic.toml`
adds a dynamics-free, multiplicative activation model without changing the
legacy `adjacency_activation` implementation.

For each subject and fixed frequency band, the cached band-filtered EEG defines
the per-channel baseline

```text
E0_i = sqrt(mean(band_filtered_EEG_i^2)).
```

For selected node `k`, the model uses the same zero-diagonal spectral adjacency
normalization as the established activation model:

```text
v = e_k + neighbor_scale * A_norm @ e_k
R_i = exp(log_gain * v_i)
E1_i = E0_i * R_i
```

Connectivity plasticity is then

```text
A_target_ij = A0_ij * (R_i * R_j)^plasticity_exponent
A_post = A0 + plasticity_fraction * (A_target - A0).
```

The default log-gain interval `[-2.302585093, 2.302585093]` is approximately
`[log(0.1), log(10)]`. The other defaults are `neighbor_scale = 1`,
`plasticity_exponent = 1`, and `plasticity_fraction = 1`.

## Cached analysis contract

The profile reads and writes beneath:

```text
D:/university/projects/worktree/eeg-static-stim-graph-classifiers/
results-adjact-signed-norej-logistic/TDBRAIN-restEC-coherence
```

Existing connectivity matrices, network measures, filtered epochs, channel
metadata, and per-band logistic classifiers are read-only inputs. New results
are isolated in `optimization-log-gain/`; the existing `optimization/`
directory is not used or modified.

Run the cached pipeline from the log-gain worktree:

```powershell
Set-Location "D:\university\projects\worktree\eeg-tdbrain-log-gain"
.\run_coherence_probability_pipeline.ps1 `
  -DatasetConfigs @(
    "tdbrain_coherence_adjacency_activation_log_gain_signed_no_rejection_logistic.toml"
  ) `
  -RunOutputDir "D:\university\projects\worktree\eeg-static-stim-graph-classifiers\results-adjact-signed-norej-logistic" `
  -SkipAnalysis
```

`-SkipAnalysis` validates the cached artifact contracts, skips `main.py`, and
still runs optimization, completeness auditing, and all applicable figures.

This computational experiment produces research hypotheses, not validated
stimulation targets or treatment recommendations.
