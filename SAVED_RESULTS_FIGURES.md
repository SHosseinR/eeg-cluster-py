# Saved-results figure suite

The paper figures can be regenerated from saved NPY and CSV artifacts without
running `main.py` or `run_optimization.py`:

```powershell
D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe generate_saved_result_figures.py `
  --dataset-config tdbrain.toml `
  --dataset-config first_paper.toml `
  --organize-existing
```

`--dataset-config` may be repeated for any number of TOML profiles.
`--organize-existing` is idempotent: it categorizes existing figure files and
writes `figure_manifest.csv` in each figure root. It should be omitted when only
regenerating the new plots.

The driver validates the saved connectivity matrices, network measures, and
optimization results before plotting. If only the band-stability summary is
missing, it rebuilds that summary from the saved per-band optimization results.
For other missing inputs, it reports the upstream pipeline stage that must be
run.

New outputs include:

- modularity-reordered Healthy/Patient connectivity matrices and ordering CSVs;
- full and optimization-cohort 3D metric-space PNG/HTML figures, independent
  Healthy/Patient clusters, assignments, and silhouette tables;
- weighted target PNG/HTML figures whose colors retain the saved rank-weighted
  score and whose radius uses the clipped median band improvement;
- an additional all-band target PNG/HTML/CSV where color is the raw all-band
  score, while radius is the node-wise sum of each band's score multiplied by
  that band's clipped median improvement;
- a cross-dataset 2x2 top-five comparison and its source table under
  `results-comparison/`.

The principal figure categories are `connectivity/`, `network_statistics/`,
`classification/`, `overview/`, `metric_space/`, `targets/`,
`target_statistics/`, and `subjects/`. Unknown legacy figures are preserved in
`misc/`.
