# TD-BRAIN log-gain paper figures and sensitivity analyses

`generate_tdbrain_log_gain_paper_results.py` is a post-analysis command. It
uses the cached connectivity matrices, classifier outputs, metadata, and 453
completed per-subject optimization files. It does **not** rerun the long EEG
pipeline or the optimizer.

Run from this worktree in PowerShell:

```powershell
D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe `
  .\generate_tdbrain_log_gain_paper_results.py `
  --paper-figures-dir "D:\university\projects\paper\figures" `
  --n-jobs 8
```

Primary outputs are written to the cached run's `final-figures` directory.
The command also mirrors the seven stable manuscript figure names to the paper
repository when `--paper-figures-dir` is supplied. Use
`--skip-age-sensitivity` for a quick figure-only regeneration after the saved
age sensitivity tables already exist.

The figure set retains the baseline-versus-candidate objective panel but omits
the selected-log-gain panel. It uses the original baseline-fitted standardized
PCA layout for group separation and patient shifts, without the redundant
``All Patient baseline`` layer. Target figures report exact unweighted counts
both numerically and on MNE montage coordinates; every scalp color scale is
set independently from that band's observed maximum.

The age sensitivity joins participant ages by stable subject ID. Within each
outer and inner training fold it estimates and removes a linear age effect
from every connectivity edge, then fits the same tuned L2-logistic family.
Age is never supplied as a classifier predictor. The primary saved classifier
and completed optimization are not changed.

The target-concentration test compares the observed unweighted best-target
counts with 100,000 draws under a uniform 26-sensor target null. This is a test
of concentration relative to random sensor choice, not evidence that the
dominant sensor is physiologically correct.
