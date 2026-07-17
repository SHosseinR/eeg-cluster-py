param(
    [string[]]$DatasetConfigs = @(
        "tdbrain_coherence.toml",
        "first_paper_coherence.toml"
    ),
    [string]$Python = "D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe",
    [switch]$AnalysisOnly
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

function Invoke-PythonChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

foreach ($datasetConfig in $DatasetConfigs) {
    Write-Host "`n========================================================================"
    Write-Host "ANALYSIS: $datasetConfig"
    Write-Host "========================================================================"
    $env:EEG_DATASET_CONFIG = $datasetConfig
    $env:EEG_STEP_TO_START = "1"
    Invoke-PythonChecked "main.py"

    if ($AnalysisOnly) {
        continue
    }

    Write-Host "`n========================================================================"
    Write-Host "CLASSIFIER-PROBABILITY OPTIMIZATION: $datasetConfig"
    Write-Host "========================================================================"
    Invoke-PythonChecked "run_optimization.py"

    # Applicable saved-result figures for a one-dimensional probability
    # objective. The historical graph-metric 3-D plots are intentionally not
    # invoked because they do not represent this experiment.
    Invoke-PythonChecked "plot_best_closeness_per_subject.py"
    Invoke-PythonChecked "plot_weighted_node_band_interactive.py"
    Invoke-PythonChecked "plot_subject_activation_and_adjacency.py" "--subject" "__first__" "--band" "delta"
    Invoke-PythonChecked "plot_band_stability_analysis.py" "--bootstrap-resamples" "10000"
}

if (-not $AnalysisOnly -and $DatasetConfigs.Count -gt 1) {
    $comparisonArguments = @("plot_top_selected_nodes.py")
    foreach ($datasetConfig in $DatasetConfigs) {
        $comparisonArguments += @("--dataset-config", $datasetConfig)
    }
    $comparisonArguments += @("--output-dir", "results-coherence-classifier-comparison")
    Invoke-PythonChecked @comparisonArguments
}

Write-Host "`nComplete. Each profile contains analysis, band classifiers, optimization, and applicable figures."
