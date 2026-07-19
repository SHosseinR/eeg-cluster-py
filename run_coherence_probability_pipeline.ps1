param(
    [string[]]$DatasetConfigs = @(
        "tdbrain_coherence.toml",
        "first_paper_coherence.toml"
    ),
    [string]$Python = "D:/Users/hosei/anaconda3/envs/eeg-graph/python.exe",
    [string]$RunOutputDir = "",
    [string]$ComparisonOutputDir = "",
    [switch]$AnalysisOnly,
    [switch]$FiguresOnly
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
if ($AnalysisOnly -and $FiguresOnly) {
    throw "-AnalysisOnly and -FiguresOnly cannot be used together."
}
$timings = [System.Collections.Generic.List[object]]::new()
$currentDataset = ""
$effectiveComparisonOutputDir = if ($ComparisonOutputDir) {
    $ComparisonOutputDir
} elseif ($RunOutputDir) {
    Join-Path $RunOutputDir "results-coherence-classifier-comparison"
} else {
    "results-coherence-classifier-comparison"
}

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
        }
    }
    finally {
        $timer.Stop()
        $timings.Add([pscustomobject]@{
            Dataset = $currentDataset
            Stage = $Stage
            Seconds = [math]::Round($timer.Elapsed.TotalSeconds, 3)
            Minutes = [math]::Round($timer.Elapsed.TotalMinutes, 3)
            Command = "$Python $($Arguments -join ' ')"
        })
        Write-Host ("TIMING {0} / {1}: {2:N2} minutes" -f $currentDataset, $Stage, $timer.Elapsed.TotalMinutes)
    }
}

foreach ($datasetConfig in $DatasetConfigs) {
    $currentDataset = $datasetConfig
    $env:EEG_DATASET_CONFIG = $datasetConfig
    $env:EEG_STEP_TO_START = "1"
    if (-not $FiguresOnly) {
        Write-Host "`n========================================================================"
        Write-Host "ANALYSIS: $datasetConfig"
        Write-Host "========================================================================"
        Invoke-PythonChecked "analysis_full" "main.py"
    }

    if ($AnalysisOnly) {
        continue
    }

    if (-not $FiguresOnly) {
        Write-Host "`n========================================================================"
        Write-Host "CLASSIFIER-PROBABILITY OPTIMIZATION: $datasetConfig"
        Write-Host "========================================================================"
        Invoke-PythonChecked "optimization_full" "run_optimization.py"
    }

    # Saved-result figures applicable to the one-dimensional probability
    # objective. The target script's historical filename says 3d, but its
    # implementation produces the requested top-down 2D maps.
    Invoke-PythonChecked "audit_optimization_completeness" "audit_optimization_completeness.py" "--dataset-config" $datasetConfig
    Invoke-PythonChecked "figures_optimization_overview_and_statistics" "regenerate_classifier_optimization_figures.py" "--dataset-config" $datasetConfig
    Invoke-PythonChecked "figure_best_closeness" "plot_best_closeness_per_subject.py"
    Invoke-PythonChecked "figure_weighted_node_band" "plot_weighted_node_band_interactive.py"
    Invoke-PythonChecked "figure_subject_example" "plot_subject_activation_and_adjacency.py" "--subject" "__first__"
    Invoke-PythonChecked "figure_fixed_pca_projection" "plot_classifier_feature_projection.py" "--dataset-config" $datasetConfig
    Invoke-PythonChecked "figure_band_stability" "plot_band_stability_analysis.py" "--bootstrap-resamples" "10000"
    Invoke-PythonChecked "figures_validity_weighted_targets" "plot_weighted_selection_target_3d.py" "--dataset-config" $datasetConfig
}

if (-not $AnalysisOnly -and $DatasetConfigs.Count -gt 1) {
    $currentDataset = "cross_dataset"
    $comparisonArguments = @("plot_top_selected_nodes.py")
    foreach ($datasetConfig in $DatasetConfigs) {
        $comparisonArguments += @("--dataset-config", $datasetConfig)
    }
    $comparisonArguments += @("--output-dir", $effectiveComparisonOutputDir)
    Invoke-PythonChecked "figure_cross_dataset_targets" @comparisonArguments
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$timingRoot = if ($RunOutputDir) {
    if ([System.IO.Path]::IsPathRooted($RunOutputDir)) {
        $RunOutputDir
    } else {
        Join-Path $repo $RunOutputDir
    }
} else {
    $repo
}
New-Item -ItemType Directory -Path $timingRoot -Force | Out-Null
$timingPath = Join-Path $timingRoot "pipeline_timings_$timestamp.csv"
$timings | Export-Csv -LiteralPath $timingPath -NoTypeInformation -Encoding UTF8
Write-Host "`nComplete. Each profile contains analysis, band classifiers, optimization, and applicable figures."
Write-Host "Runner timings: $timingPath"
