param(
    [string]$Step = "all",
    [string]$Device = "cuda",
    [string]$MainDir = "outputs/submission_main_20260711_225137",
    [string]$OutputDir = "",
    [string]$TunedConfigDir = "",
    [switch]$QuickTest,
    [switch]$SkipExisting,
    [switch]$DryRun,
    [int]$MaxParallel = 1,
    [string[]]$Seeds = @("0", "1", "2", "3", "4"),
    [string[]]$Datasets = @()
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "1" }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = "1" }
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "1" }

if (-not $OutputDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = "outputs/full_rerun_$stamp"
}
if (-not $TunedConfigDir) {
    $TunedConfigDir = Join-Path $OutputDir "tuning_hero"
}

function CommonArgs {
    $args = @("--device", $Device, "--output_dir", $OutputDir, "--max_parallel", "$MaxParallel", "--seeds") + $Seeds
    if ($Datasets.Count -gt 0) { $args += @("--datasets") + $Datasets }
    if ($QuickTest) { $args += "--quick_test" }
    if ($SkipExisting) { $args += "--skip_existing" }
    if ($DryRun) { $args += "--dry_run" }
    return $args
}

function Run-Tune {
    $args = @("--device", $Device, "--output_dir", $TunedConfigDir, "--seeds") + $Seeds
    if ($Datasets.Count -gt 0) { $args += @("--datasets") + $Datasets }
    if ($QuickTest) { $args += "--quick_test" }
    if ($SkipExisting) { $args += "--skip_existing" }
    if ($DryRun) { $args += "--dry_run" }
    python scripts/tune_hero_hyperparams.py @args
}

function Run-Suite([string]$Suite) {
    $args = @("--suite", $Suite) + (CommonArgs) + @("--tuned_config_dir", $TunedConfigDir, "--main_dir", $MainDir, "--input_dir", $OutputDir)
    python scripts/run_experiment_suite.py @args
}

function Run-Significance {
    if ($DryRun) {
        Write-Host "[dry-run] python scripts/summarize_experiment_suite.py --output_dir $OutputDir"
        Write-Host "[dry-run] python scripts/run_significance_tests.py --main_dir $OutputDir"
        return
    }
    python scripts/summarize_experiment_suite.py --output_dir $OutputDir
    python scripts/run_significance_tests.py --main_dir $OutputDir
}

function Run-Final {
    $args = @("--suite", "final") + (CommonArgs) + @(
        "--main_dir", $OutputDir,
        "--ablation_dir", $OutputDir,
        "--robustness_dir", $OutputDir,
        "--labeler_dir", $OutputDir,
        "--faithfulness_dir", $OutputDir,
        "--sensitivity_dir", $OutputDir,
        "--cost_dir", $OutputDir
    )
    python scripts/run_experiment_suite.py @args
}

function Run-Step([string]$Name) {
    switch ($Name) {
        "quick_test" {
            $args = @("--suite", "all", "--quick_test", "--output_dir", "$OutputDir/quick_test", "--device", $Device, "--max_parallel", "$MaxParallel")
            if ($SkipExisting) { $args += "--skip_existing" }
            if ($DryRun) { $args += "--dry_run" }
            python scripts/run_experiment_suite.py @args
        }
        { $_ -in @("tune", "tune_hero_hyperparams") } { Run-Tune }
        "main" { Run-Suite "main" }
        "transfer" { Run-Suite "transfer" }
        "ablation" { Run-Suite "ablation" }
        "significance" { Run-Significance }
        "robustness" { Run-Suite "robustness" }
        "labeler_comparison" { Run-Suite "labeler_comparison" }
        "faithfulness" { Run-Suite "faithfulness" }
        "sensitivity" { Run-Suite "sensitivity" }
        "cost" { Run-Suite "cost" }
        "final" { Run-Final }
        default { throw "Unknown step: $Name" }
    }
}

if ($Step -eq "all") {
    foreach ($name in @("quick_test", "tune", "main", "transfer", "ablation", "significance", "robustness", "labeler_comparison", "faithfulness", "sensitivity", "cost", "final")) {
        Run-Step $name
    }
} else {
    Run-Step $Step
}

Write-Host "Pipeline finished. Output root: $OutputDir"
