param(
    [string[]]$Stages = @(
        "quick_test",
        "tune_hero",
        "main_text_rich",
        "transfer",
        "ablation",
        "significance",
        "robustness",
        "labeler_comparison",
        "faithfulness",
        "sensitivity",
        "cost",
        "seed_stability",
        "evidence_cases",
        "final_artifacts"
    ),
    [string]$Device = "cuda",
    [string[]]$Seeds = @("0", "1", "2", "3", "4"),
    [string]$OutputRoot = "",
    [string]$TunedConfigDir = "",
    [string]$DataRoot = "data",
    [int]$MaxParallel = 1,
    [string[]]$Datasets = @(),
    [string[]]$Models = @(),
    [string[]]$Variants = @(),
    [switch]$SkipExisting,
    [switch]$RerunFailed,
    [switch]$OnlyMissing,
    [switch]$ContinueOnError,
    [switch]$DryRun,
    [switch]$GenerateMissingCommands
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "1" }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = "1" }
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "1" }

if (-not $OutputRoot) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = "outputs/review_rerun_$stamp"
}
if (-not $TunedConfigDir) {
    $TunedConfigDir = Join-Path $OutputRoot "tuning"
}
if ($OnlyMissing) { $SkipExisting = $true }
if ($RerunFailed) { $SkipExisting = $false }

function Stage-Dir([string]$Stage) {
    switch ($Stage) {
        "quick_test" { return (Join-Path $OutputRoot "quick_test") }
        "tune_hero" { return (Join-Path $OutputRoot "tuning") }
        "main_text_rich" { return (Join-Path $OutputRoot "main") }
        "transfer" { return (Join-Path $OutputRoot "transfer") }
        "ablation" { return (Join-Path $OutputRoot "ablation") }
        "significance" { return (Join-Path $OutputRoot "significance") }
        "robustness" { return (Join-Path $OutputRoot "robustness") }
        "labeler_comparison" { return (Join-Path $OutputRoot "labeler_comparison") }
        "faithfulness" { return (Join-Path $OutputRoot "faithfulness") }
        "sensitivity" { return (Join-Path $OutputRoot "sensitivity") }
        "cost" { return (Join-Path $OutputRoot "cost") }
        default { return (Join-Path $OutputRoot "final_artifacts") }
    }
}

function Common-Args([string]$OutputDir) {
    $args = @("--device", $Device, "--data_root", $DataRoot, "--max_parallel", "$MaxParallel", "--output_dir", $OutputDir, "--seeds") + $Seeds
    if ($Datasets.Count -gt 0) { $args += @("--datasets") + $Datasets }
    if ($Models.Count -gt 0) { $args += @("--models") + $Models }
    if ($Variants.Count -gt 0) { $args += @("--variants") + $Variants }
    if ($SkipExisting) { $args += "--skip_existing" }
    if ($DryRun) { $args += "--dry_run" }
    if ($ContinueOnError) { $args += "--continue_on_error" }
    return $args
}

function Run-Suite([string]$Stage, [string]$Suite) {
    $out = Stage-Dir $Stage
    if ($GenerateMissingCommands) {
        python scripts/check_experiment_completeness.py --output_dir $out --suite $Suite --generate_commands
        return
    }
    $args = @("--suite", $Suite) + (Common-Args $out) + @("--tuned_config_dir", $TunedConfigDir, "--input_dir", $OutputRoot, "--main_dir", (Join-Path $OutputRoot "main"))
    python scripts/run_experiment_suite.py @args
}

function Run-TuneHero {
    $out = Stage-Dir "tune_hero"
    $args = @("--device", $Device, "--data_root", $DataRoot, "--output_dir", $out, "--seeds") + $Seeds
    if ($Datasets.Count -gt 0) { $args += @("--datasets") + $Datasets }
    if ($SkipExisting) { $args += "--skip_existing" }
    if ($DryRun) { $args += "--dry_run" }
    python scripts/tune_hero_hyperparams.py @args
}

function Run-Significance {
    $out = Stage-Dir "significance"
    if ($DryRun) {
        Write-Host "[dry-run] summarize main/transfer and build significance in $out"
        return
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $out "summary") | Out-Null
    if (Test-Path (Join-Path $OutputRoot "main")) { python scripts/summarize_experiment_suite.py --output_dir (Join-Path $OutputRoot "main") }
    if (Test-Path (Join-Path $OutputRoot "transfer")) { python scripts/summarize_experiment_suite.py --output_dir (Join-Path $OutputRoot "transfer") }
    python -c "from pathlib import Path; import pandas as pd, sys; out=Path(sys.argv[1]); frames=[]; [frames.append(pd.read_csv(p)) for p in map(Path, sys.argv[2:]) if p.exists()]; out.mkdir(parents=True, exist_ok=True); (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(out/'all_raw_runs.csv', index=False)" (Join-Path $out "summary") (Join-Path $OutputRoot "main/summary/all_raw_runs.csv") (Join-Path $OutputRoot "transfer/summary/all_raw_runs.csv")
    python scripts/run_significance_tests.py --main_dir $out
}

function Run-FinalArtifacts {
    $out = Stage-Dir "final_artifacts"
    $args = @(
        "--main_dir", (Join-Path $OutputRoot "main"),
        "--transfer_dir", (Join-Path $OutputRoot "transfer"),
        "--significance_dir", (Join-Path $OutputRoot "significance"),
        "--ablation_dir", (Join-Path $OutputRoot "ablation"),
        "--robustness_dir", (Join-Path $OutputRoot "robustness"),
        "--labeler_dir", (Join-Path $OutputRoot "labeler_comparison"),
        "--faithfulness_dir", (Join-Path $OutputRoot "faithfulness"),
        "--sensitivity_dir", (Join-Path $OutputRoot "sensitivity"),
        "--cost_dir", (Join-Path $OutputRoot "cost"),
        "--output_dir", $out,
        "--data_root", $DataRoot,
        "--allow_missing"
    )
    python scripts/build_final_tables_and_figures.py @args
    python scripts/check_final_artifacts.py --output_root $OutputRoot --final_dir $out --allow_missing
}

foreach ($stage in $Stages) {
    Write-Host "========== stage: $stage =========="
    switch ($stage) {
        "quick_test" {
            $args = @("--suite", "all", "--quick_test") + (Common-Args (Stage-Dir "quick_test"))
            python scripts/run_experiment_suite.py @args
        }
        "tune_hero" { Run-TuneHero }
        "main_text_rich" { Run-Suite "main_text_rich" "main" }
        "transfer" { Run-Suite "transfer" "transfer" }
        "ablation" { Run-Suite "ablation" "ablation" }
        "significance" { Run-Significance }
        "robustness" { Run-Suite "robustness" "robustness" }
        "labeler_comparison" { Run-Suite "labeler_comparison" "labeler_comparison" }
        "faithfulness" { Run-Suite "faithfulness" "faithfulness" }
        "sensitivity" { Run-Suite "sensitivity" "sensitivity" }
        "cost" { Run-Suite "cost" "cost" }
        "seed_stability" { python scripts/plot_seed_stability.py --input_csv (Join-Path $OutputRoot "main/summary/all_raw_runs.csv") --output_dir (Stage-Dir "seed_stability") }
        "evidence_cases" { python scripts/build_evidence_case_table.py --input_dir (Join-Path $OutputRoot "faithfulness") --output_dir (Stage-Dir "evidence_cases") }
        "final_artifacts" { Run-FinalArtifacts }
        default { throw "Unknown stage: $stage" }
    }
}

Write-Host "All requested review experiment stages finished."
Write-Host "Final artifacts: $(Join-Path $OutputRoot 'final_artifacts')"
