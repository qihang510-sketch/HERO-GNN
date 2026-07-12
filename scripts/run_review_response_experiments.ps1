$Device = "cuda"
$MainDir = "outputs/submission_main_20260711_225137"
$SkipExisting = $false
$QuickTest = $false
$RunRobustness = $false
$RunLabeler = $false
$RunFaithfulness = $false
$RunSensitivity = $false
$RunAll = $false
$BuildFinal = $false

for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "--quick_test" { $QuickTest = $true }
        "--device" { $i++; $Device = $args[$i] }
        "--skip_existing" { $SkipExisting = $true }
        "--main_dir" { $i++; $MainDir = $args[$i] }
        "--run_robustness" { $RunRobustness = $true }
        "--run_labeler" { $RunLabeler = $true }
        "--run_faithfulness" { $RunFaithfulness = $true }
        "--run_sensitivity" { $RunSensitivity = $true }
        "--run_all" { $RunAll = $true }
        "--build_final" { $BuildFinal = $true }
        default {
            Write-Host "Unknown argument: $($args[$i])"
            exit 2
        }
    }
}

if ($RunAll) {
    $RunRobustness = $true
    $RunLabeler = $true
    $RunFaithfulness = $true
    $RunSensitivity = $true
    $BuildFinal = $true
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = "outputs/review_response_$Stamp"
if ($QuickTest) {
    $Root = "outputs/review_response_quick_$Stamp"
}
New-Item -ItemType Directory -Force -Path "$Root/logs", "$Root/summary" | Out-Null
$Failed = "$Root/summary/failed_runs.csv"
"stage,exit_code,command" | Set-Content -Encoding UTF8 $Failed

function Run-Step {
    param(
        [string]$Stage,
        [string[]]$Command
    )
    Write-Host "[$Stage] $($Command -join ' ')"
    $Log = "$Root/logs/$Stage.log"
    & $Command[0] @($Command[1..($Command.Count - 1)]) *> $Log
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        "$Stage,$Code,""$($Command -join ' ')""" | Add-Content -Encoding UTF8 $Failed
        Write-Host "[failed] $Stage exit_code=$Code (see $Log)"
    } else {
        Write-Host "[ok] $Stage"
    }
}

$Datasets = @("yelp_academic", "amazon_video")
$Seeds = @("0", "1", "2", "3", "4")
$NoiseRatios = @("0.0", "0.1", "0.2", "0.3", "0.4")
$Topks = @("1", "3", "5")
if ($QuickTest) {
    $Datasets = @("yelp_academic")
    $Seeds = @("0")
    $NoiseRatios = @("0.0", "0.1")
    $Topks = @("1")
}

$SkipArgs = @()
if ($SkipExisting) {
    $SkipArgs = @("--skip_existing")
}

if (Test-Path $MainDir) {
    Run-Step "summarize_main" @("python", "scripts/summarize_experiment_suite.py", "--output_dir", $MainDir)
    Run-Step "significance" @("python", "scripts/run_significance_tests.py", "--main_dir", $MainDir)
} else {
    "main_dir_missing,0,""$MainDir""" | Add-Content -Encoding UTF8 $Failed
    Write-Host "[missing] main_dir=$MainDir"
}

$RobustnessDir = "$Root/robustness"
$LabelerDir = "$Root/labeler_comparison"
$FaithfulnessDir = "$Root/faithfulness"
$SensitivityDir = "$Root/sensitivity"
$CostDir = "$Root/cost"

if ($RunRobustness) {
    $Cmd = @("python", "scripts/run_llm_annotation_robustness.py", "--datasets") + $Datasets + @("--seeds") + $Seeds + @("--noise_ratios") + $NoiseRatios + @("--output_dir", $RobustnessDir, "--device", $Device) + $SkipArgs
    Run-Step "robustness" $Cmd
}
if ($RunLabeler) {
    $Cmd = @("python", "scripts/run_llm_labeler_comparison.py", "--datasets") + $Datasets + @("--seeds") + $Seeds + @("--output_dir", $LabelerDir, "--device", $Device) + $SkipArgs
    Run-Step "labeler" $Cmd
}
if ($RunFaithfulness) {
    $Cmd = @("python", "scripts/run_evidence_faithfulness.py", "--datasets") + $Datasets + @("--seeds") + $Seeds + @("--topks") + $Topks + @("--input_dir", $MainDir, "--output_dir", $FaithfulnessDir, "--device", $Device)
    Run-Step "faithfulness" $Cmd
}
if ($RunSensitivity) {
    $SensArgs = @()
    if ($QuickTest) {
        $SensArgs = @("--quick_test")
    }
    $Cmd = @("python", "scripts/run_sensitivity_analysis.py", "--datasets") + $Datasets + @("--seeds") + $Seeds + @("--output_dir", $SensitivityDir, "--device", $Device) + $SkipArgs + $SensArgs
    Run-Step "sensitivity" $Cmd
}

$Cmd = @("python", "scripts/collect_cost_scalability.py", "--input_dir", $MainDir, "--output_dir", $CostDir, "--datasets") + $Datasets
Run-Step "cost" $Cmd

if ($BuildFinal) {
    Run-Step "final_artifacts" @("python", "scripts/build_final_tables_and_figures.py", "--main_dir", $MainDir, "--robustness_dir", $RobustnessDir, "--labeler_dir", $LabelerDir, "--faithfulness_dir", $FaithfulnessDir, "--cost_dir", $CostDir, "--output_dir", "outputs/final_artifacts", "--allow_missing")
    Run-Step "seed_stability" @("python", "scripts/plot_seed_stability.py", "--input_csv", "$MainDir/summary/all_raw_runs.csv", "--output_dir", "outputs/final_artifacts")
}

Write-Host "Review-response experiment driver finished."
Write-Host "Run root: $Root"
Write-Host "Final artifacts: outputs/final_artifacts"
