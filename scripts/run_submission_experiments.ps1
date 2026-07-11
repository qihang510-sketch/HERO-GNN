param(
    [ValidateSet("cuda", "cpu", "auto")]
    [string]$Device = "cuda",
    [int]$MaxParallel = 1,
    [switch]$QuickTest,
    [switch]$SkipExisting,
    [switch]$SummarizeOnly,
    [switch]$RerunMissing,
    [string]$OutputDir = "",
    [string[]]$Datasets = @("yelp_academic", "amazon_video", "fraud_yelp", "fraud_amazon", "elliptic"),
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
if ($RerunMissing) { $SkipExisting = $true }
if ($SummarizeOnly -and [string]::IsNullOrWhiteSpace($OutputDir)) {
    throw "--summarize_only requires -OutputDir"
}
if ($RerunMissing -and [string]::IsNullOrWhiteSpace($OutputDir)) {
    throw "--rerun_missing requires -OutputDir"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "outputs/submission_full_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}
if ((Test-Path $OutputDir) -and -not $SkipExisting -and -not $SummarizeOnly -and -not $RerunMissing) {
    throw "Refusing to reuse existing output dir without -SkipExisting: $OutputDir"
}
if ($QuickTest) {
    $Datasets = @("yelp_academic")
    $Seeds = @(0)
}

New-Item -ItemType Directory -Force -Path $OutputDir, "$OutputDir\raw", "$OutputDir\summary", "$OutputDir\tables", "$OutputDir\figures", "$OutputDir\logs" | Out-Null
$FailedCsv = "$OutputDir\summary\failed_runs.csv"
if (-not (Test-Path $FailedCsv)) {
    "suite,dataset,model,seed,exit_code,command,log_file,error_log" | Set-Content -Path $FailedCsv -Encoding utf8
}

function ConvertTo-SafeName {
    param([string]$Value)
    return ($Value -replace '[^A-Za-z0-9_.=-]+', '_')
}

function Add-FailedRun {
    param(
        [string]$Suite,
        [string]$Dataset,
        [string]$Model,
        [string]$Seed,
        [int]$ExitCode,
        [string]$Command,
        [string]$LogFile,
        [string]$ErrorLog
    )
    $row = [pscustomobject]@{
        suite = $Suite
        dataset = $Dataset
        model = $Model
        seed = $Seed
        exit_code = $ExitCode
        command = $Command
        log_file = $LogFile
        error_log = $ErrorLog
    }
    $row | Export-Csv -Path $FailedCsv -NoTypeInformation -Append -Encoding utf8
}

function Invoke-Step {
    param(
        [string]$Name,
        [string]$Suite,
        [string]$Dataset,
        [string]$Model,
        [string]$Seed,
        [string[]]$Args
    )
    $tag = ConvertTo-SafeName $Name
    $logFile = "$OutputDir\logs\$tag.log"
    $failDir = "$OutputDir\raw\failures\$tag"
    $errorLog = "$failDir\error.log"
    $command = "$Python " + ($Args -join " ")
    Write-Host "[RUN] $Name"
    Write-Host "[CMD] $command"
    & $Python @Args > $logFile 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        New-Item -ItemType Directory -Force -Path $failDir | Out-Null
        Copy-Item -Path $logFile -Destination $errorLog -Force
        Add-FailedRun -Suite $Suite -Dataset $Dataset -Model $Model -Seed $Seed -ExitCode $exitCode -Command $command -LogFile $logFile -ErrorLog $errorLog
        Write-Host "[FAILED] $Name exit=$exitCode log=$errorLog"
    } else {
        Write-Host "[OK] $Name"
    }
}

function Write-ExpectedConfig {
    $mainMatrix = @{
        yelp_academic = @("mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "dgp", "mled", "hero_gnn")
        amazon_video = @("mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "dgp", "mled", "hero_gnn")
        fraud_yelp = @("mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "hero_official")
        fraud_amazon = @("mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "hero_official")
        elliptic = @("mlp", "gcn", "gat", "graphsage", "bwgnn", "linkx", "hogrl", "rgtan", "hero_official")
    }
    $ablations = @("hero_gnn", "wo_risk_relevant_heterophily", "wo_mechanism_annotation", "wo_evidence_chain", "wo_llm_annotation", "wo_heterophily_filter", "wo_dual_branch_encoder", "wo_gated_fusion")
    $expected = @()
    if ($QuickTest) {
        foreach ($model in @("mlp", "gcn", "hero_gnn")) {
            $expected += [ordered]@{ suite = "main"; dataset = "yelp_academic"; model = $model; seed = 0 }
        }
    } else {
        foreach ($dataset in $Datasets) {
            foreach ($model in $mainMatrix[$dataset]) {
                foreach ($seed in $Seeds) {
                    $expected += [ordered]@{ suite = "main"; dataset = $dataset; model = $model; seed = [int]$seed }
                }
            }
        }
        $textDatasets = @($Datasets | Where-Object { $_ -in @("yelp_academic", "amazon_video") })
        foreach ($dataset in $textDatasets) {
            foreach ($model in $ablations) {
                foreach ($seed in $Seeds) {
                    $expected += [ordered]@{ suite = "ablation"; dataset = $dataset; model = $model; seed = [int]$seed }
                }
            }
            foreach ($suite in @("robustness", "labeler_comparison", "faithfulness")) {
                foreach ($seed in $Seeds) {
                    $expected += [ordered]@{ suite = $suite; dataset = $dataset; model = "hero_gnn"; seed = [int]$seed }
                }
            }
        }
    }
    $payload = [ordered]@{
        suite = "submission_full"
        created_at = (Get-Date).ToUniversalTime().ToString("o")
        datasets = $Datasets
        seeds = $Seeds
        quick_test = [bool]$QuickTest
        expected_runs = $expected
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -Path "$OutputDir\config.json" -Encoding utf8
}

function Invoke-MaterializeFailures {
    $code = @'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
failed_csv = root / "summary" / "failed_runs.csv"
failed_csv.parent.mkdir(parents=True, exist_ok=True)
existing = failed_csv.exists()
with failed_csv.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["suite", "dataset", "model", "seed", "exit_code", "command", "log_file", "error_log"])
    if not existing:
        writer.writeheader()
    for path in sorted((root / "raw").rglob("metrics.json")):
        if any(part.startswith("_project_runs") for part in path.parts):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        status = str(payload.get("status", "ok"))
        reason = str(payload.get("skip_reason", payload.get("reason", "")))
        failed = status in {"failed", "missing", "skipped"} or "experiment_failed" in reason
        if not failed:
            continue
        run_dir = path.parent
        error_log = run_dir / "error.log"
        if not error_log.exists():
            parts = [f"status={status}", f"reason={reason}", f"metrics={path}"]
            for name in ["log.txt", "run.log"]:
                log = run_dir / name
                if log.exists():
                    parts.append(log.read_text(encoding="utf-8", errors="ignore"))
            error_log.write_text("\n".join(parts) + "\n", encoding="utf-8")
        writer.writerow({
            "suite": payload.get("suite", ""),
            "dataset": payload.get("dataset", ""),
            "model": payload.get("model", payload.get("variant", "")),
            "seed": payload.get("seed", ""),
            "exit_code": "",
            "command": "",
            "log_file": str(run_dir / "log.txt") if (run_dir / "log.txt").exists() else "",
            "error_log": str(error_log),
        })
'@
    $tmp = "$OutputDir\logs\materialize_failures.py"
    $code | Set-Content -Path $tmp -Encoding utf8
    & $Python $tmp $OutputDir
}

function Invoke-EnsureRequiredTables {
    $required = @(
        "all_raw_runs.csv",
        "table_main_mean_std.csv",
        "table_transfer_mean_std.csv",
        "table_ablation_mean_std.csv",
        "table_significance.csv",
        "table_llm_robustness.csv",
        "table_llm_labeler_comparison.csv",
        "table_faithfulness.csv",
        "table_cost_scalability.csv",
        "table_missing_runs.csv"
    )
    foreach ($name in $required) {
        $path = "$OutputDir\summary\$name"
        if (-not (Test-Path $path)) {
            "status,reason`nmissing,source_table_not_generated" | Set-Content -Path $path -Encoding utf8
        }
    }
}

function Invoke-Postprocess {
    Write-ExpectedConfig
    Invoke-Step "check_completeness" "postprocess" "all" "all" "all" @("scripts\check_experiment_completeness.py", "--output_dir", $OutputDir, "--generate_commands")
    Invoke-Step "summarize_suite" "postprocess" "all" "all" "all" @("scripts\summarize_experiment_suite.py", "--output_dir", $OutputDir)
    Invoke-Step "significance_tests" "postprocess" "all" "all" "all" @("scripts\run_significance_tests.py", "--input_dir", $OutputDir, "--output_dir", "$OutputDir\summary")
    Invoke-Step "cost_scalability" "postprocess" "all" "hero" "all" (@("scripts\collect_cost_scalability.py", "--input_dir", $OutputDir, "--output_dir", $OutputDir, "--datasets") + $Datasets)
    Invoke-Step "summarize_suite_final" "postprocess" "all" "all" "all" @("scripts\summarize_experiment_suite.py", "--output_dir", $OutputDir)
    Invoke-MaterializeFailures
    Invoke-EnsureRequiredTables
}

Write-Host "Output directory: $OutputDir"
Write-Host "Device: $Device"
Write-Host "Datasets: $($Datasets -join ' ')"
Write-Host "Seeds: $($Seeds -join ' ')"
Write-Host "Max parallel requested: $MaxParallel"

if ($SummarizeOnly) {
    Invoke-Postprocess
    Write-Host "Summary-only finished: $OutputDir\summary"
    exit 0
}

Write-ExpectedConfig
$skipArgs = @()
if ($SkipExisting) { $skipArgs = @("--skip_existing") }

if ($QuickTest) {
    Invoke-Step "main_quick" "main" "yelp_academic" "mlp_gcn_hero" "0" (@("scripts\run_experiment_suite.py", "--suite", "main", "--datasets", "yelp_academic", "--models", "mlp", "gcn", "hero", "--seeds", "0", "--output_dir", $OutputDir, "--device", $Device) + $skipArgs)
    Invoke-Postprocess
    Write-Host "Quick test finished: $OutputDir"
    exit 0
}

$textDatasets = @($Datasets | Where-Object { $_ -in @("yelp_academic", "amazon_video") })
Invoke-Step "main_all" "main" "all" "all" "all" (@("scripts\run_experiment_suite.py", "--suite", "main", "--datasets") + $Datasets + @("--seeds") + [string[]]$Seeds + @("--output_dir", $OutputDir, "--device", $Device) + $skipArgs)

if ($textDatasets.Count -gt 0) {
    Invoke-Step "ablation_all" "ablation" "text_rich" "hero_ablations" "all" (@("scripts\run_experiment_suite.py", "--suite", "ablation", "--datasets") + $textDatasets + @("--seeds") + [string[]]$Seeds + @("--output_dir", $OutputDir, "--device", $Device) + $skipArgs)
    Invoke-Step "llm_robustness" "robustness" "text_rich" "hero_gnn" "all" (@("scripts\run_experiment_suite.py", "--suite", "robustness", "--datasets") + $textDatasets + @("--seeds") + [string[]]$Seeds + @("--noise_types", "relevance_flip", "mechanism_shuffle", "confidence_gaussian", "--noise_ratios", "0.1", "0.2", "0.3", "0.4", "--output_dir", $OutputDir, "--device", $Device) + $skipArgs)
    Invoke-Step "labeler_comparison" "labeler_comparison" "text_rich" "hero_gnn" "all" (@("scripts\run_experiment_suite.py", "--suite", "labeler_comparison", "--datasets") + $textDatasets + @("--seeds") + [string[]]$Seeds + @("--output_dir", $OutputDir, "--device", $Device) + $skipArgs)
    Invoke-Step "faithfulness" "faithfulness" "text_rich" "hero_gnn" "all" (@("scripts\run_experiment_suite.py", "--suite", "faithfulness", "--datasets") + $textDatasets + @("--seeds") + [string[]]$Seeds + @("--input_dir", $OutputDir, "--output_dir", $OutputDir, "--device", $Device))
}

Invoke-Step "cost_collection" "cost" "all" "hero" "all" (@("scripts\run_experiment_suite.py", "--suite", "cost", "--datasets") + $Datasets + @("--seeds") + [string[]]$Seeds + @("--input_dir", $OutputDir, "--output_dir", $OutputDir, "--device", $Device))

Invoke-Postprocess
Write-Host "Submission experiment run finished: $OutputDir"
Write-Host "Final tables: $OutputDir\summary"
