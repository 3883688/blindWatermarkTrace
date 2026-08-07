param(
    [int]$Workers = 12,
    [int]$TraceRounds = 3,
    [string]$AttackFilter = "jpeg_q30,screen_photo_sim,wechat_screenshot_sim"
)

$ErrorActionPreference = "Stop"

$previousWorkers = $env:BENCHMARK_WORKERS
$previousTraceRounds = $env:TRACE_ROUNDS
$previousAttackFilter = $env:ATTACK_FILTER
$previousBenchmarkLabel = $env:BENCHMARK_LABEL

try {
    $env:BENCHMARK_WORKERS = [string]$Workers
    $env:TRACE_ROUNDS = [string]$TraceRounds
    $env:ATTACK_FILTER = $AttackFilter
    Remove-Item Env:\BENCHMARK_LABEL -ErrorAction SilentlyContinue

    python tests\commercial_attack_benchmark.py
    exit $LASTEXITCODE
}
finally {
    if ($null -eq $previousWorkers) {
        Remove-Item Env:\BENCHMARK_WORKERS -ErrorAction SilentlyContinue
    } else {
        $env:BENCHMARK_WORKERS = $previousWorkers
    }

    if ($null -eq $previousTraceRounds) {
        Remove-Item Env:\TRACE_ROUNDS -ErrorAction SilentlyContinue
    } else {
        $env:TRACE_ROUNDS = $previousTraceRounds
    }

    if ($null -eq $previousAttackFilter) {
        Remove-Item Env:\ATTACK_FILTER -ErrorAction SilentlyContinue
    } else {
        $env:ATTACK_FILTER = $previousAttackFilter
    }

    if ($null -eq $previousBenchmarkLabel) {
        Remove-Item Env:\BENCHMARK_LABEL -ErrorAction SilentlyContinue
    } else {
        $env:BENCHMARK_LABEL = $previousBenchmarkLabel
    }
}
