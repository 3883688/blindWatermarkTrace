param(
    [int]$Workers = 12,
    [int]$SyntheticVariants = 30,
    [string]$NegativeAttacks = "jpeg_q90,jpeg_q50,jpeg_q30,rotate_3deg,rotate_10deg,browser_screenshot_sim,wechat_screenshot_sim,screen_photo_sim,gaussian_blur_1_2,median_denoise"
)

$ErrorActionPreference = "Stop"

$previousWorkers = $env:BENCHMARK_WORKERS
$previousSyntheticVariants = $env:SYNTHETIC_VARIANTS
$previousNegativeAttacks = $env:NEGATIVE_ATTACKS

try {
    $env:BENCHMARK_WORKERS = [string]$Workers
    $env:SYNTHETIC_VARIANTS = [string]$SyntheticVariants
    $env:NEGATIVE_ATTACKS = $NegativeAttacks

    python tests\commercial_negative_benchmark.py
    exit $LASTEXITCODE
}
finally {
    if ($null -eq $previousWorkers) {
        Remove-Item Env:\BENCHMARK_WORKERS -ErrorAction SilentlyContinue
    } else {
        $env:BENCHMARK_WORKERS = $previousWorkers
    }

    if ($null -eq $previousSyntheticVariants) {
        Remove-Item Env:\SYNTHETIC_VARIANTS -ErrorAction SilentlyContinue
    } else {
        $env:SYNTHETIC_VARIANTS = $previousSyntheticVariants
    }

    if ($null -eq $previousNegativeAttacks) {
        Remove-Item Env:\NEGATIVE_ATTACKS -ErrorAction SilentlyContinue
    } else {
        $env:NEGATIVE_ATTACKS = $previousNegativeAttacks
    }
}
