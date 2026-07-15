param(
    [ValidateSet('quality', 'crop', 'attack', 'negative', 'balanced')]
    [string]$Stage = 'quality',
    [ValidateRange(1, 24)][int]$Workers = 20,
    [ValidateRange(1, 20)][int]$TraceRounds = 5,
    [ValidateRange(1, 10000)][int]$NegativeVariants = 1000,
    [switch]$Reuse,
    [switch]$ConfirmLongRun
)

$ErrorActionPreference = 'Stop'
$originalLocation = Get-Location
$root = $PSScriptRoot
$qualityJson = Join-Path $root 'test_output\commercial_quality_benchmark\commercial_quality_results.json'
$cropJson = Join-Path $root 'test_output\commercial_trace_benchmark\commercial_trace_results.json'
$attackJson = Join-Path $root 'test_output\commercial_stability_benchmark\commercial_attack_results.json'
$negativeJson = Join-Path $root 'test_output\commercial_negative_benchmark\commercial_negative_results.json'
$managedEnvironment = @(
    'BENCHMARK_WORKERS', 'TRACE_ROUNDS', 'SYNTHETIC_VARIANTS', 'FIDELITY_LEVEL',
    'BENCHMARK_LABEL', 'CROPS_PER_RATIO', 'SCALE_FACTORS', 'CROP_RATIOS',
    'NEGATIVE_SCALE_FACTORS', 'NEGATIVE_CROP_RATIOS', 'SMALL_CROP_TRACE_STRENGTH',
    'SMALL_CROP_TRACE_DENSITY', 'ROBUST_WATERMARK_STRENGTH', 'ROBUST_WATERMARK_VERSION',
    'WATERMARK_AUTH_KEY', 'QUALITY_PROBE_FILTER', 'PROBE_MIN_RECALL',
    'QUALITY_MIN_SSIM', 'QUALITY_MIN_PSNR', 'FIDELITY_LEVELS', 'ATTACK_FILTER',
    'NEGATIVE_SOURCES'
)
$previousEnvironment = @{}
foreach ($name in $managedEnvironment) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json
}

function Test-JsonObject($Value) {
    return (
        $null -ne $Value -and
        ($Value -is [System.Management.Automation.PSCustomObject] -or
         $Value -is [System.Collections.IDictionary])
    )
}

function Test-SchemaVersionOne($Value) {
    if ($null -eq $Value) { return $false }
    $integralTypes = @(
        [System.SByte], [System.Byte], [System.Int16], [System.UInt16],
        [System.Int32], [System.UInt32], [System.Int64], [System.UInt64]
    )
    return $integralTypes -contains $Value.GetType() -and $Value -eq 1
}

function Test-ObjectHasProperty($Object, [string]$Name) {
    if ($Object -is [System.Collections.IDictionary]) { return $Object.Contains($Name) }
    return $null -ne $Object.PSObject.Properties[$Name]
}

function Test-IntegralValue($Value) {
    if ($null -eq $Value) { return $false }
    $integralTypes = @(
        [System.SByte], [System.Byte], [System.Int16], [System.UInt16],
        [System.Int32], [System.UInt32], [System.Int64], [System.UInt64]
    )
    return $integralTypes -contains $Value.GetType()
}

function Test-NonemptyString($Value) {
    return $Value -is [string] -and -not [string]::IsNullOrWhiteSpace($Value)
}

function ConvertFrom-CanonicalUtcTimestamp($Value) {
    if ($Value -isnot [string]) { return $null }
    $parsed = [DateTime]::MinValue
    $styles = [Globalization.DateTimeStyles]::AssumeUniversal -bor [Globalization.DateTimeStyles]::AdjustToUniversal
    if (-not [DateTime]::TryParseExact(
        $Value,
        'yyyy-MM-ddTHH:mm:ssZ',
        [Globalization.CultureInfo]::InvariantCulture,
        $styles,
        [ref]$parsed
    )) { return $null }
    return $parsed
}

function Assert-CommercialReportValue(
    $Report,
    [string]$StageName,
    [string]$ExpectedBenchmark = ''
) {
    $report = $Report
    if (-not (Test-JsonObject $report)) {
        throw "[$StageName] commercial report is invalid: report must be a JSON object"
    }
    foreach ($field in @('metadata', 'summary', 'cases', 'settings', 'verdict', 'failed_gates')) {
        if (-not (Test-ObjectHasProperty $report $field)) {
            throw "[$StageName] commercial report is invalid: missing $field"
        }
    }
    if (-not (Test-JsonObject $report.metadata)) {
        throw "[$StageName] commercial report is invalid: metadata must be a JSON object"
    }
    foreach ($field in @('schema_version', 'benchmark', 'algorithm_version', 'seed', 'generated_at', 'python_version', 'platform', 'configuration')) {
        if (-not (Test-ObjectHasProperty $report.metadata $field)) {
            throw "[$StageName] commercial report is invalid: missing metadata.$field"
        }
    }
    if (-not (Test-SchemaVersionOne $report.metadata.schema_version)) {
        throw "[$StageName] commercial report is invalid: invalid schema_version"
    }
    foreach ($field in @('benchmark', 'algorithm_version', 'generated_at', 'python_version', 'platform')) {
        if (-not (Test-NonemptyString $report.metadata.$field)) {
            throw "[$StageName] commercial report is invalid: metadata.$field must be a nonempty string"
        }
    }
    if ($null -eq (ConvertFrom-CanonicalUtcTimestamp $report.metadata.generated_at)) {
        throw "[$StageName] commercial report is invalid: metadata.generated_at must be a canonical UTC timestamp"
    }
    if (-not (Test-IntegralValue $report.metadata.seed)) {
        throw "[$StageName] commercial report is invalid: metadata.seed must be an integer"
    }
    if (-not (Test-JsonObject $report.metadata.configuration)) {
        throw "[$StageName] commercial report is invalid: metadata.configuration must be a JSON object"
    }
    $configurationAllowlist = @(
        'FIDELITY_LEVEL', 'SMALL_CROP_TRACE_STRENGTH', 'SMALL_CROP_TRACE_DENSITY',
        'ROBUST_WATERMARK_STRENGTH', 'ROBUST_WATERMARK_VERSION', 'TRACE_ROUNDS',
        'SCALE_FACTORS', 'CROP_RATIOS', 'DETECTION_WORKERS',
        'WATERMARK_DETECTION_BUDGET_SECONDS'
    )
    $configurationProperties = if ($report.metadata.configuration -is [System.Collections.IDictionary]) {
        @($report.metadata.configuration.GetEnumerator() | ForEach-Object {
            [pscustomobject]@{ Name = $_.Key; Value = $_.Value }
        })
    } else {
        @($report.metadata.configuration.PSObject.Properties)
    }
    foreach ($property in $configurationProperties) {
        if ($property.Name -isnot [string] -or $configurationAllowlist -cnotcontains $property.Name) {
            throw "[$StageName] commercial report is invalid: metadata.configuration contains unknown key"
        }
        if ($property.Value -isnot [string]) {
            throw "[$StageName] commercial report is invalid: metadata.configuration value must be a string"
        }
    }
    if (-not (Test-JsonObject $report.summary)) {
        throw "[$StageName] commercial report is invalid: summary must be a JSON object"
    }
    if ($null -eq $report.cases -or $report.cases -isnot [System.Array]) {
        throw "[$StageName] commercial report is invalid: cases must be a JSON array"
    }
    if (-not (Test-JsonObject $report.settings)) {
        throw "[$StageName] commercial report is invalid: settings must be a JSON object"
    }
    if ($report.verdict -isnot [string] -or ($report.verdict -cne 'PASS' -and $report.verdict -cne 'FAIL')) {
        throw "[$StageName] commercial report is invalid: verdict must be PASS or FAIL"
    }
    if ($null -eq $report.failed_gates -or $report.failed_gates -isnot [System.Array]) {
        throw "[$StageName] commercial report is invalid: failed_gates must be a JSON array of strings"
    }
    foreach ($gate in $report.failed_gates) {
        if ($gate -isnot [string]) {
            throw "[$StageName] commercial report is invalid: failed_gates must be a JSON array of strings"
        }
    }
    if ($report.verdict -ceq 'PASS' -and $report.failed_gates.Count -ne 0) {
        throw "[$StageName] commercial report is invalid: PASS verdict requires no failed gates"
    }
    if ($report.verdict -ceq 'FAIL' -and $report.failed_gates.Count -eq 0) {
        throw "[$StageName] commercial report is invalid: FAIL verdict requires at least one failed gate"
    }
    if (
        -not [string]::IsNullOrEmpty($ExpectedBenchmark) -and
        ($report.metadata.benchmark -isnot [string] -or
         $report.metadata.benchmark -cne $ExpectedBenchmark)
    ) {
        throw "[$StageName] commercial report is invalid: benchmark must be $ExpectedBenchmark"
    }
    return $report
}

function Assert-CommercialReport(
    [string]$StageName,
    [string]$Path,
    [string]$ExpectedBenchmark = ''
) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "[$StageName] commercial report is missing: $Path"
    }
    $snapshot = Get-StageReportSnapshot $Path
    return Assert-CommercialReportValue $snapshot.Report $StageName $ExpectedBenchmark
}

function Get-EnvironmentValue([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value
}

function ConvertTo-NumberList([string]$Value) {
    return ,@($Value.Split(',') | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { [double]$_.Trim() })
}

function ConvertTo-StringList([string]$Value) {
    return ,@($Value.Split(',') | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.Trim() })
}

function Get-ExpectedBenchmarkSeed {
    $value = Get-EnvironmentValue 'RANDOM_SEED' '20260707'
    $seed = 0L
    if (-not [Int64]::TryParse($value, [ref]$seed)) {
        throw "RANDOM_SEED must be an integer"
    }
    return $seed
}

function Get-StageExpectedSettings([string]$StageName, [string]$Fidelity) {
    switch ($StageName) {
        'quality' {
            return [ordered]@{
                fidelity_levels = ConvertTo-NumberList (Get-EnvironmentValue 'FIDELITY_LEVELS' '0.85,0.90,0.95,1.0')
                quality_min_psnr = [double](Get-EnvironmentValue 'QUALITY_MIN_PSNR' '38.0')
                quality_min_ssim = [double](Get-EnvironmentValue 'QUALITY_MIN_SSIM' '0.95')
                probe_min_recall = [double](Get-EnvironmentValue 'PROBE_MIN_RECALL' '1.0')
                small_crop_trace_strength = Get-EnvironmentValue 'SMALL_CROP_TRACE_STRENGTH' '0.35'
                small_crop_trace_density = Get-EnvironmentValue 'SMALL_CROP_TRACE_DENSITY' 'medium'
                robust_watermark_strength = Get-EnvironmentValue 'ROBUST_WATERMARK_STRENGTH' '0.74'
                robust_watermark_version = Get-EnvironmentValue 'ROBUST_WATERMARK_VERSION' '3'
                probes = ConvertTo-StringList (Get-EnvironmentValue 'QUALITY_PROBE_FILTER' 'intact')
            }
        }
        'crop' {
            $scaleFactors = ConvertTo-NumberList (Get-EnvironmentValue 'SCALE_FACTORS' '0.5,0.75,1.0,1.25,1.5,2.0')
            $cropRatios = ConvertTo-NumberList (Get-EnvironmentValue 'CROP_RATIOS' '0.3,0.5,0.8,1.0')
            return [ordered]@{
                fidelity_level = $Fidelity
                scale_factors = $scaleFactors
                crop_ratios = $cropRatios
                negative_scale_factors = ConvertTo-NumberList (Get-EnvironmentValue 'NEGATIVE_SCALE_FACTORS' ($scaleFactors -join ','))
                negative_crop_ratios = ConvertTo-NumberList (Get-EnvironmentValue 'NEGATIVE_CROP_RATIOS' ($cropRatios -join ','))
                crops_per_ratio = [int](Get-EnvironmentValue 'CROPS_PER_RATIO' '3')
                workers = $Workers
                small_crop_trace_strength = Get-EnvironmentValue 'SMALL_CROP_TRACE_STRENGTH' '0.35'
                small_crop_trace_density = Get-EnvironmentValue 'SMALL_CROP_TRACE_DENSITY' 'medium'
                robust_watermark_strength = Get-EnvironmentValue 'ROBUST_WATERMARK_STRENGTH' '1.0'
                robust_watermark_version = Get-EnvironmentValue 'ROBUST_WATERMARK_VERSION' '1'
            }
        }
        'attack' {
            $fullAttacks = @(
                'jpeg_q90', 'jpeg_q70', 'jpeg_q50', 'jpeg_q30', 'double_jpeg_70_50',
                'rotate_1deg', 'rotate_3deg', 'rotate_5deg', 'rotate_10deg',
                'gaussian_blur_1_2', 'unsharp_mask', 'median_denoise',
                'browser_screenshot_sim', 'wechat_screenshot_sim', 'additive_noise',
                'screen_photo_sim'
            )
            $attackFilter = ConvertTo-StringList (Get-EnvironmentValue 'ATTACK_FILTER' '')
            $effectiveAttacks = if ($attackFilter.Count -eq 0) {
                $fullAttacks
            } else {
                @($fullAttacks | Where-Object { $attackFilter -contains $_ })
            }
            return [ordered]@{
                trace_rounds = $TraceRounds
                workers = $Workers
                fidelity_level = $Fidelity
                robust_watermark_strength = Get-EnvironmentValue 'ROBUST_WATERMARK_STRENGTH' '1.0'
                robust_watermark_version = Get-EnvironmentValue 'ROBUST_WATERMARK_VERSION' '1'
                attack_filter = $attackFilter
                attacks = $effectiveAttacks
                negative_sources = ConvertTo-StringList (Get-EnvironmentValue 'NEGATIVE_SOURCES' '1.png,2.png,3.png,4.png,5.png')
            }
        }
        'negative' {
            return [ordered]@{
                synthetic_variants = $NegativeVariants
                negative_attacks = ConvertTo-StringList (Get-EnvironmentValue 'NEGATIVE_ATTACKS' 'jpeg_q90,jpeg_q50,jpeg_q30,rotate_3deg,rotate_10deg,browser_screenshot_sim,wechat_screenshot_sim,screen_photo_sim,gaussian_blur_1_2,median_denoise')
                fidelity_level = $Fidelity
                small_crop_trace_strength = Get-EnvironmentValue 'SMALL_CROP_TRACE_STRENGTH' '0.35'
                small_crop_trace_density = Get-EnvironmentValue 'SMALL_CROP_TRACE_DENSITY' 'medium'
                robust_watermark_strength = Get-EnvironmentValue 'ROBUST_WATERMARK_STRENGTH' '1.0'
                robust_watermark_version = Get-EnvironmentValue 'ROBUST_WATERMARK_VERSION' '1'
            }
        }
        default { throw "unknown benchmark stage: $StageName" }
    }
}

function Test-SettingValueEqual($Actual, $Expected) {
    if ($null -eq $Actual -or $null -eq $Expected) { return $null -eq $Actual -and $null -eq $Expected }
    if ($Actual -is [System.Array] -or $Expected -is [System.Array]) {
        if ($Actual -isnot [System.Array] -or $Expected -isnot [System.Array] -or $Actual.Count -ne $Expected.Count) {
            return $false
        }
        for ($index = 0; $index -lt $Actual.Count; $index++) {
            if (-not (Test-SettingValueEqual $Actual[$index] $Expected[$index])) { return $false }
        }
        return $true
    }
    $numericTypes = @(
        [System.SByte], [System.Byte], [System.Int16], [System.UInt16],
        [System.Int32], [System.UInt32], [System.Int64], [System.UInt64],
        [System.Single], [System.Double], [System.Decimal]
    )
    if ($numericTypes -contains $Actual.GetType() -and $numericTypes -contains $Expected.GetType()) {
        return [decimal]$Actual -eq [decimal]$Expected
    }
    if ($Actual -is [string] -and $Expected -is [string]) { return $Actual -ceq $Expected }
    return (ConvertTo-Json -InputObject $Actual -Compress -Depth 20) -ceq (ConvertTo-Json -InputObject $Expected -Compress -Depth 20)
}

function Test-CommercialReportReuse(
    $Report,
    [string]$ExpectedBenchmark,
    [string]$ExpectedAlgorithmVersion,
    [Int64]$ExpectedSeed,
    [System.Collections.IDictionary]$ExpectedSettings
) {
    if (-not (Test-JsonObject $Report) -or -not (Test-JsonObject $Report.metadata) -or
        -not (Test-JsonObject $Report.summary) -or $Report.cases -isnot [System.Array] -or
        -not (Test-SchemaVersionOne $Report.metadata.schema_version)) {
        return $false
    }
    if ($Report.metadata.benchmark -isnot [string] -or $Report.metadata.benchmark -cne $ExpectedBenchmark) {
        return $false
    }
    if ($Report.metadata.algorithm_version -isnot [string] -or
        $Report.metadata.algorithm_version -cne $ExpectedAlgorithmVersion) {
        return $false
    }
    if ($Report.verdict -isnot [string] -or
        ($Report.verdict -cne 'PASS' -and $Report.verdict -cne 'FAIL')) {
        return $false
    }
    $seedType = if ($null -eq $Report.metadata.seed) { $null } else { $Report.metadata.seed.GetType() }
    $integralTypes = @(
        [System.SByte], [System.Byte], [System.Int16], [System.UInt16],
        [System.Int32], [System.UInt32], [System.Int64], [System.UInt64]
    )
    if ($integralTypes -notcontains $seedType -or [Int64]$Report.metadata.seed -ne $ExpectedSeed) {
        return $false
    }
    if (-not (Test-JsonObject $Report.settings)) { return $false }
    $actualSettings = if ($Report.settings -is [System.Collections.IDictionary]) {
        @($Report.settings.GetEnumerator() | ForEach-Object {
            [pscustomobject]@{ Name = [string]$_.Key; Value = $_.Value }
        })
    } else {
        @($Report.settings.PSObject.Properties | ForEach-Object {
            [pscustomobject]@{ Name = $_.Name; Value = $_.Value }
        })
    }
    if ($actualSettings.Count -ne $ExpectedSettings.Count) { return $false }
    foreach ($name in $ExpectedSettings.Keys) {
        $matching = @($actualSettings | Where-Object { $_.Name -ceq $name })
        if ($matching.Count -ne 1) { return $false }
        $actual = $matching[0].Value
        if (-not (Test-SettingValueEqual $actual $ExpectedSettings[$name])) { return $false }
    }
    return $true
}

function Test-StageReportReuse(
    $Report,
    [string]$StageName,
    [string]$Fidelity = '',
    [bool]$RequireQualityPass = $true
) {
    $benchmark = switch ($StageName) {
        'quality' { 'quality' }
        'crop' { 'trace' }
        'attack' { 'attack' }
        'negative' { 'negative' }
        default { throw "unknown benchmark stage: $StageName" }
    }
    $algorithmVersion = Get-EnvironmentValue 'ROBUST_WATERMARK_VERSION' '1'
    $settings = Get-StageExpectedSettings $StageName $Fidelity
    if (-not (Test-CommercialReportReuse $Report $benchmark $algorithmVersion (Get-ExpectedBenchmarkSeed) $settings)) {
        return $false
    }
    if ($RequireQualityPass -and $StageName -eq 'quality' -and $Report.verdict -cne 'PASS') { return $false }
    if ($StageName -eq 'negative') {
        $workersProperty = $Report.PSObject.Properties['workers']
        if ($null -eq $workersProperty -or -not (Test-SettingValueEqual $workersProperty.Value $Workers)) {
            return $false
        }
    }
    return $true
}

function Get-ReusedReportExitCode($Report) {
    if ($Report.verdict -is [string] -and $Report.verdict -ceq 'PASS') { return 0 }
    if ($Report.verdict -is [string] -and $Report.verdict -ceq 'FAIL') { return 2 }
    return $null
}

function Get-ReusableStageReport(
    [string]$Path,
    [string]$StageName,
    [string]$ExpectedBenchmark,
    [string]$ReuseStageName,
    [string]$Fidelity = ''
) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try {
        $report = Assert-CommercialReport $StageName $Path $ExpectedBenchmark
    } catch {
        return $null
    }
    if (-not (Test-StageReportReuse $report $ReuseStageName $Fidelity)) { return $null }
    return $report
}

function Get-Sha256Hex([byte[]]$Bytes) {
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString($sha256.ComputeHash($Bytes)).Replace('-', '')
    } finally {
        $sha256.Dispose()
    }
}

function Get-StageReportSnapshot([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "commercial report is missing"
    }
    $before = Get-Item -LiteralPath $Path
    $beforeLength = $before.Length
    $beforeLastWrite = $before.LastWriteTimeUtc
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        $bytes = New-Object byte[] $stream.Length
        $offset = 0
        while ($offset -lt $bytes.Length) {
            $read = $stream.Read($bytes, $offset, $bytes.Length - $offset)
            if ($read -eq 0) { throw "commercial report changed during snapshot read" }
            $offset += $read
        }
    } finally {
        $stream.Dispose()
    }
    $after = Get-Item -LiteralPath $Path
    if ($beforeLength -ne $after.Length -or $beforeLastWrite -ne $after.LastWriteTimeUtc -or $bytes.Length -ne $after.Length) {
        throw "commercial report changed during snapshot read"
    }
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        throw "commercial report must be BOM-less UTF-8"
    }
    try {
        $utf8 = New-Object Text.UTF8Encoding($false, $true)
        $json = $utf8.GetString($bytes)
        $report = $json | ConvertFrom-Json
    } catch {
        throw "commercial report is not valid BOM-less UTF-8 JSON"
    }
    return [pscustomobject]@{
        Report = $report
        Hash = Get-Sha256Hex $bytes
        Length = [Int64]$bytes.Length
        LastWriteTimeUtc = $after.LastWriteTimeUtc
    }
}

function Get-StageReportState([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return [pscustomobject]@{ Exists = $false; LastWriteTimeUtc = $null; Hash = $null; Length = $null }
    }
    $item = Get-Item -LiteralPath $Path
    $stream = [IO.File]::OpenRead($Path)
    try {
        $sha256 = [Security.Cryptography.SHA256]::Create()
        try {
            $hash = [BitConverter]::ToString($sha256.ComputeHash($stream)).Replace('-', '')
        } finally {
            $sha256.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
    return [pscustomobject]@{
        Exists = $true
        LastWriteTimeUtc = $item.LastWriteTimeUtc
        Hash = $hash
        Length = $item.Length
    }
}

function Assert-FreshStageReport(
    [string]$StageName,
    [string]$Path,
    [string]$ExpectedBenchmark,
    [string]$ReuseStageName,
    [string]$Fidelity,
    [int]$SubprocessExit,
    [DateTime]$StartedUtc,
    $PreviousState,
    [scriptblock]$BeforeFinalIdentityCheck = $null
) {
    if ($SubprocessExit -eq 1) { throw "$StageName failed due to an execution error" }
    if ($SubprocessExit -notin @(0, 2)) { throw "$StageName returned unexpected exit code $SubprocessExit" }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "[$StageName] fresh commercial report is missing" }
    $snapshot = Get-StageReportSnapshot $Path
    $freshnessFloor = $StartedUtc.ToUniversalTime().AddSeconds(-2)
    $unchanged = (
        $null -ne $PreviousState -and $PreviousState.Exists -and
        $PreviousState.Hash -ceq $snapshot.Hash -and
        $PreviousState.Length -eq $snapshot.Length -and
        $PreviousState.LastWriteTimeUtc -eq $snapshot.LastWriteTimeUtc
    )
    if ($unchanged -or $snapshot.LastWriteTimeUtc -lt $freshnessFloor) {
        throw "[$StageName] report evidence is stale"
    }
    $report = Assert-CommercialReportValue $snapshot.Report $StageName $ExpectedBenchmark
    $generatedAt = ConvertFrom-CanonicalUtcTimestamp $report.metadata.generated_at
    if ($null -eq $generatedAt -or $generatedAt -lt $freshnessFloor -or $generatedAt -gt [DateTime]::UtcNow.AddSeconds(2)) {
        throw "[$StageName] report evidence is stale"
    }
    if (-not (Test-StageReportReuse $report $ReuseStageName $Fidelity $false)) {
        throw "[$StageName] report does not match current stage configuration"
    }
    $evidenceExit = Get-ReusedReportExitCode $report
    if ($null -eq $evidenceExit -or $evidenceExit -ne $SubprocessExit) {
        throw "[$StageName] execution/evidence consistency error"
    }
    if ($null -ne $BeforeFinalIdentityCheck) { & $BeforeFinalIdentityCheck }
    $finalSnapshot = Get-StageReportSnapshot $Path
    if (
        $finalSnapshot.Hash -cne $snapshot.Hash -or
        $finalSnapshot.Length -ne $snapshot.Length -or
        $finalSnapshot.LastWriteTimeUtc -ne $snapshot.LastWriteTimeUtc
    ) {
        throw "[$StageName] report changed during evidence validation"
    }
    return $report
}

function Invoke-FreshBenchmark(
    [string]$Name,
    [string]$ScriptPath,
    [string]$ReportPath,
    [string]$ExpectedBenchmark,
    [string]$ReuseStageName,
    [string]$Fidelity = ''
) {
    $previousState = Get-StageReportState $ReportPath
    $startedUtc = [DateTime]::UtcNow
    $exitCode = Invoke-Benchmark $Name $ScriptPath
    $null = Assert-FreshStageReport $Name $ReportPath $ExpectedBenchmark $ReuseStageName $Fidelity $exitCode $startedUtc $previousState
    return $exitCode
}

function Invoke-Benchmark([string]$Name, [string]$ScriptPath) {
    $started = Get-Date
    Write-Host "[$Name] starting at $($started.ToString('HH:mm:ss'))"
    $benchmarkOutput = & python $ScriptPath
    $exitCode = $LASTEXITCODE
    $benchmarkOutput | ForEach-Object { Write-Host $_ }
    $elapsed = (Get-Date) - $started
    Write-Host "[$Name] completed in $([Math]::Round($elapsed.TotalMinutes, 2)) minutes (exit=$exitCode)"
    if ($exitCode -notin @(0, 1, 2)) {
        throw "$Name returned unexpected exit code $exitCode"
    }
    return $exitCode
}

function Resolve-Fidelity {
    $quality = Assert-CommercialReport 'quality' $qualityJson 'quality'
    if ($null -eq $quality.recommended_fidelity) {
        throw 'No quality-approved fidelity is available. Run -Stage quality and inspect the report first.'
    }
    return [string]$quality.recommended_fidelity
}

function Run-Quality {
    $env:QUALITY_PROBE_FILTER = 'intact'
    $env:PROBE_MIN_RECALL = '1.0'
    $env:QUALITY_MIN_PSNR = '38.0'
    $env:QUALITY_MIN_SSIM = '0.95'
    $env:FIDELITY_LEVELS = '0.85,0.90,0.95,1.0'
    $env:SMALL_CROP_TRACE_STRENGTH = '0.35'
    $env:SMALL_CROP_TRACE_DENSITY = 'medium'
    $env:ROBUST_WATERMARK_STRENGTH = '0.74'
    $env:ROBUST_WATERMARK_VERSION = Get-EnvironmentValue 'ROBUST_WATERMARK_VERSION' '3'
    if ($Reuse) {
        $existing = Get-ReusableStageReport $qualityJson 'quality' 'quality' 'quality'
        if ($null -ne $existing -and $null -ne $existing.recommended_fidelity) {
            Write-Host "[quality] reusing recommended fidelity $($existing.recommended_fidelity)"
            return 0
        }
    }
    return Invoke-FreshBenchmark 'quality' (Join-Path $root 'tests\commercial_quality_benchmark.py') $qualityJson 'quality' 'quality'
}

function Run-Crop([string]$Fidelity) {
    $env:FIDELITY_LEVEL = $Fidelity
    $env:BENCHMARK_WORKERS = [string]$Workers
    $env:CROPS_PER_RATIO = '3'
    $env:SCALE_FACTORS = '0.5,0.75,1.0,1.25,1.5,2.0'
    $env:CROP_RATIOS = '0.3,0.5,0.8,1.0'
    if ($Reuse) {
        $existing = Get-ReusableStageReport $cropJson 'crop' 'trace' 'crop' $Fidelity
        if ($null -ne $existing) {
            Write-Host "[crop] reusing matching report"
            return Get-ReusedReportExitCode $existing
        }
    }
    return Invoke-FreshBenchmark 'crop' (Join-Path $root 'tests\commercial_trace_benchmark.py') $cropJson 'trace' 'crop' $Fidelity
}

function Run-Attack([string]$Fidelity) {
    $env:FIDELITY_LEVEL = $Fidelity
    $env:BENCHMARK_WORKERS = [string]$Workers
    $env:TRACE_ROUNDS = [string]$TraceRounds
    $env:BENCHMARK_LABEL = 'commercial_stability_benchmark'
    Remove-Item Env:\ATTACK_FILTER -ErrorAction SilentlyContinue
    Remove-Item Env:\NEGATIVE_SOURCES -ErrorAction SilentlyContinue
    if ($Reuse) {
        $existing = Get-ReusableStageReport $attackJson 'attack' 'attack' 'attack' $Fidelity
        if ($null -ne $existing) {
            Write-Host "[attack] reusing matching report"
            return Get-ReusedReportExitCode $existing
        }
    }
    return Invoke-FreshBenchmark 'attack' (Join-Path $root 'tests\commercial_attack_benchmark.py') $attackJson 'attack' 'attack' $Fidelity
}

function Run-Negative([string]$Fidelity) {
    $env:FIDELITY_LEVEL = $Fidelity
    $env:BENCHMARK_WORKERS = [string]$Workers
    $env:SYNTHETIC_VARIANTS = [string]$NegativeVariants
    if ($Reuse) {
        $existing = Get-ReusableStageReport $negativeJson 'negative' 'negative' 'negative' $Fidelity
        if ($null -ne $existing) {
            Write-Host "[negative] reusing matching report"
            return Get-ReusedReportExitCode $existing
        }
    }
    return Invoke-FreshBenchmark 'negative' (Join-Path $root 'tests\commercial_negative_benchmark.py') $negativeJson 'negative' 'negative' $Fidelity
}

try {
    Set-Location $root
    if ($Stage -eq 'balanced' -and -not $ConfirmLongRun) {
        throw 'Balanced mode includes a 5-round attack matrix and 1,000+ negatives. Re-run with -ConfirmLongRun after reviewing the quality/crop baseline.'
    }

    if ($Stage -in @('quality', 'balanced')) {
        $qualityExit = Run-Quality
        if ($qualityExit -eq 1) { throw 'quality failed due to an execution error' }
        $quality = Assert-CommercialReport 'quality' $qualityJson 'quality'
        if ($qualityExit -eq 2) {
            Write-Host "Quality gate failed. Report: $qualityJson"
            exit 2
        }
    }

    if ($Stage -eq 'quality') {
        Write-Host "Recommended fidelity: $($quality.recommended_fidelity)"
        exit 0
    }

    $fidelity = Resolve-Fidelity
    $quality = Assert-CommercialReport 'quality' $qualityJson 'quality'
    $env:SMALL_CROP_TRACE_STRENGTH = [string]$quality.settings.small_crop_trace_strength
    $env:SMALL_CROP_TRACE_DENSITY = [string]$quality.settings.small_crop_trace_density
    $robustStrength = [string]$quality.settings.robust_watermark_strength
    if ([string]::IsNullOrWhiteSpace($robustStrength)) { $robustStrength = '1.0' }
    $env:ROBUST_WATERMARK_STRENGTH = $robustStrength
    $robustVersion = [string]$quality.settings.robust_watermark_version
    if ([string]::IsNullOrWhiteSpace($robustVersion)) { $robustVersion = '3' }
    $env:ROBUST_WATERMARK_VERSION = $robustVersion
    Write-Host "Using fidelity: $fidelity; robust strength: $env:ROBUST_WATERMARK_STRENGTH; robust version: $env:ROBUST_WATERMARK_VERSION"

    if ($Stage -in @('crop', 'balanced')) {
        $cropExit = Run-Crop $fidelity
        if ($cropExit -eq 1) { throw 'crop failed due to an execution error' }
        $null = Assert-CommercialReport 'crop' $cropJson 'trace'
        if ($Stage -eq 'crop') { exit $cropExit }
    }
    if ($Stage -in @('attack', 'balanced')) {
        Write-Host "Next stage: $TraceRounds trace rounds x 5 images x 16 attacks, plus matching negatives."
        $attackExit = Run-Attack $fidelity
        if ($attackExit -eq 1) { throw 'attack failed due to an execution error' }
        $null = Assert-CommercialReport 'attack' $attackJson 'attack'
        if ($Stage -eq 'attack') { exit $attackExit }
    }
    if ($Stage -in @('negative', 'balanced')) {
        Write-Host "Next stage: at least $NegativeVariants synthetic negatives plus source attacks."
        $negativeExit = Run-Negative $fidelity
        if ($negativeExit -eq 1) { throw 'negative failed due to an execution error' }
        $null = Assert-CommercialReport 'negative' $negativeJson 'negative'
        if ($Stage -eq 'negative') { exit $negativeExit }
    }

    if ($cropExit -eq 2 -or $attackExit -eq 2 -or $negativeExit -eq 2) {
        exit 2
    }
    exit 0
}
finally {
    foreach ($name in $managedEnvironment) {
        $previous = $previousEnvironment[$name]
        if ($null -eq $previous) {
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        } else {
            [Environment]::SetEnvironmentVariable($name, $previous, 'Process')
        }
    }
    Set-Location $originalLocation
}
