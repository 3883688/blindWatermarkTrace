param(
    [string]$EnvPath = (Join-Path (Split-Path $PSScriptRoot -Parent) ".env")
)

$ErrorActionPreference = "Stop"

$resolved = [System.IO.Path]::GetFullPath($EnvPath)
$workspace = [System.IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
if (-not $resolved.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "EnvPath must remain inside the workspace"
}

$bytes = New-Object byte[] 48
$generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
try {
    $generator.GetBytes($bytes)
}
finally {
    $generator.Dispose()
}
$key = [Convert]::ToBase64String($bytes)

$lines = if (Test-Path -LiteralPath $resolved) {
    [System.IO.File]::ReadAllLines($resolved)
} else {
    @()
}
$output = New-Object System.Collections.Generic.List[string]
$written = $false
foreach ($line in $lines) {
    if ($line -match '^WATERMARK_AUTH_KEY=') {
        if (-not $written) {
            $output.Add("WATERMARK_AUTH_KEY=$key")
            $written = $true
        }
        continue
    }
    $output.Add($line)
}
if (-not $written) {
    $output.Add("WATERMARK_AUTH_KEY=$key")
}

$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($resolved, $output, $utf8)
Write-Output "WATERMARK_AUTH_KEY rotated: utf8_bytes=$([Text.Encoding]::UTF8.GetByteCount($key)) entries=1"
