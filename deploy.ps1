param(
  [int]$Port = 6868,
  [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw "Python is not available in PATH. Install Python 3.10+ first."
}

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "Created .env from .env.example"
}

python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\pip.exe" install -r requirements.txt
& ".\.venv\Scripts\python.exe" "tools\install_optional_gpu.py" `
  --python ".\.venv\Scripts\python.exe" `
  --requirements "requirements-gpu.txt"

New-Item -ItemType Directory -Force -Path "data","uploads","uploads\originals","uploads\watermarked" | Out-Null

Write-Host "Starting WatermarkSystem at http://$HostAddress`:$Port"
& ".\.venv\Scripts\python.exe" -m uvicorn main:app --host $HostAddress --port $Port
