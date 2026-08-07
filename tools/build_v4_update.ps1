[CmdletBinding()]
param(
    [string]$OutputDirectory = "release",
    [switch]$SkipFrontendBuild,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExpectedBranch = "V4-CPU"
$DefaultTarget = "/opt/trace-v4-docker-20260726-164548"

function Invoke-Git {
    param([Parameter(Mandatory)][string[]]$Arguments)
    $result = & git -C $Root @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: git $($Arguments -join ' ')"
    }
    return ($result | Out-String).Trim()
}

function Get-Sha256 {
    param([Parameter(Mandatory)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-PayloadFiles {
    $rootFiles = @(
        ".env.example",
        "README_DEPLOY.md",
        "deploy.sh",
        "favico.ico",
        "favicon.ico",
        "index.html",
        "logo.png",
        "main.py",
        "requirements.txt",
        "site-logo.png",
        "tools/initialize_v4.py",
        "tools/migrate_json_to_mysql.py",
        "tools/migrate_mysql_to_postgresql.py",
        "tools/migrate_v4_relational_only.py",
        "tools/prepare_deployment_env.py",
        "tools/restore_v4_backup.py"
    )
    $files = [System.Collections.Generic.List[string]]::new()
    foreach ($relative in $rootFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $relative) -PathType Leaf)) {
            throw "Required deployment file is missing: $relative"
        }
        $files.Add($relative)
    }

    $trees = @{
        "assets" = @(".css", ".js", ".png", ".ttf", ".woff", ".woff2")
        "trace_app" = @(".py")
        "watermark_v4" = @(".py")
    }
    foreach ($tree in $trees.Keys) {
        $treePath = Join-Path $Root $tree
        if (-not (Test-Path -LiteralPath $treePath -PathType Container)) {
            throw "Required deployment directory is missing: $tree"
        }
        Get-ChildItem -LiteralPath $treePath -Recurse -File | ForEach-Object {
            if ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Links are not allowed in update payloads: $($_.FullName)"
            }
            if ($trees[$tree] -contains $_.Extension.ToLowerInvariant()) {
                $relative = [IO.Path]::GetRelativePath($Root, $_.FullName).Replace("\", "/")
                if ($relative -notmatch "(^|/)(__pycache__|tests?)(/|$)") {
                    $files.Add($relative)
                }
            }
        }
    }
    return @($files | Sort-Object -Unique)
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required"
}
$branch = Invoke-Git -Arguments @("branch", "--show-current")
if ($branch -ne $ExpectedBranch) {
    throw "Update packages must be built from $ExpectedBranch; current branch is $branch"
}
$trackedChanges = Invoke-Git -Arguments @("status", "--porcelain", "--untracked-files=no")
if ($trackedChanges) {
    throw "Tracked files have uncommitted changes. Commit them before building the update package."
}

if (-not $SkipFrontendBuild) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "npm is required unless -SkipFrontendBuild is used"
    }
    Write-Host "Building frontend..."
    & npm --prefix (Join-Path $Root "frontend") run build
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed"
    }
    $trackedChanges = Invoke-Git -Arguments @("status", "--porcelain", "--untracked-files=no")
    if ($trackedChanges) {
        throw "Frontend build changed tracked artifacts. Commit the generated assets, then run this script again."
    }
}

$commit = Invoke-Git -Arguments @("rev-parse", "--short=7", "HEAD")
$fullCommit = Invoke-Git -Arguments @("rev-parse", "HEAD")
$packageName = "trace-v4-update-$commit-cpu-full"
$releaseRoot = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
    [IO.Path]::GetFullPath($OutputDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $Root $OutputDirectory))
}
$packageRoot = Join-Path $releaseRoot $packageName
$archive = Join-Path $releaseRoot "$packageName.zip"
$archiveChecksum = "$archive.sha256"

New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
$existing = @(@($packageRoot, $archive, $archiveChecksum) | Where-Object { Test-Path -LiteralPath $_ })
if ($existing.Count -gt 0 -and -not $Force) {
    throw "Release artifact already exists. Use -Force to replace: $($existing -join ', ')"
}
foreach ($path in $existing) {
    Remove-Item -LiteralPath $path -Recurse -Force
}

$payloadRoot = Join-Path $packageRoot "payload"
New-Item -ItemType Directory -Force -Path $payloadRoot | Out-Null
$files = Get-PayloadFiles
foreach ($relative in $files) {
    $source = Join-Path $Root $relative
    $destination = Join-Path $payloadRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}

$utf8NoBom = [Text.UTF8Encoding]::new($false)
$fileList = ($files -join "`n") + "`n"
[IO.File]::WriteAllText((Join-Path $packageRoot "FILES.txt"), $fileList, $utf8NoBom)

$manifest = foreach ($relative in $files) {
    $hash = Get-Sha256 -Path (Join-Path $payloadRoot $relative)
    "$hash  payload/$relative"
}
[IO.File]::WriteAllText(
    (Join-Path $packageRoot "PAYLOAD_SHA256SUMS"),
    (($manifest -join "`n") + "`n"),
    $utf8NoBom
)
[IO.File]::WriteAllText((Join-Path $packageRoot "VERSION"), "$fullCommit`n", $utf8NoBom)

$notes = @"
# V4-CPU Full Runtime Update $commit

- Branch: ``V4-CPU``
- Commit: ``$fullCommit``
- Contents: backend runtime, V4 watermark algorithm, deployment helpers, and compiled frontend assets.
- Preserved: ``.env``, PostgreSQL data, uploads, models, logs, and backups.
- Database: only creates/verifies missing V4 schema objects; it does not truncate or delete data.
- Rollback: all replaced files are backed up before installation and restored if the update fails.

Run on the server after extracting the ZIP:

``````bash
chmod +x update.sh
sudo TRACE_TARGET_DIR=$DefaultTarget ./update.sh
``````
"@
[IO.File]::WriteAllText((Join-Path $packageRoot "UPDATE_NOTES.md"), $notes, $utf8NoBom)

$updateScript = @'
#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${TRACE_TARGET_DIR:-/opt/trace-v4-docker-20260726-164548}"
APP_SERVICE="${TRACE_APP_SERVICE:-app}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
UPDATED=0
STOPPED=0

fail() { echo "ERROR: $*" >&2; exit 2; }

command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"
command -v grep >/dev/null 2>&1 || fail "grep is required"
[ -f "$PACKAGE_DIR/PAYLOAD_SHA256SUMS" ] || fail "Missing checksum manifest"
[ -f "$PACKAGE_DIR/FILES.txt" ] || fail "Missing file manifest"
(cd "$PACKAGE_DIR" && sha256sum -c PAYLOAD_SHA256SUMS)

[ -d "$TARGET_DIR/trace_app" ] || fail "Invalid project directory: $TARGET_DIR"
[ -f "$TARGET_DIR/.env" ] || fail "Existing .env is required: $TARGET_DIR/.env"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"
case "$TARGET_DIR" in "/"|"$HOME") fail "Unsafe project directory: $TARGET_DIR" ;; esac

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  fail "Docker Compose is unavailable"
fi

cd "$TARGET_DIR"
"${COMPOSE[@]}" config --services | grep -Fxq "$APP_SERVICE" || fail "Compose service not found: $APP_SERVICE"
BACKUP_DIR="$TARGET_DIR/backups/v4-update-$STAMP"
mkdir -p "$BACKUP_DIR/files"

rollback() {
  local code=$?
  trap - ERR
  echo "Update failed; restoring previous files" >&2
  if [ "$UPDATED" -eq 1 ]; then
    cp -a "$BACKUP_DIR/files/." "$TARGET_DIR/" || true
    if [ -f "$BACKUP_DIR/new-files.txt" ]; then
      while IFS= read -r relative; do
        [ -n "$relative" ] || continue
        rm -f -- "$TARGET_DIR/$relative"
      done < "$BACKUP_DIR/new-files.txt"
    fi
    "${COMPOSE[@]}" build "$APP_SERVICE" || true
  fi
  if [ "$STOPPED" -eq 1 ]; then
    "${COMPOSE[@]}" up -d --no-deps "$APP_SERVICE" || true
  fi
  echo "Rollback files: $BACKUP_DIR" >&2
  exit "$code"
}
trap rollback ERR

while IFS= read -r relative; do
  [ -n "$relative" ] || continue
  case "$relative" in /*|../*|*/../*|*/..) fail "Unsafe update path: $relative" ;; esac
  [ -f "$PACKAGE_DIR/payload/$relative" ] || fail "Payload file is missing: $relative"
  [ ! -L "$PACKAGE_DIR/payload/$relative" ] || fail "Payload links are forbidden: $relative"
  if [ -f "$TARGET_DIR/$relative" ]; then
    mkdir -p "$BACKUP_DIR/files/$(dirname "$relative")"
    cp -a "$TARGET_DIR/$relative" "$BACKUP_DIR/files/$relative"
  else
    printf '%s\n' "$relative" >> "$BACKUP_DIR/new-files.txt"
  fi
done < "$PACKAGE_DIR/FILES.txt"
cp -a "$TARGET_DIR/.env" "$BACKUP_DIR/.env"

echo "Stopping $APP_SERVICE"
"${COMPOSE[@]}" stop "$APP_SERVICE"
STOPPED=1

echo "Installing update files"
while IFS= read -r relative; do
  [ -n "$relative" ] || continue
  mkdir -p "$TARGET_DIR/$(dirname "$relative")"
  cp -a "$PACKAGE_DIR/payload/$relative" "$TARGET_DIR/$relative"
done < "$PACKAGE_DIR/FILES.txt"
cp -a "$PACKAGE_DIR/VERSION" "$TARGET_DIR/V4_VERSION"
UPDATED=1

echo "Rebuilding $APP_SERVICE"
"${COMPOSE[@]}" build "$APP_SERVICE"

echo "Creating or verifying V4 schema (no data is cleared)"
"${COMPOSE[@]}" run --rm --no-deps -T "$APP_SERVICE" python -c \
  "from sqlalchemy import create_engine; from trace_app.config import settings; from trace_app.v4.startup import initialize_v4_schema; u=settings.db_url; u=('postgresql+psycopg://'+u[len('postgresql://'):]) if u.startswith('postgresql://') else u; initialize_v4_schema(create_engine(u), require_postgres=True); print('v4 schema: ok')"

echo "Starting $APP_SERVICE"
"${COMPOSE[@]}" up -d --no-deps "$APP_SERVICE"
"${COMPOSE[@]}" ps "$APP_SERVICE"
trap - ERR
echo "Update complete: $(cat "$PACKAGE_DIR/VERSION")"
echo "Backup: $BACKUP_DIR"
'@
$updateScript = $updateScript.Replace("`r`n", "`n") + "`n"
[IO.File]::WriteAllText((Join-Path $packageRoot "update.sh"), $updateScript, $utf8NoBom)

Compress-Archive -LiteralPath $packageRoot -DestinationPath $archive -CompressionLevel Optimal
$archiveHash = Get-Sha256 -Path $archive
[IO.File]::WriteAllText($archiveChecksum, "$archiveHash  $packageName.zip`n", $utf8NoBom)

$verifyRoot = Join-Path ([IO.Path]::GetTempPath()) "trace-v4-update-verify-$([guid]::NewGuid().ToString('N'))"
try {
    Expand-Archive -LiteralPath $archive -DestinationPath $verifyRoot
    $verifiedPackage = Join-Path $verifyRoot $packageName
    $verifiedFiles = Get-Content -LiteralPath (Join-Path $verifiedPackage "FILES.txt")
    if (@(Compare-Object $files $verifiedFiles).Count -ne 0) {
        throw "Archive file manifest does not match the source manifest"
    }
    foreach ($line in Get-Content -LiteralPath (Join-Path $verifiedPackage "PAYLOAD_SHA256SUMS")) {
        if ($line -notmatch '^([0-9a-f]{64})  payload/(.+)$') {
            throw "Invalid checksum manifest line: $line"
        }
        $actual = Get-Sha256 -Path (Join-Path $verifiedPackage "payload/$($Matches[2])")
        if ($actual -ne $Matches[1]) {
            throw "Archive payload checksum mismatch: $($Matches[2])"
        }
    }
} finally {
    if (Test-Path -LiteralPath $verifyRoot) {
        Remove-Item -LiteralPath $verifyRoot -Recurse -Force
    }
}

Write-Host ""
Write-Host "Update package created and verified"
Write-Host "ZIP: $archive"
Write-Host "SHA-256: $archiveHash"
Write-Host "Files: $($files.Count)"
Write-Host ""
Write-Host "Server commands:"
Write-Host "  sha256sum -c $packageName.zip.sha256"
Write-Host "  unzip $packageName.zip"
Write-Host "  cd $packageName"
Write-Host "  chmod +x update.sh"
Write-Host "  sudo TRACE_TARGET_DIR=$DefaultTarget ./update.sh"
