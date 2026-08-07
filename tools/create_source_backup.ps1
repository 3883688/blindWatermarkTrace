[CmdletBinding()]
param(
    [string]$RootPath,
    [switch]$Force,
    [switch]$FailBeforePublish,
    [switch]$FailAclPublication
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module Microsoft.PowerShell.Utility -ErrorAction Stop

if ([string]::IsNullOrWhiteSpace($RootPath)) {
    $RootPath = Join-Path $PSScriptRoot '..'
}
$repoRoot = [System.IO.Path]::GetFullPath($RootPath)
if (-not (Test-Path -LiteralPath $repoRoot -PathType Container)) {
    throw "Repository root does not exist: $repoRoot"
}
$backupDir = Join-Path $repoRoot 'backups'
$archiveName = 'trace-v3-source-20260713.zip'
$manifestName = 'trace-v3-source-20260713.manifest.txt'
$checksumName = 'SHA256SUMS'
$archivePath = Join-Path $backupDir $archiveName
$manifestPath = Join-Path $backupDir $manifestName
$checksumPath = Join-Path $backupDir $checksumName

$existingArtifacts = @(
    @($archivePath, $manifestPath, $checksumPath) | Where-Object {
        Test-Path -LiteralPath $_
    }
)
if ($existingArtifacts.Count -gt 0 -and -not $Force) {
    throw "Backup artifacts already exist and are immutable by default; use explicit -Force only for intentional replacement."
}

$excludedDirectorySegments = @(
    '.git',
    '.venv',
    'venv',
    'backups',
    'data',
    'uploads',
    'output',
    'test_output',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.cache',
    '.playwright-cli',
    'playwright',
    'browser-profile',
    'browser_profiles',
    'logs',
    '__pycache__'
)
$excludedSecretFileNames = @(
    'credentials.json',
    'secrets.json',
    'service-account.json',
    'service_account.json'
)
$excludedArchiveExtensions = @(
    '.zip', '.7z', '.rar', '.tar', '.gz', '.tgz', '.bz2', '.xz'
)
$excludedSecretExtensions = @('.pem', '.key', '.p12', '.pfx')

function Get-NormalizedRelativePath {
    param([Parameter(Mandatory = $true)][string]$FullName)

    $rootPrefix = $repoRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $FullName.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside repository root: $FullName"
    }
    return $FullName.Substring($rootPrefix.Length).Replace('\', '/')
}

function Test-ExcludedPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $segments = $RelativePath -split '/'
    if ($segments.Count -gt 1) {
        foreach ($segment in $segments[0..($segments.Count - 2)]) {
            if ($excludedDirectorySegments -contains $segment) {
                return $true
            }
        }
    }

    $fileName = $segments[-1]
    if ($fileName -eq '.env' -or $fileName -like '.env.*' -or $fileName -like '*.log' -or $fileName -like '*.pyc') {
        return $true
    }
    if ($excludedSecretFileNames -contains $fileName) {
        return $true
    }
    $extension = [System.IO.Path]::GetExtension($fileName).ToLowerInvariant()
    return (($excludedArchiveExtensions -contains $extension) -or ($excludedSecretExtensions -contains $extension))
}

function Remove-SafeTemporaryDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedPrefix
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\', '/')
    $tempPrefix = $tempRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($tempPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove non-temporary path: $resolvedPath"
    }
    if (-not [System.IO.Path]::GetFileName($resolvedPath).StartsWith($ExpectedPrefix, [System.StringComparison]::Ordinal)) {
        throw "Refusing to remove unexpected temporary path: $resolvedPath"
    }
    Remove-Item -LiteralPath $resolvedPath -Recurse -Force
}

function Set-AndVerifyRestrictedAcl {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Directory
    )

    if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
        throw 'Restricted artifact ACL publication is supported only on Windows.'
    }

    $currentUserSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    $systemSid = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-18')
    $administratorsSid = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
    $expectedSids = @($currentUserSid.Value, $systemSid.Value, $administratorsSid.Value) | Sort-Object

    if ($Directory) {
        $acl = [System.Security.AccessControl.DirectorySecurity]::new()
        $inheritanceFlags = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
            [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
    }
    else {
        $acl = [System.Security.AccessControl.FileSecurity]::new()
        $inheritanceFlags = [System.Security.AccessControl.InheritanceFlags]::None
    }
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($sid in @($currentUserSid, $systemSid, $administratorsSid)) {
        $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $sid,
            [System.Security.AccessControl.FileSystemRights]::FullControl,
            $inheritanceFlags,
            [System.Security.AccessControl.PropagationFlags]::None,
            [System.Security.AccessControl.AccessControlType]::Allow
        )
        [void]$acl.AddAccessRule($rule)
    }
    if ($Directory) {
        [System.IO.Directory]::SetAccessControl($Path, $acl)
        $verifiedAcl = [System.IO.Directory]::GetAccessControl($Path)
    }
    else {
        [System.IO.File]::SetAccessControl($Path, $acl)
        $verifiedAcl = [System.IO.File]::GetAccessControl($Path)
    }

    $verifiedRules = @($verifiedAcl.GetAccessRules(
        $true,
        $true,
        [System.Security.Principal.SecurityIdentifier]
    ))
    $verifiedSids = @($verifiedRules | ForEach-Object { $_.IdentityReference.Value } | Sort-Object)
    $invalidRules = @($verifiedRules | Where-Object {
        $_.AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow -or
        $_.FileSystemRights -ne [System.Security.AccessControl.FileSystemRights]::FullControl -or
        $_.IsInherited
    })
    if (-not $verifiedAcl.AreAccessRulesProtected -or
        $verifiedRules.Count -ne 3 -or
        $invalidRules.Count -ne 0 -or
        $null -ne (Compare-Object -ReferenceObject $expectedSids -DifferenceObject $verifiedSids)) {
        throw "Restricted artifact ACL verification failed: $Path"
    }
}

function Apply-RestrictedArtifactAcl {
    param([Parameter(Mandatory = $true)][string]$Path)

    Set-AndVerifyRestrictedAcl -Path $Path
}

function Apply-RestrictedDirectoryAcl {
    param([Parameter(Mandatory = $true)][string]$Path)

    Set-AndVerifyRestrictedAcl -Path $Path -Directory
}

function Remove-SafeLocalDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedPrefix
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolvedPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $resolvedBackupDir = [System.IO.Path]::GetFullPath($backupDir).TrimEnd('\', '/')
    if (-not [System.IO.Path]::GetDirectoryName($resolvedPath).Equals(
        $resolvedBackupDir,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to remove local directory outside backup root: $resolvedPath"
    }
    if (-not [System.IO.Path]::GetFileName($resolvedPath).StartsWith(
        $ExpectedPrefix,
        [System.StringComparison]::Ordinal
    )) {
        throw "Refusing to remove unexpected local directory: $resolvedPath"
    }
    Remove-Item -LiteralPath $resolvedPath -Recurse -Force
}

$runId = [Guid]::NewGuid().ToString('N')
$publishTempDir = Join-Path ([System.IO.Path]::GetTempPath()) "trace-v3-backup-publish-$runId"
$localPublishDir = Join-Path $backupDir ".trace-v3-publish-$runId"
$localRollbackDir = Join-Path $backupDir ".trace-v3-rollback-$runId"
$stagingDir = Join-Path $publishTempDir 'payload'
$verificationDir = Join-Path $publishTempDir 'verification'
$tempArtifactsDir = Join-Path $publishTempDir 'artifacts'
$tempArchivePath = Join-Path $tempArtifactsDir $archiveName
$tempManifestPath = Join-Path $tempArtifactsDir $manifestName
$tempChecksumPath = Join-Path $tempArtifactsDir $checksumName
$cleanupLocalRollback = $true

try {
    $manifestEntries = @(
        Get-ChildItem -LiteralPath $repoRoot -File -Recurse | ForEach-Object {
            $relativePath = Get-NormalizedRelativePath -FullName $_.FullName
            if (-not (Test-ExcludedPath -RelativePath $relativePath)) {
                $relativePath
            }
        } | Sort-Object
    )
    if ($manifestEntries.Count -eq 0) {
        throw 'No source files were selected for backup.'
    }

    New-Item -ItemType Directory -Path $tempArtifactsDir -Force | Out-Null
    [System.IO.File]::WriteAllLines(
        $tempManifestPath,
        [string[]]$manifestEntries,
        [System.Text.UTF8Encoding]::new($false)
    )

    New-Item -ItemType Directory -Path $stagingDir | Out-Null
    foreach ($relativePath in $manifestEntries) {
        $platformRelativePath = $relativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
        $sourcePath = Join-Path $repoRoot $platformRelativePath
        $stagedPath = Join-Path $stagingDir $platformRelativePath
        $stagedParent = Split-Path -Parent $stagedPath
        New-Item -ItemType Directory -Path $stagedParent -Force | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $stagedPath
    }

    $stableTimestamp = [DateTime]::new(2000, 1, 1, 0, 0, 0, [DateTimeKind]::Utc)
    Get-ChildItem -LiteralPath $stagingDir -Force -File -Recurse | ForEach-Object {
        $_.CreationTimeUtc = $stableTimestamp
        $_.LastAccessTimeUtc = $stableTimestamp
        $_.LastWriteTimeUtc = $stableTimestamp
    }
    Get-ChildItem -LiteralPath $stagingDir -Force -Directory -Recurse |
        Sort-Object { $_.FullName.Length } -Descending |
        ForEach-Object {
            $_.CreationTimeUtc = $stableTimestamp
            $_.LastAccessTimeUtc = $stableTimestamp
            $_.LastWriteTimeUtc = $stableTimestamp
        }
    $stagingRoot = Get-Item -LiteralPath $stagingDir
    $stagingRoot.CreationTimeUtc = $stableTimestamp
    $stagingRoot.LastAccessTimeUtc = $stableTimestamp
    $stagingRoot.LastWriteTimeUtc = $stableTimestamp

    Compress-Archive -Path (Join-Path $stagingDir '*') -DestinationPath $tempArchivePath -CompressionLevel Optimal

    $archiveHash = (Get-FileHash -LiteralPath $tempArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    [System.IO.File]::WriteAllText(
        $tempChecksumPath,
        "$archiveHash  $archiveName`n",
        [System.Text.UTF8Encoding]::new($false)
    )

    New-Item -ItemType Directory -Path $verificationDir | Out-Null
    Expand-Archive -LiteralPath $tempArchivePath -DestinationPath $verificationDir
    $extractedEntries = @(
        Get-ChildItem -LiteralPath $verificationDir -File -Recurse | ForEach-Object {
            $verificationPrefix = $verificationDir.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
            $_.FullName.Substring($verificationPrefix.Length).Replace('\', '/')
        } | Sort-Object
    )

    $difference = Compare-Object -ReferenceObject ([string[]]$manifestEntries) -DifferenceObject ([string[]]$extractedEntries)
    if ($null -ne $difference) {
        $details = ($difference | Out-String).Trim()
        throw "Archive verification failed; extracted file list differs from manifest.`n$details"
    }

    if ($FailBeforePublish) {
        throw 'Intentional pre-publication failure requested by -FailBeforePublish.'
    }

    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    New-Item -ItemType Directory -Path $localPublishDir | Out-Null
    Apply-RestrictedDirectoryAcl -Path $localPublishDir
    $artifactNames = @($archiveName, $manifestName, $checksumName)
    $preservedNames = [System.Collections.Generic.List[string]]::new()
    $publishedNames = [System.Collections.Generic.List[string]]::new()
    try {
        foreach ($artifactName in $artifactNames) {
            Move-Item -LiteralPath (Join-Path $tempArtifactsDir $artifactName) -Destination (Join-Path $localPublishDir $artifactName)
            Apply-RestrictedArtifactAcl -Path (Join-Path $localPublishDir $artifactName)
        }
        New-Item -ItemType Directory -Path $localRollbackDir | Out-Null
        Apply-RestrictedDirectoryAcl -Path $localRollbackDir
        foreach ($artifactName in $artifactNames) {
            $finalPath = Join-Path $backupDir $artifactName
            if (Test-Path -LiteralPath $finalPath) {
                Move-Item -LiteralPath $finalPath -Destination (Join-Path $localRollbackDir $artifactName)
                $preservedNames.Add($artifactName)
                Apply-RestrictedArtifactAcl -Path (Join-Path $localRollbackDir $artifactName)
            }
        }
        foreach ($artifactName in $artifactNames) {
            Move-Item -LiteralPath (Join-Path $localPublishDir $artifactName) -Destination (Join-Path $backupDir $artifactName)
            $publishedNames.Add($artifactName)
        }
        if ($FailAclPublication) {
            throw 'Intentional ACL publication failure requested by -FailAclPublication.'
        }
        foreach ($artifactName in $artifactNames) {
            Apply-RestrictedArtifactAcl -Path (Join-Path $backupDir $artifactName)
        }
    }
    catch {
        $publishError = $_
        try {
            foreach ($artifactName in $publishedNames) {
                $publishedPath = Join-Path $backupDir $artifactName
                if (Test-Path -LiteralPath $publishedPath) {
                    Remove-Item -LiteralPath $publishedPath -Force
                }
            }
            foreach ($artifactName in $preservedNames) {
                $rollbackPath = Join-Path $localRollbackDir $artifactName
                if (Test-Path -LiteralPath $rollbackPath) {
                    $restorePath = Join-Path $localPublishDir $artifactName
                    Copy-Item -LiteralPath $rollbackPath -Destination $restorePath
                    Apply-RestrictedArtifactAcl -Path $restorePath
                    Move-Item -LiteralPath $restorePath -Destination (Join-Path $backupDir $artifactName)
                }
            }
            foreach ($artifactName in $preservedNames) {
                Apply-RestrictedArtifactAcl -Path (Join-Path $backupDir $artifactName)
            }
        }
        catch {
            $cleanupLocalRollback = $false
            throw "CRITICAL: artifact publication failed and rollback restoration or ACL verification could not be completed. Preserved rollback evidence remains at $localRollbackDir. Publish error: $publishError Rollback error: $_"
        }
        throw $publishError
    }

    Write-Output "Created and verified $archivePath"
    Write-Output "SHA256 $archiveHash"
}
finally {
    Remove-SafeTemporaryDirectory -Path $publishTempDir -ExpectedPrefix 'trace-v3-backup-publish-'
    Remove-SafeLocalDirectory -Path $localPublishDir -ExpectedPrefix '.trace-v3-publish-'
    if ($cleanupLocalRollback) {
        Remove-SafeLocalDirectory -Path $localRollbackDir -ExpectedPrefix '.trace-v3-rollback-'
    }
}
