$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$projectRoot = $PSScriptRoot
$exitCode = 1

function Get-SingleRuntimeDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Parent,

        [Parameter(Mandatory = $true)]
        [string]$NamePattern,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $directory = Get-ChildItem -LiteralPath $Parent -Directory |
        Where-Object { $_.Name -match $NamePattern } |
        Sort-Object -Property Name -Descending |
        Select-Object -First 1
    if (-not $directory) {
        throw "The bundled $Description directory was not found under: $Parent"
    }
    return $directory
}

function Get-TclTkRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvironmentRoot
    )

    $libraryRoot = Join-Path $EnvironmentRoot "Library\lib"
    $binaryRoot = Join-Path $EnvironmentRoot "Library\bin"
    if (-not (Test-Path -LiteralPath $libraryRoot -PathType Container)) {
        throw "The bundled Tcl/Tk library root was not found: $libraryRoot"
    }
    if (-not (Test-Path -LiteralPath $binaryRoot -PathType Container)) {
        throw "The bundled Tcl/Tk binary root was not found: $binaryRoot"
    }

    $tclLibrary = Get-SingleRuntimeDirectory $libraryRoot '^tcl[0-9]+\.[0-9]+$' 'Tcl library'
    $tkLibrary = Get-SingleRuntimeDirectory $libraryRoot '^tk[0-9]+\.[0-9]+$' 'Tk library'
    $packageLibraries = @(Get-ChildItem -LiteralPath $libraryRoot -Directory |
        Where-Object { $_.Name -match '^tcl[0-9]+$' })

    $tclDlls = @(Get-ChildItem -LiteralPath $binaryRoot -File |
        Where-Object { $_.Name -match '^tcl[0-9].*\.dll$' })
    $tkDlls = @(Get-ChildItem -LiteralPath $binaryRoot -File |
        Where-Object { $_.Name -match '^tk[0-9].*\.dll$' })
    if ($tclDlls.Count -eq 0 -or $tkDlls.Count -eq 0) {
        throw "The bundled Tcl/Tk runtime DLLs were not found under: $binaryRoot"
    }

    # Tcl's compressed-channel support uses zlib at runtime. Keep this
    # dependency beside the Tcl/Tk DLLs without copying unrelated binaries.
    $dependencyDlls = @(Get-ChildItem -LiteralPath $binaryRoot -File |
        Where-Object { $_.Name -match '^zlib.*\.dll$' })

    return [pscustomobject]@{
        TclLibrary = $tclLibrary
        TkLibrary = $tkLibrary
        PackageLibraries = $packageLibraries
        Dlls = @($tclDlls + $tkDlls + $dependencyDlls)
    }
}

function Get-TclTkCacheIdentity {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Runtime
    )

    $manifest = [System.Text.StringBuilder]::new()
    $libraryDirectories = @($Runtime.TclLibrary, $Runtime.TkLibrary) + @($Runtime.PackageLibraries)
    foreach ($directory in ($libraryDirectories | Sort-Object -Property Name)) {
        $basePath = $directory.FullName.TrimEnd('\')
        foreach ($file in (Get-ChildItem -LiteralPath $basePath -Recurse -File | Sort-Object -Property FullName)) {
            $relativePath = $file.FullName.Substring($basePath.Length).TrimStart('\')
            $fileHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            [void]$manifest.AppendLine("lib/$($directory.Name)/$relativePath|$fileHash")
        }
    }
    foreach ($file in ($Runtime.Dlls | Sort-Object -Property Name)) {
        $fileHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        [void]$manifest.AppendLine("bin/$($file.Name)|$fileHash")
    }

    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $manifestBytes = [System.Text.Encoding]::UTF8.GetBytes($manifest.ToString())
        $digestBytes = $hasher.ComputeHash($manifestBytes)
    }
    finally {
        $hasher.Dispose()
    }
    $digest = ([System.BitConverter]::ToString($digestBytes) -replace '-', '').ToLowerInvariant()
    $version = "$($Runtime.TclLibrary.Name)-$($Runtime.TkLibrary.Name)" -replace '[^A-Za-z0-9._-]', '-'

    return [pscustomobject]@{
        Digest = $digest
        DirectoryName = "tcltk-$version-$($digest.Substring(0, 16))"
    }
}

function Test-CompleteTclTkCache {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CacheDirectory,

        [Parameter(Mandatory = $true)]
        [pscustomobject]$Runtime,

        [Parameter(Mandatory = $true)]
        [string]$Digest
    )

    $marker = Join-Path $CacheDirectory ".complete"
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        return $false
    }
    if ((Get-Content -LiteralPath $marker -Raw).Trim() -ne $Digest) {
        return $false
    }

    foreach ($directory in (@($Runtime.TclLibrary, $Runtime.TkLibrary) + @($Runtime.PackageLibraries))) {
        if (-not (Test-Path -LiteralPath (Join-Path $CacheDirectory "lib\$($directory.Name)") -PathType Container)) {
            return $false
        }
    }
    foreach ($dll in $Runtime.Dlls) {
        if (-not (Test-Path -LiteralPath (Join-Path $CacheDirectory "bin\$($dll.Name)") -PathType Leaf)) {
            return $false
        }
    }
    return $true
}

function Remove-SafeCacheDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CacheRoot,

        [Parameter(Mandatory = $true)]
        [string]$Target
    )

    $resolvedRoot = [System.IO.Path]::GetFullPath($CacheRoot).TrimEnd('\')
    $resolvedTarget = [System.IO.Path]::GetFullPath($Target).TrimEnd('\')
    if (-not $resolvedTarget.StartsWith("$resolvedRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a cache directory outside the cache root: $resolvedTarget"
    }
    if (Test-Path -LiteralPath $resolvedTarget -PathType Container) {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
}

function New-TclTkCache {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CacheRoot,

        [Parameter(Mandatory = $true)]
        [string]$CacheDirectory,

        [Parameter(Mandatory = $true)]
        [pscustomobject]$Runtime,

        [Parameter(Mandatory = $true)]
        [string]$Digest
    )

    if (Test-CompleteTclTkCache $CacheDirectory $Runtime $Digest) {
        return
    }

    if (Test-Path -LiteralPath $CacheDirectory) {
        Remove-SafeCacheDirectory $CacheRoot $CacheDirectory
    }

    $stagingDirectory = "$CacheDirectory.creating-$PID-$([guid]::NewGuid().ToString('N'))"
    try {
        $stagingLib = New-Item -ItemType Directory -Path (Join-Path $stagingDirectory "lib") -Force
        $stagingBin = New-Item -ItemType Directory -Path (Join-Path $stagingDirectory "bin") -Force

        foreach ($directory in (@($Runtime.TclLibrary, $Runtime.TkLibrary) + @($Runtime.PackageLibraries))) {
            Copy-Item -LiteralPath $directory.FullName -Destination $stagingLib.FullName -Recurse -Force
        }
        foreach ($dll in $Runtime.Dlls) {
            Copy-Item -LiteralPath $dll.FullName -Destination $stagingBin.FullName -Force
        }
        [System.IO.File]::WriteAllText(
            (Join-Path $stagingDirectory ".complete"),
            $Digest,
            [System.Text.Encoding]::ASCII
        )

        try {
            # Directory.Move is atomic and fails if another launcher already
            # published this cache; Move-Item would nest the staging directory.
            [System.IO.Directory]::Move($stagingDirectory, $CacheDirectory)
        }
        catch {
            # Another launcher may have completed the same cache concurrently.
            if (-not (Test-CompleteTclTkCache $CacheDirectory $Runtime $Digest)) {
                throw
            }
        }
    }
    finally {
        if (Test-Path -LiteralPath $stagingDirectory -PathType Container) {
            Remove-SafeCacheDirectory $CacheRoot $stagingDirectory
        }
    }
}

function Remove-OldTclTkCaches {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CacheRoot,

        [Parameter(Mandatory = $true)]
        [string]$CurrentCacheDirectory
    )

    $currentFullPath = [System.IO.Path]::GetFullPath($CurrentCacheDirectory).TrimEnd('\')
    foreach ($directory in (Get-ChildItem -LiteralPath $CacheRoot -Directory -ErrorAction SilentlyContinue)) {
        $candidateFullPath = [System.IO.Path]::GetFullPath($directory.FullName).TrimEnd('\')
        if ($candidateFullPath.Equals($currentFullPath, [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }

        $isFinishedCache = $directory.Name -match '^tcltk-.*-[0-9a-f]{16}$'
        $isAbandonedStaging = (
            $directory.Name -match '^tcltk-.*-[0-9a-f]{16}\.creating-' -and
            $directory.LastWriteTimeUtc -lt [DateTime]::UtcNow.AddDays(-1)
        )
        if ($isFinishedCache -or $isAbandonedStaging) {
            try {
                Remove-SafeCacheDirectory $CacheRoot $directory.FullName
            }
            catch {
                Write-Warning "Could not remove old Tcl/Tk cache '$($directory.FullName)': $($_.Exception.Message)"
            }
        }
    }
}

function Invoke-TkWindowProbe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Python
    )

    $probeCode = "import tkinter as tk; root = tk.Tk(); root.withdraw(); root.update_idletasks(); root.destroy()"
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& $Python -c $probeCode 2>&1)
        $probeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }

    return [pscustomobject]@{
        Success = ($probeExitCode -eq 0)
        Output = (($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine).Trim()
    }
}

function Set-TclTkEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TclLibrary,

        [Parameter(Mandatory = $true)]
        [string]$TkLibrary,

        [Parameter(Mandatory = $true)]
        [string]$BinaryDirectory,

        [Parameter(Mandatory = $true)]
        [string]$BasePath
    )

    $env:TCL_LIBRARY = $TclLibrary
    $env:TK_LIBRARY = $TkLibrary
    $env:PATH = "$BinaryDirectory;$BasePath"
}

try {
    $environmentRoot = Join-Path $projectRoot "runtime\env\samroad_env"
    $python = Join-Path $environmentRoot "python.exe"
    $guiEntry = Join-Path $projectRoot "code\user_workflow_gui.py"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "The bundled Python runtime was not found: $python"
    }
    if (-not (Test-Path -LiteralPath $guiEntry -PathType Leaf)) {
        throw "The GUI entry point was not found: $guiEntry"
    }

    $runtime = Get-TclTkRuntime $environmentRoot
    $originalPath = $env:PATH
    $environmentBin = Join-Path $environmentRoot "Library\bin"
    $basePath = "$environmentRoot;$environmentBin;$(Join-Path $environmentRoot 'Scripts');$originalPath"

    # Keep the rest of the portable runtime and GIS resources in the project.
    $env:GDAL_DATA = Join-Path $environmentRoot "Lib\site-packages\rasterio\gdal_data"
    $env:PROJ_DATA = Join-Path $environmentRoot "Lib\site-packages\rasterio\proj_data"
    $env:PROJ_LIB = $env:PROJ_DATA
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    Set-TclTkEnvironment $runtime.TclLibrary.FullName $runtime.TkLibrary.FullName $environmentBin $basePath
    $directProbe = Invoke-TkWindowProbe $python

    if (-not $directProbe.Success) {
        if (-not [regex]::IsMatch($projectRoot, '[^\x00-\x7F]')) {
            throw "Tk could not create a window from the project runtime.$([Environment]::NewLine)$($directProbe.Output)"
        }
        if ([string]::IsNullOrWhiteSpace($env:PUBLIC)) {
            throw "Tk failed from the non-ASCII project path, and the PUBLIC directory is unavailable.$([Environment]::NewLine)$($directProbe.Output)"
        }

        $cacheRoot = Join-Path $env:PUBLIC "SamRoadChangeRuntime"
        [void](New-Item -ItemType Directory -Path $cacheRoot -Force)
        $identity = Get-TclTkCacheIdentity $runtime
        $cacheDirectory = Join-Path $cacheRoot $identity.DirectoryName
        New-TclTkCache $cacheRoot $cacheDirectory $runtime $identity.Digest

        $cachedTclLibrary = Join-Path $cacheDirectory "lib\$($runtime.TclLibrary.Name)"
        $cachedTkLibrary = Join-Path $cacheDirectory "lib\$($runtime.TkLibrary.Name)"
        $cachedBin = Join-Path $cacheDirectory "bin"
        Set-TclTkEnvironment $cachedTclLibrary $cachedTkLibrary $cachedBin $basePath

        $cachedProbe = Invoke-TkWindowProbe $python
        if (-not $cachedProbe.Success) {
            throw "Tk could not create a window after preparing the Tcl/Tk cache '$cacheDirectory'.$([Environment]::NewLine)$($cachedProbe.Output)"
        }
        Remove-OldTclTkCaches $cacheRoot $cacheDirectory
    }

    Push-Location $projectRoot
    try {
        & $python $guiEntry
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}
catch {
    Write-Host ""
    Write-Host "[STARTUP FAILED] $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
}

exit $exitCode
