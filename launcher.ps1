$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$projectRoot = $PSScriptRoot
$appDrive = $null
$exitCode = 1

try {
    # The bundled Tcl/Tk runtime cannot load reliably from a Chinese path.
    # A temporary drive gives this standalone project an ASCII-only runtime path.
    foreach ($letter in @("S", "T", "U", "V", "W", "X", "Y", "Z")) {
        $candidate = "${letter}:"
        if (Test-Path "${candidate}\") {
            continue
        }
        & subst.exe $candidate $projectRoot
        if ($LASTEXITCODE -eq 0) {
            $appDrive = $candidate
            break
        }
    }

    if (-not $appDrive) {
        throw "No free temporary drive is available from S: through Z:."
    }

    $appRoot = "${appDrive}\"
    $python = Join-Path $appRoot "env\samroad_env\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "The bundled Python runtime was not found: $python"
    }

    $env:PATH = "${appRoot}env\samroad_env;${appRoot}env\samroad_env\Library\bin;${appRoot}env\samroad_env\Scripts;$env:PATH"
    $env:TCL_LIBRARY = "${appRoot}env\samroad_env\Library\lib\tcl8.6"
    $env:TK_LIBRARY = "${appRoot}env\samroad_env\Library\lib\tk8.6"
    # The portable environment stores GIS databases inside the wheels rather
    # than Conda's Library\share directories.  Pointing PROJ at a missing
    # directory makes valid EPSG rasters appear as LOCAL_CS.
    $env:GDAL_DATA = "${appRoot}env\samroad_env\Lib\site-packages\rasterio\gdal_data"
    $env:PROJ_DATA = "${appRoot}env\samroad_env\Lib\site-packages\rasterio\proj_data"
    $env:PROJ_LIB = $env:PROJ_DATA
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    Push-Location $appRoot
    try {
        & $python (Join-Path $appRoot "user_workflow_gui.py")
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
finally {
    if ($appDrive) {
        & subst.exe $appDrive /d 2>$null
    }
}

exit $exitCode
