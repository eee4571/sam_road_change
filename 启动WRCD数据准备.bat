@echo off
setlocal
chcp 65001 >nul

set "WRCD_APP_ROOT=%~dp0"
set "WRCD_PYTHON=%WRCD_APP_ROOT%runtime\env\samroad_env\python.exe"
set "WRCD_SCRIPT=%WRCD_APP_ROOT%code\dev_tools\prepare_wrcd_test.py"

if not exist "%WRCD_PYTHON%" (
    echo [STARTUP FAILED] Bundled Python was not found:
    echo %WRCD_PYTHON%
    pause
    exit /b 1
)

if not exist "%WRCD_SCRIPT%" (
    echo [STARTUP FAILED] WRCD preparation script was not found:
    echo %WRCD_SCRIPT%
    pause
    exit /b 1
)

set "PATH=%WRCD_APP_ROOT%runtime\env\samroad_env;%WRCD_APP_ROOT%runtime\env\samroad_env\Library\bin;%WRCD_APP_ROOT%runtime\env\samroad_env\Scripts;%PATH%"
set "TCL_LIBRARY=%WRCD_APP_ROOT%runtime\env\samroad_env\Library\lib\tcl8.6"
set "TK_LIBRARY=%WRCD_APP_ROOT%runtime\env\samroad_env\Library\lib\tk8.6"
set "GDAL_DATA=%WRCD_APP_ROOT%runtime\env\samroad_env\Lib\site-packages\rasterio\gdal_data"
set "PROJ_DATA=%WRCD_APP_ROOT%runtime\env\samroad_env\Lib\site-packages\rasterio\proj_data"
set "PROJ_LIB=%PROJ_DATA%"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

pushd "%WRCD_APP_ROOT%"
"%WRCD_PYTHON%" "%WRCD_SCRIPT%" %*
set "WRCD_EXIT_CODE=%ERRORLEVEL%"
popd

if not "%WRCD_EXIT_CODE%"=="0" (
    echo.
    echo WRCD preparation window exited with code %WRCD_EXIT_CODE%.
    pause
)

endlocal & exit /b %WRCD_EXIT_CODE%
