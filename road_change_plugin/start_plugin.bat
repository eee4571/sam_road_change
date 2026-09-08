@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

rem UI Python only. Algorithms still use the plugin's independent environment.
set "GUI_PYTHON="
python -c "import PySide6.QtWidgets" >nul 2>&1
if not errorlevel 1 set "GUI_PYTHON=python"
if defined GUI_PYTHON goto ready

py -c "import PySide6.QtWidgets" >nul 2>&1
if not errorlevel 1 set "GUI_PYTHON=py"
if defined GUI_PYTHON goto ready

if not exist "%~dp0runtime\env\samroad_env\python.exe" goto missing
"%~dp0runtime\env\samroad_env\python.exe" -c "import PySide6.QtWidgets" >nul 2>&1
if not errorlevel 1 set "GUI_PYTHON=%~dp0runtime\env\samroad_env\python.exe"
if not defined GUI_PYTHON goto missing

:ready
if /i "%~1"=="--check" (
    echo Ready: "%GUI_PYTHON%"
    popd
    exit /b 0
)
"%GUI_PYTHON%" "%~dp0standalone.py"
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" goto done
echo.
echo Plugin failed to start or exited with an error. See details above.
pause
:done
popd
exit /b %RESULT%

:missing
echo No Python with PySide6 was found.
echo Install Python and PySide6 for the interface, then run this BAT again:
echo     python -m pip install PySide6
echo Or place a complete environment with PySide6 in runtime\env\samroad_env.
echo Models are not required just to open the interface.
if /i "%~1"=="--check" goto missing_exit
pause
:missing_exit
popd
exit /b 1
