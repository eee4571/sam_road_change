@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

rem Portable UI Python is independent of the algorithm runtime.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUTF8=1"
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "GUI_PYTHON="
if exist "%~dp0gui_env\python.exe" (
    set "GUI_PYTHON=%~dp0gui_env\python.exe"
    goto ready
)
rem Source checkout only: allow an explicitly installed host UI environment.
python -c "import PySide6.QtWidgets" >nul 2>&1
if not errorlevel 1 set "GUI_PYTHON=python"
if defined GUI_PYTHON goto ready

py -c "import PySide6.QtWidgets" >nul 2>&1
if not errorlevel 1 set "GUI_PYTHON=py"
if defined GUI_PYTHON goto ready

goto missing

:ready
"%GUI_PYTHON%" "%~dp0standalone.py" %*
set "RESULT=%ERRORLEVEL%"
if /i "%~1"=="--check" goto done
if "%RESULT%"=="0" goto done
echo.
echo Plugin failed to start or exited with an error. See details above.
pause
:done
popd
exit /b %RESULT%

:missing
echo The portable gui_env is missing. Extract the complete portable ZIP again.
echo For source development only, install Python and PySide6:
echo     python -m pip install PySide6
echo Do not install the UI dependencies into runtime.
echo Models are not required just to open the interface.
if /i "%~1"=="--check" goto missing_exit
pause
:missing_exit
popd
exit /b 1
