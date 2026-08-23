@echo off
set "TOOL_DIR=%~dp0"
set "PYTHON_EXE=%TOOL_DIR%..\..\..\runtime\env\samroad_env\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" "%TOOL_DIR%run_test.py" --recovery-only %*
