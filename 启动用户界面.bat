@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo The application failed to start. Error code: %EXIT_CODE%
    echo Please keep this window open and send the error text to the developer.
    pause
)
endlocal & exit /b %EXIT_CODE%
