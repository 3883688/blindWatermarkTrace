@echo off
setlocal

cd /d "%~dp0"

where codex.cmd >nul 2>nul
if errorlevel 1 (
    echo Codex CLI was not found. Install or repair Codex first.
    pause
    exit /b 1
)

codex.cmd resume 019f8273-2b83-7d63-841a-a0f8930d6e8f -C "%~dp0"

if errorlevel 1 (
    echo.
    echo Failed to resume the saved Codex session.
    pause
)

endlocal
