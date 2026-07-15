@echo off
setlocal

cd /d "%~dp0"

where codex.cmd >nul 2>nul
if errorlevel 1 (
    echo Codex CLI was not found. Install or repair Codex first.
    pause
    exit /b 1
)

codex.cmd resume 019f596d-97af-7053-b969-7b4c0785f825 -C "%~dp0"

if errorlevel 1 (
    echo.
    echo Failed to resume the saved Codex session.
    pause
)

endlocal
