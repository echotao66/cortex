@echo off
REM One-shot launcher for the Cortex web demo.
REM Usage:  run.bat          (port 8000)
REM         run.bat 8001     (custom port)
REM NOTE: keep this file ASCII-only -- cmd.exe parses .bat with the console
REM       codepage, so UTF-8 Chinese text here would come out garbled.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run] .venv not found. Create it first:
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

if not exist ".env" (
    echo [run] WARNING: .env not found -- copy .env.example to .env and fill in DEEPSEEK_API_KEY
)

set PORT=%1
if "%PORT%"=="" set PORT=8000

REM No ">" in echo lines -- cmd would read it as a redirect operator.
echo [run] cortex listening on http://127.0.0.1:%PORT%/
echo [run] press Ctrl+C to stop (memory + MCP subprocess are flushed on exit)

REM Open the browser a few seconds later, so uvicorn has time to boot.
start "" cmd /c "timeout /t 3 /nobreak >nul & explorer http://127.0.0.1:%PORT%/"

REM Run uvicorn in the foreground so Ctrl+C reaches it and lifespan
REM shutdown runs (saves memory to .cortex\, closes the MCP subprocess).
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port %PORT%
