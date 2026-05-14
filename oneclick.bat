@echo off
setlocal
cd /d "%~dp0."
set "PYTHONPATH=%cd%"
set "VENV_DIR=%cd%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "BOOTSTRAP_PY="

where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 -V >nul 2>nul
  if not errorlevel 1 (
    set "BOOTSTRAP_PY=py -3.13"
  ) else (
    py -3.14 -V >nul 2>nul
    if not errorlevel 1 (
      set "BOOTSTRAP_PY=py -3.14"
    ) else (
      set "BOOTSTRAP_PY=py -3"
    )
  )
)

if not defined BOOTSTRAP_PY (
  set "BOOTSTRAP_PY=python"
)

if not exist "%VENV_PY%" (
  echo Initializing virtual environment...
  %BOOTSTRAP_PY% -m venv "%VENV_DIR%"
)

if exist "%VENV_PY%" (
  "%VENV_PY%" -V >nul 2>nul
  if errorlevel 1 (
    rmdir /s /q "%VENV_DIR%"
    echo Rebuilding virtual environment...
    %BOOTSTRAP_PY% -m venv "%VENV_DIR%"
  )
)

if not exist "%VENV_PY%" (
  echo Failed to initialize virtual environment.
  exit /b 1
)

"%VENV_PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
  "%VENV_PY%" -m ensurepip --upgrade >nul
)

if not exist ".env" (
  if exist ".env.example" (
    copy /Y ".env.example" ".env" >nul
  )
)

echo [1/2] Installing dependencies...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed, trying critical runtime dependencies...
  "%VENV_PY%" -c "import httpx, dotenv, chromadb" >nul 2>nul
  if errorlevel 1 (
    "%VENV_PY%" -m pip install httpx python-dotenv chromadb
  )
)

echo [2/2] Starting interactive demo...
echo.
"%VENV_PY%" run_interactive.py testing/test1.xml --context "该车自动停车有问题?
echo.
exit /b %errorlevel%