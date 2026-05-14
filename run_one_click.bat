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
  %BOOTSTRAP_PY% -m venv "%VENV_DIR%"
)

if exist "%VENV_PY%" (
  "%VENV_PY%" -V >nul 2>nul
  if errorlevel 1 (
    rmdir /s /q "%VENV_DIR%"
    %BOOTSTRAP_PY% -m venv "%VENV_DIR%"
  )
)

if exist "%VENV_PY%" (
  "%VENV_PY%" -c "import sysconfig; soabi=(sysconfig.get_config_var('SOABI') or '').lower(); raise SystemExit(0 if 'cp314t' in soabi else 1)" >nul 2>nul
  if not errorlevel 1 (
    echo Detected free-threaded Python venv, rebuilding with compatible interpreter...
    rmdir /s /q "%VENV_DIR%"
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

echo [1/3] Installing dependencies...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed, trying critical runtime dependencies...
  "%VENV_PY%" -c "import httpx, dotenv, chromadb" >nul 2>nul
  if errorlevel 1 (
    "%VENV_PY%" -m pip install httpx python-dotenv chromadb
    if errorlevel 1 (
      echo Critical dependencies installation failed.
    )
  )
)

"%VENV_PY%" -c "import httpx" >nul 2>nul
if errorlevel 1 (
  echo Missing required dependency: httpx
  echo Please fix dependency installation in .venv, then re-run run_one_click.bat
  exit /b 1
)

"%VENV_PY%" -c "import chromadb" >nul 2>nul
if errorlevel 1 (
  echo Missing required dependency: chromadb
  echo Current Python may be incompatible. Recommended: install Python 3.13 and re-run run_one_click.bat
  exit /b 1
)

echo [2/3] Running unit tests...
"%VENV_PY%" -m unittest discover -s tests -p "test_*.py"
if errorlevel 1 (
  echo Unit tests failed.
  exit /b 1
)

echo [3/3] Starting CLI chat...
"%VENV_PY%" main.py chat
exit /b %errorlevel%
