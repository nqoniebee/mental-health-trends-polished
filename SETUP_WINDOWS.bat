@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto install
where py >nul 2>nul
if errorlevel 1 goto no_python
py -3.11 -m venv .venv
if not errorlevel 1 goto install
py -3.12 -m venv .venv
if errorlevel 1 goto no_python
:install
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe setup_project.py
if errorlevel 1 goto failed
echo Setup complete. Next time use RUN_WINDOWS.bat.
pause
exit /b 0
:no_python
echo Install Python 3.11 or 3.12 with the Windows Python launcher, then retry.
pause
exit /b 1
:failed
echo Setup did not finish. Check the error above and internet connection, then rerun this file.
pause
exit /b 1
