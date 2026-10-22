@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe goto setup
if not exist models\stress_classifier.joblib goto setup
.venv\Scripts\python.exe -m streamlit run app.py
pause
exit /b 0
:setup
echo Please run SETUP_WINDOWS.bat once before starting the app.
pause
exit /b 1
