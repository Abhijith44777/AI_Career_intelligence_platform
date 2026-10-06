@echo off
REM Local development on Windows: starts the API in a new window, then the dashboard.
cd /d "%~dp0\.."
if not exist .env (
  echo No .env found - copy .env.example to .env first.
  exit /b 1
)
python manage.py check-config
start "Career Intelligence API" cmd /k "uvicorn api:app --host 127.0.0.1 --port 8000"
timeout /t 4 >nul
streamlit run app.py
