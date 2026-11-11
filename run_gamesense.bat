@echo off
setlocal EnableExtensions
rem ============================================================
rem  GameSense launcher (optional). Normal start:  streamlit run app.py
rem  This file finds or creates a Python environment, installs what
rem  is needed (first time only) and starts the app.
rem ============================================================
cd /d "%~dp0"
title GameSense
echo.
echo  GameSense - starting from: %CD%
echo.

set "PY="
if exist "%CD%\.venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
if not defined PY if exist "D:\1_GameSense_code\GameSense\.venv\Scripts\python.exe" set "PY=D:\1_GameSense_code\GameSense\.venv\Scripts\python.exe"

if not defined PY (
  echo  No Python environment found. Creating one in this folder - this happens only once...
  where uv >nul 2>nul && uv venv --seed .venv
  if not exist "%CD%\.venv\Scripts\python.exe" ( where py >nul 2>nul && py -3 -m venv .venv )
  if not exist "%CD%\.venv\Scripts\python.exe" ( where python >nul 2>nul && python -m venv .venv )
  if exist "%CD%\.venv\Scripts\python.exe" set "PY=%CD%\.venv\Scripts\python.exe"
)

if not defined PY (
  echo.
  echo  Could not find or create Python. Install Python 3.11 or newer from https://www.python.org/downloads/
  echo  (tick "Add python.exe to PATH"), then double-click this file again.
  pause
  exit /b 1
)
echo  Using Python: %PY%

"%PY%" -c "import pandas,numpy,sklearn,scipy,joblib,yaml,streamlit,plotly" >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Installing the packages GameSense needs - this can take a few minutes, first time only...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo  The installation failed. Take a screenshot of the messages above and send it for help.
    pause
    exit /b 1
  )
)

echo.
echo  Starting GameSense. Your browser should open at http://localhost:8501
echo  Keep this window open while you use the app. Close it to stop GameSense.
echo.
"%PY%" -m streamlit run app.py
echo.
echo  GameSense stopped. If you saw an error above, send a screenshot of it.
pause
