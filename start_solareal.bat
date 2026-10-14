@echo off
title SolaREAL

:: ============================================================
:: SolaREAL - opens like an Anaconda Prompt, moves into the
:: project folder, and runs the app — exactly what you'd type
:: by hand. This window stays open so you can see any errors.
:: ============================================================

:: Where conda is installed
set "CONDA_ROOT=%USERPROFILE%\miniconda3"

:: The project folder
set "PROJECT_DIR=C:\Users\vboxuser\Desktop\solareal_app"

echo Activating conda...
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"

echo Moving to project folder...
cd /d "%PROJECT_DIR%"

echo Starting Streamlit...
echo.
streamlit run app.py

echo.
echo ----------------------------------------------------------
echo The app stopped, or a step above failed. This window stays
echo open on purpose so you can read what happened. Scroll up
echo to see the actual error if something went wrong.
echo ----------------------------------------------------------
pause
