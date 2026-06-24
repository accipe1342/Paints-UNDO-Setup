@echo off
title PaintsUndo
cd /d "%~dp0"
chcp 65001 >nul

REM Activate the conda env (rather than `conda run`) so output streams live
REM and errors / download progress are visible.
for /f "delims=" %%i in ('conda info --base 2^>nul') do set "CONDA_BASE=%%i"
if not defined CONDA_BASE (
    echo [ERROR] conda not found in PATH.
    echo Open an Anaconda Prompt and re-run, or run: conda init cmd.exe
    pause
    exit /b 1
)
call "%CONDA_BASE%\Scripts\activate.bat" paints_undo

if exist "_banner.py" ( python _banner.py ) else ( echo   PaintsUndo )
echo.
echo   Open your browser to: http://127.0.0.1:7860
echo   Models download automatically on first run (~8-10 GB)
echo.
python gradio_app.py
pause
