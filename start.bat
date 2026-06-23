@echo off
title PaintsUndo
cd /d "%~dp0"
chcp 65001 >nul
if exist "_banner.py" (
    conda run -n paints_undo python _banner.py
) else (
    echo   PaintsUndo
)
echo.
echo   Open your browser to: http://127.0.0.1:7860
echo   Models download automatically on first run (~8-10 GB)
echo.
conda run -n paints_undo python gradio_app.py
pause
