@echo off
title PaintsUndo Setup
cd /d "%~dp0"

REM ============================================================
REM  Double-click installer for Paints-UNDO.
REM  Finds conda (even if it's not on PATH), activates it, and
REM  runs setup.py. Any arguments are passed through, e.g.:
REM     setup.bat --update --install-dir H:\PaintsUndo
REM ============================================================

REM 1) Already on PATH? Use it (and activate base so python is available).
where conda >nul 2>nul
if %ERRORLEVEL%==0 (
    for /f "delims=" %%i in ('conda info --base 2^>nul') do call "%%i\Scripts\activate.bat"
    goto run
)

REM 2) Otherwise search the usual Miniconda/Anaconda install locations.
set "CANDIDATES=%USERPROFILE%\miniconda3;%USERPROFILE%\anaconda3;%USERPROFILE%\AppData\Local\miniconda3;%USERPROFILE%\AppData\Local\anaconda3;%LOCALAPPDATA%\miniconda3;%LOCALAPPDATA%\anaconda3;C:\ProgramData\miniconda3;C:\ProgramData\Anaconda3;%USERPROFILE%\Miniconda3;%USERPROFILE%\Anaconda3"
for %%P in ("%CANDIDATES:;=" "%") do (
    if exist "%%~P\Scripts\activate.bat" (
        call "%%~P\Scripts\activate.bat"
        goto run
    )
)

echo.
echo [ERROR] Could not find conda on this PC.
echo.
echo Install Miniconda first (free):
echo     https://docs.conda.io/en/latest/miniconda.html
echo Then double-click this file again.
echo.
pause
exit /b 1

:run
echo Using conda from: %CONDA_PREFIX%
echo.
python "%~dp0setup.py" %*
echo.
pause
