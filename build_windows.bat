@echo off
REM ============================================================
REM  Terminator Vision – build skript pro Windows
REM  Spustit v kořenové složce projektu
REM ============================================================

echo [1/3] Instalace zavislosti...
pip install pyinstaller opencv-python ultralytics numpy pillow pygame

echo [2/3] Sestaveni exe...
pyinstaller terminator_vision.spec --clean --noconfirm

echo [3/3] Hotovo.
echo Vystup: dist\TerminatorVision\TerminatorVision.exe
pause
