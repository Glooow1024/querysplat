@echo off
setlocal
cd /d "%~dp0"

python -c "import numpy" >nul 2>&1
if not errorlevel 1 (
    python trajectory_viewer.py
    if errorlevel 1 pause
    exit /b
)

py -c "import numpy" >nul 2>&1
if not errorlevel 1 (
    py trajectory_viewer.py
    if errorlevel 1 pause
    exit /b
)

echo No Python interpreter with NumPy was found.
echo.
echo The viewer tried both "python" and "py".
echo Check the interpreter with:
echo     python -c "import sys, numpy; print(sys.executable, numpy.__version__)"
echo.
echo Or run the viewer with the full path to the Python that has NumPy:
echo     D:\Python\Python314\python.exe trajectory_viewer.py
pause
exit /b 1
