@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Admin\Documents\GitHub\F1_XR_AI"
del /q test_done.txt 2>nul
".\venv\Scripts\python.exe" -m pytest -q -p no:cacheprovider > test_out.txt 2>&1
echo === EXITCODE %ERRORLEVEL% >> test_out.txt
echo done > test_done.txt
