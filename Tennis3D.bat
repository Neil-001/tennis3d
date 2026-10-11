@echo off
rem Double-click to start Tennis3D. tennis3d.ps1 describes what the first run installs.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tennis3d.ps1" %*
set code=%errorlevel%
if not "%code%"=="0" if not defined CI pause
exit /b %code%
