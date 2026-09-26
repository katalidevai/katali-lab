@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo === Katali Lab Chat - publish (framework-dependent single-file, win-x64) ===
dotnet publish -c Release -r win-x64 --self-contained false -p:PublishSingleFile=true -o "%~dp0publish"
if errorlevel 1 (
  echo Publish failed.
  exit /b 1
)

set "OUT=%~dp0publish\katali-lab-gui.exe"
if not exist "%OUT%" (
  echo Expected output not found: %OUT%
  exit /b 1
)

REM This folder is gui\katali-lab-gui - lab root is two levels up
set "LABROOT=%~dp0..\.."
copy /Y "%OUT%" "%LABROOT%\katali-lab-gui.exe" >nul
if exist "%LABROOT%\katali-lab-gui.exe" (
  echo Deployed %LABROOT%\katali-lab-gui.exe
) else (
  echo Published to %OUT%
  echo Copy katali-lab-gui.exe next to katali-lab.exe manually.
)

echo Done.
exit /b 0
