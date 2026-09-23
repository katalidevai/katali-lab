@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo === Katali Lab Chat — publish (framework-dependent single-file, win-x64) ===
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

REM Prefer lab root: parent of this gui folder
set "LABROOT=%~dp0.."
if exist "%LABROOT%\katali-lab.exe" (
  copy /Y "%OUT%" "%LABROOT%\katali-lab-gui.exe" >nul
  echo Copied to %LABROOT%\katali-lab-gui.exe
) else (
  copy /Y "%OUT%" "%LABROOT%\katali-lab-gui.exe" >nul 2>nul
  if exist "%LABROOT%\katali-lab-gui.exe" (
    echo Wrote %LABROOT%\katali-lab-gui.exe
  ) else (
    echo Published to %OUT%
    echo Copy katali-lab-gui.exe next to katali-lab.exe manually.
  )
)

echo Done.
exit /b 0
