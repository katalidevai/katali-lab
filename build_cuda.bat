@echo off
rem ---------------------------------------------------------------------------
rem Builds katali_cuda.dll — the OPTIONAL CUDA backend.
rem
rem Why this is a separate build: the engine (build.bat) is compiled by MinGW
rem gcc, and nvcc on Windows requires cl.exe as its host compiler — it cannot
rem use MinGW. So the backend is a standalone DLL with a flat C ABI that
rem katali-lab.exe finds at runtime via LoadLibrary. Nothing links against it,
rem which is what makes CUDA genuinely optional.
rem
rem Overridable environment:
rem   KATALI_CUDA_HOME  toolkit root (bin\nvcc.exe, include, lib\x64, nvvm)
rem   KATALI_VCVARS     vcvars64.bat that sets up cl.exe
rem   KATALI_CUDA_ARCH  -arch value, default sm_89 (Ada / RTX 4060)
rem ---------------------------------------------------------------------------
setlocal

if not defined KATALI_CUDA_HOME set "KATALI_CUDA_HOME=C:\Users\joanr\cuda\home"
if not defined KATALI_CUDA_ARCH set "KATALI_CUDA_ARCH=sm_89"

set "VCVARS=%KATALI_VCVARS%"
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%KATALI_CUDA_HOME%\bin\nvcc.exe" (
  echo nvcc not found at "%KATALI_CUDA_HOME%\bin\nvcc.exe"
  echo Set KATALI_CUDA_HOME to a CUDA Toolkit root and retry.
  exit /b 1
)
if not exist "%VCVARS%" (
  echo vcvars64.bat not found. Set KATALI_VCVARS and retry.
  exit /b 1
)

echo Setting up MSVC host compiler...
call "%VCVARS%" >nul 2>&1
if errorlevel 1 ( echo VCVARS FAILED & exit /b 1 )

echo Building katali_cuda.dll (arch=%KATALI_CUDA_ARCH%) ...
"%KATALI_CUDA_HOME%\bin\nvcc.exe" -shared -arch=%KATALI_CUDA_ARCH% -O2 --cudart static ^
  -Xcompiler "/W3" -I include -o katali_cuda.dll cuda\katali_cuda_backend.cu
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)
echo OK: katali_cuda.dll
endlocal