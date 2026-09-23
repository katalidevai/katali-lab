@echo off
setlocal
set CC=gcc
set CFLAGS=-std=c11 -O2 -Wall -Wextra -Ivendor\katali_gguf_dense\include -Iinclude -DKATALI_GGUF_SDK_BUILD
set SRC=vendor\katali_gguf_dense\src\gguf_reader.c vendor\katali_gguf_dense\src\gguf_dtype.c vendor\katali_gguf_dense\src\gguf_simd_x86.c vendor\katali_gguf_dense\src\gguf_threads.c vendor\katali_gguf_dense\src\gguf_kernels.c vendor\katali_gguf_dense\src\gguf_tokenizer.c vendor\katali_gguf_dense\src\gguf_sampler.c vendor\katali_gguf_dense\src\gguf_model.c vendor\katali_gguf_dense\src\gguf_backend.c vendor\katali_gguf_dense\src\gguf_session.c vendor\katali_gguf_dense\src\gguf_prof.c vendor\katali_gguf_dense\src\gguf_server.c vendor\katali_gguf_dense\src\katali_gguf_sdk.c vendor\katali_gguf_dense\src\katali_dense_cuda.c src\katali_cuda.c src\platform.c
%CC% %CFLAGS% -o katali-lab-qwen3.exe vendor\katali_gguf_dense\src\gguf_cli.c %SRC% -lm -lpsapi -lws2_32
if errorlevel 1 exit /b 1
echo OK: katali-lab-qwen3.exe
endlocal






