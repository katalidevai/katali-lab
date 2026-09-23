@echo off
setlocal
set CC=gcc
set CFLAGS=-std=c11 -O2 -Wall -Wextra -mavx2 -mfma -mf16c -D_FILE_OFFSET_BITS=64 -Iinclude -Ivendor\katali_gguf\include
set SRC=src\platform.c src\gguf_reader.c src\q4.c src\ecache.c src\ops.c src\model.c src\moe_index.c src\moe_ffn.c src\attn_gqa.c src\attn_delta.c src\forward.c src\host.c src\katali_cuda.c src\vram_cache.c src\api_server.c src\main.c
set VSRC=vendor\katali_gguf\src\gguf_reader.c vendor\katali_gguf\src\gguf_dtype.c vendor\katali_gguf\src\gguf_simd_x86.c vendor\katali_gguf\src\gguf_threads.c vendor\katali_gguf\src\gguf_kernels.c vendor\katali_gguf\src\gguf_prof.c vendor\katali_gguf\src\gguf_qwen35.c vendor\katali_gguf\src\gguf_tokenizer.c vendor\katali_gguf\src\gguf_sampler.c
set OUT=katali-lab.exe

echo Building %OUT% ...
%CC% %CFLAGS% %SRC% %VSRC% -o %OUT% -lm -lws2_32
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)
echo OK: %OUT%
endlocal
