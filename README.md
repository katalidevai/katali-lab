# katali-lab

`katali-lab` is an elastic native inference engine for large quantized Qwen3.5 MoE models. It adapts the proven katali2 elastic expert-cache and SSD/mmap ideas to ordinary GGUF model files.

KATALI scales across whatever hardware is present: **CPU + system RAM + SSD** by default, extended to **GPU + CPU + system RAM + SSD** when a compatible NVIDIA CUDA GPU is available. Models larger than available RAM *or* VRAM still run through elastic memory management.

CUDA is **optional and never required**. The engine is built by MinGW gcc and **does not link against any CUDA library**; the GPU backend is a separate `katali_cuda.dll` (built by nvcc + MSVC, since nvcc cannot use a MinGW host compiler) that is discovered at runtime with `LoadLibrary`. Without that DLL, without a driver, or without an NVIDIA GPU, the engine runs the unchanged CPU path and reports why.

**Measured status:** the CUDA kernels are correct (`rel_L2 ≈ 2e-7` against the engine's own CPU kernels on real GGUF tensors — see `cuda-check` and `cuda-check-moe`) and the kernels run **6.9–10.9× faster than the CPU** on DRAM-bound expert traffic (`katali-lab.exe cuda-bench`). Layer-level MoE fusion is implemented: one GPU call per MoE layer instead of one per expert, cutting CUDA API calls/token **8×**, kernel launches **6×** and device syncs **24×**, all with byte-identical output.

It is nonetheless **still slower end-to-end than CPU**, so the GPU MoE path stays **opt-in** (`KATALI_CUDA_MOE=1`). See `docs/CUDA_PLAN.md` §12–§15.

**Stage A (this phase): DP4A + q8_1 activations.** The inner GEMV loop no longer dequantizes every weight element to float. The layer's activation is quantized **once** to q8_1 on the device and every selected expert's quants are dotted against it with `__dp4a` (the technique llama.cpp uses). Measured: the fused layer at the engine's operating point (`n_sel=8`) went **10.0 → 3.20 ms per 24 calls (3.1×)**, 36.3 → **114 GB/s** against a 134 GB/s no-arithmetic ceiling, with the q8_1 quantization *counted* (0.27 ms, 8 %). End-to-end the GPU arm rose **1.720 → 2.245 tok/s (+31 %)** in a matched A/B (and 1.705 → 2.268 in a second run), with the GPU MoE window per token falling **209.7 → 70.1 ms** and the CPU's GPU wait **221.7 → 79.8 ms**. Output was **byte-identical** to CPU and to the fp32 CUDA path on the reference prompt (`8EC7BAC655`), and a new synthetic test (`cuda-dp4a-selftest`) proves the kernels match the fp32 kernels to **rel_L2 8.3e-07** when the activation is exactly representable; on real weights the residual 7.2e-03 is q8_1 activation rounding, with 0 elements outside `1e-3·(1+|ref|)`. Enabled with `KATALI_CUDA_DP4A=1`; **default OFF**, so the default path is unchanged. See `docs/CUDA_PLAN.md` §17.

**Stage B (this phase): attention was re-examined and is *not* the next target.** Measurement split the 210 ms CPU attention bucket: **147 ms is the 30 DeltaNet/GDN layers**, **44.6 ms is the 10 full-attention layers** — the only part a FlashAttention-style port would address — and the host-side VRAM tier costs **48.5 ms**, more than all full attention combined. The KV cache lives in system RAM (`40 KB per token of context`), and the dense attention weights are 1.05 GB, so GPU attention needs a new residency tier before it saves anything. Recommendation, kill criterion and the smallest PoC are in `docs/CUDA_PLAN.md` §18.

**The measured root cause is not the kernel.** A cold-memory benchmark (`cuda-bench-cold`, rotating experts over a working set far larger than L2) shows the *same* fused kernel reaching **31.9 GB/s** — 240 GB/s is available per the raw stream test — while inside the real engine every layer runs at **2.8 GB/s** with zero cache misses. The reason: during inference the GPU is 50 % *busy* but sits at its **idle power state — 210 MHz core / 405 MHz memory versus 2505 / 8501 MHz at boost — a 21× lower memory clock**, because the workload is a ~6 % duty cycle (≈0.5 ms GPU per ≈8 ms CPU). Confirmed by holding the GPU busy with a second process: decode rose 1.837 → **2.537 tok/s** and the per-layer GPU window halved, 3.776 → **1.922 ms**, *while that process consumed 94 % of the machine*.

Kernel work was also tried and **rejected on measurement**: a 4-way unrolled column loop gained 11–15 % at n_sel≤4 but **lost 22 % at n_sel=8**, the engine's operating point, so `KATALI_GEMV_UNROLL=1` remains the default. The next phase should raise the GPU duty cycle (async overlap of GPU MoE with CPU attention), not micro-tune GEMV. Default behaviour is byte-for-byte the original CPU path (re-verified: 3.05 tok/s, `Manila`, 0 CUDA calls).

## Current models

| Model | Status | CPU profile | Official model | GGUF source |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | Supported | Automatic routed-expert fusion; mmap/cache profile | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | [bartowski GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF) |
| Qwen3.5-122B-A10B | Supported | Raw mmap; expert cache disabled by default; fusion disabled for sustained CPU decode | [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | [bartowski GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-122B-A10B-GGUF) |
| Qwen3-Coder-Next 80B-A3B | Supported | Hybrid DeltaNet/attention; 512-expert elastic cache; CPU-first profile | [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | [Qwen GGUF](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) |

The official model cards are the source of truth for model configuration and licensing. GGUF repositories are community conversions; verify quantization, shard completeness, and tokenizer files before use.

## Validated flagship

| Model | Status | Why it is next | CPU reality |
|---|---|---|---|
| Qwen3.5-397B-A17B | Validated CPU smoke test | Same `qwen3_5_moe` family; loaded and generated successfully through Katali-lab | 231.21 GiB Q4_K_M; measured about 0.11 tok/s decode |

Links: [official model](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), [GGUF search](https://huggingface.co/models?apps=llama.cpp&other=base_model%3Aquantized%3AQwen%2FQwen3.5-397B-A17B).

The 397B smoke test confirmed hidden size 4096, 61 layers, 512 experts, 10 routed experts plus 1 shared expert, and seven-shard GGUF loading. It is compatible, but not yet performance-practical on CPU.

## Latest CUDA model verification

The current CUDA-enabled executable was tested against both supported Qwen3.5 models on an RTX 4060 (8 GB VRAM), with `KATALI_CUDA_MOE=1`:

| Model | Result | Prefill | Decode | GPU result |
|---|---|---:|---:|---|
| Qwen3.5-35B-A3B Q4_K_M | `Manila` | 3.46 tok/s | 3.11 tok/s | 640 GPU experts, 0 CPU experts, 0 fallbacks |
| Qwen3.5-122B-A10B Q4_K_M | `Manila` | 0.465 tok/s | 0.267 tok/s | 768 GPU experts, 0 CPU experts, 0 fallbacks |

These are short smoke-test measurements, not universal benchmarks. The 122B model runs correctly through the elastic SSD → RAM → VRAM path, but its larger working set causes substantially more transfer and eviction pressure on this machine.

Qwen3-Coder-Next is also integrated and validated. Its Q4_K_M GGUF is approximately 45.09 GiB locally, with 80B total parameters and 3B active parameters. On the development RTX 4060, the matching smoke test measured **0.692 tok/s prefill and 0.484 tok/s decode with CUDA**, versus **0.880 tok/s prefill and 2.028 tok/s decode with CUDA disabled**. CPU mode is currently faster for this model; use `KATALI_CUDA=0` to disable GPU detection, or `KATALI_CUDA_MOE=1` to enable the experimental VRAM expert tier.

## Future model roadmap

1. **Qwen3.8-Flash-Next** — next parked flagship and new architecture target.
2. **Qwen3.5 quantization variants** — evaluate Q3/Q4/Q5 GGUF variants for CPU RAM and SSD tradeoffs.
3. **Smaller Qwen3.5 text models** — useful regression and low-RAM test targets.
4. **Other architectures** — explicitly deferred until the Qwen3.8 backend is understood.

Out of scope for this roadmap: EQS as a user-facing model format, changing trained MoE top-k to create a misleading “small mode,” and non-NVIDIA GPU backends (ROCm, Vulkan, Metal).

## CPU principles

- Prefer GGUF plus memory mapping so model weights can remain SSD-backed.
- Keep the trained routing configuration, including top-k, unchanged.
- Treat SSD-bound operation as a first-class mode; only add more asynchronous pipeline behavior after measuring a real SSD-bound machine.
- Benchmark sustained generation, not only one-token or warm-cache results.
- Keep model-specific optimizations behind shape/configuration checks rather than assuming that a larger sibling behaves like the 35B model.

## CUDA support — implemented, optional, and measured

The CUDA mode uses the same elastic model rather than requiring the whole model to fit in VRAM.

At startup the executable detects whether an NVIDIA CUDA device and a usable CUDA runtime are available:

```text
CUDA available:  SSD -> system RAM expert cache -> GPU VRAM -> CUDA compute
CUDA unavailable: SSD -> system RAM expert cache -> CPU compute
```

Inspect what the engine found:

```bat
katali-lab.exe cuda-info                  REM device, capability, VRAM, driver/runtime
katali-lab.exe cuda-check                 REM GPU kernels vs CPU kernels on real tensors
katali-lab.exe cuda-bench                 REM CPU vs GPU throughput for the decode ops
```

To build the optional backend you need `nvcc` (CUDA Toolkit) and `cl.exe`; `build_cuda.bat` produces `katali_cuda.dll`:

```bat
build.bat          REM engine (MinGW gcc) — unaffected by CUDA
build_cuda.bat     REM katali_cuda.dll (nvcc + MSVC) — optional
```

Environment: `KATALI_CUDA=0` disables detection, `KATALI_CUDA_DLL` overrides the DLL path, `KATALI_CUDA_MOE=1` enables the experimental VRAM expert tier, and `KATALI_VRAM_GB` caps its budget.

The application remains one executable. Active expert weights can move to VRAM while cold experts remain in the system-RAM cache or on SSD. Without CUDA—or if CUDA initialization fails—the engine automatically falls back to the CPU path. GPU kernels currently cover the quantized GEMV/matmul family and routed MoE experts; Gated DeltaNet and full-attention GPU kernels are not implemented yet.

## Local builds

The repository build produces:

- `katali-lab.exe` — command-line CPU inference engine
- `katali-lab-gui.exe` — Windows GUI frontend

The published GitHub repository intentionally contains only these two executable builds. Source, models, logs, and benchmark artifacts remain local.

## How to use `katali-lab.exe`

Open Command Prompt or PowerShell in the folder containing the executable.

Check the build and run the built-in quantization test:

```bat
katali-lab.exe info
katali-lab.exe selftest
```

Inspect a model before generating:

```bat
katali-lab.exe inspect C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
```

Generate with the 35B model:

```bat
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf "What is the capital of the Philippines? Reply with only the city name." --max 8 --pin 25 --cache-gb 8
```

Generate with the multi-shard 122B model by passing its split directory:

```bat
katali-lab.exe generate C:\models\Qwen_Qwen3.5-122B-A10B-Q4_K_M\Qwen_Qwen3.5-122B-A10B-Q4_K_M "Explain why the sky appears blue in one short sentence." --max 8 --pin 25 --cache-gb 12
```

For CPU-only operation, use memory mapping and the model-specific profiles in `docs/35B_OPTIMIZE.md` and `docs/122B_OPTIMIZE.md`. `--max` limits generated tokens, `--pin` controls the pinned-cache percentage, and `--cache-gb` sets the expert-cache budget.

## Local HTTP API

Start the built-in loopback API server:

```bat
katali-lab.exe api --port 8080
```

It listens only on `127.0.0.1`. Check that it is alive:

```bat
curl http://127.0.0.1:8080/health
```

Generate text using either `POST /generate` or the OpenAI-compatible `POST /v1/chat/completions` route:

```bat
curl -X POST http://127.0.0.1:8080/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"C:\\models\\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf\",\"prompt\":\"What is the capital of the Philippines?\",\"max_tokens\":8}"
```

The `model` field is optional and defaults to the 35B model path shown above. The request accepts `prompt` (or `content`) and `max_tokens` (or `max`). The API returns a JSON response containing the generated text in `choices[0].message.content`.

## Build verification

The current published executables were tested locally after the final rebuild. `katali-lab.exe selftest` passed, 35B generation completed coherently at 2.25 tok/s for one decode token, and the 122B bounded smoke test completed at 1.04 tok/s for one decode token. These are CPU measurements on the development desktop, not guaranteed performance on other machines.

## Next architecture target: Qwen3.8-Flash-Next

After the Qwen3.5-397B smoke test, the next planned model family is **Qwen3.8-Flash-Next**. It is not a drop-in Qwen3.5 model: current GGUF metadata identifies it as `qwen4exp`, with a newer sparse-expert design and a large n-gram embedding table. It therefore needs a separate architecture backend while reusing Katali-lab's general elastic storage ideas.

Official model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). CPU-oriented GGUF reference: [Unsloth Qwen3.8-Flash-Next GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

Approximate GGUF storage options:

| Quantization | Approximate size |
|---|---:|
| IQ3_M | 93 GiB |
| Q4_K_S | 105 GiB |
| Q4_K_M | 111 GiB / 120 GB |
| Q4_K_L | 130 GiB / 139 GB |
| Q5_K_M | 125 GiB / 135 GB |
| Q6_K | 157 GiB / 168 GB |
| Q8_0 | 175 GiB / 188 GB |
| BF16 | 330 GiB / 354 GB |

An optional MTP sidecar is approximately 2.6 GB. The first Qwen3.8 milestone is compatibility only: inspect the GGUF, load it, and complete one short CPU smoke generation. Optimization comes later.

### Qwen3.5-397B smoke result

The seven-shard Q4_K_M model was loaded successfully on the CPU/SSD path. A bounded generation completed at approximately **0.11 tokens/sec decode** after a roughly **356-second prefill**. A second 8-token request reached the model's thinking output but stopped before the final answer because the output limit was intentionally short. This confirms compatibility, not usable performance.

## References

- [katali2 reference notes](docs/FUTURE.md)
- [35B CPU optimization profile](docs/35B_OPTIMIZE.md)
- [122B CPU optimization profile](docs/122B_OPTIMIZE.md)
- [GGUF parity checklist](docs/KATALI2_GGUF_PARITY.md)
- [Qwen3.5 collection](https://huggingface.co/collections/Qwen/qwen35)
