# katali-lab

`katali-lab` is a CPU-only native inference engine for large quantized Qwen3.5 MoE models. It adapts the proven katali2 elastic expert-cache and SSD/mmap ideas to ordinary GGUF model files.

This project intentionally targets CPU execution. GPU, CUDA, ROCm, and GPU-offload work are out of scope.

## Current models

| Model | Status | CPU profile | Official model | GGUF source |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | Supported | Automatic routed-expert fusion; mmap/cache profile | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | [bartowski GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF) |
| Qwen3.5-122B-A10B | Supported | Raw mmap; expert cache disabled by default; fusion disabled for sustained CPU decode | [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | [bartowski GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-122B-A10B-GGUF) |

The official model cards are the source of truth for model configuration and licensing. GGUF repositories are community conversions; verify quantization, shard completeness, and tokenizer files before use.

## Parked flagship

| Model | Status | Why it is next | CPU reality |
|---|---|---|---|
| Qwen3.5-397B-A17B | Parked, next integration | Same `qwen3_5_moe` family and the natural scale-up from 122B | Expect roughly 200+ GB for a solid 4-bit build, very high RAM demand, and slower CPU decode |

Links: [official model](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), [GGUF search](https://huggingface.co/models?apps=llama.cpp&other=base_model%3Aquantized%3AQwen%2FQwen3.5-397B-A17B).

Before downloading 397B, inspect its configuration and GGUF metadata. Confirm hidden size, layer count, expert count, active top-k, tensor naming, shard layout, and MTP/NextN metadata against the existing loader. Do not begin with a full download until the CPU memory and storage budget is accepted.

## Future model roadmap

1. **Qwen3.5-397B-A17B** — primary next target. Extend the existing Qwen3.5 MoE path only where inspection proves a real shape or metadata difference.
2. **Qwen3.5 quantization variants** — evaluate Q3/Q4/Q5 GGUF variants for CPU RAM and SSD tradeoffs; preserve accuracy before chasing speed.
3. **Smaller Qwen3.5 text models** — useful regression and low-RAM test targets if an official or reliable GGUF is available, but not the main flagship path.
4. **Other architectures** — explicitly deferred. They require a separate loader/forward path and are not part of the current `qwen35moe` CPU scope.

Out of scope for this roadmap: GPU backends, CUDA/ROCm, EQS as a user-facing model format, and changing trained MoE top-k to create a misleading “small mode.”

## CPU principles

- Prefer GGUF plus memory mapping so model weights can remain SSD-backed.
- Keep the trained routing configuration, including top-k, unchanged.
- Treat SSD-bound operation as a first-class mode; only add more asynchronous pipeline behavior after measuring a real SSD-bound machine.
- Benchmark sustained generation, not only one-token or warm-cache results.
- Keep model-specific optimizations behind shape/configuration checks rather than assuming that a larger sibling behaves like the 35B model.

## Planned automatic CUDA support

Katali-lab is CPU-first today, but the planned GPU mode will use the same elastic model rather than requiring the whole model to fit in VRAM.

At startup, the executable will detect whether an NVIDIA CUDA device and usable CUDA runtime are available:

```text
CUDA available:  SSD -> system RAM expert cache -> GPU VRAM -> CUDA compute
CUDA unavailable: SSD -> system RAM expert cache -> CPU compute
```

The application will remain one executable. With CUDA available, active experts and compute state will move to VRAM while cold experts remain in the system-RAM cache or on SSD. Without CUDA—or if CUDA initialization fails—the engine will automatically fall back to the existing CPU path and report the reason.

The planned CUDA backend will add GPU kernels for quantized matrix-vector operations, routed MoE experts, Gated DeltaNet, full attention, and state transfers. The CPU/SSD path remains the correctness reference and fallback. CUDA support is intentionally limited to NVIDIA hardware; Vulkan, ROCm, and other GPU backends are out of scope for this plan.

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
