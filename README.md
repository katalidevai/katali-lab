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

## Local builds

The repository build produces:

- `katali-lab.exe` — command-line CPU inference engine
- `katali-lab-gui.exe` — Windows GUI frontend

The published GitHub repository intentionally contains only these two executable builds. Source, models, logs, and benchmark artifacts remain local.

## References

- [katali2 reference notes](docs/FUTURE.md)
- [35B CPU optimization profile](docs/35B_OPTIMIZE.md)
- [122B CPU optimization profile](docs/122B_OPTIMIZE.md)
- [GGUF parity checklist](docs/KATALI2_GGUF_PARITY.md)
- [Qwen3.5 collection](https://huggingface.co/collections/Qwen/qwen35)
