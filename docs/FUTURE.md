# Future plan — katali-lab

Engine scope stays **GGUF `qwen35moe` only** (hybrid DeltaNet + GQA + MoE, elastic expert cache, no EQS, no llama.cpp as engine).

## Done
| Model | Notes |
|-------|--------|
| Qwen3.6-35B-A3B Q4_K_M | Coherent generate; ~1–3 tok/s CPU |
| Qwen3.5-122B-A10B Q4_K_M | Multi-shard; MTP trunk skip 49→48; coherent; ~0.3–0.7 tok/s |

## Next flagship target
**Qwen3.5-397B-A17B** (GGUF, preferably Q4_K_M / solid 4-bit)

Why this one:
- Same family / arch (`qwen35moe`) as 35B and 122B — expected drop-in on the existing forward path with dim scaling.
- Real flagship step after 122B (397B total / ~17B active).

Expect when we pick it up:
- Multi-shard GGUF open (already supported).
- Larger dims (e.g. hidden ~4096, more layers, 512 experts / 10+1 active — confirm from GGUF inspect).
- MTP / NextN trunk skip (same class of bug as 122B).
- Disk ~200+ GB for 4-bit; RAM preferably ~256 GB class.
- CPU decode much slower than 122B.
- Text path only first (no vision).

## Explicitly not the next target
- Dense / small Qwen3.5 sizes that are not `qwen35moe` — need a new forward path.
- EQS packs — that is the katali2 reference track, not katali-lab.
- Downloading 397B until we choose to start that work.

Last updated: 2026-09-22

## Speed (active focus)
See [OPTIMIZE.md](OPTIMIZE.md) — beat usable llama.cpp CPU speed on the same GGUFs without losing accuracy.

