# Research: Colibrì (JustVugg/colibri) — 2026-09-22

**Repo:** https://github.com/JustVugg/colibri  
**Site:** https://justvugg.github.io/colibri/  
**License:** Apache 2.0  
**Related fork/lineage:** javimosch/machin-anvil (renamed from machin-colibri; inspired by colibri philosophy)

## What it is
Pure-C MoE inference engine (“tiny engine, immense model”). Runs frontier MoEs by treating **VRAM / RAM / NVMe as one placement hierarchy**. Dense weights stay resident; **routed experts stream from disk** with LRU + learned pin + prefetch. GPU optional (CUDA / Metal / Vulkan) for speed only — semantics stay the same.

Slogan match to katali: small engine, big MoE, CPU-first, measure everything, don’t silently change the model for speed.

## Models (sibling engines, one `.c` each)
GLM-5.2/5.3 (744B), GLM-5.3-Flash, Inkling, Kimi K3, DeepSeek V4(+.1) Flash, Qwen3.8-Flash-Next, **Qwen3.6-35B-A3B**, OLMoE.

Qwen3.6 path is hybrid **Gated Attention + Gated DeltaNet** — same family as katali-lab’s `qwen35moe` interest — but Colibrì uses **its own int4-gs64 containers**, not GGUF Q4_K_M.

## Speed philosophy (steal these ideas, not the format)
1. **Placement ≠ semantics** — less RAM only slows; never changes router/precision quietly.
2. **JIT for weights** — routing heat → pin hot experts; one-layer-ahead prefetch (~71.6% predictable).
3. **I/O in the engine** — batch-union experts, async pipe, O_DIRECT (drive-dependent), dual-SSD mirror for bandwidth.
4. **Right-size cache** — small-RAM boxes are RAM-capped not disk-capped; oversized cache can hurt (aligns with katali2 RAM-cache findings).
5. **`--topp` can cut expert bytes/token** on disk-bound boxes (fewer experts read) — quality claim: measure.
6. **Speculation (MTP) must earn keep** — int8 MTP head required; int4 drafts die; can lose when hit-rate high.
7. **Quality gates** — gs64 group scales beat per-row int4; rotation/E8 ablations in-repo (#81).
8. **Profile phases** — disk vs matmul split published (e.g. disk-bound 0.41 tok/s even with fast NVMe).

## Honest numbers (from their docs)
| Setup | Decode (approx) |
|-------|-----------------|
| 25 GB cold stream | 0.05–0.1 tok/s |
| 128 GB CPU warm | ~1.8 tok/s |
| Qwen3.6 CPU profile example | ~2.9 tok/s (mixed prompt+gen wall) |
| Qwen3.6 + CUDA expert tier (2×8GB) | 1.44 → **10.05 tok/s** (bit-identical) |
| Full multi-GPU residency | several–~9 tok/s class |

So Colibrì’s win is **huge MoE on modest RAM via streaming**, not automatically beating llama.cpp on a fully-resident 35B GGUF.

## Relevance to katali-lab
| Take | Do not take |
|------|-------------|
| Expert LRU + learned pin + layer-ahead prefetch | Their safetensors/int4 container as default |
| Batch-union unique experts per token/batch | Replacing GGUF Q4_K_M path |
| Overlap disk fill with compute (reader pool) | Silent quant changes for speed |
| Phase profiling (disk vs matmul) | Assuming their tok/s on GGUF without measuring |
| gs64 / quality A/B discipline | Linking their engine as runtime |
| Dual-SSD / O_DIRECT experiments if 122B/397B disk-bound | Claiming Colibrì is a drop-in for lab |

**Positioning:** Colibrì = multitier **streaming MoE research engine**. katali-lab = **GGUF qwen35moe elastic CPU** (no EQS, no llama engine). Closest conceptual cousin to katali2 elastic cache; best reference for **122B/397B when RAM < weights**. For **35B fully in RAM**, still prioritize AVX2 batched MoE matmul + threads (OPTIMIZE.md P1–P2) — Colibrì’s RAM-resident Qwen3.6 CUDA tier shows expert compute, not only I/O, matters.

## Local note
Desktop has `katali2_colibri_stage` (staging name); treat as optional local experiment folder, not the upstream repo.

## Links
- https://github.com/JustVugg/colibri
- https://justvugg.github.io/colibri/
- https://github.com/javimosch/machin-anvil (inspired lineage)
