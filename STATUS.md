# STATUS — katali-lab (2026-09-22)

Full snapshot of product goal, **katali2 reference usage**, GGUF ports, benches, and open work.

Companion docs:

- `docs/KATALI2_GGUF_PARITY.md` — EQS↔GGUF checklist
- `docs/OPTIMIZE.md` — ecache ports / env
- `docs/LAPTOP_8GB_GGUF.md` — real 8 GB laptop runbook
- `docs/FUTURE.md` — 397B parked
- `docs/SMOKE_122B.md` — 122B golden notes

---

## 1. How we use the katali2 reference

**Reference tree (authoritative finished engine):**

`C:\Users\joanr\Desktop\katali2-backup-pre-rac`

| Role | Detail |
|------|--------|
| What it is | Finished **EQS** elastic MoE engine (ExpertCache, pin/LFU, `ensure_many`, prefetch, range reads, freq warm-start, layer-major prefill, laptop-proven) |
| What it is not | Not the download format we want for users (EQS packs require conversion) |
| How katali-lab uses it | **Behavior reference only** — copy ExpertCache / host choreography / I/O policy; **reimplement fills on GGUF mmap spans** (`FillFn` / `ResolveFn`), never ship EQS as the product path |
| Proven on hardware | User ran katali2 on a **real 8 GB laptop** (~2 GB headroom) with **Qwen3 235B**; same-style question answered after **~800 s** |
| Key headers/sources consulted | `include/expert_cache.h`, `src/expert_cache.c`, `src/host_qwen.c`, `src/host_load.c`, `include/eqs.h`, `docs/expert_range_coalescing.md`, `docs/laptop_optimization_20260921.md` |
| Secondary references | Colibri / other engines only to fill gaps; **katali2 wins on conflicts** for decode defaults (READY/PIPE/STREAM stay off until SSD-bound) |
| Mapping rule | `eqs_pread` → GGUF fill/resolve; `dense.eqs` → always-mapped GGUF dense tensors; `experts.eqs` → `ecache` over gate/up/down spans; one packed EQS expert key → **three GGUF keys** (`expert*3+which`) |

**Product sentence:** keep katali2’s engine; swap EQS for GGUF so models are easy to download.

---

## 2. North star

| Item | Decision |
|------|----------|
| Product | Pure-C **GGUF** MoE for `qwen35moe` — no llama.cpp as engine, no EQS as user format |
| Bar | ~**8 GB RAM** laptops; accuracy + speed so users don’t default to llama.cpp |
| Targets | **35B-A3B** and **122B-A10B** GGUF first; **397B-A17B** parked |
| Active experts | Keep trained **top-k = 8** (35B ≈ **~3B active**). Do not ship lowered top-k as “2B mode” |
| Fake sim | Desktop Job Object 8 GB on a 32 GB PC ≠ real 8 GB laptop (page cache / mmap WS) |

Workspace: `C:\Users\joanr\Desktop\katali-lab` on `DESKTOP-46LDJSM` (~32 GB dev box).

---

## 3. Models

| Model | Path | Notes |
|-------|------|-------|
| 35B Q4_K_M | `C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` (~20.75 GiB) | layers=41/trunk=40, hidden=2048, experts=256, **active=8+shared**, intermediate=512 |
| 122B Q4_K_M | `C:\models\Qwen_Qwen3.5-122B-A10B-Q4_K_M\…` | Multi-shard; ~10B active; slower (~0.3 tok/s class on desktop) |
| 397B | Not downloaded | `docs/FUTURE.md` |

Golden: Philippines prompt → **Manila**.

---

## 4. Ports completed (katali2 behavior → GGUF)

| Port | katali2 source idea | Lab status |
|------|---------------------|------------|
| #1 | `ExpertCache` core | `ecache` pin/LFU/workers/`ensure_many`/prefetch |
| #2 | STREAM=0 demand path | pin → ensure → MLP; `last_eids` prefetch |
| #3 | Private residency vs RAM cap | `KATALI_ECACHE_MMAP=1` owned=0 pointers |
| #3b | `drop_unpinned` / `spec_pin_limit` / `freq_decay` | Wired post-prefill + decode pins |
| #4 | Async pool + `freq_load`/`save`/`prefetch_hottest` | `pf_iss`/`pf_done` live; `hot_experts.kcache` |
| #5 | Layer-major `prefill_batch` | Default ON; `KATALI_PREFILL_BATCH=0` A/B |
| #6 | `ExpertRangeFillFn` / `KATALI_EC_RANGE_READ` | Opt-in; default OFF |
| #7 | `wait_idle` / `hot_resident` | Warmup drain before prefill; laptop doc |

Deferred (same as katali2 optional path): READY/PIPE/STREAM, packed one-key, Job-Object-as-122B-laptop.

---

## 5. Bench ledger (35B unless noted)

### Desktop P0 floor

~**3.4 tok/s** decode (6 threads, easy full-RAM regime) — floor, not the product bar.

### Soft/hard 8 GB (pre-mmap lessons)

Hard Job + private copies can die; mmap residency keeps Job runs alive (~3.1–3.2 tok/s Manila). Job WS can still show ~tens of GB — not a real laptop.

### After ports #4–#7 (typical warm desktop, MMAP=1, cache-gb 8, top-k=8)

| Milestone | Prefill | Decode | Notes |
|-----------|---------|--------|-------|
| #4 workers | ~3.2 | ~3.3 | pf_iss 0→16k |
| #5 layer-major | ~3.16 | ~3.40 | pf_iss ~9k vs ~16k per-token |
| #6 RANGE ON | ~3.04 | ~3.15 | pf_iss **266**; text identical to OFF |
| #7 cold/warm | ~3.13–3.14 | ~3.28–3.37 | warmup leftover=0; hot resident ~3.9k/7710 |

### Top-k research only (not product)

K=8→5→4 raised tok/s; Manila held on short smoke — **do not change default top-k**.

### 122B

Coherent + Manila in smoke docs; desktop Job-8 “laptop” run cancelled as invalid sim.

---

## 6. Env cheat sheet

```bat
set KATALI_ECACHE_MMAP=1
set KATALI_EC_WORKERS=8
rem disk-bound laptop: set KATALI_EC_RANGE_READ=1
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines? Reply with only the city name after a short think." ^
  --max 64 --pin 25 --cache-gb 8
```

See `docs/LAPTOP_8GB_GGUF.md` for the real 8 GB procedure.

---

## 7. Next

1. Run GGUF on the **real 8 GB laptop** (katali2 EQS success is the bar).
2. If SSD-bound → keep RANGE_READ=1; only then consider READY/PIPE from katali2.
3. Same recipe for **122B** on that laptop.
4. 397B stays parked.

---

## 8. One-line summary

**katali2-backup-pre-rac is the finished EQS reference; katali-lab ports its elastic engine onto GGUF (ports #1–#7) so users can download models easily — next proof is a real ~8 GB laptop GGUF run, not more desktop soft-cap benches.**
