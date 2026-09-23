# Running Qwen3.6-35B-A3B Q4_K_M GGUF on an 8GB Windows laptop

Target lab: `katali-lab` with ecache ports #1–#7 overlaid. Model example:

`C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`

## Why mmap-resident matters

Under a hard **Job Object 8GB** commit limit, private ecache (`owned=1`) mallocs a
second copy of every expert weight blob on top of the GGUF mmap → OOM / empty
output. Port #3 fixes that:

```bat
set KATALI_ECACHE_MMAP=1
```

Entries store pointers into the GGUF map (`owned=0`). Logical `used/cap` stay
slot-based for pin/LFU policy; the OS charges faulted working set separately.

If private init returns `NOMEM`, host auto-retries mmap and logs the fallback.

## Recommended baseline (soft or hard 8GB)

```bat
set KATALI_ECACHE_MMAP=1
set KATALI_EC_WORKERS=8
rem leave top-k at 8 — do NOT lower
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8
```

Expect: `residency=mmap`, Manila in stdout, `pf_iss`/`pf_done` > 0 with batch
prefill (port #5), `phase: ecache warmup idle leftover=0` (port #7).

## Hard Job Object 8GB

Apply an 8GiB Job Object to the process (parent / launcher), then:

```bat
set KATALI_ECACHE_MMAP=1
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 96 --cache-gb 8
```

Success: non-empty output (not OOM); `residency=mmap`. Prefer `--max 64..96`
on first hard-cap pass.

## Cold vs warm A/B (port #7)

Open-time `prefetch_hottest` is async. Port #7 `ecache_wait_idle` drains it
before prefill and resets stats so the run's hits/misses are forward-only.

```bat
set KATALI_ECACHE_MMAP=1

rem === A: COLD ===
del /q C:\models\hot_experts.kcache 2>nul
set KATALI_EC_WARMUP=0
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8

rem === B: WARM (kcache written by A close) ===
set KATALI_EC_WARMUP=1
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8
```

| | A cold | B warm |
|---|--------|--------|
| `freq_load` | none yet | N entries |
| `prefetch_hottest` | skipped | queued N |
| `warmup idle leftover` | 0 | 0 |
| `hot-experts resident(>=2)` | low | higher |
| text | Manila | Manila (must match) |
| report | pf tok/s + decode tok/s | same metrics |

Optional: `KATALI_EC_FREQ=C:\models\hot_experts.kcache` to pin the path.

## Knobs (safe on 8GB)

| Env | Default | Note |
|-----|---------|------|
| `KATALI_ECACHE_MMAP` | 0 (auto NOMEM→mmap) | **Set 1** under hard 8GB |
| `KATALI_EC_WORKERS` | 8 | Prefetch readers |
| `KATALI_EC_WARMUP` | ON | `0` skips prefetch_hottest |
| `KATALI_EC_PIN` | ON | `0` disables speculative pins |
| `KATALI_EC_PIN_FRAC` | 25 | % of slots for speculative pins |
| `KATALI_EC_RANGE_READ` | OFF | Port #6 coalesce; A/B separately |
| `KATALI_PREFILL_BATCH` | ON | `0` = per-token prefill |
| `KATALI_NO_ECACHE` | OFF | Raw mmap; no pin/ensure/prefetch |

## Do not

- Lower top-k below 8 for “speed”
- Enable READY / PIPE / STREAM / `ensure_many_ex`
- Expect 122B under Job Object 8GB (out of scope)
- Pack experts to one EQS-style key (keep 3-key GGUF)

## What to paste back

From stderr for A and B:

1. `phase: ecache … residency=mmap …`
2. `phase: ecache warmup idle leftover=…`
3. `phase: prefill …` tok/s
4. `speed: decode …` tok/s
5. `ecache: hits=… misses=… … pf_iss=… pf_done=…`
6. `ecache: hot-experts resident(>=2 uses)=N/cap`
7. stdout text (Manila)
