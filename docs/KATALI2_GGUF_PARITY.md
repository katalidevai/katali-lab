# katali2 (EQS) → katali-lab (GGUF) parity checklist

Reference: `C:\Users\joanr\Desktop\katali2-backup-pre-rac`  
Lab: `C:\Users\joanr\Desktop\katali-lab`

Mapping rule of thumb:

| katali2 (EQS) | katali-lab (GGUF) |
|---------------|-------------------|
| `eqs_pread` / FillFn into cache slot | GGUF mmap FillFn (`memcpy`) or ResolveFn (`owned=0`) |
| `dense.eqs` always-mapped dense tensors | GGUF dense tensors always via `KataliGgufTensor->data` (mmap) |
| `experts.eqs` streamed expert blobs | `ecache` over gate/up/down expert spans (3 keys / expert) |
| one packed expert key `(layer, eid)` | three weight keys `(layer, eid*3+{0,1,2})` |

## Done

| Area | katali2 | lab status |
|------|---------|------------|
| Expert cache core | `ExpertCache` pin/LFU/workers | **Done** — `ecache` (port #1) |
| Soft pin / unpin / unpin_all | yes | **Done** |
| `ensure_many` + FillFn | yes | **Done** (3×topk weight keys) |
| Async prefetch workers | yes | **Done** (`KATALI_EC_WORKERS`) |
| MoE demand path STREAM=0 | pin → ensure_many → get | **Done** (port #2 `moe_ffn` / `forward`) |
| Prior-token `last_eids` + prefetch | yes | **Done** (port #2) |
| mmap-resident fill (`owned=0`) | N/A (EQS pread) | **Done** (port #3 `KATALI_ECACHE_MMAP=1`) |
| `pins_enabled` / `KATALI_EC_PIN` | yes | **Done** (port #3b) |
| `spec_pin_limit` / `PIN_FRAC` / `PIN_MAX` | yes | **Done** (port #3b; uses `c->pin_frac` if env unset) |
| Post-prefill `freq_decay` + pin budget + `drop_unpinned` | `host_qwen` prefill end | **Done** (port #3b `host_generate`) |
| Decode end-of-token limited speculative pin | `host_qwen_forward_token` | **Done** (port #3b `forward.c`) |
| `freq_load` / `freq_save` / `prefetch_hottest` | yes | **Done** (port #4; `KATALI_EC_FREQ` or `<gguf_dir>/hot_experts.kcache`) |
| `ensure_many` queues async workers | prefetch pool | **Done** (port #4; bumps `prefetch_issued`/`prefetch_done`) |
| Win32 worker wake (SetEvent) | yes | **Done** (port #4; katali2-faithful WaitForSingleObject) |
| Layer-major batch prefill | `host_qwen_prefill_batch` | **Done** (port #5; union experts/layer + `ensure_many`; `KATALI_PREFILL_BATCH=0` disables) |
| Range fill + adjacent coalesce | `ExpertRangeFillFn` / `KATALI_EC_RANGE_READ` | **Done** (port #6; GGUF mmap span touch + split/resolve; default OFF) |
| `wait_idle` / `hot_resident` | yes | **Done** (port #7; generate-start drain + end log) |
| Dense weights always mapped | `dense.eqs` | **Done** — GGUF mmap tensors (no ecache) |
| Top-k default | model / 8 | **Keep top-k=8** (do not lower) |

## TODO / intentionally deferred

| Area | Notes |
|------|-------|
| EQS fill → GGUF fill | Fill path done; no EQS file format in lab |
| READY / PIPE / STREAM / `ensure_many_ex` | **Do not enable** until demand path measured SSD-bound |
| `protect_keys` (READY-AS-COMPLETE) | READY-only |
| Layer bias / knob sweeps | Deferred |
| Packed one-key expert blob | Would require GGUF repack; keep 3-key |
| Job-Object 122B | **Out of scope** for this smoke |

## Env (parity-relevant)

- `KATALI_ECACHE_MMAP=1` — mmap-resident ecache (`owned=0`)
- `KATALI_EC_PIN=0` — disable speculative prior-token pin set
- `KATALI_EC_PIN_FRAC=1..100` — % of ecache slots for speculative pins (default 25)
- `KATALI_EC_PIN_MAX=N` — absolute speculative pin count override
- `KATALI_EC_WORKERS=N` — prefetch readers (default 8)
- `KATALI_NO_ECACHE=1` — raw mmap; no pin/ensure/prefetch

## Smoke target (35B Manila)

```bat
set KATALI_ECACHE_MMAP=1
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8
```

Expect: `residency=mmap`, `phase: prefill … (layer-major)`, `phase: ecache post-prefill drop_unpinned … (batch)`, `pf_iss`/`pf_done` > 0, Manila in stdout, top-k unchanged (8).

Env: `KATALI_PREFILL_BATCH=0` forces per-token prefill (A/B). Default ON.

### Port #6 A/B (range off vs on)

```bat
set KATALI_ECACHE_MMAP=1
rem A: default range OFF
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8

rem B: range ON (text must match A)
set KATALI_EC_RANGE_READ=1
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8
```

Expect: both Manila; phase `range_read=OFF` vs `ON`; report pf/tok/s both ways.

### Port #7 A/B (cold vs warm hot-set)

```bat
set KATALI_ECACHE_MMAP=1
rem A: cold — delete hot_experts.kcache, WARMUP=0
del /q C:\models\hot_experts.kcache 2>nul
set KATALI_EC_WARMUP=0
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8

rem B: warm — kcache from A present, WARMUP default ON
set KATALI_EC_WARMUP=1
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8
```

Expect: both Manila; `phase: ecache warmup idle leftover=0`; B reports higher
`hot-experts resident(>=2 uses)`; compare pf/tok/s. See `LAPTOP_8GB_GGUF.md`.

