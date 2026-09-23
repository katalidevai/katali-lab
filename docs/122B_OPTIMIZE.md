# 122B optimization profile

This profile is limited to Qwen3.5-122B-A10B and leaves the 35B defaults
unchanged.

## Current profile

```bat
set KATALI_ECACHE_MMAP=1
set KATALI_PREFILL_BATCH=1
set KATALI_NO_ECACHE=1
```

The practical default is now raw mmap plus layer-major prompt prefill. The
expert cache remains available as an opt-in experiment, but the measured
8-token comparison was faster without it. Top-k and model arithmetic are
unchanged.

The cache profile still supports Windows `PrefetchVirtualMemory` worker
prefetch, but it is not enabled by the default launcher because cache
bookkeeping and evictions outweighed the benefit in the measured workload.

The default now also fuses the gate/up matvec dispatch across all routed
experts for each token. The down projections remain separate because they use
different activation vectors. It is automatic for 35B-A3B, where it improved
sustained decode. It remains opt-in for 122B-A10B because the 32-token control
showed 0.624 tok/s fused versus 0.855 tok/s unfused.

## Current bottleneck

An opt-in `KATALI_PROFILE=1` raw-mmap smoke measured 15,730 matvec calls,
6.41 seconds in matvec kernels, 60.23 GiB of Q4_K traffic, and 0.87 seconds
of thread synchronization wait. The next optimization target is therefore
batched MoE matmul and dispatch reduction, not more SSD-cache tuning.

## Measurement

Short 122B smoke (`hi`, `--max 1`, 48 layers, 9 prompt tokens):

| Profile | Prefill | Decode | Exit | Text |
|---|---:|---:|---:|---|
| Range OFF, 8 workers | 0.800 tok/s | 1.043 tok/s | 0 | `<think>` |
| Range ON, cold, 8 workers | **0.955 tok/s** | **1.165 tok/s** | 0 | `<think>` |
| Async mmap prefetch, range OFF | **1.074 tok/s** | **1.296 tok/s** | 0 | `<think>` |
| Raw mmap, no expert cache, 8 tokens | **1.083 tok/s** | **1.311 tok/s** | 0 | `<think>` |
| Raw mmap + routed-expert fusion, 8 tokens | **1.272 tok/s** | **1.380 tok/s** | 0 | `<think>` |

The range-on run reduced cold prefill time from 11.25 s to 9.43 s. Both runs
used mmap residency, batch prefill, pin 25%, and the same model. The range-on
run also completed with no worker leftovers and preserved the model output
path.

Worker-count tests were not accepted as tuning evidence because simultaneous
processes contended for the same storage; keep the established eight workers.

Capacity sweep with range reads ON and eight workers:

| Cache | Prefill | Decode | Evictions |
|---:|---:|---:|---:|
| 8 GiB | 0.973 tok/s | 1.193 tok/s | 2,665 |
| **12 GiB** | **0.978 tok/s** | **1.209 tok/s** | 1,380 |
| 16 GiB | 0.958 tok/s | 1.210 tok/s | 95 |

## Run

```bat
deploy_122b_optimized.bat
```

This script builds and runs only the 122B profile. It does not change the
global 35B defaults or enable READY/PIPE/STREAM behavior. The bundled smoke is
intentionally `--max 1`; use a separately timed short `--max 8` run for quality
and warm-cache comparison. Reserve a full-length generation for final review,
not for knob selection.
