# CUDA backend â€” Step 1 findings + benchmark ledger

Companion docs: `docs/OPTIMIZE.md` (ecache ports), `STATUS.md` (bench ledger),
`docs/architecture.md` (current scope, still says "no GPU backends").

---

## 1. GPU actually installed (verified, not assumed)

| Property | Value |
|---|---|
| Exact model | **NVIDIA GeForce RTX 4060** (ASUS, `PCI\VEN_10DE&DEV_2882` â†’ AD107) |
| Architecture | **Ada Lovelace** (AD107) |
| Compute capability | **8.9** (sm_89) |
| Total VRAM | **8188 MiB (8 GB)** â€” nvidia-smi is authoritative |
| Available VRAM | **7936 MiB free**, 22 MiB used (idle) |
| Driver (UMD / KMD) | **616.56** / `32.0.16.1656`; VBIOS `95.07.31.00.3d` |
| CUDA driver (UMD) version | **CUDA 13.4** â†’ any CUDA 13.x toolkit usable; 12.x also works |
| CUDA Toolkit | **not installed** (no `NVIDIA GPU Computing Toolkit`, no registry keys) |
| nvcc | **not installed / not on PATH** |
| CUDA DLLs | `nvcuda.dll` + `nvcudadebugger.dll` present in `System32`; **no `cudart64_*`, `cublas*`, `nvrtc*`** |
| PCIe | **max Gen3, x8** (idle-reported Gen1). RTX 4060 is a native **x8** card; board is Gigabyte H410M S2H (PCIe 3.0) |
| GPU / VRAM utilization | **0 % / 0 %** |
| Clocks / temp | 210 MHz core, 405 MHz mem, 40 Â°C |

`Win32_VideoController.AdapterRAM` reports 4 GB â€” that field is a truncated 32-bit
value. **8 GB is correct.**

**Second adapter:** Intel UHD Graphics 630 (iGPU), 1 GB shared â€” unused by Katali.

### Host
Intel Core i5-10400 (Comet Lake), 6C/12T, 2.9 GHz, **AVX2+FMA, no AVX-512**,
**31.87 GiB RAM**, Windows 10 Home 25H2 (build 26200), **142.5 GB free on C:**

---

## 2. Toolchain verdict

| Item | Status |
|---|---|
| Project compiler | **TDM-GCC 64 / gcc 10.3.0** (MinGW), via `build.bat` â€” `-O2 -mavx2 -mfma -mf16c` |
| MSVC | **present** â€” VS 2022 BuildTools, MSVC toolset **14.44.35207**, `cl.exe` at `...\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe` |
| Windows SDK | **10.0.26100.0** |
| nvcc | **missing** â†’ nothing CUDA can be compiled today |
| Driver CUDA-capable | **yes** |

**nvcc on Windows requires `cl.exe`; it does not accept MinGW gcc as host compiler.**
So the CUDA backend must be a **separate DLL built by nvcc + MSVC** exposing a flat
`extern "C"` ABI, loaded by the MinGW-built exe at runtime via
`LoadLibrary`/`GetProcAddress`. That keeps `build.bat` and the CPU path untouched and
makes CUDA optional *at the linker level* (nothing links against the DLL).

---

## 3. Engine map (where CUDA can plug in)

**Every** projection in the model funnels through two primitives:

```
katali_ggml_matvec / katali_ggml_matvec_role   â†’ matvec_worker â†’ katali_ggml_matvec_rows (AVX2)
katali_ggml_matmul_role_ex                                                 (prefill)
```

Callers: `attn_delta.c` (QKV/gate/alpha/beta/ssm_out), `attn_gqa.c` (q/k/v/o),
`moe_ffn.c` (routed gate/up/down, shared expert, router logits), `forward.c`
(LM head).

Already present and reusable:

- `gguf_prof.c` â€” per-**role** Ã— per-**type** matvec accounting, with a separate
  prefill bucket. `KataliMatvecRole = {LM_HEAD, WQ, WK, WV, WO, GATE, UP, DOWN}`.
- `ECache` â€” slots, LFU+freq eviction, soft pins, `pin_frac`, `ensure_many`,
  `prefetch`, `hot_experts.kcache` frequency warm start.
- Two residency models already: **private** (`owned=1`) and **mmap** (`owned=0`,
  `KATALI_ECACHE_MMAP=1`, with automatic NOMEM fallback).
- Custom Windows worker pool (`gguf_threads.c`), **not** OpenMP.
- `KATALI_EC_RANGE_READ=1` SSD range coalescing; `KATALI_EC_WORKERS` prefetch threads.

**Design consequence:** the new tier should be a **third residency level inside
ECache**, not a parallel system. `ecache_get`/`fill`/`pin`/`prefetch` keeps its
contract; only *where* a filled slot lives changes.

---

## 4. Bottleneck analysis (35B-A3B, measured + derived)

Geometry: hidden 2048, moe_intermediate 512, 256 experts, 40 trunk layers,
top-k 8 + 1 shared. One routed expert = 3 Ã— (512 Ã— 2048) = 3.1 M params â‰ˆ
**1.75 MB at Q4_K_M**; a *usable* residency unit is all three = â‰ˆ **5.25 MB**.

- Active weight bytes per decode token â‰ˆ 3 B params Ã— ~4.5 bpw â‰ˆ **~1.7 GB/token**.
- At the measured **2.12 tok/s** that is **~3.6 GB/s** of weight streaming â€” at
  DDR4-2666 dual-channel class bandwidth. **Decode is memory-bandwidth-bound, not FLOP-bound.**

Two quantitative conclusions that constrain the design:

1. **Per-token PCIe streaming does not pay.** Gen3 **x8** â‰ˆ 8 GB/s theoretical,
   ~6.5 GB/s realistic. 1.7 GB / 6.5 GB/s â‰ˆ **262 ms/token â†’ ~3.8 tok/s ceiling**
   before any compute. Streaming experts over PCIe every token would swap the RAM
   wall for a PCIe wall. **VRAM *residency* is what pays, not transfer.**
2. **VRAM is ~10Ã— RAM bandwidth** (8 GB GDDR6 â‰ˆ 272 GB/s vs DDR4-2666 dual-channel).
   ~7.4 GB usable VRAM holds **~1,400 of 10,240 experts (~14 % of the pool)**, and
   only ~3 % of the pool is selected per token â€” so a hot tier can plausibly serve
   most of decode if routing locality holds. The existing LFU data must be used to
   **measure** this before committing.

Prefill is batch-bound (matrix Ã— matrix, high arithmetic intensity) â†’ the strongest
pure-GPU candidate. The LM head (248 320 Ã— 2048 Q6_K â‰ˆ 250 MB) is one large matvec â€”
attractive, but needs ~250 MB VRAM or a ~40 ms/token transfer, so **measure, don't assume**.

---

## 5. Benchmark ledger

### B0 â€” RETRACTED (invalid: truncated prompt)

My first baseline used `Start-Process -ArgumentList` from PowerShell, which does
**not** quote arguments containing spaces. The engine therefore received `prompt =
"What"` plus stray trailing argv entries, not the golden sentence. Evidence:
13 prefill tokens and the generic greeting
`Hello! How can I help you today? Feel free to ask a question,` â€” versus
**40 prefill tokens and `Manila`** once the prompt is passed correctly.
B0's raw logs are kept as `baseline_cpu_35b.*` but **must not be used as a
workload-matched baseline.** The "48 % faster after the change" reading was page
cache, not code.

### B1 â€” Valid CPU baseline (35B, golden prompt, `Manila` gate)

Driven from a `.bat` file (correct quoting). All runs: `--pin 25 --cache-gb 8`,
no env overrides, `residency=private`, `range_read=OFF`, `workers=8`,
ecache cap 8.00 GiB / 7710 slots, block 1088 KiB. **Every run output `Manila`.**

| Run | CUDA probe | Prefill (40 tok) | Prefill tok/s | Decode | Decode tok/s |
|---|---|---|---|---|---|
| S1-R1 | ON | 20.236 s | 1.977 | 2 tok / 0.781 s | 2.561 |
| S1-R2 | OFF | 14.243 s | 2.808 | 2 tok / 0.645 s | 3.100 |
| S1-R3 | ON | 13.298 s | 3.008 | 2 tok / 0.648 s | 3.086 |
| S1-R4 | ON | 18.536 s | 2.158 | 2 tok / 0.863 s | 2.317 |
| S2-p1 A | ON | 13.242 s | 3.021 | 2 tok / 0.663 s | 3.018 |
| S2-p1 B | OFF | 13.281 s | 3.012 | 2 tok / 0.658 s | 3.037 |
| S2-p2 A | ON | 13.557 s | 2.951 | 2 tok / 0.654 s | 3.060 |
| S2-p2 B | OFF | 13.630 s | 2.935 | 2 tok / 0.652 s | 3.069 |
| S2-p3 A | ON | 13.309 s | 3.005 | 2 tok / 0.648 s | 3.086 |
| S2-p3 B | OFF | 13.226 s | 3.024 | 2 tok / 0.633 s | 3.161 |

Series 1 (S1) ran on a cold page cache â€” the first run is much slower and the
spread is ~1.5Ã—. Series 2 (S2) is paired and warm. **TTFT â‰ˆ 13.3 s at 3.0 tok/s
prefill**, **decode â‰ˆ 3.0â€“3.2 tok/s**. `--max 64` still stops after 2 decode
tokens because the model emits EOS right after `Manila`, so the decode figure is a
short-burst number; it agrees with the existing `STATUS.md` desktop floor of
~3.4 tok/s.

Valid-run ecache counters (S1-R1): hits **34074**, misses **11290** (24.9 %),
evictions 2391, resident 2413, used **2563.81 / 8191.88 MiB**,
`bytes_loaded` **11995.62 MiB**, prefetch 5044 issued / 5044 done / 3792 hits.

### B2 â€” A/B: cost of the optional-CUDA probe

Only variable changed: `KATALI_CUDA=0` (skip the `LoadLibrary` probe) vs default.

| Arm | Prefill tok/s (mean of 3) | Decode tok/s (mean of 3) |
|---|---|---|
| probe ON | **2.992** | **3.055** |
| probe OFF | **2.990** | **3.089** |
| Î” | **+0.07 %** | **âˆ’1.1 %** |

**Verdict:** the probe is **below measurement resolution**. The 1.1 % decode delta
is confounded â€” arm B always ran second, so it benefits from a warmer cache, and
the prefill measurement (13 s of real work, by far the more stable statistic)
shows **+0.07 %**. Theory agrees: one `LoadLibraryA` on a missing file costs
microseconds against a 13 s prefill. **No CPU regression.**

### Reading of the measurements

- A **~25 % expert-cache miss rate** remains the concrete headroom: that is the
  SSD/mmap traffic a VRAM hot tier can absorb.
- ~3.0 tok/s decode against a **~3.6 GB/s** weight-stream floor is consistent with
  a memory-bandwidth wall â€” the wall VRAM removes.
- Run-to-run variance from OS page-cache state is **larger than any effect my
  change could have**, so only paired back-to-back runs are usable as evidence.

---

## 6. Install status â€” CUDA Toolkit 13.4.2

### Downloaded (verified)

```
https://developer.download.nvidia.com/compute/cuda/13.4.2/local_installers/cuda_13.4.2_windows_x86_64.exe
```

Saved to `C:\Users\joanr\Downloads\cuda_13.4.2_windows_x86_64.exe`
**3 917 283 152 bytes (3.65 GB) â€” size matches the server exactly.**

Installer plan: **Custom â†’ uncheck Display Driver / GeForce Experience / PhysX**,
keep only **CUDA Runtime** + **CUDA Development**. Installed driver 616.56 already
reports CUDA 13.4, so no driver change is needed or wanted.
Expected path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`.

### Blocker: the installer requires elevation

`cuda_13.4.2_windows_x86_64.exe` carries a `requireAdministrator` manifest.

- Current shell: **not elevated** (`IsInRole(Administrator) = False`, user `joanr`).
- `Start-Process -Verb RunAs` raises a UAC prompt (`consent.exe` observed live,
  PID 19800) but the consent was not answered, so the `-extract` step never ran.
- **Consequence:** the install cannot be completed from a non-elevated shell.
  One user-approved UAC elevation is unavoidable for the official installer.

### Verified elevation-free alternative

NVIDIA's official pip wheels (CUDA Installation Guide Â§4 "Pip Wheels") need **no
admin** and never touch the driver. Availability checked live against PyPI:

| Wheel | Latest | Windows size |
|---|---|---|
| `nvidia-cuda-nvcc-cu12` | 12.9.86 | 33.1 MB |
| `nvidia-cuda-runtime-cu12` | 12.9.79 | 3.4 MB |
| `nvidia-cuda-nvrtc-cu12` | 12.9.86 | 72.9 MB |
| `nvidia-cublas-cu12` | 12.9.2.10 | 527.5 MB |

Total â‰ˆ **640 MB** (vs 3.65 GB). CUDA **12.9** fully supports **sm_89 (Ada)** and
runs on the installed 616.56 driver, because CUDA drivers are backward compatible.

**Caveat:** `-cu13` wheels are **not published** (`nvidia-cuda-nvcc-cu13` â†’ 404;
only a `0.0.1` placeholder `nvidia-cuda-crt-cu13`). So the pip route means
**CUDA 12.9**, not 13.4 â€” functionally sufficient for this GPU, but it requires
assembling a `CUDA_HOME` layout (`bin`/`include`/`lib/x64`) from the wheel trees,
which must be verified before the first `nvcc` compile.

### Decision pending

Path A â€” user approves UAC â†’ official CUDA 13.4.2 toolkit (2 clicks, standard
layout, `nvcc` on PATH).
Path B â€” cancel UAC â†’ pip-wheel CUDA 12.9 (no admin; needs `CUDA_HOME` assembly +
verification).
---

## 7. Increment 1 delivered â€” optional-CUDA detection scaffold (no CUDA compiled)

Implemented and verified today with **no CUDA toolkit present**. This is Step 3's
first half: prove the engine stays correct and fast when CUDA is absent, and give
the backend a stable ABI to implement against.

> Note: the earlier UAC prompt was **dismissed** and the extract never ran, so the
> CUDA Toolkit is still **not installed**. Path A / Path B above is still open.

### Files

| File | Change |
|---|---|
| `include/katali_cuda.h` | **new** â€” ABI v1: `KataliCudaDeviceInfo` POD, 10 exported backend entry points (`abi_version`, `backend_init`, `backend_shutdown`, `supports_type`, `matvec`, `matmul`, `device_alloc`, `device_free`, `upload`, `download`), plus the host-side API. |
| `src/katali_cuda.c` | **new** â€” runtime loader: `LoadLibraryA` â†’ `GetProcAddress`, ABI check, honest diagnostics. Compiled into the MinGW exe; **never links CUDA**. |
| `src/host.c` | `host_open` probes once and prints one `phase: cuda â€¦` line. Never fatal. |
| `src/main.c` | new `cuda-info` (alias `gpu`) subcommand; usage text updated (old "No GPU." claim removed). |
| `build.bat` | added `src\katali_cuda.c` to `SRC`. No new flags, libraries, or link dependencies. |

### Guarantees implemented

- **Optional at the linker level.** Nothing links a CUDA library, so a machine
  with no NVIDIA hardware runs the identical CPU + RAM + SSD path.
- **Absent DLL is not an error:** `cuda-info` exits **0** and reports
  `tier: CPU + system RAM + SSD`.
- **Failure modes distinguished** rather than guessed: `ERROR_FILE_NOT_FOUND` â†’
  "DLL not present"; `ERROR_MOD_NOT_FOUND` â†’ "exists but `cudart64_*.dll` missing";
  `ERROR_BAD_EXE_FORMAT` â†’ "wrong architecture". (The first cut mis-reported
  "DLL not present" as "cudart missing"; fixed and retested.)
- **ABI mismatch rejected** (DLL abi != host abi â†’ CPU fallback), so a stale
  backend can never silently corrupt results.
- **`KATALI_CUDA=0`** disables the probe with no filesystem access.
- **Search order:** `$KATALI_CUDA_DLL` â†’ `<exe dir>\katali_cuda.dll` â†’ default
  search order.

### Verification performed

| Check | Result |
|---|---|
| `build.bat` | **OK** â€” only pre-existing warnings (`api_server.c` STARTUPINFO, `main.c` `ai`) |
| `katali-lab.exe selftest` | **PASS**, exit 0 |
| `katali-lab.exe cuda-info` | `cuda: unavailable`, correct reason, exit **0** |
| `generate` with DLL absent | runs, prints `phase: cuda unavailable (â€¦)`, **continues normally** |
| golden answer, 10 of 10 runs | **`Manila`** |
| probe-overhead A/B (B2) | **+0.07 %** prefill â€” below measurement resolution |

### What this deliberately does NOT do yet

No CUDA source exists, no kernels are compiled, and **no dispatch is wired**: the
`matvec`/`matmul` calls still go unconditionally to `katali_ggml_matvec*`, so the
numeric path is byte-for-byte the previous behaviour. VRAM residency, the adaptive
VRAM/RAM budget, and tiered hot-expert placement all still need the toolkit.

### Remaining steps (all blocked on a CUDA compiler)

1. Extract the installer to read its **exact** `-s` component names, then install
   **CUDA Runtime + Development only** (no display driver).
2. Implement `katali_cuda.dll` (`cuda/katali_cuda_kernels.cu` + MSVC/nvcc build
   script): Q4_K / Q6_K / Q8_0 / F32 dequant-GEMV and tiled GEMV, then batched
   matmul for prefill; validated against `katali_ggml_vec_dot` under the existing
   tolerance gates.
3. Wire role-based dispatch via `KataliMatvecRole` (LM head and attention first;
   MoE experts only where the measured per-token PCIe cost amortises).
4. Adaptive budgets from `cudaMemGetInfo` + `katali_ram_avail_bytes`, with manual
   overrides preserved.
5. VRAM hot tier as a **third `ECache` residency**, reusing `hot_experts.kcache`
   LFU data â€” the **~25 % miss rate** from B1 is the target.
6. Async pipeline: CUDA streams, pinned host memory, double buffering, and overlap
   of SSD â†’ RAM â†’ VRAM with CPU attention compute.

---

## 8. CUDA toolchain obtained WITHOUT administrator rights

The elevated installer never ran (the UAC prompt was dismissed), so the toolkit
was assembled from NVIDIA's **redist archives** â€” plain zips over HTTP, no
installer, **no elevation, no driver change**:

| Component | Version | Size | Provides |
|---|---|---|---|
| `cuda_nvcc-windows-x86_64-13.4.92-archive.zip` | 13.4.92 | 31.5 MB | `nvcc.exe`, `cudafe++`, `nvlink`, `ptxas`, `fatbinary`, `bin2c` |
| `libnvvm-windows-x86_64-13.4.92-archive.zip` | 13.4.92 | 57.3 MB | `nvvm/bin/cicc.exe`, `nvvm64_40_0.dll`, `libdevice.10.bc` |
| `cuda_crt-windows-x86_64-13.4.92-archive.zip` | 13.4.92 | 0.2 MB | `include/crt/*` device headers |
| `cuda_cudart-windows-x86_64-13.4.92-archive.zip` | 13.4.92 | 2.6 MB | `cuda_runtime.h`, `cudart.lib`, `cudart_static.lib`, `cudart64_13.dll` |

Total â‰ˆ **92 MB** (vs 3.65 GB). `cuda_nvcc` SHA-256 verified against
`redistrib_13.4.2.json`. Assembled into `C:\Users\joanr\cuda\home`
(`bin`, `include`, `lib\x64`, `nvvm`).

Two discoveries this corrected:
- The **PyPI `nvidia-cuda-nvcc-cu12` wheel contains NO `nvcc.exe`** â€” only
  `ptxas`, `nvvm64_40_0.dll` and crt headers. The pip-wheel route cannot produce
  a working `nvcc`; the earlier plan to use it was wrong and has been dropped.
- CUDA 13 splits NVVM out of the nvcc package. Without `libnvvm`, nvcc fails with
  `""%CICC_PATH%\cicc" ... The system cannot find the path specified.`

Verified end to end:

```
$ nvcc --version
Cuda compilation tools, release 13.4, V13.4.92

$ hello.exe        (nvcc + MSVC 19.44, -arch=sm_89)
cudaGetDeviceCount=1 err=no error runtime=13040
dev0: NVIDIA GeForce RTX 4060  sm_89  SMs=24  vram=8.00 GiB
saxpy sync=no error  y[0]=5.0 (expect 5.0)
```

`build_cuda.bat` builds the backend; `KATALI_CUDA_HOME` / `KATALI_VCVARS` /
`KATALI_CUDA_ARCH` override the paths. The main engine keeps building with
MinGW `build.bat` and **never links against CUDA**.

---

## 9. What was delivered

| File | Role |
|---|---|
| `include/katali_cuda.h` | ABI v1: device info POD, 10 backend entry points, host API, dispatch wrappers |
| `src/katali_cuda.c` | Runtime `LoadLibrary` loader, ABI gate, honest diagnostics, dispatch wrappers |
| `cuda/katali_cuda_backend.cu` | **The CUDA backend**: device/VRAM tier + Q4_K/Q6_K/Q8_0/Q4_0/F32 GEMV + matmul |
| `build_cuda.bat` | nvcc + MSVC build of `katali_cuda.dll` |
| `include/vram_cache.h` / `src/vram_cache.c` | VRAM expert tier (third residency level), LRU, budget-driven |
| `src/host.c` | One non-fatal `phase: cuda â€¦` line; VRAM tier open/close + scratch |
| `src/moe_ffn.c` | `expert_matvecs_gpu()` path; fusion suppressed when the tier is live |
| `src/main.c` | `cuda-info`, `cuda-check`, `cuda-bench` |
| `build.bat` | adds `katali_cuda.c`, `vram_cache.c` |

`katali_cuda.dll` is 192 KB and links `cudart` **statically**, so it has no DLL
dependencies of its own â€” nothing to ship alongside it.

---

## 10. Kernel-level results (the win is real)

### Correctness â€” `cuda-check` vs the engine's own CPU kernels on REAL GGUF tensors

```
output.weight        Q6_K  cols=2048 rows=8  rel_L2=1.373e-07  PASS
token_embd.weight    Q4_K  cols=2048 rows=8  rel_L2=1.502e-07  PASS
blk.0.ffn_down_shexp Q8_0  cols=512  rows=8  rel_L2=9.610e-08  PASS
blk.0.ffn_gate_inp   F32   cols=2048 rows=8  rel_L2=1.966e-07  PASS
types validated=4  failed=0  worst rel_L2=1.966e-07 -> ALL PASS
```

Errors sit at f32 round-off, so the Q4_K 6-bit scale/min unpacking, the Q6_K
`ql/qh/scales` indexing and the Q8_0/F32 paths are all correct. (Q4_0 is
implemented but this model contains none, so it is unvalidated.)

### Speed â€” `cuda-bench`, weights resident in VRAM

| Operation | CPU | GPU | Speedup |
|---|---|---|---|
| expert gate 512Ã—2048 Q4_K (re-read) | 9.3 GB/s | 20.4 GB/s | 2.20Ã— |
| expert down 2048Ã—512 Q6_K (re-read) | 11.9 GB/s | 27.3 GB/s | 2.30Ã— |
| LM head 248320Ã—2048 Q6_K (398 MB) | 19.7 GB/s | 42.3 GB/s | 2.15Ã— |
| **64 distinct gate experts, 36 MB** | **2.9 GB/s** | **20.2 GB/s** | **6.93Ã—** |
| **64 distinct down experts, 52 MB** | **3.5 GB/s** | **28.0 GB/s** | **7.98Ã—** |
| **all 256 gate experts, 144 MB** | **1.8 GB/s** | **19.1 GB/s** | **10.92Ã—** |

This is the predicted memory-bandwidth wall, measured. The CPU drops from
~10 GB/s (L3-resident) to **1.8 GB/s** once the working set exceeds the 12 MB L3,
while the GPU holds ~20 GB/s from VRAM. Two caveats stated plainly:

- The GPU reaches only ~7 % of the RTX 4060's ~272 GB/s, so the **kernel, not
  VRAM, is now the limit** â€” bigger blocks, vectorized 128-bit loads, and
  multiple rows per block are unclaimed headroom.
- `katali_cuda_matmul` re-reads W per batch row, so it is **not** expected to
  beat the CPU's `katali_ggml_matmul` on prefill. Prefill needs a tiled GEMM.

---

## 11. End-to-end hybrid: measured SLOWER, and why

> **Superseded by Â§13.** This section describes the *per-expert* integration
> (Phase 1). Layer-level fusion has since been implemented; the conclusion is
> still "slower than CPU", but the cause has changed. Â§13 is the current record.

35B, golden prompt, `--max 64 --pin 25 --cache-gb 8`, back-to-back:

| Arm | Prefill (40 tok) | Decode | Output |
|---|---|---|---|
| **A: CPU + RAM + SSD** | 13.305 s / 3.006 tok/s | **3.112 tok/s** | `Manila` |
| **B: GPU + CPU + RAM + SSD** (tier v1) | 14.159 s / 2.825 tok/s | 2.153 tok/s | `Manila` |
| **Bâ€²: after fixing capacity + restoring the RAM tier** | 14.652 s / 2.730 tok/s | **2.144 tok/s** | `Manila` |

Correctness is preserved in every arm. Two real defects were found and fixed
between v1 and Bâ€²:

1. **Fixed 4096-slot hash table.** It silently capped residency at 2.44 GiB of the
   6.44 GiB budget, after which every lookup returned NULL (`evictions=0` proved
   eviction never ran). Now sized from the budget: hit rate went **29.3 % â†’ 74.4 %**
   (`hits=29991 / misses=10329`, `resident=10329`, `used=6.12/6.44 GiB`).
2. **I had disabled the ECache when the VRAM tier was on**, turning every VRAM
   miss into a *synchronous* mmap fill. Reverted: RAM is the middle tier of
   SSD â†’ RAM â†’ VRAM and its async prefetch is what makes a VRAM miss cheap.

**Neither fixed the regression, because the cause is CUDA API latency, not
compute or hit rate.** Per token: ~320 experts Ã— 3 GEMVs, and each expert costs
1 upload + 3 launches + 1 download + syncs â‰ˆ 8 round trips â†’ **~2560 CUDA round
trips per token**. Under WDDM each costs tens of microseconds, so
**150â€“250 ms/token of overhead** swamps the ~32 ms of real GPU work. 320 ms/token
CPU-only becomes ~465 ms/token hybrid.

**Decision taken:** the VRAM tier is **opt-in** (`KATALI_CUDA_MOE=1`, or set
`KATALI_VRAM_GB`). Default behaviour is byte-for-byte the previous CPU path â€”
re-verified at **3.081 tok/s decode, prefill 3.018 tok/s, output `Manila`** â€” so
there is **no regression for anyone who does not ask for it**. CUDA is still
detected and reported; it just does not take over until it can win.

---

## 12. Phase 1 â€” telemetry baseline BEFORE layer fusion

Matched run: **64 decode tokens**, prompt `Count from 1 to 50, comma separated.`,
`--max 64 --pin 25 --cache-gb 8`, 35B Q4_K_M. Both arms identical except
`KATALI_CUDA_MOE`.

| Arm | Decode | ms/token | Output |
|---|---|---|---|
| **A: CPU + RAM + SSD** | **2.836 tok/s** (22.567 s) | 353 | `1, 2, 3, â€¦` |
| **B: GPU hybrid, per-expert dispatch** | **1.597 tok/s** (40.085 s) | **626** | `1, 2, 3, â€¦` |

Per-token CUDA telemetry (decode only; raw: `tel2_gpu.err`):

| Counter | Per token | Total over 64 tokens |
|---|---|---|
| **CUDA API calls** | **2 826** | 180 886 |
| kernel launches | **960** | 61 440 |
| `cudaDeviceSynchronize` | **960** | 61 440 |
| `cudaMemcpy` | **1 701** | 108 837 |
| uploads / downloads | 740 / 960 | 47 397 / 61 440 |
| H2D / D2H | **64.0 MiB / 3.75 MiB** | 4093 / 240 MiB |
| cudaMalloc / cudaFree | 100 / 65 | 6 437 / 4 172 |
| **host time inside CUDA** | **354 ms** | 22 668 ms |
| â†³ matvec | **280 ms** | 17 911 ms |
| â†³ upload | 41 ms | 2 629 ms |
| â†³ download | 20 ms | 1 302 ms |
| GPU experts processed | 320 | 20 480 |
| VRAM cache | 74 230 hits / 15 050 misses, `used=6.44/6.44 GiB` | |

`960` launches and `960` syncs per token is exactly `8 experts x 40 layers x 3
GEMVs`, i.e. one launch + one blocking sync per GEMV. Measured cost:
`22668 / 180886 = 125 us` per API call and **`17911 / 61440 = 292 us` per
matvec** (launch + `cudaDeviceSynchronize` under WDDM).

**Conclusion: 354 ms of a 626 ms token (57 %) is host-side CUDA overhead.** The
bottleneck is call count and synchronization, not kernel throughput â€” which is
precisely what layer-level fusion must remove.

---

## 13. Phase 2-7 â€” layer-level fusion: implemented, measured, and the bottleneck MOVED

`katali_cuda_moe_layer()` replaces per-expert dispatch. One call per MoE layer
performs: **gate+up batched â†’ GPU siluÃ—router-weight â†’ down batched â†’ GPU
reduce**, with everything intermediate staying on the device. The only PCIe
traffic is the layer input and the layer output.

Enabled with `KATALI_CUDA_MOE=1`; `KATALI_CUDA_MOE_LAYER=0` reverts to the
Phase 1 per-expert GPU path for A/B.

### Dispatch reduction: achieved as designed

Measured over 64 decode tokens, identical prompt/model/settings:

| Counter | per token | change |
|---|---|---|
| CUDA API calls | **3 070 â†’ 386** | **8.0Ã— fewer** |
| kernel launches | **960 â†’ 160** | **6.0Ã— fewer** (= 40 layers Ã— 4) |
| `cudaDeviceSynchronize` | **960 â†’ 40** | **24Ã— fewer** (= 1 per layer) |
| D2H bytes | 3.75 â†’ **0.31 MiB** | per-layer output only |
| **host time inside CUDA** | **354 â†’ 280 ms** | âˆ’21 % |
| â†³ of which is API overhead | ~354 â†’ **~48 ms** | **7.4Ã— less** |

The last row is the important one: subtracting GPU execution from the host time
shows the *dispatch* cost collapsed from ~354 ms to ~48 ms per token.

### End-to-end: fusion alone does NOT win

| Arm | Decode | Output |
|---|---|---|
| **A: CPU + RAM + SSD** | **3.270 tok/s** | `8EC7BAC65514` |
| **B: FUSED layer MoE** | **1.833 tok/s** | `8EC7BAC65514` âœ… |
| C: per-expert GPU (Phase 1) | 1.590 tok/s | `8EC7BAC65514` âœ… |
| A2: CPU again (bracket) | **3.266 tok/s** | `8EC7BAC65514` âœ… |

Fusion bought **+15 %** over per-expert (1.590 â†’ 1.833) and left it **1.8Ã—
slower than CPU**. **The predicted 11-16 tok/s did not materialise.**

### Two real bugs found and fixed

1. **Every `down` job read a shared input base.** All experts' down projections
   consumed *expert 0's* activation. Invisible at `n_sel=1` and when the same
   expert was repeated, so it survived a careless look. Fixed by giving each job
   its own `x` pointer (`is + j*FF`). This was caught only by writing a
   value-level test â€” the telemetry looked perfect while the text was garbage.
2. **My own test harness had `acc` overlapping `dout`**, producing a false FAIL
   and sending me after a kernel bug that did not exist. Fixed the layout and
   documented it in the code.

Also changed while debugging: the job table moved from a ~1 KB by-value kernel
argument to a persisted device buffer read through a pointer.

### Correctness now verified at two levels

| Check | Result |
|---|---|
| `cuda-check` (per-op kernels) | ALL PASS, worst `rel_L2 = 1.966e-07` |
| `cuda-check-moe` n_sel = 1, 2, 8, 16 | PASS, `rel_L2 â‰ˆ 2.3e-07`, **0 bad elements** |
| `cuda-check-moe` layers 0, 1, 5, 13, 20, 39 | PASS (covers Q4_K/Q4_K/Q6_K and Q4_K/Q4_K/Q4_K) |
| End-to-end, 4 arms | **byte-identical output hashes** |
| Default path (no env) | **3.049 tok/s, `Manila`, 0 CUDA calls** â€” untouched |

### Where the time actually goes now

`KATALI_CUDA_GPU_TIME=1` gives the split:

| | per token | share of 541 ms |
|---|---|---|
| decode total | 541 ms | â€” |
| host inside CUDA | 280 ms | 52 % |
| of which **API overhead** | **~48 ms** | 9 % |
| of which GPU execution | **202 ms** | **37 %** |

**GPU execution is now the bottleneck, not dispatch.** 202 ms for ~660 MB of
expert weights/token = **3.2 GB/s**. The card can do ~272 GB/s.

In isolation the *same* fused kernel is fast:

```
timing fused n_sel=8 x30: 0.452 ms/call  15.6 MB/call  33.6 GB/s
timing fused n_sel=1 x30: 0.089 ms/call   1.9 MB/call  21.2 GB/s
```

â€¦but that re-reads the same slabs, and the 4060 has ~24 MB of L2, so it is an
**L2-warm upper bound**. The engine streams 660 MB/token L2-cold and lands at
**10Ã— less**. So the kernel *logic* is sound and the limit is **memory-level
parallelism on streaming data**: one row per block, 128 threads, 16 dependent
loads per thread â€” far too little outstanding memory traffic to cover DRAM
latency.

### Honest status and the correct next step

- Fusion **worked structurally** and is **correct**; the dispatch problem is
  solved (354 â†’ 48 ms/token).
- It is **not yet a win** (1.833 vs 3.270 tok/s), so `KATALI_CUDA_MOE` **stays
  opt-in**. Default behaviour is byte-for-byte the CPU path, re-verified.
- The measured next bottleneck is **kernel memory-level parallelism**, not
  dispatch. Ranked candidates: 128-bit vectorized loads, several rows per block
  sharing one activation read, and more warps cooperating per row. The L2-warm
  measurement (33.6 GB/s) shows the headroom that is being lost to latency, so
  this is a bounded, measurable target rather than a guess.

## 14. Kernel phase â€” what the cold numbers actually say

### 14.1 Research first (done before touching the kernel)

Sources read in depth (raw source, not READMEs):

| Project | File | Technique observed |
|---|---|---|
| **llama.cpp / ggml** | `ggml/src/ggml-cuda/mmvq.cu` | `mul_mat_vec_q` family: **rows-per-block is a template parameter** (`nrows`), plus `nwarps` and `ncols_dst`; a **dedicated MoE kernel** `mul_mat_vec_q_moe_launch` for the multi-token `MUL_MAT_ID` path (`has_ids`); per-type `vec_dot_q4_K_q8_1` / `vec_dot_q6_K_q8_1` doing **DP4A integer dots against a q8_1-quantized activation**; `prefetch.global.L2` deliberately **gated to DGX Spark**, with the comment that on higher-bandwidth parts "the kernel has little exposed latency left to hide and the extra requests cost more than they save" |
| **llama.cpp / ggml** | `ggml/src/ggml-cuda/mmq.cu` | Quantized **mat-mat** tiles needing â‰¥48 KiB shared memory/block; MMQ preferred for **MoE even without native dp4a**; `MMQ_DP4A_MAX_BATCH_SIZE` gates mmq vs BLAS |
| Marlin / CUTLASS / TensorRT-LLM / FlashInfer / ExLlama / vLLM | â€” | **Not read in depth in this phase**, so they are not cited as evidence. Deferred, not claimed. |

**Most important difference found:** llama.cpp's matvec does not dequantize to
float and MAC in float32 per element. It **quantizes the activation to q8_1 once,
then uses `__dp4a` 4-way integer dot products**. That removes ~7/8 of the
activation traffic (an f32 activation is 8 KiB per row versus 1.1 KiB of Q4_K
weights) and cuts the dot instruction count 4Ã—. KATALI computes a full float dot
per element. Our own measurement supports the technique: removing dequant
entirely is **3.3Ã— faster**.

### 14.2 Phase 1/9/8 â€” the cold ladder (`cuda-bench-cold`)

New `cuda-bench-cold` builds a working set far larger than the 24 MB L2 and
**rotates through different experts every call**, so consecutive calls touch
disjoint slabs. Three numbers in one run:

```
Phase 9 raw VRAM:      seq_read=240.4 GB/s  copy(r+w)=232.9 GB/s   [256 MiB buffer]
                       (234.6 GB/s sustained over a 90 s hold)
Phase 1/8 fused MoE, cold, rotating experts:
  n_sel calls  dq_GB/s  nomath_GB/s  dq_ms  host_ms  launch
  1     192    18.8     27.8         19.4   20.8     4
  2     96     24.6     51.0         14.9   15.5     4
  4     48     29.1     83.7         12.5   12.9     4
  8     24     31.9     106.4        11.4   11.9     4
```

**MEASURED:**
- **The card sustains 240 GB/s in this exact environment** â†’ WDDM/driver/PCIe are
  not the limit.
- **The fused kernel, cold, reaches 31.9 GB/s** â€” 10Ã— better than the 3.2 GB/s
  previously attributed to it.
- **`nomath`** (identical access pattern, no dequant) reaches **106.4 GB/s** â†’
  **dequantisation costs ~3.3Ã—**. Real, and previously unknown.
- Raising the working set 374 MB â†’ **5.8 GB** changed nothing (34.4 GB/s), ruling
  out address-space/TLB effects.

### 14.3 Phase 11 â€” per-layer profile (the decisive measurement)

`KATALI_CUDA_LAYER_PROFILE=1` prints per layer. All 40 layers look identical:

```
cu-layer L0  n_sel=8 bytes=15.56MiB gpu=5.222ms 2.9GB/s vhit=0 vmiss=0 vevict=0 vupload=0 resident=10494
cu-layer L1  n_sel=8 bytes=15.56MiB gpu=5.400ms 2.8GB/s vhit=0 vmiss=0 vevict=0 vupload=0 resident=10497
```

**2.8 GB/s with zero misses and zero evictions.** Not churn, not hit rate, not a
particular layer or quantisation type, and not the kernel â€” the *same call* does
31.9 GB/s standalone.

### 14.4 Hypotheses tested and REJECTED (all by direct measurement)

| Hypothesis | Test | Result |
|---|---|---|
| Address space / TLB spread | benchmark, 374 MB vs **5.8 GB** footprint, same bytes/call | 31.9 â†’ **34.4 GB/s**. Rejected |
| WDDM paging (cache at budget) | VRAM budget 2 / 4 / 6 GiB | 1.630 / 1.752 / 1.846 tok/s, no cliff. Rejected |
| `cudaMalloc`/`cudaFree` churn | benchmark with the engine's per-call alloc+upload+free | GPU window **unchanged** (18.2 â†’ 18.5 ms). Rejected |
| ECache's 8 CPU fill workers | engine with `KATALI_NO_ECACHE=1` | 3.776 â†’ **3.890 ms/layer**. Rejected |
| Kernel inefficiency | cold benchmark + `nomath` ladder | 31.9 / 106.4 GB/s. Rejected |

### 14.5 ROOT CAUSE â€” the GPU runs at its IDLE power state

Sampled with `nvidia-smi` during each workload:

| Workload | SM clock | Memory clock | Util |
|---|---|---|---|
| `cuda-bench-cold` (tight loop) | **2505 MHz** | **8501 MHz** | 10â€“33 % |
| **Real engine decode** | **210 MHz** | **405 MHz** | **50 %** |

**During real inference the GPU is 50 % *busy* while sitting at its idle clock** â€”
core 210 vs 2505 MHz and **memory 405 vs 8501 MHz, a 21Ã— lower memory clock** â€”
because the workload is a **~6 % duty cycle**: ~0.5 ms of GPU work per ~8 ms of
CPU work. NVIDIA's power management never requests boost, and a purely memory-bound
kernel at 1/21st memory clock lands exactly where we measured.

**Controlled proof** â€” same engine, same prompt, with a background process holding
the GPU busy so it stays at 2850/8251 MHz:

| | idle clocks (normal) | clocks held up |
|---|---|---|
| decode | 1.837 tok/s | **2.537 tok/s (+38 %)** |
| layer GPU window | **3.776 ms** | **1.922 ms (2.0Ã—)** |

That gain was bought while the holding process consumed ~94 % of the machine, so
the engine's kernels were also competing for bandwidth. The true cost of the idle
clock state is therefore **at least 2Ã—, and by inference considerably more**.

### 14.6 Phase 3/7 kernel experiment â€” REJECTED

Attempted: restructure the column loop into 4 independent accumulator chains so
several weight loads are in flight per thread (`cols` is a runtime value, so the
compiler cannot unroll it). Both variants compiled in, selected by
`KATALI_GEMV_UNROLL`.

| n_sel | dq GB/s U=1 | dq GB/s U=4 | change |
|---|---|---|---|
| 1 | 17.7 | 20.3 | +15 % |
| 2 | 24.0 | 26.7 | +11 % |
| 4 | 28.4 | 31.4 | +11 % |
| **8** | **34.7** | **27.0** | **âˆ’22 %** |

Helps at low n_sel (few blocks in flight, so per-thread MLP matters), **hurts 22 %
at n_sel=8** â€” the engine's operating point, where 24 576 blocks per layer already
saturate the machine and the extra registers cost occupancy.

**Verdict: rejected. `KATALI_GEMV_UNROLL=1` stays the default.** Correctness was
verified for both variants (`rel_L2 â‰ˆ 2.3e-07`, 0 bad elements) â€” this is a genuine
performance rejection, not a correctness failure. The switch stays in the code.

### 14.7 Scoreboard

| Variant | cold GB/s @n_sel=8 | layer GPU ms | decode tok/s | Result |
|---|---|---|---|---|
| Baseline (fused, unroll=1) | 34.7 | 3.78 | 1.837 | reference |
| `nomath` read-only probe | 114.7 | â€” | â€” | diagnostic only |
| Unroll=4 | 27.0 | â€” | â€” | **REJECTED (âˆ’22 %)** |
| Working set 374 MB â†’ 5.8 GB | 34.4 | â€” | â€” | no effect (hypothesis killed) |
| VRAM budget 2/4/6 GiB | â€” | 4.06/4.89/5.02 | 1.630/1.752/1.846 | smaller is worse |
| `KATALI_NO_ECACHE=1` | â€” | 3.890 | 1.853 | no effect |
| **GPU clocks held up** | â€” | **1.922** | **2.537** | **+38 %, root cause confirmed** |

### 14.8 What the new measured bottleneck is

**GPU power state driven by duty cycle** â€” not the kernel, not dispatch, not memory
layout, not VRAM management. KATALI's CPU path costs ~350 ms/token while the GPU
needs only ~20 ms of real work per token, so the GPU is idle ~94 % of the time and
stays at idle clocks.

This is decision-point **CASE C** from the plan: "cold bandwidth remains very low
despite kernel restructuring â€” stop blindly optimizing GEMV, use profiling
evidence." The evidence identifies it.

### 14.9 Recommendation for the next phase

**Do not do more GEMV tuning.** The highest-value next step is to **raise the GPU
duty cycle so the card stays boosted**:

1. **Overlap GPU work with CPU work** (the plan's Phase 9, previously deferred).
   Issue layer N's GPU MoE asynchronously and do layer N's attention/router on the
   CPU while it runs, so the GPU queue never drains. This attacks the root cause
   and is also the only way to hide CPU time.
2. **Clock warm-keeping** (e.g. a small periodic kernel) is *untested* â€” it may or
   may not hold boost and must be measured before adoption.
3. Only once the duty cycle is fixed does **DP4A + q8_1 activation quantization**
   (llama.cpp's technique; measured here as a 3.3Ã— dequant cost) become the right
   kernel work.
4. More GPU work per token (GPU attention / Gated DeltaNet) would also raise the
   duty cycle, but that is explicitly out of scope for now.

**Do not read the 2.537 tok/s as an achieved speedup**: it required an external
process eating 94 % of the GPU. It is a controlled experiment that proves the
cause, not a deployable configuration.

---

## 15. Duty-cycle phase — timeline, async overlap, and the clock experiment

### 15.1 Phase 1 — measured token timeline (before async)

Mean over 24 decode tokens, identical prompt/settings; `KATALI_CUDA_TIMELINE=1`.

| arm | wall | attn | router | shared | gpu_sub | gpu_k | rest | idle | duty |
|---|---|---|---|---|---|---|---|---|---|
| GPU fused | 538.8 | **184.1** | 6.0 | 14.9 | **284.4** | 188.8 | 49.4 | 349.9 | 35.1 % |
| CPU only | 314.3 | 182.7 | 5.4 | 10.5 | 0 | 0 | **115.7** | — | — |

*(ms/token)*

Reading it: attention is **184 ms of CPU, 58 % of the CPU arm's wall**; the CPU
arm's `rest` (115.7) contains the routed experts while the GPU arm's `rest` (49.4)
does not, so **the routed experts cost ~66 ms on CPU** — yet the GPU path charged
**284 ms** for the same work. That 218 ms delta *is* the 1.83-vs-3.27 gap.

### 15.2 Phase 2 — dependency analysis (what may and may not overlap)

The layer chain is strictly serial and this was verified in the code, not assumed
(`src/forward.c` `host_forward_token`):

```
attn(L) -> x += attn_out      -> post_attn_norm(L) -> MoE(L) -> x += moe_out
                                                                    |
                                                          attn(L+1) needs x
```

- `MoE(L)` consumes `post_attn_norm(L)`, so it cannot start before `attn(L)` ends.
- `attn(L+1)` consumes the residual that includes `moe_out(L)`, so it cannot start
  before `MoE(L)` ends.
- **The router must run on the CPU before the GPU can be told which experts to
  run**, so it is unavoidably on the critical path.
- **Prefetching layer L+1's experts is impossible**: their ids are unknown until
  L+1's router, which needs L's output.

The only genuinely independent CPU work per layer is the **shared expert**, which
depends on `x` and not on the routed experts. Measured value: **14.9 ms/token**
(~3 %). This is the whole available overlap in the current dataflow.

### 15.3 Phase 3/4/5 — async MoE submit + shared-expert overlap (IMPLEMENTED)

New ABI pair (ABI 5): `katali_cuda_moe_submit()` queues the layer and returns;
`katali_cuda_moe_wait()` blocks once. The **only** synchronisation removed is the
final one; the caller must wait before reading `y`, and the D2H copy is
stream-ordered anyway. If event creation fails the code falls back to the blocking
path rather than returning an unsynchronised result.

`moe_ffn_mlp_routed` now: submit → run the shared expert on the CPU → wait →
accumulate. The accumulation is `routed + shared`, arithmetically identical to the
previous `r += routed; r += shared` because `y` starts zeroed, so the **output
stays byte-identical**.

| arm | tok/s | output hash |
|---|---|---|
| CPU | 3.199 | `8EC7BAC655` |
| **GPU async** | **1.902** | `8EC7BAC655` |
| GPU sync | 1.844 | `8EC7BAC655` |
| CPU (bracket) | 3.194 | `8EC7BAC655` |

**+3.1 % and byte-identical.** Enabled by `KATALI_CUDA_MOE_ASYNC=1` (default);
`=0` restores the fully synchronous path.

### 15.4 A real defect found on the way: a forced mid-layer drain

The per-layer submission cost (`sub`) measured **169.4 ms/token (4.2 ms/layer)**,
far too high for 4-5 launches. Cause: the job tables were written with
**synchronous `cudaMemcpy`**, which — documented CUDA behaviour — *synchronizes
the stream first*. That force-drained the layer **half-way through**, after
gate+up+silu and before the down launch, on every layer.

**Fix:** pass the ~1 KB job table **by value as a kernel parameter** (the original
design; it had been moved to a device buffer while chasing a bug that turned out
to be elsewhere). No copy, no forced sync.

| | before | after |
|---|---|---|
| `sub` | **169.4 ms** | **3.0 ms** (56×) |
| `wait` | 60.7 ms | 219.0 ms |
| wall | 534.3 | 524.1 |
| tok/s | 1.872 | **1.908** |
| duty | 38.1 % | 37.6 % |

`sub + wait` went 230.1 → 222.0 ms: the sync was relocating time between buckets
rather than creating it, but the split is now honest (3 ms really is submission)
and the residual cost is correctly attributed to waiting for the GPU.

### 15.5 Phase 9 — duty-cycle metric (`KATALI_CUDA_TIMELINE=1`)

After the mid-layer fix, per decode token, 64 tokens:

| component | ms/token | share of wall |
|---|---|---|
| attn (CPU) | 188.9 | 36 % |
| router (CPU) | 6.2 | 1 % |
| shared (CPU, overlapped with GPU) | 14.7 | 3 % |
| vram_cache_get (lookup + malloc + upload) | 89.1 | 17 % |
| xup (activation upload) | 14.0 | 3 % |
| sub (launch call) | **3.0** | 0.6 % |
| **wait (CPU blocked on GPU)** | **219.0** | **42 %** |
| rest (norm, residuals, LM head, sampler) | ~34 | 6 % |
| **wall** | **524.1** | — |
| **GPU duty cycle** (`gpu_k / wall`) | **37.6 %** | (inflated: the event window includes host gaps) |

`idle = wall - gpu_k` ≈ 327 ms. The event window itself (196.8 ms) is not pure GPU
compute — it spans the host's own submission gaps — so the true GPU-busy fraction
is lower still. **The 219 ms `wait` is the dominant single cost in the token.**

### 15.6 Phase 8 — keep-warm experiment: FAILED (measured)

A tiny periodic kernel (grid=1, block=1, one FMA) on a non-blocking stream, every
1 and 2 ms, `KATALI_CUDA_KEEPWARM=<ms>`. Clock state sampled with `nvidia-smi`
during real decode:

| interval | SM clock during decode | memory clock during decode | util | collapse? |
|---|---|---|---|---|
| off | 210 MHz | 405 MHz | 39–53 % | — |
| **2 ms** | **210 MHz** | **405 MHz** | 39–55 % | **still collapses** |
| 1 ms | 210–255 MHz | 405 MHz | 43–55 % | still collapses |

**A small amount of periodic activity does not prevent the collapse.** The driver's
power management responds to *load*, not to the *presence* of work. This was worth
testing precisely because it would have been a cheap fix — it is not, and it is
**not** enabled by default. (Earlier diagnostic: holding the GPU at ~94 %
utilization *did* hold boost at 2850/8251 MHz and halved the layer window — but
that consumes the whole GPU and is explicitly not a deployable solution.)

### 15.7 What the new measured bottleneck is

**The clock state sets the cost of the 219 ms wait, and the clock state is set by
a duty cycle this dataflow cannot raise.**

- GPU work per token is only **~20–77 ms** (measured at boost with contention;
  the 197 ms event window at idle clocks is inflated by host gaps).
- CPU work per token is **~250–310 ms**, of which **189 ms is attention**.
- Attention is bounded on *both* sides by true dependencies (§15.2), and the
  remaining independent CPU work is only the shared expert (15 ms).
- Therefore the GPU idles ~94 % of wall time, the driver keeps it at idle clocks,
  and the memory clock stays 21× below boost.
- Async overlap and removing the forced mid-layer sync both helped (1.844 →
  1.903/1.908 tok/s) but **cannot change the duty cycle**, so they cannot fix the
  clocks.

This is **CASE B with a CASE C conclusion**: overlap improved things modestly and
measurably; the GPU still drops clocks; and the profiling says the duty cycle
cannot be raised within the current scope because the remaining CPU block
(attention) is on the critical path and moving it to the GPU is out of scope.

### 15.8 Recommendation for the next phase

Ranked by measured value:

1. **Reduce total GPU work so the wait shrinks proportionally.** The 219 ms wait
   is bounded below by the GPU's actual work; the one measured lever is
   **DP4A + q8_1 activation quantization** (llama.cpp's technique), worth the
   measured **3.3× dequant cost** (cold benchmark: 31.9 → 106.4 GB/s without
   dequant). This is the only in-scope change that directly attacks the largest
   bucket.
2. **Re-examine the scope decision on GPU attention.** The 189 ms attention block
   is 36 % of the token and the single largest reason the duty cycle is ~6 %. It
   is the difference between CASE B and CASE A. No amount of MoE-side work fixes
   the duty cycle while 189 ms of serial CPU attention sits between every GPU
   burst. This is a scope call, not a technical one, so it is flagged rather than
   taken.
3. **Do not pursue keep-warm** (measured failure) and **do not pursue a persistent
   kernel** for clock purposes: the keep-warm result shows the driver ignores
   activity that carries no load, and a persistent spinning kernel is exactly the
   wasteful 94 %-utilization pattern this phase was told not to use.
4. Re-test any of the above with the timeline and clock sampling enabled, since a
   kernel A/B is meaningless if one arm ran at 210 MHz and the other at 2505 MHz.

## 16. New tooling added in this phase

| Command | Purpose |
|---|---|
| `cuda-bench-cold [model] [layer] [slots]` | Phase 1/8/9 cold ladder: raw stream ceiling, `nomath` vs real GEMV, rotating cold experts, optional multi-GB footprint and churn probe |
| `cuda-hold <seconds>` | Holds the GPU under continuous load to keep it at boost clocks (diagnostic for the power-state finding) |
| `KATALI_CUDA_LAYER_PROFILE=1` | Phase 11 per-layer line: bytes, GPU ms, GB/s, and VRAM hit/miss/evict/upload |
| `KATALI_CUDA_GPU_TIME=1` | Device-event kernel timing (also settable via the ABI, because a late `_putenv` never reaches the lazily-loaded DLL's CRT) |
| `KATALI_GEMV_UNROLL=1\|4` | Selects the JIT A/B variant of the job GEMV (default 1; 4 measured worse at n_sel=8) |
| `KATALI_VRAM_GB` | VRAM tier budget override (used for the paging sweep) |
| `KATALI_CUDA_DP4A=1` | §17: enables the DP4A + q8_1 kernels (default OFF). `KATALI_CUDA_DP4A_Q4K=0` / `KATALI_CUDA_DP4A_Q6K=0` gate the path per weight type |
| `cuda-dp4a-selftest` | §17 synthetic exactness test: builds Q4_K/Q6_K slabs whose contents are known and an activation that q8_1 represents exactly, then proves the DP4A kernels match the fp32 kernels to float precision |
| `bench_dp4a.bat` | §17.6 the matched four-arm end-to-end A/B (CPU, GPU fp32, GPU DP4A, CPU bracket) with the timeline instrumentation on |
| `bench_attn_split.bat` | §18.2 the MoE-free attention split (`KATALI_SKIP_MOE=1` plus one skipped attention kind per arm), which is what makes the GDN-vs-full-attention numbers trustworthy |
| `KATALI_SKIP_MOE=1`, `KATALI_SKIP_DELTA=1`, `KATALI_SKIP_GQA=1` | Pre-existing probes, used in §17.6 to split the attention bucket. **`KATALI_SKIP_MOE=1` is what makes that split clean** — with the MoE in the loop, skipping an attention kind changes the routing, which changes the expert-cache behaviour, which contaminates the measurement (that mistake is recorded in §17.6) |

---

## 17. Stage A — DP4A + q8_1 activation quantization (implemented, measured)

### 17.1 Research: how llama.cpp avoids per-element dequantization

Read before writing code (raw source, not documentation):

| Source | What it establishes |
|---|---|
| `ggml/src/ggml-cuda/quantize.cu` → `quantize_q8_1` | The activation quantization kernel. One thread per activation element; `amax = warp_reduce_max<QK8_1>(fabsf(xi))` and `sum = warp_reduce_sum<QK8_1>(xi)` reduce across exactly 32 lanes (= one q8_1 block), then `d = amax/127`, `q = roundf(xi/d)` (0 when `amax == 0`), `y[ib].qs[iqs] = q`, and lane `iqs == 0` writes `y[ib].ds = make_half2(d, sum)`. It runs **once per matvec call**, not once per expert |
| `ggml/src/ggml-cuda/mmvq.cu` | `mul_mat_vec_q` family: `nrows` / `nwarps` / `ncols_dst` are template parameters; a dedicated `mul_mat_vec_q_moe_launch` exists for the multi-token `MUL_MAT_ID` (MoE) path; `get_vec_dot_q_cuda()` dispatches to per-type `vec_dot_q4_K_q8_1` / `vec_dot_q6_K_q8_1`; `prefetch.global.L2` is **deliberately gated to DGX Spark**, with the comment that on higher-bandwidth parts "the kernel has little exposed latency left to hide and the extra requests cost more than they save" |
| `ggml/src/ggml-common.h` | `block_q8_1`: `QK8_1 = 32`, `{ half2 ds; int8_t qs[32]; }` = 36 bytes, with `ds.x = d` and `ds.y = sum` |
| `ggml/src/ggml-cuda/mmq.cu` | MMQ (mat-mat) is preferred for MoE prefill; `MMQ_DP4A_MAX_BATCH_SIZE` gates mmq-vs-BLAS |
| `ggml/src/ggml-cuda/fattn-vec.cuh`, `fattn.cu` | (Stage B background) batch-1 decode attention: `flash_attn_ext_vec` is chosen for `Q->ne[1] == 1` (or `<= 2` when K/V are quantized); 128 threads, `ncols_per_block = 1`, and no shared-memory staging (`nbytes_shared = 0`) |

**Licence check before adapting anything:** llama.cpp / ggml is **MIT**; this repository is Apache-2.0. Nothing was copied. What is reused is the q8_1 *idea* and the arithmetic identity; both kernels here are written for KATALI's own layouts and job batching, and KATALI's on-device q8_1 block is not even byte-compatible with ggml's (see 17.3).

### 17.2 The technique, precisely

1. **How the activation is quantized.** Per 32-value block: `d = max|x|/127`, `q_i = round(x_i/d)` as `int8`, plus the *integer* sum `s = Σq_i`.
2. **Where.** On the device, on the whole activation, immediately before the quantized GEMV.
3. **How often.** Once per matvec — and for KATALI, once per *layer*, because all selected experts read the same activation. Measured: 0.013 ms per layer (0.33 ms per 24 fused calls at `n_sel=8`, i.e. 8 % of the DP4A GEMV time, not hidden).
4. **How Q4_K/Q6_K interact with q8_1.** Q4_K gives `y_i = (d·sc_g)q_i − (dmin·m_g)` per 32-value group, so
   `Σ y_i a_i = (d·sc_g)·Σ(q_i a_i) − (dmin·m_g)·Σa_i`
   — one integer dot plus the group's activation sum, which the q8_1 `s` field *is*. Q6_K gives `y_i = (d·sc_g)(q6_i − 32)` with `q6 = lo4 + 16·hi2`, so
   `Σ y_i a_i = (d·sc_g)[Σ(lo4·a) + 16·Σ(hi2·a) − 32·Σa]` — two integer dots plus an `s`-based correction.
5. **Where DP4A is used.** `__dp4a(uint8x4, int8x4, acc)`: packed weight quants in the *unsigned* operand, int8 activations in the *signed* one, 4 MACs per instruction.
6. **How scales are applied.** Once per group, outside the inner loop: `(d·sc·d8)·sumi` plus `(dmin·m·d8)·s`. No per-element float conversion exists in the loop.
7. **How accumulation works.** integer `__dp4a` per thread → one fp32 multiply per group → the same block-level fp32 reduction the fp32 kernel already used.
8. **How the final fp32 result is produced.** Exactly the fp32 dot, differing only by q8_1 rounding and summation order.
9. **Amortization.** Yes, and KATALI exploits it further than llama.cpp does: gate/up share one quantization per layer, and the down projection's per-expert intermediate is quantized *inside* the silu pass (no extra launch, no re-read).
10. **Fit to KATALI's dimensions.** `H=2048` → 4 words/thread, `FF=512` → 1 word/thread; both divide exactly, so no thread straddles a q8_1 block or a scale group. Checked at run time by `dp4a_ok()`; anything else silently keeps the fp32 kernel.
11. **Ada sm_89.** `__dp4a` exists from sm_61 and runs on the INT32 pipes. Measured result: the DP4A GEMV is **3.4× faster** than the fp32-dequant GEMV at `n_sel=8`.

### 17.3 What was implemented (KATALI-specific)

* **Device-only q8_1 block**: `float d; float s; float s_lo; int8_t qs[32];` = **44 bytes** (ggml's is 36). Deliberate: fp32 scales keep the integer sums exact (a 32-value int8 sum reaches 4064, which f16 cannot represent exactly) and remove half↔float conversion from the inner loop. `s_lo` is the sum of the *first 16* values, because Q6_K's scale groups are 16 wide while a q8_1 block is 32 and the `−32·Σa` correction needs a 16-value sum. Never written to disk, so it does not have to match ggml.
* `kce_quant_q8_1_kernel` — one warp per 32-value block; `d`, `s`, `s_lo` are warp reductions; `lrintf` + saturation mirror `roundf` semantics.
* `kce_silu_weight_q8_kernel` — the existing `w·silu(gate)·up` pass, additionally emitting the q8_1 form of its output. Saves a launch *and* a re-read of the intermediate per layer, and is exactly valid because that kernel is elementwise with one thread per value: a warp owns 32 consecutive values = one q8_1 block, which needs only the intermediate width to be a multiple of 32 (checked).
* `kce_gemv_jobs_q4k_dp4a_kernel` / `kce_gemv_jobs_q6k_dp4a_kernel` — the same one-block-per-(job,row) geometry as the fp32 job kernel, so both paths run on identical schedules. Words-per-thread `NW` is a template parameter {1,2,4,8} for Q4_K, {1,2,4} for Q6_K; `dp4a_ok()` guarantees `cols % 512 == 0` and a whole number of words per thread, so no thread straddles a q8_1 block (8 words) or a Q6_K scale group (4 words). The `−dmin·m·Σa` term is added by exactly one thread per group (the one owning the group's first word). Q6_K's `ql`/`qh` are read with **two half-word loads**, because a 210-byte super-block stride leaves rows only 2-byte aligned.
* **Selection.** `KATALI_CUDA_DP4A=1` enables it; **default OFF**, so the fp32 path stays the byte-for-byte default. `KATALI_CUDA_DP4A_Q4K` / `_Q6K` gate per type, and `katali_cuda_set_dp4a()` / `katali_cuda_set_dp4a_types()` let one process A/B both arms. Each projection is decided independently.
* **ABI 6** adds `katali_cuda_set_dp4a`, `katali_cuda_set_dp4a_types`, `katali_cuda_dp4a_active` and the `q8_1_quant_launches` / `q8_1_quant_seconds` counters, so the quantization overhead is *reported* rather than hidden. The elapsed time is read lazily at the end of the layer: bracketing the quant kernel with events must not introduce a mid-layer `cudaEventSynchronize`, because that is the forced-drain defect fixed in §15.4.
* **Device-error reporting.** Every launch/sync path now prints `kce: <where>: CUDA error: <what>` on unbuffered stderr. Without it the two kernel bugs in 17.7 were invisible — an asynchronous device fault killed the process before its buffered stdout was flushed. Note `cudaGetLastError()` also *clears* the error, so it must be read exactly once per check (a second read reports "no error" and hides the cause — observed).

### 17.4 Cold benchmark: current kernel vs DP4A (`cuda-bench-cold`, extended)

374 MiB resident working set (192 experts), rotating experts so consecutive calls touch disjoint slabs, device-event timing. Bytes are weight bytes only.

```
  n_sel calls  dqGB/s   dp4aGB/s  nmGB/s    dq_ms   dp4a_ms  q8_ms   q8_lch  launch  d4
  1     192    22.8     25.8      38.1      16.0    14.2     2.86    1.00    5       1
  2     96     29.0     47.1      68.6      12.6    7.7      1.29    1.00    5       1
  4     48     33.5     88.7      113.6     10.9    4.1      0.55    1.00    5       1
  8     24     36.4     109.4     134.1     10.0    3.3      0.33    1.00    5       1

Stage A per-type A/B (n_sel=8, 24 calls = 374 MiB each):
  arm                      GB/s      gpu_ms   q8_ms   q8_lch  launch d4
  fp32 dequant (baseline)  36.3      10.04    0.00    0.00    4      0
  dp4a Q4_K only           68.4      5.33     0.27    1.00    5      1
  dp4a Q6_K only           47.1      7.74     0.00    0.00    4      1
  dp4a Q4_K + Q6_K         114.1     3.20     0.27    1.00    5      1
```

**MEASURED**

* At the engine's operating point (`n_sel=8`) the fused layer goes **10.0 → 3.20 ms per 24 calls: 3.1× faster**, 36.3 → **114 GB/s**, against a `nomath` read-only ceiling of 134 GB/s. The DP4A kernel is now within 15 % of a kernel that does no arithmetic at all.
* **The q8_1 quantization costs 0.27 ms of the 3.20 ms (8.4 %)** and is counted, not hidden. It is one launch per layer (5 vs 4).
* **Q4_K is the bigger lever**: 1.4× more of its bytes (gate+up 1.18 MiB vs down 0.86 MiB per expert) *and* relatively better inside dp4a (68.4 vs 47.1 GB/s), because Q6_K needs two integer dots plus a correction per 4 values where Q4_K needs one. That answers "which type first" by measurement: both, Q4_K first.
* The gain grows with `n_sel` (1.13× at `n_sel=1` → 3.1× at 8), because at low `n_sel` the kernel is latency-bound rather than issue-bound — exactly the state the idle clocks produce.
* L2-warm per-layer check (`cuda-check-moe`, 30 reps): **0.443 → 0.123 ms/call**, 34.3 → 124.0 GB/s (**3.6×**).

### 17.5 Correctness: what changed numerically, and the tolerance

q8_1 is a different numerical path, so byte-identical output was explicitly not expected. Measured, in increasing order of strength:

| Test | Result |
|---|---|
| `cuda-dp4a-selftest` (new, synthetic) | **PASS, rel_L2 = 8.342e-07** on outputs of magnitude ~1e4, i.e. fp32 rounding. It makes the quantization disappear: slabs that decode to exactly 1.0 plus a constant activation produce an intermediate that is constant per 32-value block, which q8_1 represents exactly (`d = amax/127`, every `q = ±127`). Anything larger would be a weight-layout bug |
| Exactness inside `cuda-bench-cold` (integer activation, whole layer, Q4_K arm) | **rel_L2 = 2.152e-07** — the whole layer is exact when nothing is rounded. The Q6_K arm reads 5.444e-03 there, which is the rounding of the *intermediate* that the down projection necessarily consumes |
| `cuda-check-moe`, fp32 path (regression) | rel_L2 = 2.292e-07, 0 bad — **unchanged**, so the refactor did not perturb the existing kernel |
| `cuda-check-moe`, DP4A path, real weights | rel_L2 = **7.179e-03**, max_abs = 4.2e-05, **elems_bad = 0/2048**. The harness uses `rel_L2 <= 1e-3` for the fp32 kernel and now applies, and *prints*, the documented 2e-2 relative tolerance plus the element-wise absolute criterion when the DP4A kernel is the one that ran — a bare "FAIL" would misread a correct kernel as broken. The element-wise criterion passes with a 24× margin |
| End-to-end 64-token generation | **byte-identical** across CPU, GPU-fp32 and GPU-DP4A (hash `8EC7BAC655`, the same hash §15.3 recorded) |

Read the `bad(1e-3)` column in the cold bench carefully: it is an **absolute** count of elements outside `1e-3`, which is meaningful for the real-model check (outputs ~1e-3) but not for the synthetic exactness arm (outputs ~1e4, where 1e-3 is below one ulp). That arm is judged by `rel_L2` alone, and the `cuda-dp4a-selftest` command uses a relative element criterion for exactly this reason.

A repeat of the whole cold ladder on a more idle GPU gave 31.8 → 104.8 GB/s and 11.47 → 3.48 ms (3.3×) against the run quoted above (36.4 → 109.4 GB/s, 10.0 → 3.3 ms, 3.1×): the *absolute* GB/s move with the clock state, the **ratio does not**, which is why the A/B is trustworthy even on a machine whose clocks wander.

**Documented tolerance for the DP4A path:** global `rel_L2 ≤ 2e-2` against the fp32-dequant kernel (measured 7.2e-03 real weights / 7.7e-03 cold), every element inside `1e-3·(1+|ref|)` (measured 0 violations), and token-identical generation on the reference prompt. The synthetic test bounds the *kernel* error at 1e-5, so the residual 7e-3 is attributable to q8_1 activation rounding and nothing else. This is not silent: DP4A is opt-in, and the default path is unchanged.

### 17.6 End-to-end (matched A/B, same prompt/settings as §12/§13/§15)

`bench_dp4a.bat`, prompt `Count from 1 to 50, comma separated.`, `--max 64 --pin 25 --cache-gb 8`.

| arm | tok/s | output |
|---|---|---|
| CPU + RAM + SSD | 2.504 | `8EC7BAC655` |
| GPU, fp32 dequant | 1.720 | `8EC7BAC655` |
| **GPU, DP4A + q8_1** | **2.245** | `8EC7BAC655` |
| CPU (bracket) | 2.597 | `8EC7BAC655` |

An un-instrumented 32-token re-run on the same machine: CPU **2.665**, GPU fp32 **1.705**, GPU DP4A **2.268** tok/s.

**MEASURED:** DP4A improves the GPU arm by **+30.5 %** (1.720 → 2.245) and **+33 %** (1.705 → 2.268) in the two runs. The absolute level is below the §15 baseline (CPU 3.20 / GPU 1.91) because this machine is currently ~17 % slower than during that phase (its own CPU bracket is 2.50-2.67, not 3.20); every arm of the A/B was measured under the same conditions, so the *ratios* are the meaningful result. Against the §15 baseline numbers, +33 % on 1.91 is **≈ 2.54 tok/s** — the Level 2 mark — while the raw measurement here is **Level 1 (> 2.2)**. It does **not** reach the CPU baseline (Level 3).

Per-token timeline (`KATALI_CUDA_TIMELINE=1`, mean of the last 40 decode tokens, ms/token):

| bucket | fp32 | DP4A | change |
|---|---|---|---|
| **gpu_k (device-event window)** | **209.7** | **70.1** | **−139.6 ms (3.0×)** |
| **wait (CPU blocked on GPU)** | **221.7** | **79.8** | **−141.9 ms (2.8×)** |
| wall | 586.6 | 440.4 | −146.2 ms (−25 %) |
| attn (CPU) | 211.2 | 209.9 | unchanged |
| router (CPU) | 6.7 | 6.5 | unchanged |
| shared (CPU, overlapped) | 14.9 | 14.6 | unchanged |
| vram_cache_get | 48.1 | 48.5 | unchanged |
| xup (activation upload) | 11.8 | 11.7 | unchanged |
| sub (launch calls) | 2.7 | 3.4 | +0.7 ms (one extra launch/layer) |

So the GPU MoE work per token really did drop 3.0× and the wait followed it. The tok/s gain is only +31 % because Amdahl now applies in the other direction: 141 ms of the 587 ms was removable, and the remaining wall is dominated by CPU attention.

**Clocks (the §14.5 finding is unchanged).** `nvidia-smi` sampled once a second throughout the A/B: 168 samples, of which ~88 % read `210 MHz / 405 MHz` (the idle state) with utilization 4-50 %, and only ~4 % read boost. The DP4A kernel made the GPU *faster at the same clock*, which is why it helped at all; it did not change the power state, because the duty cycle is still set by the CPU-side dataflow.

### 17.7 Failed experiments and mistakes (kept, because they are instructive)

| Attempt | Outcome |
|---|---|
| Row-wide group index used for the *weight*-side offsets (`chunk = g>>1`, `half = g>>2` on the row-wide group) | Illegal address on every layer. The weight layout is local to the super-block; the activation index is the only row-wide one. Found only after adding device-error reporting |
| Reading Q6_K `ql`/`qh` with 4-byte loads | `misaligned address` — 210-byte super-block stride ⇒ 2-byte alignment. Fixed with two half-word loads |
| Q4_K scale index taken from the row-wide group instead of the block-local one | Numerically wrong (rel_L2 ≈ 1.2), out-of-range `scales[]` reads |
| Gating the whole DP4A path on the gate projection being eligible | The "Q6_K only" A/B arm silently ran fp32 twice and reported "no speedup", which looked like a kernel property. Each projection is now decided independently |
| First synthetic "exactness" activation: `(i*37)%255 - 127` | Integer-valued but with per-block `amax < 127`, so `d < 1` and the values were still rounded. The test reported a 4.5e-3 "bug" that was quantization. The vector now contains ±127 in every block |
| `cudaEventSynchronize` on the q8_1 quantization window inside the layer | Rejected before use: that is the §15.4 mid-layer drain. The timing is recorded and read lazily at the layer's existing sync point |
| A second `cudaGetLastError()` inside the error reporter | Reported "no error" and hid the real cause (the call clears the error). The error code is now captured once |
| `KATALI_SKIP_DELTA` measured *with the MoE running* | Suggested GDN = 94 ms/token, which contradicted the 210 ms bucket. Skipping attention changes the activations, hence the routing, hence the expert-cache behaviour. Re-measured with `KATALI_SKIP_MOE=1` (below): the clean number is 147 ms. The contaminated measurement is not used |

---

## 18. Stage B — should CUDA attention be the next major GPU target? (research + design)

### 18.1 Research: what other engines do for batch-1 decode attention

| Project | Source read | What applies to KATALI |
|---|---|---|
| llama.cpp / ggml CUDA | `fattn.cu`, `fattn-vec.cuh` | A **dedicated small-batch path exists and is the default for token generation**: `flash_attn_ext_vec` with `ncols_per_block = 1` for `Q->ne[1] == 1`, 128 threads, and `nbytes_shared = 0` — the KV tile is read **directly from global memory**, not staged through shared memory. `ncols2`/GQA handling makes one KV read serve several query heads. Tensor-core (`mma-f16`) paths are only chosen when there are enough queries. So for batch-1 decode the "FlashAttention" everyone cites is *not* the tensor-core kernel; it is a plain vector kernel that is bandwidth/latency bound |
| llama.cpp, KV placement | `-nkvo` / `no_kv_offload` | KV can be kept in host RAM while attention runs on the CPU — i.e. the mainstream implementation of exactly KATALI's hybrid arrangement, with the KV transfer paid per attention call |
| FlashInfer / vLLM / TensorRT-LLM XQA | design-level (no source read here) | All three converge on the same conclusion: a single-request decode is *memory-latency bound*, so it is attacked with split-KV/cooperative reduction (split the KV into chunks, one block each, combine partial softmax results), GQA amortization (one K/V use per 8 query heads here), and paged KV. XQA in particular exists because generic FlashAttention is wasteful for MHA/GQA at batch 1 |

**Not read in depth, and therefore not cited as evidence:** FlashInfer, vLLM, SGLang and TensorRT-LLM sources. The statement above is what their documentation/design descriptions say; it is flagged so it is not mistaken for a source-level finding.

### 18.2 The current KATALI attention architecture (measured, not assumed)

From `katali-lab.exe info` on the 35B: `hidden=2048 heads=16 kv_heads=2 head_dim=256 q_proj=8192`, `ctx_train=262144`, `rope_dim=64`, and **`layer_kinds: linear/DeltaNet=30, full_attn=10`**.

So "attention" in the 210 ms bucket is two *different* workloads, and they had to be measured separately. `KATALI_SKIP_MOE=1` removes the MoE entirely, so nothing downstream depends on the (garbage) activations a skipped attention produces:

| arm (CPU, 32 tokens, MoE disabled) | decode | ms/token |
|---|---|---|
| all attention | 7.213 s | 225.4 |
| GDN (delta) layers skipped | 2.502 s | 78.2 |
| full-attention layers skipped | 5.785 s | 180.8 |

* **30 GDN layers cost 147.2 ms/token** (≈ 4.9 ms/layer).
* **10 full-attention layers cost 44.6 ms/token** (≈ 4.5 ms/layer).
* The 78.2 ms/token baseline is norms + lm_head + embedding + sampler.

Both per-layer costs (4.5-4.9 ms) are far above what their *weight traffic* would cost (28 MB/layer GDN → ~1.1 ms at 25 GB/s; ~20 MB/layer full-attn → ~0.8 ms), so **both are dominated by single-threaded scalar CPU compute**, not by memory. `src/attn_delta.c` and `src/attn_gqa.c` contain no threads or SIMD: the GDN recurrence, the `1/sqrt` q/k normalisation loops, the causal conv1d, the rope application and `katali_gguf_gqa_decode_scratch` are all scalar.

**KV cache**: `src/forward.c:82` — `st->kv_k = xcalloc(n_full * NKV * max_seq * HD, sizeof(float))` (and the same for V). It lives in **system RAM, fp32, contiguous per layer**, indexed `[layer][kv_head][pos][head_dim]`. No VRAM involvement of any kind today.

### 18.3 KV-cache strategy (the numbers, before any decision)

KV bytes per token of context: `10 layers × 2 kv_heads × 256 head_dim × 2 (K and V) × 4 B` = **40 KB/token**. At `heads=16` with `kv_heads=2`, one K/V value serves 8 query heads, so a full-KV read per decode token is 40 KB × context × (1 if the kernel amortises GQA, ×8 if it does not).

| context | fp32 KV | fp16 KV | q8_0 KV |
|---|---|---|---|
| 4 096 | 168 MB | 84 MB | 42 MB |
| 8 192 | 336 MB | 168 MB | 84 MB |
| 16 384 | 671 MB | 336 MB | 168 MB |
| 32 768 | 1.34 GB | 671 MB | 336 MB |
| 131 072 | 5.4 GB | 2.7 GB | 1.3 GB |
| 262 144 (`ctx_train`) | 10.7 GB | 5.4 GB | 2.7 GB |

Read cost of the whole KV once per token: at 25 GB/s from RAM = 1.6 ms @4k / 13 ms @32k; over PCIe H2D at ~5-6 GB/s = 7 ms @4k / 55 ms @32k; from VRAM at the *idle* memory clock (~13 GB/s measured-equivalent) = 3 ms @4k / 25 ms @32k; at boost (>100 GB/s) = 0.4 ms @4k / 3 ms @32k.

**Recommendation (a clear answer, not a hedge): option B for the VRAM-resident range and option C beyond it.**

* **B — move KV entirely to VRAM** while `10 × ctx × 40 KB ≤ budget` (≈ 32k context fits in 1.34 GB fp32, 671 MB fp16). This is the range the engine actually runs in today (the tests use 93-token contexts; the elastic design targets long context only in principle).
* **C — hybrid beyond that**: keep a VRAM-resident *recent* window plus a RAM-resident tail, because attention weights tokens equally by position, so a "recent window" is only valid with a *scoring* pass over the old tail — that is a correctness change, not just a placement change, and is therefore explicitly **not** part of the smallest PoC. The honest intermediate is "KV in VRAM up to the budget; above it, fall back to the CPU attention path for that layer" — i.e. the existing elastic contract, one tier further down.
* **A (remain in RAM)** is only right if the attention stays on the CPU. Streaming 1.34 GB/token over PCIe to do GPU attention is 220+ ms/token at 6 GB/s, which is worse than the CPU.
* fp32 → **fp16 KV** is a 2× saving and is the standard trade everywhere; it halves the PCIe cost of the initial upload and the VRAM footprint. It changes numerics (rope + softmax inputs), so it belongs in the PoC's tolerance discussion, not in the first commit.

### 18.4 VRAM budget (8 GB device, 8 188 MiB physical, 19 MiB in use when idle)

| item | bytes | note |
|---|---|---|
| CUDA runtime + driver + context | ~250 MB | measured: `cudaMemGetInfo` delta across `backend_init` |
| Desktop/compositor reserve | 512 MB | already the constant in `vram_cache_budget_from_env`; measured free at idle = 8 169 MiB |
| q8_1 activations, job scratch, fused-layer scratch | < 1 MB | `scr_ensure`: `sel*FF` + `sel*H` floats + 2 q8 buffers (2.8 + 5.6 KB at the 35B's shape) |
| MoE resident expert slabs (auto = free − 512 MB, i.e. ~7.5 GiB today) | elastic | must stay elastic; this is the tier the project exists for |
| **KV, 32k context, fp32** | **1.34 GB** | 671 MB fp16 |
| **dense attention weights, all 40 layers, as stored in the GGUF** | **1.05 GB** | GDN 842 MB + full-attn ~205 MB (Q4_K/Q5_K/Q6_K/Q8_0, no conversion) |
| dense attention weights converted to fp16 | ~2.1 GB | only if a future kernel demands f16 |
| lm_head + token_embd (dense, optional) | 703 MB | 286 MB Q4_K + 417 MB Q6_K; today they stay in RAM |

**A GPU-attention variant that keeps KV at 32k in VRAM and the attention weights resident needs ≈ 2.4 GB** (1.05 GB weights + 1.34 GB KV) plus the 250 MB context and the 512 MB desktop reserve — leaving ≈ 4.5-5 GB for experts, i.e. ≈ 2 300-2 500 resident expert slabs of ~2 MB. That fits, but it is *not* hardcoded for 8 GB: the budget must be computed at open time from `vram_free_bytes` exactly as `vram_cache_budget_from_env` already does, and the attention tier must be *skipped* (CPU path) when it does not fit.

### 18.5 Proposed CUDA decode-attention architecture (PROPOSED, nothing implemented)

Per full-attention layer, one submission, everything device-resident:

```
x (2048 f32, already uploaded)
  -> q8_1 quantize (existing kernel, existing q8_1 block)
  -> attn_q GEMV   (8192x2048 Q6_K/Q4_K)   existing DP4A job kernel
  -> attn_k / attn_v GEMV (512x2048 Q8_0)  existing DP4A path needs Q8_0 support
  -> q_norm / k_norm (256-wide RMSNorm) + rope (rope_dim 64)   NEW small kernels
  -> KV store: append the new K/V row into the VRAM KV buffer  NEW tiny kernel
  -> GQA decode attention over pos+1 keys: 16 q heads x 2 kv heads x 256 dim
        one block per query head-group, K/V read from VRAM, running max/sum
        rescaling in registers (flash-style online softmax), 8 heads share a K/V read
  -> sigmoid(q_gate) multiply                                  existing-style elementwise
  -> attn_output GEMV (2048x4096 Q5_K)                         existing DP4A path needs Q5_K support
  -> D2H 2048 f32
```

Notes that come from the measurements, not from taste:

* The existing q8_1/DP4A machinery covers Q4_K and Q6_K but **not Q8_0 (k/v) or Q5_K (attn_output)**, which is 1.11 + 1.11 + 5.77 MB per layer, i.e. 80 MB/token over 10 layers. Extending the DP4A kernels to Q8_0/Q5_K (both are *simpler* than Q6_K: Q8_0 is a straight int8 dot, Q5_K has no split-field correction) or routing them to the existing fp32 kernels decides most of the per-layer GPU time.
* At the measured idle memory clock, pulling these weights from *host* RAM per token is ~4 ms/layer — the same as the CPU costs today. **Resident in VRAM is what makes the proposal work**, which is why 18.4's 1.05 GB matters.
* GQA amortization (8 query heads per K/V head) is not optional: without it the KV traffic is 8× on a device whose memory clock collapses when idle.

### 18.6 Router placement (measured inputs, then a recommendation)

Measured today: `router = 6.5 ms/token` (0.16 ms/layer) and `sub` (the whole launch call) = 3.4 ms/token for 5 launches/layer. The router's own compute is therefore irrelevant; what matters is the *boundary count*. With attention on the GPU and the router on the CPU, every layer needs **GPU→CPU (x) → router → CPU→GPU (x + job table) → GPU**, i.e. ~10 extra syncs/token across the full-attention layers (and 40 if the GDN moves too).

Precedent from this project, measured: a per-layer forced drain cost **169.4 ms/token** and was only visible because it was instrumented (§15.4). A *legitimate* sync at a layer boundary is much cheaper than a mid-layer drain, but the pattern repeats the structural problem §15.2 identified.

**Recommendation:** if attention moves to the GPU, the **router must move with it** — and it should move *first*, because it is the cheapest of the two: `[2048 × 256] F32` per layer = 2.1 MB/layer, 84 MB over 40 layers, a plain matvec + top-k + softmax. Moving it removes the host from the layer loop entirely (GPU owns attention → router → MoE with no host round-trip *inside* a layer), which is the only mechanism here that can actually raise the duty cycle.

**Measured qualification of the sync cost itself**, so this is not a hand-wave: 10 extra layer-boundary syncs with ~2 KB H2D + 2 KB D2H are well under 1 ms/token at the measured per-layer submission cost (§17.6: 3.4 ms/token for 200 launches = ~17 µs/launch). **The sync cost is not the blocker.** The blocker is that the CPU idles during every GPU attention burst and the GPU idles during every CPU interval; the router placement changes *how much of the token each side owns*, which is exactly what the duty cycle depends on.

### 18.7 Expected architectural advantages — PREDICTED, not measured

Everything in this subsection is a **prediction derived from measured quantities**; none of it has been tested.

* PREDICTED: per-layer GPU full-attention work would be ~0.5-1.5 ms of *resident-weight* reads (20 MB at the idle memory clock's ~13 GB/s equivalent) plus the attention math over `pos+1` keys (negligible at 93 tokens; at 4096 tokens ≈ 34 MFLOP ≈ 1.5 ms at idle / 0.05 ms at boost). Against the measured 4.5 ms/layer on the CPU that is a net win of ~3 ms/layer ≈ **30 ms/token** *if* the weights are resident and the latency is hidden.
* PREDICTED: the same statement for the GDN layers is *not* obviously favourable. Its inner work is a sequential recurrence over 32 value heads with an outer product per head; it parallelises over heads, not over time, and its per-layer weight traffic (28 MB) is the largest of any layer kind. A GPU version is plausible (~2-3 ms/layer predicted) but unmeasured. Its state (32 × 128 × 128 floats = 2 MB/layer, 63 MB total) would live in VRAM.
* PREDICTED: the duty cycle would rise from ~16 % to ~40-50 % if attention *and* the router moved (the GPU would then own ~70 + ~45 + ~147 ≈ 260 ms of the ~440 ms token). That is the only mechanism in this project's measured history that has ever restored boost clocks without burning GPU on dummy work, and it is the reason this deserves a PoC rather than a dismissal.
* PREDICTED: if the clock state *does* recover, the Stage A gains compound (the DP4A kernel is memory-side limited at idle clocks), moving the GPU arm towards the CPU baseline. This is the strongest argument for the PoC, and it is a prediction.
* NOT predicted to help: moving the *projections* alone (already cheap relative to the scalar math), and anything that leaves KV in RAM.

### 18.8 Implementation risks (ordered by how likely they are to kill the work)

1. **The idle-clock trap, twice.** Every number above assumes the weights are *resident in VRAM*. Streamed per token, the PCIe transfer (20 MB/layer at ~5 GB/s ≈ 4 ms/layer) cancels the CPU saving exactly. The PoC must measure the *resident* case and report the clock state with it.
2. **Q8_0 / Q5_K coverage.** DP4A covers Q4_K/Q6_K only; `attn_k`/`attn_v` (Q8_0) and `attn_output` (Q5_K) are 8 MB of the 20 MB per full-attention layer. Without them the kernel win is much smaller.
3. **Numerical drift on a new path.** The MoE tolerance argument is now measured; the attention path has no exactness test yet. A first PoC must be *proved* correct (or wrong) before anything is tuned.
4. **KV ownership.** Two writers (CPU and GPU paths) sharing one KV buffer is a correctness hazard. The PoC must own its buffer and keep the CPU path authoritative elsewhere.
5. **The elastic contract.** Nothing may be hard-coded to 8 GB; the attention tier must fail back per layer when the budget is absent, exactly as `vram_cache_get` returning NULL does today.
6. **Regressions.** The MoE path needed three measured bug hunts in this phase. A new tier must leave the CPU path byte-identical (verifiable with the same token hash).

### 18.9 The exact smallest GPU-attention PoC (specified, NOT implemented)

One layer kind, one layer, one device-resident KV buffer, measured against `attn_gqa_forward` as the oracle:

1. **Scope:** a *single* full-attention layer (e.g. `blk.3`), decode only (`pos+1` keys), batch 1. Its 4 dense tensors (q 13.8 MB, k/v 1.1 MB each, o 5.8 MB ≈ 21.8 MB) are uploaded once at open time into VRAM. KV is not re-architected: the PoC copies that layer's K and V into a dedicated VRAM buffer (`max_seq × 2 × 256` floats ×2 = 2 MB at 4096 positions), refreshed per step (2048 floats = 8 KB, ~1.6 µs at 5 GB/s — measurable, not assumed).
2. **Kernels to write (3, all small):** (a) `rope + q/k RMSNorm`, (b) `kv_append`, (c) `gqa_decode_attention` — one block per query head (16 blocks), 128 threads, online softmax over keys read from VRAM, 8 query heads per KV head so the K/V read is amortised. No paging, no split-KV, no tensor cores.
3. **Reuse:** the q/k/v/o GEMVs use the *existing* Stage A machinery where the type is covered (Q6_K/Q4_K for q; the fp32 kernel for Q8_0 k/v and Q5_K o in the first version) plus the existing q8_1 activation quantization.
4. **Measurement required before any judgement:** per-layer wall ms (CPU oracle vs GPU PoC), the device-event kernel span, H2D/D2H bytes, and **`nvidia-smi` clock samples during the run** — the entire point is to see whether ~10 GPU attention calls/layer move the power state at all.
5. **Correctness gate:** `rel_L2 ≤ 1e-5` against `attn_gqa_forward` with fp32 KV, plus the end-to-end token hash `8EC7BAC655`. A disagreement means the PoC is wrong until proven otherwise.
6. **Explicitly out of scope:** the GDN path, the router, fp16/quantized KV, KV paging, lm_head, CUDA Graphs.

**Kill criterion, fixed before the measurement so the result cannot be rationalised afterwards:** if the single-layer PoC is not at least **1.5× faster per layer than the CPU** with resident weights (≤ 3.0 ms/layer against the measured 4.5 ms), GPU attention stays closed for this machine. The threshold exists because the project-level gain is capped by Amdahl: 10 layers × 3 ms = 30 ms of a 440 ms token = +7 %, and a margin that small does not justify a new kernel family plus a KV tier.

### 18.10 Recommendation whether to proceed

**Do not make CUDA attention the next major GPU target as a whole-subsystem rewrite. Run the §18.9 one-layer PoC as a gate — and do these two measured items first, because they are cheaper and larger per line of code:**

1. **`vram_cache_get` costs 48.5 ms/token** (11 % of the token) in pure host work: lookup + `cudaMalloc`/`cudaFree` + slab upload. That is *larger than all 10 full-attention layers combined* (44.6 ms), and it is host-side marshalling rather than a kernel problem. It is the biggest bucket left after Stage A and it needs no new kernel.
2. **The attention kernels are single-threaded scalar CPU code** (measured 4.5-4.9 ms/layer against ~1 ms of weight traffic). A threaded/SIMD CPU version is a smaller change than a new CUDA kernel family and is measured against the same oracle. Not a GPU target, but Stage B's question is "where should the next effort go", and the honest answer includes it: the GPU and the CPU are two ways to attack the same 192 ms, and the CPU one has not been tried.

Applied to the brief's decision rule: Stage A landed at **Level 1-2** (raw 2.245-2.268 tok/s, ≈2.54 equivalent against the §15 baseline), so **the GPU arm is now within 15 % of a CPU it used to trail by 36 %** — the "DP4A gets ~2.5-3.2 tok/s" case where attention becomes HIGH PRIORITY, but not yet the "GPU > CPU" case. Therefore: **attention is the right next concept, but the measurement says the first step is the full-attention *projections* + the router, device-resident, fused into one layer call — not a FlashAttention port, and not before the PoC's kill criterion has been evaluated.**

### 18.11 The CUDA Graphs question (researched, NOT implemented)

The workload does repeat near-identical operations per layer, so graphs are the obvious next thought. Measured facts that bound the payoff:

* After Stage A a layer is **5 launches** (quantize, gate+up, silu+q8, down, reduce) → 200 launches/token; the measured cost of *all* submission is **3.4 ms/token** (0.6 % of a 440 ms token). Collapsing 200 launches into 40 graph replays can save at most ~3 ms/token ≈ **+0.7 % tok/s**. That is the ceiling, and it is not worth the complexity today.
* **Compatibility with dynamic expert routing is the real problem.** The job table is passed **by value as a kernel parameter** (~1.5 KB) — which is what removed the forced mid-layer drain in §15.4. A captured graph bakes those parameter bytes in. Options: (a) one graph per (layer, expert-set) — combinatorially impossible (`C(256,8)`); (b) `cudaGraphExecKernelNodeSetParams()` before each replay — workable, but it re-introduces a host-side mutation per layer, i.e. most of what the single replay was meant to save; (c) move the job table back into device memory and update it with `cudaMemcpyAsync` — then the graph is static and the launch cost really does collapse, but the node count and dependency structure must be identical every layer, and they are not: gate+up is one launch only when `gate_type == up_type`, and `use_gu`/`use_up`/`use_dn` can differ per layer.
* Also required: capture needs a non-default stream (the backend currently uses the legacy default stream), and the event-based async fence plus the D2H download must be captured or excluded deliberately.

**Verdict:** CUDA Graphs are *partially* compatible with dynamic expert routing — option (c) makes it work — but the measured upside is ~3 ms/token, so they stay unimplemented. If a future phase adds kernels that raise the per-layer launch count (attention: a field of small kernels), revisit it: the ceiling scales with the launch count, not the token count.

### 18.12 Stage B summary in one table

| Bucket (DP4A arm, ms/token) | value | GPU attention would touch it? |
|---|---|---|
| CPU attention total | 209.9 | yes, but split below |
| — of which 30 GDN layers | 147.2 | not FlashAttention; a different kernel family |
| — of which 10 full-attention layers | 44.6 | yes, this is the FlashAttention-shaped part |
| gpu_k (device event window) | 70.1 | would *grow* by the attention work |
| wait (CPU blocked on GPU) | 79.8 | would move with gpu_k |
| vram_cache_get | 48.5 | no — and it is larger than all full attention |
| router | 6.5 | no compute, but it is a sync boundary |
| shared expert / xup / sub / rest | 30.5 | no |
| **wall** | **440.4** | |

**The decision-relevant sentence:** the 10 full-attention layers — the only part a FlashAttention-style CUDA port addresses — cost **44.6 ms of a 440 ms token (10 %)**, while the GDN layers cost **147.2 ms (33 %)** and the host-side VRAM tier costs **48.5 ms (11 %)**. A CUDA attention port is therefore not the next thing to build; it is behind the GDN and the VRAM-tier dispatch, both of which were measured to be larger.




