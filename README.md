# katali-lab

`katali-lab` is a specialized native inference engine for selected large and small quantized Qwen models. Each flagship model receives architecture-specific optimization, measurement, and correctness validation rather than a generic one-size-fits-all path. It adapts the proven katali2 elastic expert-cache and SSD/mmap ideas to ordinary GGUF model files.

KATALI scales across whatever hardware is present: **CPU + system RAM + SSD** by default, extended to **GPU + CPU + system RAM + SSD** when a compatible NVIDIA CUDA GPU is available. Models larger than available RAM *or* VRAM still run through elastic memory management.

**Public hardware policy:** Qwen3-1.7B and Qwen3-4B are CPU-first models intended to run on ordinary 8 GB laptops with system RAM and SSD-backed mmap. Qwen3-8B also starts CPU-first; CUDA is an optional accelerator when detected and measured faster. No model requires a GPU, fixed VRAM size, or a specific NVIDIA card. Dense Qwen3 CUDA is opt-in with `KATALI_CUDA=1`; leaving it unset keeps the CPU path.
The dense Qwen3 backend uses a **4,096-token default KV budget** for laptop-safe memory use; larger contexts remain explicitly configurable with `--ctx`. On Qwen3-4B this reduces nominal default resident memory from about 11.5 GB to about 1.16 GB before optional CUDA allocations.


CUDA is **optional and never required**. The engine is built by MinGW gcc and **does not link against any CUDA library**; the GPU backend is a separate `katali_cuda.dll` (built by nvcc + MSVC, since nvcc cannot use a MinGW host compiler) that is discovered at runtime with `LoadLibrary`. Without that DLL, without a driver, or without an NVIDIA GPU, the engine runs the unchanged CPU path and reports why.

**Measured status:** the CUDA kernels are correct (`rel_L2 ≈ 2e-7` against the engine's own CPU kernels on real GGUF tensors — see `cuda-check` and `cuda-check-moe`) and the kernels run **6.9–10.9× faster than the CPU** on DRAM-bound expert traffic (`katali-lab.exe cuda-bench`). Layer-level MoE fusion is implemented: one GPU call per MoE layer instead of one per expert, cutting CUDA API calls/token **8×**, kernel launches **6×** and device syncs **24×**, all with byte-identical output.

The current winning CUDA stack is model-specific and experimental: Qwen3.6-35B-A3B reached about **6.0–6.1 tok/s at 16 tokens** and **5.9–6.0 tok/s at 32 tokens**, versus about 3.7 tok/s CPU, with byte-identical greedy output. Keep it opt-in (`KATALI_CUDA_MOE=1`); stable defaults remain CPU-first. See the latest archived reports under `_gdn_bench`, `_moe_device_bench`, and `_moe_overlap_bench`.

**Stage A (this phase): DP4A + q8_1 activations.** The inner GEMV loop no longer dequantizes every weight element to float. The layer's activation is quantized **once** to q8_1 on the device and every selected expert's quants are dotted against it with `__dp4a` (the technique llama.cpp uses). Measured: the fused layer at the engine's operating point (`n_sel=8`) went **10.0 → 3.20 ms per 24 calls (3.1×)**, 36.3 → **114 GB/s** against a 134 GB/s no-arithmetic ceiling, with the q8_1 quantization *counted* (0.27 ms, 8 %). End-to-end the GPU arm rose **1.720 → 2.245 tok/s (+31 %)** in a matched A/B (and 1.705 → 2.268 in a second run), with the GPU MoE window per token falling **209.7 → 70.1 ms** and the CPU's GPU wait **221.7 → 79.8 ms**. Output was **byte-identical** to CPU and to the fp32 CUDA path on the reference prompt (`8EC7BAC655`), and a new synthetic test (`cuda-dp4a-selftest`) proves the kernels match the fp32 kernels to **rel_L2 8.3e-07** when the activation is exactly representable; on real weights the residual 7.2e-03 is q8_1 activation rounding, with 0 elements outside `1e-3·(1+|ref|)`. Enabled with `KATALI_CUDA_DP4A=1`; **default OFF**, so the default path is unchanged. See `docs/CUDA_PLAN.md` §17.

**Stage B (this phase): attention was re-examined and is *not* the next target.** Measurement split the 210 ms CPU attention bucket: **147 ms is the 30 DeltaNet/GDN layers**, **44.6 ms is the 10 full-attention layers** — the only part a FlashAttention-style port would address — and the host-side VRAM tier costs **48.5 ms**, more than all full attention combined. The KV cache lives in system RAM (`40 KB per token of context`), and the dense attention weights are 1.05 GB, so GPU attention needs a new residency tier before it saves anything. Recommendation, kill criterion and the smallest PoC are in `docs/CUDA_PLAN.md` §18.

**The measured root cause is not the kernel.** A cold-memory benchmark (`cuda-bench-cold`, rotating experts over a working set far larger than L2) shows the *same* fused kernel reaching **31.9 GB/s** — 240 GB/s is available per the raw stream test — while inside the real engine every layer runs at **2.8 GB/s** with zero cache misses. The reason: during inference the GPU is 50 % *busy* but sits at its **idle power state — 210 MHz core / 405 MHz memory versus 2505 / 8501 MHz at boost — a 21× lower memory clock**, because the workload is a ~6 % duty cycle (≈0.5 ms GPU per ≈8 ms CPU). Confirmed by holding the GPU busy with a second process: decode rose 1.837 → **2.537 tok/s** and the per-layer GPU window halved, 3.776 → **1.922 ms**, *while that process consumed 94 % of the machine*.

Kernel work was also tried and **rejected on measurement**: a 4-way unrolled column loop gained 11–15 % at n_sel≤4 but **lost 22 % at n_sel=8**, the engine's operating point, so `KATALI_GEMV_UNROLL=1` remains the default. The next phase should raise the GPU duty cycle (async overlap of GPU MoE with CPU attention), not micro-tune GEMV. Default behaviour is byte-for-byte the original CPU path (re-verified: 3.05 tok/s, `Manila`, 0 CUDA calls).

## Qwen3.8-27B support

`Qwen3.8-27B-Q4_K_M.gguf` is now a supported model. It is a **fourth**
architecture variant, not a Qwen3.6-35B sibling: `general.architecture` is
`qwen35`, the trunk is 64 blocks (**48 GDN/DeltaNet + 16 full attention**) with one
MTP block appended, and the feed-forward is **dense** - there are no
`ffn_*_exps` tensors and no `expert_count` metadata key at all.

The engine previously treated it as a 256-expert MoE, because it started from the
`qwen35moe` defaults and only cleared them when a model had neither expert *nor*
ssm tensors, and `katali_gguf_get_int()` returns the caller's default for a key
that is missing. The result was an empty expert index and
`host_generate: layer-major prefill failed` before the first token. Models are now
classified from **tensor evidence**, so this dense hybrid layout is detected and
served by the dense FFN path.

Measured on the development machine (RTX 4060 present but unused; CPU only):

| Metric | Value |
|---|---|
| Load / open | 0.57 s (mmap, lazy) |
| Prefill, 40-token prompt | 0.735-0.790 tok/s |
| Prefill, 75-token prompt (batched) | 1.14 tok/s |
| Decode | 0.81-0.85 tok/s |
| Weight traffic per decoded token | **15.3 GiB** (a dense model re-reads every layer) |
| Runtime context | **2048 by default, configurable with `--ctx N`** (validated up to 65536; memory-budgeted and refused when unsafe). The metadata advertises 262144 - that is a training ceiling, not something this runtime serves |
| Greedy answer correctness | `Manila`, byte-identical across runs |

Because the FFN is dense, one decoded token must stream ~15.3 GiB of weights, so
CPU matvec throughput - not routing and not the GPU - is the binding constraint.
The batched prefill introduced for this layout streams those weights once for the
whole prompt instead of once per token: **1.41x** on a 75-token prompt
(95.7 s -> 67.9 s) with byte-identical output.

Qwen3.8-27B is **not** recommended for interactive use on a CPU-only machine: a
decoded token costs ~1.2 s. Use it with a bounded `--max`.

## Context configuration and the CUDA rebuild

**The runtime context is now explicit.** `generate` and `open` accept `--ctx N`, and
the HTTP API accepts `"ctx": N`. The default is 2048 (unchanged behaviour) and the
minimum is 256. Every request is validated and budgeted **before** anything is
allocated, and the decision is printed as a `context:` block:

```text
context:
  requested: 0 (default)
  effective: 2048
  ctx_train_metadata: 262144   [metadata ceiling - NOT served by this runtime]
  kv_bytes: 256.00 MiB
  kv_location: system RAM
  estimated_ram: 464.06 MiB (KV + state + scratch; not reclaimable)
  estimated_vram: 0.00 MiB
  status: ok
```

Two bounds are enforced: the hard (non-reclaimable) allocation must fit in
available RAM minus reserves, and it plus the resident weights must fit in total
RAM minus reserves or the engine would thrash. Reserves are never consumed
(10 % of RAM, minimum 1 GiB, plus 768 MiB for the front-ends).

Measured on Qwen3.8-27B: `--ctx` 2048/4096/8192/16384/32768/65536 are **accepted**
(KV 256 MiB to 8 GiB), 128 is **refused** (below minimum), 262144 is **clamped to
94044** with the numbers that forced it, and 400000 is **refused** (beyond the
trained context). 1M context is **not implemented and not claimed** - it would need
YaRN/rope support that does not exist here.

**CUDA can be rebuilt.** `build_cuda.bat` is present (it had been archived under
`_gpu_review/`; the root copy was restored and its LF-only line endings fixed).
The pruned toolkit lives at `C:\Users\joanr\cuda\home` - `nvcc` 13.4.92,
`ptxas`, `cudafe++`, the headers, and the NVVM library as
`home\nvvm\bin\x64\nvvm64_40_0.dll` (**not** `libnvvm.dll`, which is why an
earlier search concluded it was missing). MSVC `cl.exe` 14.44 is the host compiler.

The rebuilt `katali_cuda.dll` passes every gate: `cuda-dp4a-selftest` at
`rel_L2=8.342e-07` (identical to the recorded prebuilt value), `cuda-check` on the
27B across F32/Q4_0/Q8_0/Q4_K/Q6_K on real tensors, and `cuda-check-moe` on the
35B at `rel_L2=2.292e-07`. See [docs/PHASE2_REPORT.md](docs/PHASE2_REPORT.md).

## Persistent attention/GDN residency tier (opt-in)

A dense decoder re-reads every weight on every token, so the attention/GDN
projections (32.1 % of the per-token byte budget on the 27B) are a candidate for
persistent VRAM residency. `src/dense_tier.c` is that tier: a **residency table**
over the one existing device allocator (`katali_cuda_malloc`), not a second
allocator and not an LRU cache.

```text
KATALI_DENSE_GPU=1                       request the tier (default: off, no effect)
KATALI_DENSE_TIER=attn,gdn,lm            which groups to admit (also: all, ffn*)
KATALI_DENSE_GPU=auto                    planner picks the largest safe wired set
KATALI_DENSE_TIER_GB=N                   cap the device budget explicitly
```

Admission is **whole-layer**: a layer whose attention set is only partly resident
would pay a device round trip *and* the CPU stream for the same activation, so a
layer is costed before a byte is allocated and declined entirely if it does not
fit. Roles with no device kernel behind them are never admitted - `DTR_ALPHA`/
`DTR_BETA` (48-float outputs: one GPU round trip costs more than the CPU matvec
it replaces), the norm weights (consumed by in-place host RMSNorm, not by a
matvec), and the dense FFN (out of scope) - so the reported size is residency that
is actually *used* rather than "fits on paper".

The tier requires model evidence, not a model name: a **dense** model (0 experts)
that has **GDN/linear-attention** layers. Qwen3.8-27B qualifies; Qwen3-1.7B/4B/8B
(pure full-attention dense) do not, and the MoE models use the existing expert
VRAM tier instead. With `KATALI_DENSE_GPU` unset nothing is opened for any model.

Two correctness invariants are enforced structurally rather than by convention:

* **Activation identity is `(generation, layer, act)`.** `dense_tier_token_begin()`
  is called at every token and every prefilled position, and the activation buffer
  is only reused inside one generation *and* for the same named activation
  (`DTR_ACT_LAYER_IN` / `DTR_ACT_ATTN_OUT` / `DTR_ACT_SSM_OUT` / `DTR_ACT_LM_IN`).
  `dense-tier-test` proves both halves: the same key with a new generation must
  change the result, and the same key *without* a new generation must return the
  previous answer (the hazard is demonstrated, not merely avoided).
* **Every device step has an unconditional CPU fallback.** `dense_tier_proj()` is
  the only place a dense projection chooses its route.

`katali-lab.exe dense-tier-test [model.gguf]` runs the per-role CPU-vs-GPU
`rel_L2` report, the generation-guard proof, and a multi-token sequence with the
tier ON versus BYPASSED in one process.

## GPU matvec microbenchmark (`gpu-matvec-bench`)

```bat
katali-lab.exe gpu-matvec-bench 5
```

A standalone benchmark of the device matvec path - synthetic slabs, no model file,
no tokenizer, no mmap - so a per-call claim is reproducible in ~90 seconds. Per
shape it reports wall ms/call, event-measured device kernel ms/call, effective
GB/s and launches / syncs / H2D / D2H per call, and it measures the device's raw
read/copy bandwidth in the same process for a same-conditions reference. The two
shapes that matter run under four conditions: back-to-back, whole round trip,
100 ms spacing (the live decode's call spacing) and 11 CPU threads streaming RAM
(the live decode's memory-system load).

Its first result overturned the Phase 2 explanation. The 17-34 MiB attention/GDN
projections cost **0.49-1.02 ms** standalone (35-40 GB/s) and stay there for
30 000 calls, while the live engine pays **4.8-6.7 ms per crossing** for the same
work. Neither idle spacing, nor CPU memory contention, nor 24 s of sustained load,
nor reducing the engine to 2 CPU threads reproduces the gap - so the cost is per
*crossing*, not per byte, and the next work is batching crossings
(208/token -> 64) rather than kernel tuning. See
[docs/PHASE3_REPORT.md](docs/PHASE3_REPORT.md).

At model open the engine prints the plan it selected, on **stderr**, so stdout
stays reserved for generated text and the HTTP API / GUI keep working unchanged:

```text
hardware-plan:
  cpu: Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz, 6 cores / 12 threads
  simd: sse4.2+avx2+fma+f16c (binary built for avx2+fma+f16c)
  threads: 12
  ram_total: 32629.93 MiB   ram_avail: 24228.00 MiB   ram_budget: 12174.26 MiB
  cuda: NVIDIA GeForce RTX 4060 (Ada Lovelace, sm_89, 24 SMs); vram total 8187.50 MiB free 7107.00 MiB budget 5330.25 MiB
  cuda_enabled: yes   vram_tier: off
  storage: >=4771 MiB/s sequential (OS cache warm; lower bound)
  model: qwen35  class=hybrid GDN + dense FFN  layers=65 (trunk 64)
  model_file: 16634.37 MiB   weight_bytes_per_token: 15703.85 MiB
  model_mode: cpu
  gpu_layers: 0   (no GPU layer tier for this model class)
  kv_cache_location: system RAM
  ctx_runtime: 2048 (requested 0, status ok, KV 0.25 GiB)
  ctx_train_metadata: 262144   [training ceiling; the runtime serves 2048]
  mmap: on (whole file, read-only), ECache copies into private slots
  ecache: on   workers=0   pin=25%
  prefetch: ssd=off route=off
  reason: dense model: a decoded token re-reads every layer's weights (15703 MiB per token), so the win is CPU matvec throughput and a batched prefill, not per-call GPU work. CUDA stays available but unused.
```

`model_mode` is `cpu`, `hybrid` or `gpu`; `weight_bytes_per_token` is the decode
memory budget (1092 MiB for the 35B MoE against 15704 MiB for the dense 27B, which
is why the MoE decodes ~3x faster per token despite being a larger file); and
`kv_cache_at_ctx` states the KV memory a 2K/8K/32K/64K context would need so a
context claim can be checked instead of believed.

Full hardware modes, flag reference, benchmark tables, correctness results,
reverted experiments and known limitations:
**[docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md)**.

## Specialized flagship models
Katali-lab focuses its deepest optimization work on a small number of flagship models. These receive dedicated CPU/RAM/SSD and CUDA execution profiles, model-specific scheduling, and measured regression gates.

| Model | Specialization status | Verified result |
|---|---|---|
| **Qwen3.6-35B-A3B** | Primary large flagship; hybrid GDN/MoE CPU + CUDA profile | CUDA GDN stack approximately 6.0–6.1 tok/s short decode and 5.9–6.0 tok/s at 32 tokens |
| **Qwen3-1.7B** | Primary small flagship; dense CPU + CUDA profile | **20.65 tok/s at 16 tokens; 23.59 tok/s at 32 tokens** on RTX 4060, versus 3.26 tok/s CPU |

“Specialized” means the model has been profiled and optimized as its own architecture. Optimizations are not assumed to transfer automatically to other model families.

## Supported compatibility models

| Model | Status | CPU profile | Official model | GGUF source |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | Supported | Automatic routed-expert fusion; mmap/cache profile | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | [bartowski GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF) |
| Qwen3.5-122B-A10B | Supported | Raw mmap; expert cache disabled by default; fusion disabled for sustained CPU decode | [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | [bartowski GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-122B-A10B-GGUF) |
| Qwen3-Coder-Next 80B-A3B | Supported | Hybrid DeltaNet/attention; 512-expert elastic cache; CPU-first profile | [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | [Qwen GGUF](https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF) |
| Qwen3-Coder-30B-A3B | Supported | Standard full-attention MoE; 128-expert elastic cache | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | [GGUF source](https://huggingface.co/Zoed/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen3-4B | Supported | Dense Qwen3 CUDA/CPU backend; model-specific profiling in progress | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [Official GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |
| **Qwen3.8-27B** | Supported | Hybrid GDN + **dense** FFN; tensor-evidence architecture detection; batched prefill; CPU-first | Qwen3.8-27B | `C:\models\qwen38-27b\Qwen3.8-27B-Q4_K_M.gguf` |

Qwen3-4B is a supported dense-model compatibility target. The current RTX 4060 smoke test reached **3.98 tok/s at 16 tokens** and **6.15 tok/s at 32 tokens** with CUDA, versus approximately **1.21** and **1.28 tok/s** on CPU; the 32-token CPU/CUDA output was byte-identical. It remains below the two specialized flagship profiles.

The official model cards are the source of truth for model configuration and licensing. GGUF repositories are community conversions; verify quantization, shard completeness, and tokenizer files before use.

## Additional compatibility validation

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

The unified executable was revalidated after restoring the combined `ssm_in`/`ssm_ba` tensor mapping used by Coder-Next: a current CPU smoke test reached **2.31 tok/s decode**, while CUDA reached **1.86 tok/s decode** with 480 GPU experts and zero fallbacks. Both modes generated successfully.

The current CPU release also includes a cache-friendly DeltaNet state-loop optimization. Two matched 64-token Coder-Next runs measured **1.758 and 1.696 tok/s decode**, compared with the previous **1.601 tok/s** baseline. The release also fixes standard Qwen3-Coder-30B full-attention handling: Qwen3 uses a plain Q projection, while Qwen3.5 uses fused Q+gate. Both 30B and 35B now pass the Manila quality smoke test, and `q4_selftest` passes.

For the Coder-Next CUDA path, async MoE submission is the selected profile: a matched two-token A/B measured **2.016 tok/s** async versus **1.451 tok/s** synchronous. Increasing the VRAM budget to 7.5 GiB reached **1.857 tok/s**, and enabling DP4A reached **1.520 tok/s**, so neither is enabled as a default optimization. The current bottleneck is expert-cache upload/synchronization rather than the GEMV kernel.

Qwen3-Coder-30B-A3B is integrated and validated as standard Qwen3 MoE. Its Q4_K_M GGUF is approximately 17.28 GiB. The smoke test measured **3.62 tok/s CPU decode** and **6.29 tok/s CUDA decode** on the RTX 4060, with 384 GPU experts and zero GPU fallbacks. On the corrected CPU path, matched 16-token runs reached **4.64 and 5.06 tok/s decode** with `--cache-gb 8 --pin 50`; this is the recommended 30B CPU profile. The same cache at `--pin 25` measured 4.39 tok/s, while 12 GiB/50% measured 4.13 tok/s.

### Qwen3-Coder-30B CPU/GPU side-by-side

Same model, prompt, `--max 8`, cache settings, and RTX 4060 test machine:

| Mode | Answer | Prefill | Decode | Expert execution |
|---|---:|---:|---:|---|
| CUDA GPU | `4` | 1.30 tok/s | 1.43 tok/s | 320 GPU experts, 0 fallbacks |
| CPU + system RAM + SSD | `4` | 2.73 tok/s | 3.07 tok/s | CUDA disabled |

This short test favors CPU because GPU startup and transfer overhead dominate. Longer prompts or sustained generation can change the result.

Captured sustained test, same prompt and `--max 64`: CUDA produced **2.109 tok/s decode** after **3.441 tok/s prefill**, while CPU + system RAM + SSD produced **3.540 tok/s decode** after **3.041 tok/s prefill**. Both returned a coherent answer. For this 30B model and RTX 4060, CPU remains the faster default for this workload; CUDA remains available for experimentation and larger workloads.

## Future model roadmap

1. **Qwen3.6-35B-A3B** — current flagship and primary optimization target.
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

The application remains one executable. Active expert weights can move to VRAM while cold experts remain in the system-RAM cache or on SSD. Without CUDA—or if CUDA initialization fails—the engine automatically falls back to the CPU path. GPU kernels currently cover the quantized GEMV/matmul family, routed MoE experts, and the experimental Qwen3.5 Gated DeltaNet/GDN path. Full GQA remains CPU-side because the measured CUDA port was slower. These results are not automatically transferable to Qwen3-Coder-Next, Qwen3.5-122B, dense models, or other architectures.

## Latest CUDA experiment status (2026-09-23)

The latest verified winner is the Qwen3.6-35B-A3B Q4_K_M path on the development RTX 4060:

```text
KATALI_VRAM_POOL=1
KATALI_UPLOAD_BATCH=1
KATALI_CUDA_MOE=1
KATALI_CUDA_DP4A=1
KATALI_CUDA_GDN=1
```

Measured on the Manila paragraph benchmark: approximately **6.0–6.1 tok/s at max16** and **5.9–6.0 tok/s at max32**, compared with approximately **3.7 tok/s CPU**. The generated text passed byte-identical greedy oracle checks. The improvement is primarily from moving the 30 GDN/DeltaNet layers to CUDA and using DP4A fused MoE kernels.

These measurements are **specific to Qwen3.6-35B-A3B** and are not a claim about every model. The 122B, Coder-Next, 397B, dense models, and different MoE/GDN layouts require separate validation.

The following experiments were tested for correctness but reverted because they did not beat the winning stack: full GQA CUDA, standalone GPU RMSNorm/residual/router, activation residency, device-side MoE boundary, and MoE stream overlap. Their reports remain archived locally. They are not enabled by default and are not part of the stable release.
## Dense Qwen3 backend

The official repository includes a dense Qwen2/Qwen3 backend migrated from Katali-GGUF. Build it with:

```bat
build-qwen3.bat
```

This produces `katali-lab-qwen3.exe`, which supports dense Qwen3 models such as Qwen3-1.7B. Example:

```bat
katali-lab-qwen3.exe inspect C:\models\qwen3-1.7b-gguf\qwen3-1.7b-q4_k_m.gguf
katali-lab-qwen3.exe run C:\models\qwen3-1.7b-gguf\qwen3-1.7b-q4_k_m.gguf --prompt "What is the capital of the Philippines? Answer one word." --max-tokens 8 --no-think
```

The dense backend is intentionally separate from the specialized Qwen3.6-35B MoE executable while the common backend dispatcher is being consolidated.
Current Qwen3-1.7B CUDA result: the architecture-gated dense CUDA matvec path reaches **20.65 tok/s at 16 tokens** and **23.59 tok/s at 32 tokens** on the RTX 4060, versus approximately **3.26 tok/s CPU**. Five CPU/CUDA greedy oracle cases are byte-identical. The current optimization caches quantized weights on the GPU and batches QKV/gate-up projections; full device-resident RMSNorm/attention/activation execution remains the next engineering phase.
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

The request accepts `model` (**required** - the server no longer substitutes one
hardcoded path, because that silently ran the wrong model), `prompt` (or
`content`), `max_tokens` (or `max`) and `stream`.

Non-streaming replies are a single OpenAI-shaped JSON object with the text in
`choices[0].message.content`; the text is JSON-escaped, so quotes, backslashes and
newlines in the model output cannot corrupt the reply.

Set `"stream": true` to receive `text/event-stream` frames as the model writes
them (the generator flushes after each token):

```bat
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"C:\\models\\qwen38-27b\\Qwen3.8-27B-Q4_K_M.gguf\",\"prompt\":\"hi\",\"max_tokens\":8,\"stream\":true}"
:: data: {"content":"The"}
:: data: {"content":" capital"}
:: data: [DONE]
```

`GET /health` reports `{"status":"ok","cpu_only":<bool>,"cuda":"<probe result>","cuda_device_usable":<bool>}`.
The API runs generation in a child `katali-lab.exe generate` process, so it uses
exactly the CLI's planner and prefill behaviour; the child's stderr is kept out of
the reply.

## Build verification

The current published executables were tested locally after the final rebuild. `katali-lab.exe selftest` passed, 35B generation completed coherently at 2.25 tok/s for one decode token, and the 122B bounded smoke test completed at 1.04 tok/s for one decode token. These are CPU measurements on the development desktop, not guaranteed performance on other machines.

## Next architecture target: Qwen3.8-Flash-Next

The current flagship is **Qwen3.6-35B-A3B**. After stabilizing and optimizing it, the next planned model family is **Qwen3.8-Flash-Next**. It is not a drop-in Qwen3.5 model: current GGUF metadata identifies it as `qwen4exp`, with a newer sparse-expert design and a large n-gram embedding table. It therefore needs a separate architecture backend while reusing Katali-lab's general elastic storage ideas.

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

- [Benchmark report: hardware, modes, flags, measurements, correctness, limitations](docs/BENCHMARK_REPORT.md)
- [Phase 1 audit: repository, hardware and Qwen3.8-27B findings](docs/PHASE1_AUDIT.md)
- [katali2 reference notes](docs/FUTURE.md)
- [35B CPU optimization profile](docs/35B_OPTIMIZE.md)
- [122B CPU optimization profile](docs/122B_OPTIMIZE.md)
- [GGUF parity checklist](docs/KATALI2_GGUF_PARITY.md)
- [Qwen3.5 collection](https://huggingface.co/collections/Qwen/qwen35)



