# Expert cache — katali-lab ports

## Key space (unchanged)

**Three cache keys per routed expert:** `key = expert * 3 + which`
(`0=gate`, `1=up`, `2=down`).

GGUF keeps separate gate/up/down tensors; packing to one EQS-style blob is out of
scope. `ensure_many` / `prefetch` / `pin` take the same weight keys.

## Port #1 — ExpertCache core

- Fixed-slot cache, LFU+freq eviction, soft-pin, workers, `ensure_many`, prefetch
- GGUF mmap → private `memcpy` fill (default)
- Compat `ecache_get(src,len)` for existing callers

## Port #2 — MoE demand path (STREAM=0 choreography)

After top-k routing in `moe_ffn.c`:

1. Soft-pin the 3 weight keys for each of the K routed experts
2. `ecache_ensure_many` on that 3×K key list (parallel cold fills via FillFn)
3. MLP via `ecache_get` (should hit; sync fill only on rare miss)
4. Record `HostModel.last_eids[layer][0..K)` for the next token

In `forward.c` (per layer, before attn):

- Speculative `ecache_prefetch` of prior-token routes for this layer and next
  (3 keys × topk), overlapping SSD with attention compute

End of token:

- `ecache_unpin_all` then soft-pin all `last_eids` weight keys so LFU prefers them

**Not enabled:** READY / PIPE / STREAM / `ensure_many_ex` / knob sweeps.

## Port #3 — mmap-resident fill (`owned=0`)

**Goal:** under a hard Job Object / laptop RAM cap, stop private `malloc`+`memcpy`
of expert weight blobs (double-charge vs the GGUF mmap) while keeping the same
ExpertCache pin/LFU/`ensure_many`/prefetch policy of *which* experts are
"resident".

### Residency models

| Mode | Entry | Weight commit | When |
|------|-------|---------------|------|
| **private** (default) | `owned=1`, `data` = malloc slot | `slots × block_bytes` private | RAM plentiful / soft `--cache-gb` |
| **mmap** | `owned=0`, `data` → GGUF mmap span | no weight malloc; OS WS on fault/touch | `KATALI_ECACHE_MMAP=1` **or** private `ecache_init` returns `NOMEM` (auto-fallback) |

Evict mmap entry: clear pointer only (never `free`). Evict private: keep buffer
for reuse. `ecache_free` frees only `owned=1` buffers.

Prefetch / ensure in mmap mode: `resolve_fn` installs the mmap pointer and
lightly touches pages (~4KiB stride) to warm the working set.

### Accounting choice (documented)

`ecache_stats` **used/cap** still report **logical** resident bytes
(`n_valid × block_bytes`). Pin/LFU/ensure capacity policy is unchanged.

Under Job Object the *private* model OOMs because commit = mmap + copies.
Mmap mode does **not** malloc weight blobs; faulted WS is charged by the OS
separately from the logical counter. We do **not** track faulted WS bytes in
the cache — policy stays slot-based.

### Env

- `KATALI_ECACHE_MMAP=1` — force mmap-resident (recommended for hard 8GB Job)
- Auto: if private slot alloc fails with `KATALI_ERR_NOMEM`, host retries mmap
  and logs `falling back to mmap-resident`
- `KATALI_NO_ECACHE=1` — raw mmap pointers; no pin/ensure/prefetch
- `KATALI_EC_WORKERS=N` — prefetch reader count (default 8)
- `KATALI_SKIP_SHARED=1` — skip shared expert MLP

Stderr phase line includes `residency=mmap` or `residency=private`.

### Smoke (Windows lab)

```bat
rem soft (private OK if RAM allows; or set KATALI_ECACHE_MMAP=1)
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 128 --cache-gb 8

rem hard Job Object 8GB — require mmap mode
set KATALI_ECACHE_MMAP=1
katali-lab.exe generate ... --max 96 --cache-gb 8
```

Expect: non-empty output + Manila when max is enough; phase shows `residency=mmap`.

## Left for later

- Measure decode tok/s + eviction delta vs port #2 soft8 (~3.01 tok/s, evictions~6393)
- READY/PIPE/STREAM only if demand path is still SSD-bound after mmap-resident
- Do not knob-sweep pin%

## Port #3b — post-prefill drop + speculative pin budget

Mirror of katali2 `expert_cache_drop_unpinned` / `spec_pin_limit` / `pins_enabled`
and the `host_qwen` post-prefill + end-of-token pin loops.

### APIs

- `ecache_pins_enabled()` — `KATALI_EC_PIN=0` disables speculative set
- `ecache_spec_pin_limit(c)` — `PIN_MAX` or `PIN_FRAC` (else `c->pin_frac`, default 25%)
- `ecache_freq_decay(c, rounds)` — post-prefill LFU reset (4 rounds)
- `ecache_drop_unpinned(c)` — invalidate non-pinned residents after prefill

### Host wiring

1. After prefill token loop: `freq_decay(4)` → `unpin_all` → pin last_eids under
   `lim/nL` budget (3 weight keys per expert) → `drop_unpinned`
2. End of every forward token: `unpin_all` → same limited speculative pin (no drop)

**Still off:** READY / PIPE / STREAM.



## Port #4 — ensure_many→worker queue + freq persistence

katali2 parity for async fill accounting and multi-turn hot-set warm-start.

### ensure_many + prefetch

- When `n_workers > 0`, cold misses in `ecache_ensure_many` are **queued onto the
  same prefetch worker pool** (not only one-shot CreateThread batches).
- Each successful queue push increments `prefetch_issued`; worker completion
  increments `prefetch_done` (and `freq_bump` on success).
- Win32 workers wait on the wake **event** (`WaitForSingleObject`), matching
  katali2 — fixes the port #3 CV+1ms-event hybrid that could starve the queue.
- Mmap mode: workers use `resolve_fn` (owned=0); FillFn still required for
  private mode / start_workers.

### freq_load / freq_save / prefetch_hottest

- Binary format identical to katali2 `hot_experts.kcache` (magic `KTHF` / v1).
- Path: `KATALI_EC_FREQ` if set, else `<dirname(gguf)>/hot_experts.kcache`.
- `host_open`: `freq_load` then optional `prefetch_hottest(capacity)` unless
  `KATALI_EC_WARMUP=0`.
- `host_close`: `freq_save` before stop_workers.

### Still off

READY / PIPE / STREAM / `ensure_many_ex` / Job-Object 122B / top-k changes.


## Port #5 — layer-major batch prefill (union + ensure_many)

Mirror of katali2 `host_qwen_prefill_batch` / architecture §12.3 for GGUF lab.

### Shape

1. Embed all prompt tokens.
2. For each layer: attention + post-attn RMSNorm **position-sequential** (GQA/delta KV order).
3. Route **all** positions (same softmax→top-k as `moe_ffn_forward`).
4. **Union** unique expert ids across the prompt at that layer → pin 3 keys each →
   one `ecache_ensure_many` (port #4 worker queue fills the cold set).
5. MLP each position via `moe_ffn_mlp_routed` (`ecache_get` hits) + shared expert.
6. Unpin all but last-position routes; post-batch `freq_decay` + soft-pin + `drop_unpinned`.

Decode (`host_forward_token`) unchanged. Top-k stays 8. No READY/PIPE/STREAM.

### Env

- `KATALI_PREFILL_BATCH=0` — force per-token prefill (default ON)
- `KATALI_PREFILL_VERBOSE=1` — layer/union stderr
- Existing: `KATALI_ECACHE_MMAP`, `KATALI_EC_PIN*`, `KATALI_EC_WORKERS`, `KATALI_NO_ECACHE`

### Why union (not only look-ahead)

GGUF keys are 3× per expert; port #4 `ensure_many` already queues workers.
Unioning the prompt's experts per layer one-shots the I/O for that layer while
keeping arithmetic identical to the per-token path (`moe_ffn_route` shared).

## Port #6 — ExpertRangeFillFn + adjacent offset coalescing

Opt-in contiguous range reads for cold `ensure_many` misses (katali2
`docs/expert_range_coalescing.md` / `ExpertRangeFillFn`).

### APIs

- `ECacheRangeFillFn` — fill `n` keys from one contiguous file/mmap span
- `ecache_set_range_fill(c, fn, ud)` — host wires GGUF range fill

### ensure_many behavior

When **`KATALI_EC_RANGE_READ=1`** and `range_fill` is set:

1. Sync-claim cold misses (skip async worker queue for this batch)
2. Sort by absolute GGUF file offset (`offset_fn`)
3. Coalesce runs where `off[i+1] == off[i] + nbytes[i]` (actual span length;
   gate/up/down may differ — do **not** use `block_bytes` for adjacency)
4. `range_fill` once per run of length ≥ 2; singles use normal `fill_slot`
5. Sparse batches (no adjacent pair) stay on single-key fills (no slower
   serialized empty range loop)

Default **OFF** — unchanged worker-queue / parallel-fill path.

### GGUF host range fill

`host_ecache_range_fill`: compute `[lo, hi)` over the coalesced keys, touch the
mmap span once (~4 KiB stride), then:

| Mode | Action |
|------|--------|
| **private** | `memcpy` slices into owned slot buffers |
| **mmap** (`dests[i]=NULL`) | touch only; `ensure_many` then `resolve_fn` packs pointers |

Keys remain **3 per expert** (`expert*3+which`). Adjacent runs typically appear
within the same which-tensor (consecutive experts).

### Env

- `KATALI_EC_RANGE_READ=1` — enable coalescing (default OFF)
- `KATALI_EC_RANGE_TRACE=1` — stderr `groups/jobs/max/total/has_adj`
- Phase line: `range_read=ON|OFF`

### Still off

READY / PIPE / STREAM / `ensure_many_ex` flags / top-k changes / packed one-key.

## Port #7 — wait_idle + hot_resident

katali2 polish: drain async warmup before generate, and report how much of the
hot set is actually resident.

### APIs

- `ecache_wait_idle(c, timeout_ms)` — spin/wait until worker queue + in-flight
  `loading` slots are empty. Returns leftover busy count (0 = idle).
  `timeout_ms < 0` waits forever; `0` polls once.
- `ecache_hot_resident(c, min_count)` — count of valid resident slots whose
  freq table count is `>= min_count` (katali2 uses 2).

### Host wiring

1. `host_generate` start (after tokenize, before prefill): `wait_idle(15000)` then
   `reset_stats` so open-time `prefetch_hottest` does not race prefill I/O and
   does not pollute hit/miss/bytes for the run.
2. End-of-generate ecache line: `hot-experts resident(>=2 uses)=N/cap`.

### Cold A/B (parent, 8GB laptop)

```bat
set KATALI_ECACHE_MMAP=1
rem A cold: no hot_experts.kcache (delete it) + optional KATALI_EC_WARMUP=0
del /q C:\models\hot_experts.kcache 2>nul
set KATALI_EC_WARMUP=0
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8

rem B warm: keep kcache from A, WARMUP on (default), expect leftover=0 and
rem higher hot-experts resident after open wait_idle
set KATALI_EC_WARMUP=1
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8
```

Expect: both Manila; A shows `freq_load: none yet` / `leftover=0` quickly;
B shows `prefetch_hottest queued N`, `leftover=0`, larger `hot-experts resident`.
Report pf/tok/s both ways. Top-k=8. No READY/PIPE/STREAM.

See `docs/LAPTOP_8GB_GGUF.md` for the hard Job Object recipe.


## Thinking (Qwen)

- Default **no think** (empty `<think></think>` prefill in `format_prompt`).
- Re-enable: `KATALI_THINK=1`. See `docs/NO_THINK.md`.
