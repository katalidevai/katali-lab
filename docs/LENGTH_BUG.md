# Length-dependent prefill logits (Qwen3.5/3.6 MoE GGUF)

## Symptom
Chat-templated `generate --max 1`: ~9 prompt tokens → top5 has `think` / English.
~13+ tokens → digits / `?` / garbage. llama.cpp on the **same GGUF** stays coherent.

## Root cause (cliff2)
llama.cpp `conversion/qwen.py` `_reorder_v_heads` stores GDN **value heads in tiled
order** so `ggml_repeat` / fused GDN can use `kh = h % n_k_heads`:

| Layout | V-head order | Key pairing |
|--------|----------------|-------------|
| HF / EQS | `[k0v0, k0v1, k1v0, k1v1, …]` | `kh = h / rep` |
| **GGUF (this file)** | `[k0v0, k1v0, …, k0v1, k1v1, …]` | **`kh = h % n_k`** |

Lab previously used HF interleave (`h / rep`) on GGUF tensors → wrong K paired
with each V every step → recurrent state drifts → sharp cliff ~11–13 chat tokens.

## Fix
`vendor/katali_gguf/src/gguf_qwen35.c` → `katali_qwen35_delta_heads`:
default `kh = h % key_heads`. Opt-in HF layout: `KATALI_GDN_INTERLEAVE=1`.

## Ruled out (identical bad logits after rebuild)
- Decay-first vs fused GDN (already decay-first; parent null result)
- SIMD / ecache
- `SKIP_MOE` (worse)

## Related (keep)
- `ssm_a` = `-exp(A_log)` → `decay = exp(a * softplus(α+dt))`
- QSCALE `q /= sqrt(key_dim)`
- Causal conv: `w[0]`=oldest (HF); `KATALI_CONV_FLIP=1` for AB
- MoE: softmax-all → top-k → renorm

## Verify
```
deploy_cliff2.bat
```
Expect `ab_lensweep`: 14+ token prompts → `think` or English in top5, not digits.
Also run `ab_maxlayer.bat` / `ab_convflip.bat` / `ab_dump_gdn.bat`.
