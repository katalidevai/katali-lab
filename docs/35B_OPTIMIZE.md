# 35B optimized profile

The Qwen3.5 35B-A3B profile uses the routed-expert gate/up fusion path automatically.

Recommended settings:

```bat
set KATALI_ECACHE_MMAP=1
set KATALI_PREFILL_BATCH=1
set KATALI_NO_ECACHE=0
```

Do not force `KATALI_MOE_FUSED_EXPERTS=0`; the runtime enables fusion for the 35B-shaped model automatically. Set `KATALI_MOE_FUSED_EXPERTS=1` only when explicitly testing the path.

Measured on the local Q4_K_M model with an 8-token generation, fusion improved prefill from 6.162s to 3.298s and decode from 2.638 to 3.476 tok/s. The default smoke run completed coherently at about 3.08 tok/s decode.

The 35B and 122B models share the Qwen3.5 MoE family, but the 122B model has a larger expert hidden dimension. The optimization is therefore selected by shape rather than assumed to transfer between sizes.
