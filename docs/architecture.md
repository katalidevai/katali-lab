# Architecture — elastic GGUF Q4 MoE

```
CLI (main.c)
  |
  v
host_*  (HostModel)
  |-- gguf_reader : mmap GGUF, KV metadata, tensor table
  |-- model       : Qwen3.6-A3B arch + tensor name map
  |-- ecache      : byte-LRU for expert Q4 blobs (elastic)
  |-- q4 / ops    : CPU kernels (Q4_0 / Q4_K later)
  v
GGUF file on disk   (Q4 quantized; no EQS)
```

## Elastic residency

- **Always mapped**: embeddings, norms, attention, routers, shared expert, lm_head (dense side of the GGUF)
- **On demand**: routed expert gate/up/down Q4 payloads via `ecache`
- Pin fraction / soft-pin hooks mirror katali2 policy knobs without EQS I/O

## Explicit non-goals (this base)

- No EQS1 / dense.eqs / experts.eqs
- No GPU backends
- No llama.cpp / ggml linkage — GGUF is a file format only
