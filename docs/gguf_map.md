# Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf map

Path: `C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` (~20.75 GiB)

- `general.architecture` = `qwen35moe`
- blocks = 41 (40 transformer + MTP-style tail)
- experts = 256, used = 8, shared expert present
- full_attention_interval = 4 (full attn at blk 3,7,...,39 and 40)
- other blocks = Gated DeltaNet / SSM (`ssm_*`, `attn_qkv`, `attn_gate`)
- Expert packs: `ffn_gate_exps`/`ffn_up_exps` = Q4_K, `ffn_down_exps` = Q6_K
- Shared expert: Q8_0; router `ffn_gate_inp` = F32

Q4_K_M is a **mixed** quant recipe, not pure Q4_0.
