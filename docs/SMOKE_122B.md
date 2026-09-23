# 122B golden smoke

**Model:** `C:\models\Qwen_Qwen3.5-122B-A10B-Q4_K_M\Qwen_Qwen3.5-122B-A10B-Q4_K_M`  
**Command shape:**
```bat
katali-lab.exe generate <model-dir> "What is the capital of the Philippines? After thinking, reply with one short sentence naming the city." --max 256 --pin 25
```

## Expected shape (quality)
- Opens with `<think>` (or equivalent thinking).
- Reasoning states the capital is **Manila** (not garbage / `!!!!`).
- Ideally closes thinking and answers with a short sentence naming Manila.
- Stderr ends with `speed: decode_tokens=… tok/s=…` (and after speed knobs: `phase: ecache …`, `phase: prefill …`).

## Recorded runs (2026-09-22)

### Run A — `--max 128` (first e2e)
- Prompt: `What is the capital of the Philippines? Answer briefly after thinking.`
- Result: coherent thinking; correctly identifies **Manila**; hit token cap still inside think (no final spoken answer yet).
- Speed: **~0.300 tok/s** (128 decode / 426.6 s). Wall ~525.5 s including load.
- Exit 0. Artifacts: `gen122_ph_out.txt`, `gen122_ph_err.txt`.

### Run B — `--max 256` (follow-up for closed answer)
- See `gen122_ph2_out.txt` / `gen122_ph2_err.txt` when finished.

## Regression check
Fail if: NaN/`!!!!`, empty think forever with nonsense, or stdout is only punctuation.
Pass if: Manila appears in think and/or final answer with real tokens.

(See also 35B A/B in STATUS.md.)


### Run B — `--max 256` (cancelled)
- Started 2026-09-22; cancelled by user (~12+ min) — too slow to wait for closed answer.
- Partial output already had Manila drafts in thinking (`gen122_ph2_out.txt`).
- Conclusion: 122B quality OK; CPU decode ~0.3 tok/s is the limiter, not a correctness hole to chase further right now.
