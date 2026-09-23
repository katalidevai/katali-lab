# No-think default (Qwen3 / 3.5 / 3.6 MoE)

## Root cause

Qwen3-family chat models **default to thinking mode**. With a plain ChatML wrap
(`<|im_start|>user…<|im_end|>\n<|im_start|>assistant\n`), the model often opens
with a `<think>…</think>` block before the visible answer.

Hugging Face's hard switch (`enable_thinking=False`) is implemented by **prefilling
an empty think block** after the assistant header:

```
<|im_start|>assistant
<think>

</think>

```

katali-lab previously only did that when `KATALI_PREFILL_THINK=1` (debug). Default
was thinking-on.

## Fix (engine)

In `src/host.c` `format_prompt`:

- **Default:** if `think_open` / `think_close` token IDs exist, prefill empty
  `<think>\n\n</think>\n\n` (Qwen hard-disable).
- **Re-enable thinking:** `KATALI_THINK=1` or `KATALI_ENABLE_THINK=1`.
- **Debug:** `KATALI_PREFILL_THINK_OPEN=1` still forces open-only.
- **Legacy:** `KATALI_PREFILL_THINK=1` forces empty prefill even when think is on.

Decode also **suppresses** residual `<think>…</think>` tokens on stdout unless
thinking is enabled (band-aid for older prompts / partial leaks).

## Fix (GUI)

- Strips residual `<think>…</think>` from streamed stdout (honors `KATALI_THINK=1`).
- Compact left-aligned log chat (no bubbles): **You** / **Model** labels.
- Status strip under the input shows live elapsed:
  - `Loading… 3.1s` until first token
  - `Generating… 12.3s` while decoding
  - `Last reply: 48.2s · 3.1 tok/s` when done (tok/s from CLI `speed:` line when present)

## Knobs

| Env | Effect |
|-----|--------|
| *(default)* | No think (empty think prefill + strip) |
| `KATALI_THINK=1` | Allow thinking; show think tokens |
| `KATALI_ENABLE_THINK=1` | Same as `KATALI_THINK=1` |
| `KATALI_PREFILL_THINK=1` | Force empty prefill even if think on |
| `KATALI_PREFILL_THINK_OPEN=1` | Debug: open think only |
| `KATALI_RAW_PROMPT=1` | Skip ChatML wrap entirely |

## Smoke (on Joan's PC after deploy)

```bat
cd /d C:\Users\joanr\Desktop\katali-lab
build.bat
set KATALI_ECACHE_MMAP=1
katali-lab.exe generate C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf ^
  "What is the capital of the Philippines?" --max 64 --cache-gb 8
```

Expect answer containing **Manila** and **no** `<think>` tags.

Re-enable think A/B:

```bat
set KATALI_THINK=1
katali-lab.exe generate ...
```
