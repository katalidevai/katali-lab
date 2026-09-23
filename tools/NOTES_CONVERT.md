# HF safetensors → GGUF Q4

Source checkpoint (downloaded):

`C:\models\Qwen3.6-35B-A3B`

Runtime for this project requires a **single (or sharded) GGUF** with Q4 quant (`Q4_0` first; `Q4_K_M` later).

Suggested destination under the workspace:

`C:\Users\joanr\Desktop\katali-lab\models\Qwen3.6-35B-A3B-Q4_0.gguf`

Do **not** pack to EQS. Conversion tooling can be llama.cpp `convert_hf_to_gguf.py` + quantize, or a future Katali-native converter. Lock tensor names against the real GGUF before wiring `model.c` name maps.
