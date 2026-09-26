# Katali Lab Chat

Dark-theme Windows chat app for `katali-lab` — local GGUF inference with multi-turn ChatML.

Model weights are **not** included. Download GGUFs yourself (Hugging Face or another source), then point the app at them.

## Requirements

- Windows x64
- [.NET 9](https://dotnet.microsoft.com/download/dotnet/9.0) runtime (framework-dependent build)
- `katali-lab.exe` next to this GUI (or discoverable in a parent folder)
- At least one GGUF model (file or multi-shard folder)

## Recommended model

**Qwen3.6-35B-A3B** (Q4_K_M) is the recommended laptop chat model. Example filename after download:

```
Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
```

You can keep models anywhere. The app also scans `C:\models` when that folder exists. On first launch — or when the last-used model is missing — it **prefers a 35B model if one is already present**. It never requires a preinstalled path.

Product generation defaults (independent of whether a 35B file is on disk):

| Setting | Default | Notes |
|--------|---------|--------|
| Max tokens | 512 | Comfortable chat length at ~2 tok/s |
| Expert cache | 8 GiB | Measured 35B laptop profile |
| Pinned cache | 25% | Same profile |
| Thinking | Off | Faster replies |
| Memory-map cache | On | Recommended |
| GPU MoE | Off | CPU is typically faster for short 35B runs |

Selecting a 35B model only fills **empty** cache/pin fields — saved settings are never overwritten.

## Get a model

1. Download a Qwen3.6-35B-A3B Q4_K_M GGUF (or another supported model).
2. In the app, click **Browse…** (single file) or **Folder…** (multi-shard).
3. Or place `.gguf` files under `C:\models` and click **Refresh models**.

## Build

```bat
dotnet publish -c Release -r win-x64 --self-contained false -p:PublishSingleFile=true -o publish
```

Or run `build-gui.bat` from this folder. Output is copied next to `katali-lab.exe` (lab root).

## Run

1. Place `katali-lab-gui.exe` beside `katali-lab.exe`.
2. Add a model (see above).
3. Open the GUI, pick the model, type a message, **Send** (Enter). Shift+Enter for a newline.

## Features

- **Multi-turn chat** — history embedded as Qwen ChatML each turn
- **Sessions** — sidebar chats under `%AppData%\KataliLab\chats\`
- **Context meter** — approximate prompt tokens (`chars ÷ 4`) vs an 8k soft budget; warns above ~6k
- **Markdown** — assistant replies render bold, italic, code, lists, and links after generation (plain text while streaming)
- **Message actions** — right-click **Copy** / **Retry** / **Edit last user**
- **Settings** — generation, thinking, system prompt, cache, GPU MoE

## Settings file

`%AppData%\KataliLab\gui-settings.json`

## Tips for 35B (~2 tok/s)

- Keep chats short; start a **New chat** when the context meter turns red.
- Leave **thinking** and **GPU MoE** off unless you need them.
- Prefer **Stop** over closing the window during a run.

## Layout

```
katali-lab\
  katali-lab.exe
  katali-lab-gui.exe
  gui\katali-lab-gui\   ← this source tree
```
