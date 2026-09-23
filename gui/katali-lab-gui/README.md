# Katali Lab Chat

Professional dark-theme WPF chat GUI for `katali-lab.exe`.

## Requirements

- Windows x64
- .NET 9 SDK (`net9.0-windows`)

## Build (framework-dependent single-file)

From this folder on Windows:

```bat
dotnet publish -c Release -r win-x64 --self-contained false -p:PublishSingleFile=true -o publish
```

Or run:

```bat
build-gui.bat
```

That writes `katali-lab-gui.exe` into the lab root (`..\` when this folder is `gui\`) or into `publish\` as fallback.

## Deploy / run

1. Place `katali-lab-gui.exe` next to `katali-lab.exe` (lab root), **or** keep it under `gui\` — the GUI searches upward for `katali-lab.exe`.
2. Models live under `C:\models`:
   - Any `*.gguf` file → selectable entry
   - Any directory containing `*.gguf` (including one nested level) → entry pointing at that directory (multi-shard)
3. Double-click `katali-lab-gui.exe`, pick a model, type a prompt, Send.

## Behavior

- **Send** runs: `katali-lab.exe generate <modelPath> <prompt> --max <N>`
- Streams stdout into the assistant bubble (UTF-8)
- Parses stderr lines starting with `speed:` into the status bar (SpeedText)
- Status phases: `idle` → `loading model…` → (`prefill…` if reported) → `generating…` → `done` (or `stopping…` / `stopped`)
- **Stop** immediately disables itself, kills the process tree (`taskkill /T /F`), clears streaming state, re-enables Send
- **Clear chat** removes all bubbles (disabled while generating); model / max-tokens unchanged
- **Refresh models** rescans `C:\models`
- **Last model** remembered in `%AppData%\KataliLab\gui-settings.json`
- **Screenshot**: `katali-lab-gui.exe --screenshot C:\path\to\out.png` (RenderTargetBitmap PNG, then exit)

## Suggested layout on disk

```
C:\Users\joanr\Desktop\katali-lab\
  katali-lab.exe
  katali-lab-gui.exe
  gui\
    katali-lab-gui\     ← this source tree
      KataliLabGui.csproj
      ...
```

See `POLISH_NOTES.md` for polish #2 details.
