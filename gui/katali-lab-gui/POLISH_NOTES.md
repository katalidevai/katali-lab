# Katali Lab Chat GUI — Polish #2

## Settings path

Last selected model is persisted to:

```
%AppData%\KataliLab\gui-settings.json
```

(`Environment.SpecialFolder.ApplicationData` → typically `C:\Users\<you>\AppData\Roaming\KataliLab\gui-settings.json`)

JSON shape:

```json
{
  "lastModelPath": "C:\\models\\SomeModel.gguf"
}
```

Saved on model dropdown change, on Send, and on window close. Restored on startup if that path still appears in the scanned model list.

## Files changed / added

| File | Change |
|------|--------|
| `GuiSettings.cs` | **New** — load/save `gui-settings.json` |
| `MainWindow.xaml.cs` | Remember last model; status phases; snappier Stop; `--screenshot` hook |
| `MainWindow.xaml` | `SelectionChanged` on model combo; Stop/Send `MinHeight=56` + `VerticalAlignment=Stretch` |
| `README.md` | Document settings path, status phases, screenshot flag |
| `POLISH_NOTES.md` | This file |

Unchanged: `App.xaml`, `App.xaml.cs`, `KataliLabGui.csproj`, `Models/ModelEntry.cs`, `build-gui.bat`, `.gitignore`.

## Behavior polish

1. **Remember last model** — restore from settings if still in list after refresh.
2. **Status clarity** — phases: `idle` / `loading model…` / `prefill…` / `generating…` / `done` / `stopping…` / `stopped`. Parses stderr `speed: …` into SpeedText. Approximates loading→generating if no explicit phase lines.
3. **Snappier Stop** — immediately disables Stop, sets `stopping…`, `taskkill /T /F`, appends `[stopped]`, clears streaming chrome, re-enables Send.
4. **Screenshot** — `katali-lab-gui.exe --screenshot path.png` renders via `RenderTargetBitmap` then exits.

## Build (on Windows)

```bat
cd gui\katali-lab-gui
build-gui.bat
```

Or:

```bat
dotnet publish -c Release -r win-x64 --self-contained false -p:PublishSingleFile=true -o publish
```
