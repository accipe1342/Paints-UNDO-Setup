# Paints-UNDO (Easy Install Fork)

This is a fork of [lllyasviel/Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO) with a setup script that handles installation, dependency fixes, and GPU detection automatically.

## What's different from the original

- `setup.py` — one-command install that auto-detects your GPU and installs the correct PyTorch CUDA version
- Fixes a `gradio_client` bug that causes UI buttons to be greyed out (`APIInfoParseError: Cannot parse schema True`)
- Fixes a localhost binding issue that prevents the Gradio UI from loading
- Fixes Step 3 video generation on RTX 5000 series (and other GPUs where xformers lacks CUDA support) by patching `diffusers_vdm/vae.py` and `diffusers_vdm/attention.py` to fall back to PyTorch native attention
- Fixes Step 3 video saving which fails on newer torchvision versions (`write_video` missing) by switching to `imageio`
- Pins dependency versions known to work together
- Verifies PyTorch can see your GPU after install
- Warns if your VRAM is too low before you start
- `--update` flag to re-apply patches without reinstalling

### Added features (beyond the original)

- **Multi-model WD tagger with a Step 1 picker** — choose one or several taggers in the UI; multiple are run as an ensemble and merged by max confidence. Newer, more accurate models than the original (EVA02-Large / ViT-Large v3) are the default.
- **GPU tagging** — the WD tagger runs on CUDA (via `onnxruntime-gpu`, CUDA-matched at install) with automatic CPU fallback.
- **Cancel buttons** for Step 2 (key frames) and Step 3 (video) so you can stop a long run without killing the app.
- **Input validation** — friendly messages instead of tracebacks when no image is uploaded, no operation steps are selected, or there are too few key frames.
- **VRAM-capped key frames** — Step 2 generates in sub-batches (output unchanged) so large step selections don't OOM.
- **High-VRAM mode** — on GPUs with >= 20 GB, models stay resident for faster runs (auto-enabled by the installer).

## Requirements

- Python 3.7+ (just to run the setup script — any Python will do)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed and in PATH
- [Git](https://git-scm.com/) installed and in PATH
- NVIDIA GPU with up-to-date drivers (RTX 2000 series or newer recommended, 10GB+ VRAM for best results)

## Install

**Windows (easiest) — double-click `setup.bat`.** It finds conda automatically (even if it is not on your PATH), then runs the installer. You will be prompted for an install location.

**From a terminal (any platform):**
```
python setup.py
```

You will be prompted for an install location. Everything else is automatic.

> On Windows, run `setup.py` from an **Anaconda Prompt** (or just use `setup.bat`). Double-clicking `setup.py` directly will fail because conda is not on PATH.

**Or specify the path directly:**
```
setup.bat --install-dir C:\AI\PaintsUndo
python setup.py --install-dir C:\AI\PaintsUndo
python setup.py --install-dir ~/PaintsUndo
```

First launch downloads the models automatically (~8-10 GB):
- `lllyasviel/paints_undo_single_frame` — keyframe generation (SD1.5-based)
- `lllyasviel/paints_undo_multi_frame` — video interpolation model

## Launch

After setup, a launch script is written to your install directory. Run it from there:

**Windows** — run `start.bat` from your install folder, or double-click it in Explorer.

**Linux** — run `bash start.sh` from your install folder.

Then open `http://127.0.0.1:7860` in your browser.

> Note: the `start.bat` and `start.sh` in this repo are templates. The actual launch scripts are written to your install directory by `setup.py`.

## Usage

**Step 1** — Upload your image, optionally choose which **tagger model(s)** to use in the dropdown, then click Generate Prompt. Selecting multiple runs them as an ensemble and merges the tags. Models download on first use.

**Step 2** — Click Generate Key Frames. Default operation steps (400, 600, 800, 900, 950, 999) work well. Steps 900 and 950 produce the most useful loose gesture sketches.

**Step 3** — Video interpolation between keyframes. Works on all supported GPUs. May still OOM on cards with less than 12 GB VRAM — if so, reduce Image Width and Height in Step 2 before generating.

Steps 2 and 3 each have a **Cancel** button to stop a running generation.

### Advanced configuration (environment variables)

Set these before launching (or add them to your `start` script) to change behavior without editing code:

| Variable | Default | Effect |
| --- | --- | --- |
| `PAINTS_UNDO_TAGGERS` | `eva02-large-v3,vit-large-v3` | Comma-separated taggers for Step 1. Options: `moat-v2, vit-v3, vit-large-v3, swinv2-v3, convnext-v3, eva02-large-v3`. The Step 1 dropdown overrides this. |
| `PAINTS_UNDO_HIGH_VRAM` | `0` | Set to `1` to keep all models resident on the GPU (faster, needs more VRAM). Auto-set by the installer on >= 20 GB cards. |
| `PAINTS_UNDO_KEYFRAME_BATCH` | `6` | Max key frames generated per sub-batch in Step 2. Lower it if Step 2 runs out of memory. |

## Updating patches

If you upgrade any packages and the UI breaks, re-apply the patches without reinstalling:

```
python setup.py --update --install-dir C:\AI\PaintsUndo
```

## Troubleshooting

**Buttons are greyed out / `APIInfoParseError`**
Re-run `python setup.py --update` to re-apply the gradio_client patch.

**`When localhost is not accessible` error**
On Windows, reset your network stack as administrator:
```
netsh int ipv4 reset
netsh int ipv6 reset
```
Then reboot.

**`conda not found` error**
You need to run the script from an Anaconda Prompt on Windows, or a terminal where `conda init` has been run on Linux.

**OOM on Step 2**
Reduce Image Width and Height in the UI. Try 448x448 instead of 512x512.

**Step 3 OOM error**
Reduce Image Width and Height in the UI in Step 2 before running Step 3. Try 384x448 instead of 512x640.

**WSL (Windows Subsystem for Linux)**
Not recommended. Run natively on Windows instead for best GPU support.

## Notes

- Models are saved to `<install_dir>/hf_download/` on first launch
- The conda environment (`paints_undo`) is fully isolated from all other Python environments on your system
- AMD and Intel GPUs are not supported — CUDA is NVIDIA-only

## Credits

Original project by [lllyasviel](https://github.com/lllyasviel). This fork adds installation tooling plus the added features listed above.
