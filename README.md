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

## Requirements

- Python 3.7+ (just to run the setup script — any Python will do)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed and in PATH
- [Git](https://git-scm.com/) installed and in PATH
- NVIDIA GPU with up-to-date drivers (RTX 2000 series or newer recommended, 10GB+ VRAM for best results)

## Install

```
python setup.py
```

You will be prompted for an install location. Everything else is automatic.

**Or specify the path directly:**
```
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

**Step 1** — Upload your image and click Generate Prompt. Runs WD14 tagger automatically.

**Step 2** — Click Generate Key Frames. Default operation steps (400, 600, 800, 900, 950, 999) work well. Steps 900 and 950 produce the most useful loose gesture sketches.

**Step 3** — Video interpolation between keyframes. Works on all supported GPUs. May still OOM on cards with less than 12 GB VRAM — if so, reduce Image Width and Height in Step 2 before generating.

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

Original project by [lllyasviel](https://github.com/lllyasviel). This fork adds installation tooling only.
