# Paints-UNDO (Easy Install Fork)

This is a fork of [lllyasviel/Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO) with a setup script that handles installation, dependency fixes, and GPU detection automatically.

## What's different from the original

- `setup.py` — one-command install that auto-detects your GPU and installs the correct PyTorch CUDA version
- Fixes a `gradio_client` bug that causes the UI buttons to be greyed out (`APIInfoParseError: Cannot parse schema True`)
- Fixes a localhost binding issue that prevents the Gradio UI from loading
- Pins dependency versions known to work together on Windows and Linux

## Requirements

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed and in PATH
- [Git](https://git-scm.com/) installed and in PATH
- NVIDIA GPU with drivers installed (RTX 2000 series or newer recommended)

## Install

```
python setup.py
```

You'll be prompted to choose an install location. That's it.

First launch will download the models automatically (~8-10GB):
- `lllyasviel/paints_undo_single_frame` — the keyframe generation model (SD1.5-based)
- `lllyasviel/paints_undo_multi_frame` — the video interpolation model

## Launch

After setup, run:

**Windows:**
```
start.bat
```

**Linux:**
```
bash start.sh
```

Then open your browser to `http://127.0.0.1:7860`

## Usage

**Step 1** — Upload your image and click Generate Prompt. This runs WD14 tagger and fills the prompt box automatically.

**Step 2** — Click Generate Key Frames. Default operation steps (400, 600, 800, 900, 950, 999) work well. Steps 900 and 950 produce the most useful loose gesture sketches.

**Step 3** — Video interpolation between keyframes. Skip this if you only need the keyframe images, or if you have less than 16GB VRAM.

## Notes

- Step 3 (video generation) may fail on RTX 5000 series GPUs due to xformers compatibility with compute capability 12.0. Steps 1 and 2 work fine.
- Models are saved to `<install_dir>/hf_download/` on first launch.
- The conda environment (`paints_undo`) is fully isolated and won't affect any other Python environment on your system.

## Credits

Original project by [lllyasviel](https://github.com/lllyasviel). This fork only adds installation tooling.
