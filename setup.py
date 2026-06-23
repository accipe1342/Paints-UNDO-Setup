"""
PaintsUndo Setup Script
=======================
Cross-platform setup for lllyasviel/Paints-UNDO
Supports Windows and Linux, auto-detects NVIDIA GPU and picks correct CUDA version.

Requirements before running:
  - Miniconda or Anaconda installed and in PATH
  - Git installed and in PATH
  - NVIDIA GPU with drivers installed

Usage:
  python setup.py
  python setup.py --install-dir /path/to/install
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

REPO_URL = "https://github.com/lllyasviel/Paints-UNDO.git"
ENV_NAME = "paints_undo"
PYTHON_VERSION = "3.10"

# Maps GPU compute capability to PyTorch CUDA index URL
# Compute capability -> (cu_tag, display_name)
CUDA_MAP = {
    (12, 0): ("cu128", "CUDA 12.8 (RTX 5000 series)"),
    (8,  9): ("cu124", "CUDA 12.4 (RTX 4000 series)"),
    (8,  6): ("cu121", "CUDA 12.1 (RTX 3000 series)"),
    (7,  5): ("cu118", "CUDA 11.8 (RTX 2000 series)"),
    (7,  0): ("cu118", "CUDA 11.8 (Volta)"),
}

PYTORCH_INDEX_BASE = "https://download.pytorch.org/whl"

REQUIREMENTS = [
    "huggingface_hub==0.23.4",
    "diffusers==0.28.0",
    "transformers==4.41.1",
    "gradio==4.26.0",
    "gradio-client==0.15.1",
    "starlette==0.27.0",
    "bitsandbytes==0.43.1",
    "accelerate==0.30.1",
    "peft==0.11.1",
    "protobuf==3.20",
    "opencv-python",
    "tensorboardX",
    "safetensors",
    "pillow",
    "einops",
    "onnxruntime",
    "av",
]

IS_WINDOWS = platform.system() == "Windows"

# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd, check=True, capture=False):
    """Run a shell command, print it, and return CompletedProcess."""
    print(f"  >> {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        shell=IS_WINDOWS,
    )


def conda_run(args, check=True):
    """Run a command inside the paints_undo conda environment."""
    cmd = ["conda", "run", "--no-capture-output", "-n", ENV_NAME] + args
    return run(cmd, check=check)


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def ok(msg):
    print(f"  [OK] {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def error(msg):
    print(f"  [ERROR] {msg}")
    sys.exit(1)


# ── GPU Detection ─────────────────────────────────────────────────────────────

def detect_gpu():
    """
    Use nvidia-smi to get the GPU compute capability.
    Returns (major, minor) tuple or None if detection fails.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, None, "nvidia-smi not found"

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        line = result.stdout.strip().split("\n")[0]
        parts = line.split(",")
        if len(parts) < 2:
            return None, None, f"Unexpected nvidia-smi output: {line}"

        gpu_name = parts[0].strip()
        cap = parts[1].strip()  # e.g. "8.9" or "12.0"
        major, minor = cap.split(".")
        return int(major), int(minor), gpu_name

    except Exception as e:
        return None, None, str(e)


def pick_cuda_tag(major, minor):
    """
    Given compute capability (major, minor), return the best matching cu tag.
    Falls back to cu118 if nothing matches exactly.
    """
    # Try exact match first
    if (major, minor) in CUDA_MAP:
        return CUDA_MAP[(major, minor)]

    # Find highest supported capability <= detected
    best = None
    best_cap = (0, 0)
    for cap, info in CUDA_MAP.items():
        if cap <= (major, minor) and cap > best_cap:
            best = info
            best_cap = cap

    if best:
        return best

    # Absolute fallback
    return ("cu118", "CUDA 11.8 (fallback)")


# ── Conda checks ─────────────────────────────────────────────────────────────

def check_prereqs():
    section("Checking prerequisites")

    if not shutil.which("conda"):
        error(
            "conda not found in PATH.\n"
            "  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html\n"
            "  Then re-run this script from an Anaconda Prompt (Windows)\n"
            "  or a terminal with conda initialized (Linux)."
        )
    ok("conda found")

    if not shutil.which("git"):
        error(
            "git not found in PATH.\n"
            "  Install Git: https://git-scm.com/\n"
            "  Then re-run this script."
        )
    ok("git found")

    # Accept conda TOS (newer Miniconda versions require this)
    for channel in [
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
        "https://repo.anaconda.com/pkgs/msys2",
    ]:
        run(["conda", "tos", "accept", "--override-channels", "--channel", channel], check=False)
    ok("conda TOS accepted")


# ── Clone ─────────────────────────────────────────────────────────────────────

def clone_repo(install_dir: Path):
    section(f"Cloning Paints-UNDO to {install_dir}")

    if (install_dir / ".git").exists():
        print("  Repo already exists - pulling latest changes.")
        run(["git", "-C", str(install_dir), "pull"])
    else:
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", REPO_URL, str(install_dir)])

    ok(f"Repo ready at {install_dir}")


# ── Conda env ─────────────────────────────────────────────────────────────────

def create_env():
    section(f'Creating conda environment "{ENV_NAME}" (Python {PYTHON_VERSION})')
    run(["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"])
    ok("Environment ready")


# ── PyTorch ───────────────────────────────────────────────────────────────────

def install_pytorch():
    section("Detecting GPU and installing PyTorch")

    major, minor, gpu_info = detect_gpu()

    if major is None:
        warn(f"GPU detection failed: {gpu_info}")
        warn("Falling back to CUDA 11.8 (cu118). If this is wrong, re-run and manually edit the index URL.")
        cu_tag, cu_desc = "cu118", "CUDA 11.8 (fallback)"
    else:
        cu_tag, cu_desc = pick_cuda_tag(major, minor)
        print(f"  Detected GPU: {gpu_info}")
        print(f"  Compute capability: {major}.{minor}")
        print(f"  Selected: {cu_desc}")

    index_url = f"{PYTORCH_INDEX_BASE}/{cu_tag}"
    print(f"  PyTorch index URL: {index_url}")

    conda_run(["pip", "install", "torch", "torchvision", "--index-url", index_url])
    conda_run(["pip", "install", "xformers", "--index-url", index_url])
    ok("PyTorch + xformers installed")


# ── Requirements ──────────────────────────────────────────────────────────────

def install_requirements():
    section("Installing requirements")
    conda_run(["pip", "install"] + REQUIREMENTS)
    ok("Requirements installed")


# ── Patches ───────────────────────────────────────────────────────────────────

GRADIO_APP_PATCH = """
import re, sys
path = sys.argv[1]
content = open(path, 'r', encoding='utf-8').read()
patched = re.sub(
    r"block\\.queue\\(\\)\\.launch\\([^)]*\\)",
    "block.queue().launch(server_name='127.0.0.1', share=False)",
    content
)
if patched == content:
    print('  gradio_app.py: nothing to patch (already patched or pattern not found)')
else:
    open(path, 'w', encoding='utf-8').write(patched)
    print('  gradio_app.py: patched server_name to 127.0.0.1')
"""

GRADIO_CLIENT_PATCH = """
import sys, os
import gradio_client
base = os.path.dirname(gradio_client.__file__)
path = os.path.join(base, 'utils.py')
content = open(path, 'r', encoding='utf-8').read()
changed = False

patches = [
    (
        'def _json_schema_to_python_type(schema: Any, defs=None) -> str:',
        'def _json_schema_to_python_type(schema: Any, defs=None) -> str:\\n    if not isinstance(schema, dict): return "any"'
    ),
    (
        'def get_type(schema: dict):',
        'def get_type(schema: dict):\\n    if not isinstance(schema, dict): return "any"'
    ),
    (
        "f\\"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}\\"",
        "f\\"str, {_json_schema_to_python_type(schema['additionalProperties'] if isinstance(schema['additionalProperties'], dict) else {}, defs)}\\""
    ),
]

for old, new in patches:
    if old in content and new not in content:
        content = content.replace(old, new)
        changed = True

if changed:
    open(path, 'w', encoding='utf-8').write(content)
    print('  gradio_client/utils.py: boolean schema bug patched')
else:
    print('  gradio_client/utils.py: already patched or pattern not found')
"""


def apply_patches(install_dir: Path):
    section("Applying patches")

    # Write patch scripts to temp files
    patch_app = install_dir / "_patch_gradio_app.py"
    patch_client = install_dir / "_patch_gradio_client.py"

    patch_app.write_text(GRADIO_APP_PATCH, encoding="utf-8")
    patch_client.write_text(GRADIO_CLIENT_PATCH, encoding="utf-8")

    gradio_app = install_dir / "gradio_app.py"
    conda_run(["python", str(patch_app), str(gradio_app)])
    conda_run(["python", str(patch_client)])

    # Clean up temp patch files
    patch_app.unlink(missing_ok=True)
    patch_client.unlink(missing_ok=True)

    ok("Patches applied")


# ── Launch script ─────────────────────────────────────────────────────────────

def write_launch_script(install_dir: Path):
    section("Writing launch script")

    if IS_WINDOWS:
        launch_path = install_dir / "run_paints_undo.bat"
        launch_path.write_text(
            "@echo off\n"
            "title PaintsUndo\n"
            f'cd /d "{install_dir}"\n'
            "echo Starting PaintsUndo...\n"
            "echo Models download automatically on first run (~8-10GB)\n"
            "echo Gradio UI will open at http://127.0.0.1:7860\n"
            "echo.\n"
            f"conda activate {ENV_NAME}\n"
            "python gradio_app.py\n"
            "pause\n",
            encoding="utf-8"
        )
        ok(f"Launch script written: {launch_path}")
    else:
        launch_path = install_dir / "run_paints_undo.sh"
        launch_path.write_text(
            "#!/bin/bash\n"
            f'cd "{install_dir}"\n'
            "echo Starting PaintsUndo...\n"
            "echo Models download automatically on first run (~8-10GB)\n"
            "echo Gradio UI will open at http://127.0.0.1:7860\n"
            f"conda activate {ENV_NAME}\n"
            "python gradio_app.py\n",
            encoding="utf-8"
        )
        launch_path.chmod(0o755)
        ok(f"Launch script written: {launch_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(install_dir: Path):
    launch = "run_paints_undo.bat" if IS_WINDOWS else "run_paints_undo.sh"
    print()
    print("=" * 60)
    print("  Setup complete!")
    print()
    print("  First launch downloads ~8-10GB of models:")
    print("    - lllyasviel/paints_undo_single_frame")
    print("    - lllyasviel/paints_undo_multi_frame")
    print(f"  Models saved to: {install_dir / 'hf_download'}")
    print()
    print("  To launch:")
    print(f"    {install_dir / launch}")
    print("  OR manually:")
    print(f"    conda activate {ENV_NAME}")
    print(f"    cd {install_dir}")
    print("    python gradio_app.py")
    print()
    print("  Then open: http://127.0.0.1:7860")
    print("=" * 60)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PaintsUndo setup script")
    parser.add_argument(
        "--install-dir",
        type=str,
        default=None,
        help="Directory to install PaintsUndo into (default: prompted interactively)",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  PaintsUndo Setup")
    print("=" * 60)

    # Get install directory
    if args.install_dir:
        install_dir = Path(args.install_dir)
    else:
        print()
        print("  Where would you like to install PaintsUndo?")
        if IS_WINDOWS:
            print("  Examples: H:\\PaintsUndo   C:\\AI\\PaintsUndo")
        else:
            print("  Examples: /home/user/PaintsUndo   ~/AI/PaintsUndo")
        print()
        raw = input("  Install path: ").strip()
        if not raw:
            error("No path entered.")
        install_dir = Path(raw)

    print(f"\n  Install location: {install_dir}")

    check_prereqs()
    clone_repo(install_dir)
    create_env()
    install_pytorch()
    install_requirements()
    apply_patches(install_dir)
    write_launch_script(install_dir)
    print_summary(install_dir)


if __name__ == "__main__":
    main()
