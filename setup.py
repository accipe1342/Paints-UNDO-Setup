"""
PaintsUndo Setup Script
=======================
Cross-platform setup for lllyasviel/Paints-UNDO
Supports Windows and Linux, auto-detects NVIDIA GPU and picks correct CUDA version.

Requirements before running:
  - Python 3.7+ (system Python is fine, just to run this script)
  - Miniconda or Anaconda installed and in PATH
  - Git installed and in PATH
  - NVIDIA GPU with drivers installed

Usage:
  python setup.py
  python setup.py --install-dir /path/to/install
  python setup.py --update        (re-apply patches to existing install)
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── Python version check ─────────────────────────────────────────────────────

if sys.version_info < (3, 7):
    print("[ERROR] This script requires Python 3.7 or newer.")
    print(f"        You are running Python {sys.version}")
    print("        Install Python 3.10 or newer from https://www.python.org/")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────

REPO_URL  = "https://github.com/lllyasviel/Paints-UNDO.git"
ENV_NAME  = "paints_undo"
PYTHON_VERSION = "3.10"
MIN_VRAM_GB  = 8   # hard warning threshold — below this Step 2 will almost certainly fail
SOFT_VRAM_GB = 10  # soft warning threshold — below this Step 2 may OOM on larger images

# Compute capability -> (cu_tag, friendly name)
CUDA_MAP = {
    (12, 0): ("cu128", "CUDA 12.8 (RTX 5000 series)"),
    (8,  9): ("cu124", "CUDA 12.4 (RTX 4000 series)"),
    (8,  6): ("cu121", "CUDA 12.1 (RTX 3000 series)"),
    (7,  5): ("cu118", "CUDA 11.8 (RTX 2000 series)"),
    (7,  0): ("cu118", "CUDA 11.8 (Volta)"),
    (6,  1): ("cu118", "CUDA 11.8 (GTX 1000 series)"),
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
    "imageio",
    "imageio-ffmpeg",
]

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = r"""
  ███████████             ███              █████            █████  █████
  ▒▒███▒▒▒▒▒███           ▒▒▒              ▒▒███            ▒▒███  ▒▒███
   ▒███    ▒███  ██████   ████  ████████   ███████    █████  ▒███   ▒███  ████████    ███████   ██████
   ▒██████████  ▒▒▒▒▒███ ▒▒███ ▒▒███▒▒███ ▒▒▒███▒    ███▒▒  ▒███   ▒███ ▒▒███▒▒███  ███▒▒███  ███▒▒███
   ▒███▒▒▒▒▒▒    ███████  ▒███  ▒███ ▒███   ▒███    ▒▒█████ ▒███   ▒███  ▒███ ▒███ ▒███ ▒███ ▒███ ▒███
   ▒███         ███▒▒███  ▒███  ▒███ ▒███   ▒███ ███ ▒▒▒▒███▒███   ▒███  ▒███ ▒███ ▒███ ▒███ ▒███ ▒███
   █████       ▒▒████████ █████ ████ █████  ▒▒█████  ██████ ▒▒████████   ████ █████▒▒████████▒▒██████
  ▒▒▒▒▒         ▒▒▒▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒ ▒▒▒▒▒    ▒▒▒▒▒  ▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒   ▒▒▒▒ ▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd, check=True):
    """Run a shell command, print it, return CompletedProcess."""
    display = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"  >> {display}")
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        shell=IS_WINDOWS,
    )


def conda_run(args, check=True):
    """Run a command inside the paints_undo conda environment.
    Output goes directly to terminal (not captured) so user sees live progress.
    """
    cmd = ["conda", "run", "-n", ENV_NAME] + args
    display = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"  >> {display}")
    return subprocess.run(
        cmd,
        check=check,
        shell=IS_WINDOWS,
        # No text=True, no stdout/stderr redirect — inherits parent terminal directly
    )


def conda_env_exists():
    """Return True if the paints_undo conda env already exists."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, check=True,
            shell=IS_WINDOWS
        )
        return ENV_NAME in result.stdout
    except Exception:
        return False


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def info(msg): print(f"  [INFO] {msg}")


def error(msg):
    print()
    print(f"  [ERROR] {msg}")
    print()
    sys.exit(1)


def ask_yes_no(prompt, default=True):
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(f"  {prompt}{suffix}").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# ── GPU Detection ─────────────────────────────────────────────────────────────

def detect_gpus():
    """
    Use nvidia-smi to detect all NVIDIA GPUs.
    Returns list of dicts: [{name, major, minor, vram_mb}]
    Returns empty list if nvidia-smi not found or no GPUs detected.
    """
    if not shutil.which("nvidia-smi"):
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name = parts[0]
            cap  = parts[1]   # e.g. "8.9"
            vram = parts[2]   # MiB as string

            major_s, minor_s = cap.split(".")
            gpus.append({
                "name":    name,
                "major":   int(major_s),
                "minor":   int(minor_s),
                "vram_mb": int(vram),
            })
        return gpus

    except Exception as e:
        warn(f"nvidia-smi query failed: {e}")
        return []


def pick_cuda_tag(major, minor):
    """
    Given compute capability, return best matching (cu_tag, description).
    Picks the highest supported capability that is <= the detected one.
    Falls back to cu118.
    """
    if (major, minor) in CUDA_MAP:
        return CUDA_MAP[(major, minor)]

    best, best_cap = None, (0, 0)
    for cap, entry in CUDA_MAP.items():
        if cap <= (major, minor) and cap > best_cap:
            best, best_cap = entry, cap

    return best if best else ("cu118", "CUDA 11.8 (fallback)")


def check_gpu():
    """
    Detect GPUs, print info, check VRAM, warn about unsupported vendors.
    Returns (cu_tag, cu_desc) for the primary GPU.
    """
    section("GPU Detection")

    # Check for non-NVIDIA GPU presence as a warning
    if IS_LINUX:
        try:
            lspci = subprocess.run(["lspci"], capture_output=True, text=True, check=False)
            if "AMD" in lspci.stdout or "Radeon" in lspci.stdout:
                warn("AMD GPU detected. PaintsUndo requires an NVIDIA GPU.")
                warn("AMD GPUs are not supported — CUDA is NVIDIA-only.")
            if "Intel" in lspci.stdout and "VGA" in lspci.stdout:
                warn("Intel integrated GPU detected. This is not sufficient for PaintsUndo.")
        except Exception:
            pass

    gpus = detect_gpus()

    if not gpus:
        warn("No NVIDIA GPU detected via nvidia-smi.")
        warn("Make sure NVIDIA drivers are installed and nvidia-smi is in PATH.")
        warn("Falling back to CUDA 11.8 (cu118) — change manually if needed.")
        return "cu118", "CUDA 11.8 (fallback — no GPU detected)"

    # Print all detected GPUs
    print(f"  Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        vram_gb = gpu["vram_mb"] / 1024
        vram_str = f"{vram_gb:.1f} GB VRAM"
        cap_str  = f"compute {gpu['major']}.{gpu['minor']}"
        print(f"    GPU {i}: {gpu['name']} — {vram_str} — {cap_str}")

    # Use GPU 0 (primary) for CUDA version selection
    primary = gpus[0]
    cu_tag, cu_desc = pick_cuda_tag(primary["major"], primary["minor"])

    # VRAM check
    vram_gb = primary["vram_mb"] / 1024
    if vram_gb < MIN_VRAM_GB:
        warn(f"Primary GPU has {vram_gb:.1f} GB VRAM.")
        warn("PaintsUndo Step 2 (keyframes) needs ~10 GB minimum.")
        warn("Step 1 (prompt generation) may still work.")
    elif vram_gb < SOFT_VRAM_GB:
        warn(f"Primary GPU has {vram_gb:.1f} GB VRAM.")
        warn("Step 2 may run out of memory on some images. Try smaller resolutions.")
    else:
        ok(f"Primary GPU: {primary['name']} ({vram_gb:.1f} GB VRAM)")

    ok(f"Selected PyTorch build: {cu_desc}")
    return cu_tag, cu_desc


# ── Prerequisites ─────────────────────────────────────────────────────────────

def check_prereqs():
    section("Checking prerequisites")

    # conda
    if not shutil.which("conda"):
        error(
            "conda not found in PATH.\n"
            "  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html\n"
            "  Windows: run this script from Anaconda Prompt\n"
            "  Linux:   run `conda init bash` then restart your terminal"
        )
    ok("conda found")

    # git
    if not shutil.which("git"):
        error(
            "git not found in PATH.\n"
            "  Install Git from https://git-scm.com/ then re-run."
        )
    ok("git found")

    # Accept conda TOS (newer Miniconda requires this)
    channels = [
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
        "https://repo.anaconda.com/pkgs/msys2",
    ]
    for ch in channels:
        run(["conda", "tos", "accept", "--override-channels", "--channel", ch], check=False)
    ok("conda TOS accepted")


# ── Clone ─────────────────────────────────────────────────────────────────────

def clone_repo(install_dir: Path):
    section(f"Cloning Paints-UNDO to {install_dir}")

    if (install_dir / ".git").exists():
        info("Repo already exists — pulling latest changes.")
        run(["git", "-C", str(install_dir), "pull"])
    else:
        try:
            install_dir.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            error(f"Cannot create install directory: {install_dir}\n  {e}\n  Check that the drive exists and you have write permission.")
        info("Cloning repository (~3 MB)...")
        run(["git", "clone", REPO_URL, str(install_dir)])

    ok(f"Repo ready at {install_dir}")


# ── Conda env ─────────────────────────────────────────────────────────────────

def create_env():
    section(f'Conda environment "{ENV_NAME}"')

    if conda_env_exists():
        info(f'Environment "{ENV_NAME}" already exists — skipping creation.')
        info("To reinstall from scratch: conda env remove -n paints_undo")
    else:
        info(f"Creating environment with Python {PYTHON_VERSION}...")
        run(["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"])
        ok("Environment created")


# ── PyTorch ───────────────────────────────────────────────────────────────────

def install_pytorch(cu_tag):
    section("Installing PyTorch + xformers")

    index_url = f"{PYTORCH_INDEX_BASE}/{cu_tag}"
    info(f"Index URL: {index_url}")
    info("Downloading PyTorch (~2.5 GB) — this may take several minutes...")

    conda_run(["pip", "install", "torch", "torchvision", "--index-url", index_url])
    conda_run(["pip", "install", "xformers", "--index-url", index_url])
    ok("PyTorch + xformers installed")


# ── Requirements ──────────────────────────────────────────────────────────────

def install_requirements():
    section("Installing requirements")
    info("Installing pinned dependencies...")
    conda_run(["pip", "install"] + REQUIREMENTS)
    ok("Requirements installed")


# ── Verify PyTorch ────────────────────────────────────────────────────────────

VERIFY_SCRIPT = """
import sys
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA OK: {name} ({vram:.1f} GB)")
    else:
        print("CUDA NOT AVAILABLE - torch installed but GPU not accessible")
        sys.exit(1)
except ImportError:
    print("torch not found")
    sys.exit(1)
"""

def verify_pytorch(install_dir: Path):
    section("Verifying PyTorch + CUDA")

    verify_file = install_dir / "_verify_torch.py"
    verify_file.write_text(VERIFY_SCRIPT, encoding="utf-8")

    try:
        result = conda_run(["python", str(verify_file)], check=False)
        if result.returncode == 0:
            ok("PyTorch CUDA verification passed.")
        else:
            warn("PyTorch CUDA verification failed.")
            warn("PaintsUndo may not be able to use your GPU.")
            warn("Check that your NVIDIA drivers are up to date.")
    except Exception as e:
        warn(f"Verification step failed: {e}")
    finally:
        verify_file.unlink(missing_ok=True)


# ── Patches ───────────────────────────────────────────────────────────────────

GRADIO_APP_PATCH = r"""
import re, sys
path = sys.argv[1]
content = open(path, 'r', encoding='utf-8').read()
patched = re.sub(
    r"block\.queue\(\)\.launch\([^)]*\)",
    "block.queue().launch(server_name='127.0.0.1', share=False)",
    content
)
if patched == content:
    print('  gradio_app.py: already patched or pattern not found')
else:
    open(path, 'w', encoding='utf-8').write(patched)
    print('  gradio_app.py: patched server_name -> 127.0.0.1')
"""

GRADIO_CLIENT_PATCH = """
import os
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
    print('  gradio_client/utils.py: already patched')
"""

VAE_PATCH = r"""
import sys
path = sys.argv[1]
content = open(path, 'r', encoding='utf-8').read()

OLD = '''def chunked_attention(q, k, v, batch_chunk=0):
    # if batch_chunk > 0 and not torch.is_grad_enabled():
    #     batch_size = q.size(0)
    #     chunks = [slice(i, i + batch_chunk) for i in range(0, batch_size, batch_chunk)]
    #
    #     out_chunks = []
    #     for chunk in chunks:
    #         q_chunk = q[chunk]
    #         k_chunk = k[chunk]
    #         v_chunk = v[chunk]
    #
    #         out_chunk = torch.nn.functional.scaled_dot_product_attention(
    #             q_chunk, k_chunk, v_chunk, attn_mask=None
    #         )
    #         out_chunks.append(out_chunk)
    #
    #     out = torch.cat(out_chunks, dim=0)
    # else:
    #     out = torch.nn.functional.scaled_dot_product_attention(
    #         q, k, v, attn_mask=None
    #     )
    out = xformers.ops.memory_efficient_attention(q, k, v)
    return out'''

NEW = '''def chunked_attention(q, k, v, batch_chunk=0):
    try:
        out = xformers.ops.memory_efficient_attention(q, k, v)
    except Exception:
        # Fall back to chunked scaled_dot_product_attention to avoid OOM
        chunk_size = 256
        out_chunks = []
        for i in range(0, q.shape[0], chunk_size):
            q_c = q[i:i + chunk_size].unsqueeze(1)
            k_c = k[i:i + chunk_size].unsqueeze(1)
            v_c = v[i:i + chunk_size].unsqueeze(1)
            out_chunks.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q_c, k_c, v_c, attn_mask=None
                ).squeeze(1)
            )
        out = torch.cat(out_chunks, dim=0)
    return out'''

if OLD in content and NEW not in content:
    content = content.replace(OLD, NEW)
    open(path, 'w', encoding='utf-8').write(content)
    print('  diffusers_vdm/vae.py: xformers fallback + chunked OOM fix applied')
else:
    print('  diffusers_vdm/vae.py: already patched or pattern not found')
"""

ATTENTION_PATCH = r"""
import sys
path = sys.argv[1]
content = open(path, 'r', encoding='utf-8').read()

OLD = '    out = xformers.ops.memory_efficient_attention(q, k, v)\n\n    out = ('
NEW = '''    try:
        out = xformers.ops.memory_efficient_attention(q, k, v)
    except Exception:
        q_t = q.unsqueeze(1)
        k_t = k.unsqueeze(1)
        v_t = v.unsqueeze(1)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t).squeeze(1)

    out = ('''

if OLD in content and NEW not in content:
    content = content.replace(OLD, NEW)
    open(path, 'w', encoding='utf-8').write(content)
    print('  diffusers_vdm/attention.py: xformers fallback applied')
else:
    print('  diffusers_vdm/attention.py: already patched or pattern not found')
"""

UTILS_PATCH = r"""
import sys
path = sys.argv[1]
content = open(path, 'r', encoding='utf-8').read()

OLD = '    torchvision.io.write_video(output_filename, x, fps=fps, video_codec=\'h264\', options={\'crf\': \'1\'})'
NEW = '''    import imageio
    imageio.mimwrite(output_filename, x.numpy(), fps=fps, codec='h264', quality=9)'''

if OLD in content and NEW not in content:
    content = content.replace(OLD, NEW)
    open(path, 'w', encoding='utf-8').write(content)
    print('  diffusers_vdm/utils.py: write_video -> imageio patch applied')
else:
    print('  diffusers_vdm/utils.py: already patched or pattern not found')
"""


def apply_patches(install_dir: Path):
    section("Applying patches")

    patch_app        = install_dir / "_patch_gradio_app.py"
    patch_client     = install_dir / "_patch_gradio_client.py"
    patch_vae        = install_dir / "_patch_vae.py"
    patch_attention  = install_dir / "_patch_attention.py"
    patch_utils      = install_dir / "_patch_utils.py"

    patch_app.write_text(GRADIO_APP_PATCH, encoding="utf-8")
    patch_client.write_text(GRADIO_CLIENT_PATCH, encoding="utf-8")
    patch_vae.write_text(VAE_PATCH, encoding="utf-8")
    patch_attention.write_text(ATTENTION_PATCH, encoding="utf-8")
    patch_utils.write_text(UTILS_PATCH, encoding="utf-8")

    try:
        conda_run(["python", str(patch_app),       str(install_dir / "gradio_app.py")])
        conda_run(["python", str(patch_client)])
        conda_run(["python", str(patch_vae),       str(install_dir / "diffusers_vdm" / "vae.py")])
        conda_run(["python", str(patch_attention), str(install_dir / "diffusers_vdm" / "attention.py")])
        conda_run(["python", str(patch_utils),     str(install_dir / "diffusers_vdm" / "utils.py")])
    finally:
        patch_app.unlink(missing_ok=True)
        patch_client.unlink(missing_ok=True)
        patch_vae.unlink(missing_ok=True)
        patch_attention.unlink(missing_ok=True)
        patch_utils.unlink(missing_ok=True)

    ok("Patches applied")


# ── Launch script ─────────────────────────────────────────────────────────────

def write_launch_script(install_dir: Path):
    section("Writing launch script")

    # Write banner helper so start scripts can print it via Python
    banner_file = install_dir / "_banner.py"
    banner_file.write_text(
        f'BANNER = {repr(BANNER)}\n'
        'if __name__ == "__main__":\n'
        '    import sys\n'
        '    if hasattr(sys.stdout, "reconfigure"):\n'
        '        sys.stdout.reconfigure(encoding="utf-8", errors="replace")\n'
        '    print(BANNER)\n',
        encoding="utf-8"
    )

    if IS_WINDOWS:
        path = install_dir / "start.bat"
        path.write_text(
            "@echo off\n"
            "title PaintsUndo\n"
            f'cd /d "{install_dir}"\n'
            "chcp 65001 >nul\n"
            f"conda run -n {ENV_NAME} python _banner.py\n"
            "echo.\n"
            "echo   Open your browser to: http://127.0.0.1:7860\n"
            "echo   Models download automatically on first run (~8-10 GB)\n"
            "echo.\n"
            f"conda run -n {ENV_NAME} python gradio_app.py\n"
            "pause\n",
            encoding="utf-8"
        )
    else:
        path = install_dir / "start.sh"
        path.write_text(
            "#!/bin/bash\n"
            f'cd "{install_dir}"\n'
            "python3 _banner.py 2>/dev/null || echo '  PaintsUndo'\n"
            "echo\n"
            "echo '  Open your browser to: http://127.0.0.1:7860'\n"
            "echo '  Models download automatically on first run (~8-10 GB)'\n"
            "echo\n"
            "CONDA_BASE=$(conda info --base 2>/dev/null)\n"
            'if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then\n'
            '    source "$CONDA_BASE/etc/profile.d/conda.sh"\n'
            f"    conda activate {ENV_NAME}\n"
            "    python gradio_app.py\n"
            "else\n"
            f"    conda run -n {ENV_NAME} python gradio_app.py\n"
            "fi\n",
            encoding="utf-8"
        )
        path.chmod(0o755)

    ok(f"Launch script written: {path}")
    return path


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(install_dir: Path, launch_path: Path):
    print()
    print("=" * 60)
    print("  Setup complete!")
    print()
    print("  First launch will download ~8-10 GB of models:")
    print("    - lllyasviel/paints_undo_single_frame")
    print("    - lllyasviel/paints_undo_multi_frame")
    print(f"  Models saved to: {install_dir / 'hf_download'}")
    print()
    print("  To launch:")
    print(f"    {launch_path}")
    print()
    print("  Then open: http://127.0.0.1:7860")
    print("=" * 60)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PaintsUndo cross-platform setup script"
    )
    parser.add_argument(
        "--install-dir",
        type=str,
        default=None,
        help="Where to install PaintsUndo (prompted if not given)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Re-apply patches to an existing install without reinstalling",
    )
    args = parser.parse_args()

    print(BANNER)
    print(f"  Setup  |  Python {sys.version.split()[0]}  |  {platform.system()}")
    print()

    # Get install directory
    if args.install_dir:
        install_dir = Path(args.install_dir).expanduser().resolve()
    else:
        print()
        if args.update:
            print("  Where is your existing PaintsUndo install?")
        else:
            print("  Where would you like to install PaintsUndo?")
        if IS_WINDOWS:
            print("  Examples:  H:\\PaintsUndo    C:\\AI\\PaintsUndo")
        else:
            print("  Examples:  ~/PaintsUndo    /opt/PaintsUndo")
        print()
        raw = input("  Install path: ").strip()
        if not raw:
            error("No path entered.")
        install_dir = Path(raw).expanduser().resolve()

    print(f"\n  Install location: {install_dir}")

    # Update-only mode — just re-apply patches
    if args.update:
        if not (install_dir / "gradio_app.py").exists():
            error(f"No PaintsUndo install found at {install_dir}")
        if not conda_env_exists():
            error(f'Conda environment "{ENV_NAME}" not found. Run setup.py without --update first.')
        apply_patches(install_dir)
        ok("Update complete.")
        return

    # Full install
    check_prereqs()
    cu_tag, _ = check_gpu()
    clone_repo(install_dir)
    create_env()
    install_pytorch(cu_tag)
    install_requirements()
    verify_pytorch(install_dir)
    apply_patches(install_dir)
    launch_path = write_launch_script(install_dir)
    print_summary(install_dir, launch_path)

    # Offer to launch immediately
    print()
    if ask_yes_no("Launch PaintsUndo now?"):
        print()
        info("Starting PaintsUndo — open http://127.0.0.1:7860 in your browser")
        info("Models will download on first use (~8-10 GB)")
        print()
        os.chdir(install_dir)
        conda_run(["python", "gradio_app.py"])


if __name__ == "__main__":
    main()
