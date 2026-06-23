#!/bin/bash
cd "$(dirname "$0")"
python3 _banner.py 2>/dev/null || echo '  PaintsUndo'
echo
echo '  Open your browser to: http://127.0.0.1:7860'
echo '  Models download automatically on first run (~8-10 GB)'
echo
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate paints_undo
    python gradio_app.py
else
    conda run -n paints_undo python gradio_app.py
fi
