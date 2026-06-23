#!/bin/bash
cd "$(dirname "$0")"
echo "Starting PaintsUndo..."
echo "Gradio UI will open at http://127.0.0.1:7860"
echo
conda activate paints_undo
python gradio_app.py
