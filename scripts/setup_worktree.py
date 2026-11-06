"""Check an explicit prepared ComfyUI runtime without modifying it or starting services."""
import os
from pathlib import Path
import subprocess
import sys

runtime = os.environ.get('COMFYUI_PYTHON')
core = Path(os.environ.get('COMFYUI_ROOT', ''))
if not runtime or not Path(runtime).is_file() or not (core / 'comfy/sd.py').is_file():
    sys.exit('Set COMFYUI_PYTHON to a prepared Python and COMFYUI_ROOT to a ComfyUI source checkout. No packages or models are installed.')
subprocess.run([runtime, '-c', "import sys, torch, safetensors, numpy, PIL; assert sys.version_info >= (3,10); print('Ready:', sys.version.split()[0], 'torch', torch.__version__, 'safetensors', safetensors.__version__)"] , check=True)
print('Readiness only. No model, environment, serving source or shared state was changed. See docs/development.md for separate checks.')
