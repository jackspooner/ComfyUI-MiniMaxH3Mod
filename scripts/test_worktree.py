"""Run existing model-free regressions from an isolated checkout, using native ComfyUI."""
import os
from pathlib import Path
import sys
import types
import unittest

root = Path(__file__).resolve().parents[1]
core = Path(os.environ.get('COMFYUI_ROOT', ''))
if not (core / 'comfy/sd.py').is_file():
    sys.exit('Set COMFYUI_ROOT to the intended ComfyUI source checkout.')
sys.path[:0] = [str(core), str(root), str(root / 'tests')]
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
torch.set_num_threads(1)

# Existing tests import custom_nodes.<checkout-directory>.nodes. Resolve that exact
# task checkout in this process rather than copying it into a serving installation.
package = sys.modules.setdefault('custom_nodes', types.ModuleType('custom_nodes'))
package.__path__ = [str(root.parent), *getattr(package, '__path__', [])]
suite = unittest.TestLoader().discover(str(root / 'tests'))
suite.addTests(unittest.TestLoader().discover(str(root / 'checks'), top_level_dir=str(root / 'checks')))
suite.addTests(unittest.TestLoader().loadTestsFromNames(['compatibility_regression_test', 'issue_regression_test']))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
