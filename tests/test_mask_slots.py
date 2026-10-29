"""Collector unit tests only: no workflow validation, model loads or inference."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", ROOT.parents[1]))
sys.path.insert(0, str(COMFY_ROOT))
from comfy.cli_args import args
args.cpu = True
import folder_paths
import torch

spec = importlib.util.spec_from_file_location("refmod_mask_slots_tests", ROOT / "__init__.py",
                                            submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
with tempfile.TemporaryDirectory() as temporary:
    with patch.object(folder_paths, "models_dir", temporary):
        spec.loader.exec_module(package)
N = sys.modules[spec.name + ".nodes"]


class MaskSlotsTests(unittest.TestCase):
    def test_numeric_order_and_different_sizes(self):
        masks = {f"mask_{index}": torch.full((1, 8 + index, 12), index / 24)
                 for index in reversed(range(24))}
        result = N.MiniMaxH3RefModMaskListMulti.execute(masks)[0]
        self.assertEqual(len(result), 24)
        self.assertEqual([tuple(value.shape) for value in result], [(1, 8 + i, 12) for i in range(24)])
        for index, value in enumerate(result):
            self.assertAlmostEqual(value.mean().item(), index / 24, places=6)

    def test_batch_splits_and_single_mask_broadcast_input(self):
        batch = torch.stack([torch.zeros(8, 12), torch.ones(8, 12)])
        result = N.MiniMaxH3RefModMaskListMulti.execute({"mask_0": batch})[0]
        self.assertEqual([value.mean().item() for value in result], [0, 1])
        self.assertEqual(len(N.MiniMaxH3RefModMaskListMulti.execute({"mask_0": batch[:1]})[0]), 1)

    def test_rejects_missing_slots_and_non_masks(self):
        for masks in ({}, {"mask_1": torch.zeros(8, 12)}, {"mask_0": "not a mask"},
                      {f"mask_{index}": torch.zeros(8, 12) for index in range(25)}):
            with self.assertRaises(ValueError):
                N.MiniMaxH3RefModMaskListMulti.execute(masks)

    def test_old_list_collector_and_node_registration(self):
        mask = torch.ones(1, 8, 12)
        result = N.MiniMaxH3RefModMaskList().collect([mask])[0]
        torch.testing.assert_close(result[0], mask)
        self.assertIs(N.NODE_CLASS_MAPPINGS["MiniMaxH3RefModMaskListMulti"], N.MiniMaxH3RefModMaskListMulti)


if __name__ == "__main__":
    unittest.main()
