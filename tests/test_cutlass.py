"""Dependency resolution, offline headers, and a real CUDA integration smoke test."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.build_offline_bundle import copy_cutlass
from tiny_transformer._cutlass import cutlass_include_paths
from tiny_transformer.check_cutlass import run_smoke_test


class CutlassHostTests(unittest.TestCase):
    def test_default_checkout_and_offline_copy(self):
        # Verify the shipped dependency and its offline distribution together.
        with patch.dict(os.environ, {}, clear=True):
            includes = cutlass_include_paths()
        self.assertTrue((Path(includes[0]) / "cute/tensor.hpp").is_file())
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            info = copy_cutlass(staging)
            self.assertEqual(len(info["commit"]), 40)
            root = staging / "third_party/cutlass"
            self.assertTrue((root / "LICENSE.txt").is_file())
            self.assertFalse((root / ".git").exists())
            with patch.dict(os.environ, {"CUTLASS_PATH": str(root)}):
                offline_includes = cutlass_include_paths()
            self.assertEqual(Path(offline_includes[0]), root.resolve() / "include")
            self.assertEqual((root / "include/cute/tensor.hpp").read_bytes(),
                             (Path(includes[0]) / "cute/tensor.hpp").read_bytes())

    def test_invalid_override_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CUTLASS_PATH": directory}):
                with self.assertRaisesRegex(RuntimeError, "git submodule update"):
                    cutlass_include_paths()

    def test_partial_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include/cutlass").mkdir(parents=True)
            (root / "include/cutlass/cutlass.h").touch()
            with patch.dict(os.environ, {"CUTLASS_PATH": str(root)}):
                with self.assertRaisesRegex(RuntimeError, "include/cute/tensor.hpp"):
                    cutlass_include_paths()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and nvcc")
class CutlassCudaTests(unittest.TestCase):
    def test_compile_and_run(self):
        run_smoke_test()


if __name__ == "__main__":
    unittest.main()
