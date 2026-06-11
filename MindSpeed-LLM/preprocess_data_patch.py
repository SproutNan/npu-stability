#!/usr/bin/env python3
"""Wrapper for preprocess_data.py that patches torch.jit.script before imports."""

import torch

# transfer_to_npu monkey-patches torch in ways that break TorchScript.
# Megatron's @jit_fuser uses torch.jit.script on PyTorch < 2.2.
# Replace it with a no-op to avoid the compilation error.
def _noop_jit_script(fn=None, **kwargs):
    if fn is not None:
        return fn
    return lambda f: f

torch.jit.script = _noop_jit_script

# Now run the actual preprocessing script
import runpy
runpy.run_module("preprocess_data", run_name="__main__")
