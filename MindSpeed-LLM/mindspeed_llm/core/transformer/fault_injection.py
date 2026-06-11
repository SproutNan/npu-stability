import logging
import os

import torch

logger = logging.getLogger(__name__)

_FAULT_DTYPE_WARNED = {"bitshift": False, "round2nearest": False}


def _warn_dtype_once(op_name, x, shift):
    """Log dtype and mantissa info once per op type."""
    if _FAULT_DTYPE_WARNED[op_name]:
        return
    _FAULT_DTYPE_WARNED[op_name] = True
    extra = ""
    if x.dtype == torch.float32:
        effective_bits = max(23 - shift, 0)
    elif x.dtype == torch.bfloat16:
        effective_bits = max(7 - shift, 0)
        if shift > 7:
            extra = ", also_masks_exponent_low_bits=1"
    elif x.dtype == torch.float16:
        effective_bits = max(10 - shift, 0)
    else:
        effective_bits = "?"
    rank = os.environ.get("RANK", "0")
    logger.warning(
        f"[Fault Injection] rank={rank} {op_name}: tensor dtype={x.dtype}, "
        f"shift={shift}, effective_mantissa_bits={effective_bits}, "
        f"shape={tuple(x.shape)}{extra}"
    )


def bitshift(x, shift: int = 7):
    _warn_dtype_once("bitshift", x, shift)
    mask = -1 << shift

    if x.dtype == torch.float32:
        n_int = x.view(torch.int32)
        return (n_int & mask).view(torch.float32)

    elif x.dtype == torch.bfloat16 or x.dtype == torch.float16:
        n_int = x.view(torch.int16)
        return (n_int & mask).view(x.dtype)

    else:
        logger.warning(f"[Fault Injection] bitshift: unsupported dtype {x.dtype}, returning unchanged")
        return x


def round2nearest(x, pos: int = 7):
    _warn_dtype_once("round2nearest", x, pos)
    mask_val = -1 << pos
    add_val = 1 << (pos - 1)

    if x.dtype == torch.float32:
        n_int = x.view(torch.int32)
        return ((n_int + add_val) & mask_val).view(torch.float32)

    elif x.dtype == torch.float16:
        n_int = x.view(torch.int16)
        return ((n_int + add_val) & mask_val).view(torch.float16)

    elif x.dtype == torch.bfloat16:
        n_int = x.view(torch.int16)
        return ((n_int + add_val) & mask_val).view(torch.bfloat16)

    else:
        logger.warning(f"[Fault Injection] round2nearest: unsupported dtype {x.dtype}, returning unchanged")
        return x
