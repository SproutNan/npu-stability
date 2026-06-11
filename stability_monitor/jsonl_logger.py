import json
from typing import Optional, Dict, Any


def _is_rank_zero() -> bool:
    try:
        import torch.distributed as dist
    except ImportError:
        return True
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class JsonlLogger:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._fh = None
        if path and _is_rank_zero():
            self._fh = open(path, "a", buffering=1)

    def log(self, record: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record, default=_default) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _default(o):
    try:
        import torch

        if isinstance(o, torch.Tensor):
            return o.item() if o.numel() == 1 else o.tolist()
    except Exception:
        pass
    raise TypeError(f"Type not serializable: {type(o)}")
