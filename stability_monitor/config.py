from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MonitorConfig:
    router_cadence: int = 10
    delta_windows: Tuple[int, ...] = (5, 10, 20, 50, 100, 500, 1000)
    w_cadence: int = 500

    per_head: bool = True
    head_aggregation: Tuple[str, ...] = ("mean", "max", "p95")

    output_path: Optional[str] = None
