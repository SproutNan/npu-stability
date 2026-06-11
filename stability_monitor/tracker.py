import torch
from typing import Dict, List, Optional


class MultiWindowWeightTracker:
    """Per-step snapshot store with refcounting for multi-window delta-W.

    Each window w in ``windows`` fires every w steps (cadence = window).
    dW = W_t - W_{t-w} is computed from the snapshot stored w steps ago.

    Snapshots are stored once even when multiple windows fire at the same
    step.  Refcounting ensures a step is pruned as soon as no future window
    will look it up again.

    Call ``update()`` once per key at each step, then ``finalize_step()``
    once at the end of the step to advance window pointers and prune.
    """

    def __init__(self, windows: List[int]):
        if not windows:
            raise ValueError("windows must be non-empty")
        self.windows = sorted(windows)
        self._store: Dict[int, Dict[str, torch.Tensor]] = {}  # step -> {key: cpu_tensor}
        self._refcounts: Dict[int, int] = {}                   # step -> # windows pointing here
        self._window_ptrs: Dict[int, int] = {w: 0 for w in self.windows}
        # per-step state (cleared each step change)
        self._dedup_step: Optional[int] = None
        self._dedup_keys: set = set()
        self._active_windows: set = set()  # windows that fired at this step

    # ── initialisation (W_0) ────────────────────────────────────────────

    def init_step_zero(self, key: str, W: torch.Tensor) -> None:
        """Store W_0 for *key*.  Called once per registered weight key."""
        if 0 not in self._store:
            self._store[0] = {}
        self._store[0][key] = W.detach().cpu().clone()

    def finalize_step_zero(self) -> None:
        """Set refcount at step 0 to len(windows).  All windows start here."""
        if 0 in self._store:
            self._refcounts[0] = len(self.windows)

    # ── main update ─────────────────────────────────────────────────────

    def update(self, key: str, W: torch.Tensor, step: int) -> Dict[int, torch.Tensor]:
        """Return {window: dW} for windows that fire at *step*.

        Only the first call per (key, step) is processed (micro-batch dedup).
        Window pointers and refcounts are NOT updated here — call
        ``finalize_step(step)`` after all keys for this step have been seen.
        """
        # ── per-step dedup ──
        if step != self._dedup_step:
            self._dedup_step = step
            self._dedup_keys.clear()
            self._active_windows.clear()
        if key in self._dedup_keys:
            return {}
        self._dedup_keys.add(key)

        # ── which windows fire? ──
        active = [w for w in self.windows if step % w == 0]
        if not active:
            return {}
        self._active_windows.update(active)

        # ── store current W ──
        if step not in self._store:
            self._store[step] = {}
        self._store[step][key] = W.detach().cpu().clone()

        # ── compute deltas ──
        results: Dict[int, torch.Tensor] = {}

        for w in active:
            lookback = step - w  # mathematical lookback, not affected by other keys
            if lookback >= 0:
                W_old = self._store.get(lookback, {}).get(key)
                if W_old is not None:
                    results[w] = W.detach() - W_old.to(W.device)

        return results

    def finalize_step(self, step: int) -> None:
        """Advance window pointers and update refcounts (called once per step)."""
        if step not in self._store:
            return

        # Init refcount for this step if not yet set
        if step not in self._refcounts:
            self._refcounts[step] = 0

        for w in self._active_windows:
            old_ptr = self._window_ptrs[w]
            if old_ptr == step:
                continue  # already advanced (shouldn't happen, but be safe)

            self._window_ptrs[w] = step

            # Release old
            if old_ptr in self._refcounts:
                self._refcounts[old_ptr] -= 1
                if self._refcounts[old_ptr] <= 0:
                    self._prune_step(old_ptr)

            # Claim new
            self._refcounts[step] += 1

    # ── accessors ────────────────────────────────────────────────────────

    def get_snapshot(self, key: str, step: int) -> Optional[torch.Tensor]:
        """Return the CPU tensor for *key* at *step*, or None."""
        store = self._store.get(step)
        if store is None:
            return None
        return store.get(key)

    def get_lookback_step(self, window: int) -> Optional[int]:
        """Return the step that *window* is currently using as its lookback."""
        return self._window_ptrs.get(window)

    # ── internal ─────────────────────────────────────────────────────────

    def _prune_step(self, step: int) -> None:
        """Delete all tensors at *step* — no window references it anymore."""
        self._store.pop(step, None)
        self._refcounts.pop(step, None)
