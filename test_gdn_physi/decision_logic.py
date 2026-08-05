"""
Two-signal runtime decision logic (`mem_phys_prompt_zh.md` Sec 5): combine
memory distance `d` (novelty vs. the learned "known normal" dictionary)
with structure residual `r` (whether the physical/relational constraints
still hold) to resolve the ambiguity neither signal can resolve alone --
"this doesn't match anything I've memorized" is consistent with BOTH a
genuinely new (but still physically valid) normal work mode AND a real
fault, and only `r` tells them apart.

Decision table (Sec 5's five cases):
  1. d small                              -> KNOWN_NORMAL
  2. d large AND r broken                 -> HARD_ALARM (immediate)
  3. d large AND r holds, brief/isolated   -> SOFT_ALERT (observe, don't memorize yet)
  4. d large AND r holds, persists         -> PENDING_LABEL (queue for confirmation)
  5. pending item confirmed normal         -> grow codebook (new prototype),
     never alarms again for this regime; confirmed anomalous -> alarm.

This module implements the STATEFUL part (persistence tracking, queueing,
codebook growth) -- the model itself (`fl_model.py`) only produces d and r
per window; this class turns a d/r stream into decisions.
"""
from dataclasses import dataclass, field
from enum import Enum

import torch


class Decision(Enum):
    KNOWN_NORMAL = "known_normal"
    HARD_ALARM = "hard_alarm"
    SOFT_ALERT = "soft_alert"
    PENDING_LABEL = "pending_label"


@dataclass
class StreamMonitor:
    """Tracks one data stream (one robot) over time. Call `step(d, r, z)`
    once per window in temporal order."""

    d_threshold: float
    r_threshold: float
    persistence_windows: int = 5  # consecutive "novel but relation-consistent" windows before escalating soft-alert -> pending-label
    max_queue: int = 64

    _consecutive_novel_consistent: int = field(default=0, init=False)
    _pending_queue: list = field(default_factory=list, init=False)  # list of z (or z-summary) awaiting human confirmation

    def step(self, d: float, r: float, z=None) -> Decision:
        relation_holds = r <= self.r_threshold
        novel = d > self.d_threshold

        if not novel:
            self._consecutive_novel_consistent = 0
            return Decision.KNOWN_NORMAL

        if not relation_holds:
            self._consecutive_novel_consistent = 0
            return Decision.HARD_ALARM

        # novel AND relation holds
        self._consecutive_novel_consistent += 1
        if self._consecutive_novel_consistent >= self.persistence_windows:
            if z is not None and len(self._pending_queue) < self.max_queue:
                self._pending_queue.append(z)
            return Decision.PENDING_LABEL
        return Decision.SOFT_ALERT

    def confirm_pending(self, confirmed_normal: bool):
        """Human (or a downstream supervised classifier) resolves the
        oldest pending item. Returns the resolved z (for codebook growth
        by the caller) if confirmed normal, else None (treat as the
        anomaly it looked like it might be)."""
        if not self._pending_queue:
            return None
        z = self._pending_queue.pop(0)
        self._consecutive_novel_consistent = 0
        return z if confirmed_normal else None

    @property
    def pending_count(self):
        return len(self._pending_queue)


def grow_codebook(memory, new_prototype: torch.Tensor, replace_least_used: bool = True):
    """Online codebook growth (Sec 5, case 5): once a novel-but-relation-
    consistent regime is confirmed normal, add it as a new prototype so it
    stops re-triggering PENDING_LABEL every time it recurs. `memory` is a
    DiscretePrototypicalMemory (gdn_memory_model.py). Since the codebook
    has a fixed capacity M, this replaces the least-used existing
    prototype (by `usage_count`) rather than growing M unboundedly --
    matches Sec 7's open question about "字典长度的衰减策略," resolved here
    with the simplest reasonable policy (LRU-by-frequency eviction)."""
    with torch.no_grad():
        if replace_least_used:
            idx = int(memory.usage_count.argmin())
            memory.codebook[idx] = new_prototype.to(memory.codebook.device)
            memory.usage_count[idx] = 0.0
        else:
            raise NotImplementedError("fixed-capacity codebook only supports replacement, not true growth")
