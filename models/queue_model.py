# models/queue_model.py
# Priority Queue for EV Charging
# CN Concepts: Queueing theory, priority scheduling

import heapq
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class EV:
    """Represents one EV at the charging station."""
    ev_id: int
    soc: float                    # current SOC (0-1)
    target_soc: float = 0.80      # target SOC at departure
    battery_kwh: float = 40.5     # battery capacity
    max_charge_kw: float = 7.2    # max AC charge rate
    efficiency: float = 0.92
    arrival_step: int = 0
    deadline_step: int = 8        # steps until departure
    ev_type: str = "Tata_Nexon"
    arrival_soc: float = 0.0

    def steps_remaining(self, current_step: int) -> int:
        return max(0, self.deadline_step - current_step)

    def urgency(self, current_step: int,
                timescale_hr: float = 0.25) -> float:
        """
        Urgency score = (needed SOC) / (time available × charge rate)
        Higher = more urgent = higher queue priority
        """
        needed = max(0, self.target_soc - self.soc)
        steps_left = self.steps_remaining(current_step)
        time_left = max(0.01, steps_left * timescale_hr)
        max_achievable = (self.max_charge_kw * time_left
                          * self.efficiency) / self.battery_kwh
        if needed <= 0:
            return 0.0
        if steps_left <= 1:
            return 1.0
        return float(np.clip(needed / max(needed, max_achievable),
                             0.0, 1.0))

    def needs_charging(self) -> bool:
        return self.soc < self.target_soc


@dataclass(order=True)
class PriorityEntry:
    """Heap entry: lower priority value = higher urgency = served first."""
    priority: float
    ev: EV = field(compare=False)


class EVPriorityQueue:
    """
    Min-heap priority queue for EVs.
    Highest urgency EV served first.
    CN concept: priority queuing (similar to QoS priority queues)
    """

    def __init__(self):
        self._heap = []
        self._ev_map = {}       # ev_id → EV for fast lookup
        self._counter = 0       # tiebreaker

    def add(self, ev: EV, current_step: int):
        """Add EV to queue with urgency-based priority."""
        urgency = ev.urgency(current_step)
        # Negate urgency so highest urgency = lowest heap value
        priority = -urgency
        entry = PriorityEntry(priority=priority, ev=ev)
        heapq.heappush(self._heap, entry)
        self._ev_map[ev.ev_id] = ev

    def update_priorities(self, current_step: int):
        """Recompute all urgency scores (call each timestep)."""
        evs = [entry.ev for entry in self._heap]
        self._heap = []
        for ev in evs:
            self.add(ev, current_step)

    def get_top_n(self, n: int) -> list:
        """Return top-n most urgent EVs without removing them."""
        sorted_entries = sorted(self._heap,
                                key=lambda e: e.priority)
        return [entry.ev for entry in sorted_entries[:n]]

    def remove(self, ev_id: int):
        """Remove EV from queue (on departure or charge complete)."""
        self._heap = [e for e in self._heap
                      if e.ev.ev_id != ev_id]
        heapq.heapify(self._heap)
        self._ev_map.pop(ev_id, None)

    def get_ev(self, ev_id: int) -> Optional[EV]:
        return self._ev_map.get(ev_id)

    def size(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def get_state_vector(self, n_slots: int,
                         current_step: int) -> np.ndarray:
        """
        Return fixed-size state vector for RL:
        [soc_1, urgency_1, deadline_1, soc_2, urgency_2, deadline_2, ...]
        Padded with zeros if fewer than n_slots EVs present.
        """
        top_evs = self.get_top_n(n_slots)
        state = np.zeros(n_slots * 3, dtype=np.float32)
        for i, ev in enumerate(top_evs[:n_slots]):
            state[i*3]   = ev.soc
            state[i*3+1] = ev.urgency(current_step)
            state[i*3+2] = ev.steps_remaining(current_step) / 96.0
        return state


if __name__ == "__main__":
    queue = EVPriorityQueue()

    ev1 = EV(ev_id=1, soc=0.20, deadline_step=4)   # very urgent
    ev2 = EV(ev_id=2, soc=0.60, deadline_step=12)  # not urgent
    ev3 = EV(ev_id=3, soc=0.30, deadline_step=6)   # moderate

    for ev in [ev1, ev2, ev3]:
        queue.add(ev, current_step=0)

    print("Queue priority order (highest urgency first):")
    for ev in queue.get_top_n(3):
        u = ev.urgency(0)
        print(f"  EV{ev.ev_id}: SOC={ev.soc:.0%}, "
              f"deadline={ev.deadline_step}, urgency={u:.3f}")
