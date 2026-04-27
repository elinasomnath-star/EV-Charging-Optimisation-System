# baselines/schedulers.py
# Baseline Algorithms for comparison with DQN
# 1. Greedy (highest urgency first)
# 2. First Come First Served (FCFS)

import numpy as np
from models.queue_model import EVPriorityQueue


class GreedyScheduler:
    """
    Greedy algorithm: always charge the most urgent EV first.
    Urgency = (required SOC - current SOC) / time left
    DAA concept: greedy choice property
    """

    def __init__(self, n_chargers: int, max_kw: float = 7.2):
        self.n_chargers = n_chargers
        self.max_kw = max_kw

    def schedule(self, queue: EVPriorityQueue,
                 available_kw: float,
                 current_step: int) -> dict:
        """
        Returns: {ev_id: power_kw} for this timestep
        """
        top_evs = queue.get_top_n(self.n_chargers)
        schedule = {}
        remaining_kw = available_kw

        # Sort by urgency descending
        sorted_evs = sorted(
            top_evs,
            key=lambda ev: ev.urgency(current_step),
            reverse=True
        )

        for ev in sorted_evs:
            if remaining_kw <= 0:
                break
            power = min(ev.max_charge_kw, remaining_kw)
            if power > 0:
                schedule[ev.ev_id] = power
                remaining_kw -= power

        return schedule


class FCFSScheduler:
    """
    First Come First Served: simplest baseline.
    What current apps (Bolt.Earth etc) do.
    """

    def __init__(self, n_chargers: int, max_kw: float = 7.2):
        self.n_chargers = n_chargers
        self.max_kw = max_kw

    def schedule(self, queue: EVPriorityQueue,
                 available_kw: float,
                 current_step: int) -> dict:
        """Sort by arrival time (earliest arrival first)."""
        top_evs = queue.get_top_n(self.n_chargers * 2)
        schedule = {}
        remaining_kw = available_kw

        sorted_evs = sorted(top_evs,
                            key=lambda ev: ev.arrival_step)

        for ev in sorted_evs[:self.n_chargers]:
            if remaining_kw <= 0:
                break
            power = min(ev.max_charge_kw, remaining_kw)
            if power > 0:
                schedule[ev.ev_id] = power
                remaining_kw -= power

        return schedule
