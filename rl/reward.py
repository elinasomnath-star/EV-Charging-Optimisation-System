# rl/reward.py
# Reward Function for EVOCS DQN Scheduler

from models.tariff_model import TariffModel
from models.queue_model import EVPriorityQueue


def compute_reward(charging_plan: dict,
                   departed_evs: list,
                   cost: float,
                   grid_violated: bool,
                   hour: float,
                   queue: EVPriorityQueue,
                   timestep: int,
                   tariff_model: TariffModel) -> float:
    """
    Reward signal for DQN agent.

    Positive rewards:
    - EV charged to target SOC before deadline
    - Charging during cheap tariff slot

    Negative rewards:
    - EV missed deadline (left with low SOC)
    - Grid constraint violated
    - Charging during peak tariff unnecessarily
    - EV waiting with high urgency but not being charged
    """
    reward = 0.0

    # +++ Reward: Proportional to SOC gained
    for ev in departed_evs:
        soc_gained = max(0.0, ev.soc - ev.arrival_soc)
        reward += 10.0 * soc_gained
        if ev.soc >= ev.target_soc:
            reward += 5.0          # bonus for fully serving EV

    # +++ Reward: charging during cheap tariff
    if charging_plan and tariff_model.is_cheap(hour):
        reward += 1.0

    # --- Penalty: charging cost (scaled down heavily to ensure net positive episode rewards)
    reward -= cost * 0.001

    # --- Penalty: grid violation
    if grid_violated:
        reward -= 3.0

    # --- Penalty: charging during expensive peak unnecessarily
    if charging_plan and tariff_model.is_peak(hour):
        top_evs = queue.get_top_n(len(charging_plan))
        avg_urgency = sum(ev.urgency(timestep)
                          for ev in top_evs) / max(1, len(top_evs))
        if avg_urgency < 0.4:
            reward -= 0.5 * len(charging_plan)

    # --- Penalty: cost (small, to encourage cost efficiency)
    reward -= cost * 0.02

    # --- Penalty: high-urgency EVs sitting uncharged
    if queue.size() > 0:
        top_evs = queue.get_top_n(min(3, queue.size()))
        for ev in top_evs:
            urgency = ev.urgency(timestep)
            if urgency > 0.8 and ev.ev_id not in charging_plan:
                reward -= 1.0

    # --- Penalty: doing nothing when urgent EV waiting
    if not charging_plan and queue.size() > 0:
        top_evs = queue.get_top_n(1)
        if top_evs and top_evs[0].urgency(timestep) > 0.5:
            reward -= 0.5

    return float(reward)
