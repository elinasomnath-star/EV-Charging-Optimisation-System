# models/soc_model.py
# Battery SOC Model for EV Charging Simulation

import numpy as np

class SOCModel:
    """
    Tracks battery State of Charge for each EV.
    SOC(t+1) = SOC(t) + (P × Δt × η) / C
    """

    def __init__(self, capacity_kwh: float = 40.5,
                 efficiency: float = 0.92,
                 timescale_min: float = 15):
        self.capacity_kwh = capacity_kwh
        self.efficiency = efficiency
        self.timescale_hr = timescale_min / 60.0

    def update(self, soc: float, power_kw: float) -> float:
        """
        Update SOC for one timestep.
        Args:
            soc: current SOC (0.0 to 1.0)
            power_kw: charging power in kW (0 = not charging)
        Returns:
            new SOC (0.0 to 1.0)
        """
        delta = (power_kw * self.timescale_hr * self.efficiency) \
                / self.capacity_kwh
        return float(np.clip(soc + delta, 0.0, 1.0))

    def urgency_score(self, soc: float,
                      deadline_steps: int,
                      target_soc: float = 0.80) -> float:
        """
        Urgency = how much charging is needed vs time available.
        High urgency → EV needs to charge soon.
        Range: 0.0 (no urgency) to 1.0 (critical)

        Used by priority queue and RL state.
        """
        needed_kwh = max(0, (target_soc - soc) * self.capacity_kwh)
        time_available_hr = max(0.01, deadline_steps * self.timescale_hr)
        max_possible_kwh = (self.capacity_kwh * 7.2
                            * time_available_hr
                            * self.efficiency
                            / self.capacity_kwh)
        if needed_kwh <= 0:
            return 0.0  # already charged
        if time_available_hr <= self.timescale_hr:
            return 1.0  # last chance
        return float(np.clip(needed_kwh / max(needed_kwh,
                             max_possible_kwh), 0.0, 1.0))

    def time_to_full(self, soc: float,
                     power_kw: float,
                     target_soc: float = 0.80) -> float:
        """
        Estimate hours needed to reach target SOC at given power.
        """
        if power_kw <= 0 or soc >= target_soc:
            return 0.0
        needed_kwh = (target_soc - soc) * self.capacity_kwh
        return needed_kwh / (power_kw * self.efficiency)

    def will_miss_deadline(self, soc: float,
                           deadline_steps: int,
                           power_kw: float,
                           target_soc: float = 0.80) -> bool:
        """
        Check if EV will miss target SOC by deadline
        even if charged continuously at given power.
        """
        hours_needed = self.time_to_full(soc, power_kw, target_soc)
        hours_available = deadline_steps * self.timescale_hr
        return hours_needed > hours_available


if __name__ == "__main__":
    soc_model = SOCModel(capacity_kwh=40.5, efficiency=0.92)

    soc = 0.30
    print(f"Initial SOC: {soc:.0%}")
    for step in range(8):
        soc = soc_model.update(soc, power_kw=7.2)
        print(f"  Step {step+1} (15 min): SOC = {soc:.1%}")

    print()
    urgency = soc_model.urgency_score(0.20, deadline_steps=4)
    print(f"Urgency (SOC=20%, 4 steps left): {urgency:.2f}")
    urgency2 = soc_model.urgency_score(0.70, deadline_steps=8)
    print(f"Urgency (SOC=70%, 8 steps left): {urgency2:.2f}")
