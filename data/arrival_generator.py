# data/arrival_generator.py
# EV Arrival Generator for NH-44 Highway Stations
# Simulates highway traffic patterns (Bangalore → Salem corridor)

import numpy as np
from models.queue_model import EV

# Indian EV specs from NH-44 app survey
EV_TYPES = [
    {"name": "Tata_Nexon_EV",  "battery_kwh": 40.5,
     "max_kw": 7.2, "max_dc_kw": 30.0, "efficiency": 0.92, "weight": 0.35},
    {"name": "Tata_Tiago_EV",  "battery_kwh": 24.0,
     "max_kw": 3.3, "max_dc_kw": 20.0, "efficiency": 0.91, "weight": 0.25},
    {"name": "MG_ZS_EV",       "battery_kwh": 50.3,
     "max_kw": 7.4, "max_dc_kw": 50.0, "efficiency": 0.93, "weight": 0.20},
    {"name": "Ola_S1_Pro",     "battery_kwh": 4.0,
     "max_kw": 1.5, "max_dc_kw": 1.5,  "efficiency": 0.90, "weight": 0.20},
]
EV_WEIGHTS = [ev["weight"] for ev in EV_TYPES]


class ArrivalGenerator:
    """
    Generates EV arrivals based on NH-44 highway traffic patterns.
    Peak hours: morning (7-9 AM) and evening (5-8 PM).
    """

    # NH-44 highway: higher traffic on weekday mornings and evenings
    HOURLY_ARRIVAL_RATE = {
        0: 0.5,  1: 0.3,  2: 0.2,  3: 0.2,
        4: 0.3,  5: 0.5,  6: 1.5,  7: 3.5,
        8: 4.0,  9: 2.5,  10: 1.5, 11: 1.5,
        12: 2.0, 13: 1.5, 14: 1.5, 15: 2.0,
        16: 3.0, 17: 4.5, 18: 5.0, 19: 4.0,
        20: 3.0, 21: 2.0, 22: 1.0, 23: 0.5
    }

    def __init__(self, station_id: int = 1,
                 n_chargers: int = 4,
                 timescale_min: int = 15,
                 station_charger_kw: float = 7.2,
                 station_charger_type: str = "AC_Type2",
                 config_arrival: dict = None,
                 random_seed: int = 42):
        self.station_id = station_id
        self.n_chargers = n_chargers
        self.timescale_hr = timescale_min / 60.0
        self.station_charger_kw = station_charger_kw
        self.station_charger_type = station_charger_type
        
        if config_arrival is None:
            config_arrival = {}
        self.min_soc = config_arrival.get("min_soc_on_arrival", 0.15)
        self.max_soc = config_arrival.get("max_soc_on_arrival", 0.65)
        self.min_stay = config_arrival.get("min_stay_hours", 1.0)
        self.max_stay = config_arrival.get("max_stay_hours", 3.0)
        self.target_soc = config_arrival.get("target_soc", 0.80)

        self.rng = np.random.default_rng(random_seed)
        self._ev_counter = 0

    def _timestep_to_hour(self, timestep: int) -> int:
        return int((timestep * self.timescale_hr)) % 24

    def _sample_ev_type(self) -> dict:
        idx = self.rng.choice(len(EV_TYPES), p=EV_WEIGHTS)
        return EV_TYPES[idx]

    def get_arrivals(self, timestep: int) -> list:
        """
        Sample EV arrivals for current timestep.
        Returns list of EV objects to add to queue.
        """
        hour = self._timestep_to_hour(timestep)
        rate_per_hour = self.HOURLY_ARRIVAL_RATE.get(hour, 1.0)
        rate_per_step = rate_per_hour * self.timescale_hr

        # Poisson arrivals
        n_arrivals = self.rng.poisson(rate_per_step)

        arrivals = []
        for _ in range(n_arrivals):
            ev_type = self._sample_ev_type()

            # Highway EVs arrive with low-medium SOC
            # (they have been driving, need top-up)
            soc = float(self.rng.uniform(self.min_soc, self.max_soc))

            # Stay time based on config
            stay_hr = float(self.rng.uniform(self.min_stay, self.max_stay))
            stay_steps = max(2, int(stay_hr / self.timescale_hr))
            deadline_step = timestep + stay_steps

            # Determine charging power limit based on station type
            if "DC" in self.station_charger_type:
                ev_max_kw = ev_type.get("max_dc_kw", ev_type["max_kw"])
            else:
                ev_max_kw = ev_type["max_kw"]
            
            effective_max_kw = min(self.station_charger_kw, ev_max_kw)

            self._ev_counter += 1
            ev = EV(
                ev_id=self._ev_counter,
                soc=soc,
                target_soc=self.target_soc,
                battery_kwh=ev_type["battery_kwh"],
                max_charge_kw=effective_max_kw,
                efficiency=ev_type["efficiency"],
                arrival_step=timestep,
                deadline_step=deadline_step,
                ev_type=ev_type["name"],
                arrival_soc=soc
            )
            arrivals.append(ev)

        return arrivals

    def reset(self):
        """Reset counter for new episode."""
        self._ev_counter = 0


if __name__ == "__main__":
    gen = ArrivalGenerator(station_id=1, n_chargers=4)
    total = 0
    print("Simulated EV arrivals over 24 hours (NH-44 Station 1):")
    for step in range(96):
        arrivals = gen.get_arrivals(step)
        hour = step * 15 // 60
        minute = (step * 15) % 60
        if arrivals:
            for ev in arrivals:
                total += 1
                print(f"  {hour:02d}:{minute:02d}  EV{ev.ev_id:03d}"
                      f"  {ev.ev_type:<15}"
                      f"  SOC={ev.soc:.0%}"
                      f"  Stay={ev.deadline_step - step} steps")
    print(f"\nTotal arrivals: {total} EVs in 24 hours")
