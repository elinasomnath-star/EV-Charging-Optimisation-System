# iot/guidance_system.py
# IoT Central Guidance System for NH-44 EVs
# Computes optimal charging station based on distance, SOC, and queue length.

class GuidanceSystem:
    def __init__(self, driving_efficiency_kwh_per_km: float = 0.15,
                 min_arrival_soc: float = 0.05):
        self.driving_eff = driving_efficiency_kwh_per_km
        self.min_arrival_soc = min_arrival_soc

    def recommend_station(self, ev_soc: float, ev_battery_kwh: float,
                          current_km: float, station_statuses: list) -> dict:
        """
        Recommend the best station for an EV.
        station_statuses: list of dicts with keys:
            id, name, location_km, n_chargers, charger_kw, queue_size
        Returns:
            dict containing the recommended station and arrival SOC, or None.
        """
        best_station = None
        best_score = -float('inf')
        best_arrival_soc = 0.0

        for st in station_statuses:
            dist = st["location_km"] - current_km
            
            # Can't go backwards
            if dist < 0:
                continue
            
            # Check reachability (allow all to route, handle low SOC later)
            energy_needed = dist * self.driving_eff
            soc_needed = energy_needed / ev_battery_kwh
            arrival_soc = ev_soc - soc_needed
            
            # Calculate Score
            # 1. Closer is slightly preferred
            dist_score = -0.1 * dist
            
            # 2. Shorter queue relative to chargers is highly preferred
            queue_score = -5.0 * (st["queue_size"] / max(1, st["n_chargers"]))
            
            # 3. Faster charging is preferred
            speed_score = 0.5 * st["charger_kw"]
            
            score = dist_score + queue_score + speed_score
            
            if score > best_score:
                best_score = score
                best_station = st
                best_arrival_soc = arrival_soc
                
        if best_station:
            return {
                "station": best_station,
                "arrival_soc": best_arrival_soc,
                "score": best_score,
                "distance": best_station["location_km"] - current_km
            }
        return None
