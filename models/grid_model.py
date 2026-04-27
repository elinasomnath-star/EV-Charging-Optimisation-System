# models/grid_model.py
# Indian Grid Constraint Model
# Simulates transformer limits for highway stations

class GridModel:
    """
    Rule-based transformer safety model.
    Ensures: Total_EV_Load + Base_Load <= Transformer_Limit
    """

    def __init__(self, transformer_kva: float = 63,
                 power_factor: float = 0.95):
        self.transformer_kva = transformer_kva
        self.transformer_kw = transformer_kva * power_factor

    def get_base_load(self, hour: float) -> float:
        """
        Rule-based base load (kW) for highway station by hour.
        Includes station lighting, amenities, office equipment.
        """
        if 6 <= hour < 9:   return 8.0   # morning
        if 9 <= hour < 17:  return 6.0   # daytime
        if 17 <= hour < 22: return 10.0  # evening peak
        return 4.0                        # night

    def available_ev_capacity(self, hour: float) -> float:
        """Return max kW available for EV charging at given hour."""
        return self.transformer_kw - self.get_base_load(hour)

    def check_feasibility(self, charging_powers: list,
                          hour: float) -> tuple:
        """
        Check if proposed charging plan is within grid limits.
        Returns: (feasible: bool, headroom_kw: float)
        """
        base = self.get_base_load(hour)
        total_ev = sum(charging_powers)
        total = base + total_ev
        feasible = total <= self.transformer_kw
        headroom = self.transformer_kw - total
        return feasible, headroom

    def throttle(self, charging_powers: list,
                 hour: float) -> list:
        """
        Scale down charging powers proportionally
        if grid limit is exceeded.
        CN concept: congestion control
        """
        feasible, _ = self.check_feasibility(charging_powers, hour)
        if feasible:
            return charging_powers

        available = self.available_ev_capacity(hour)
        total_requested = sum(charging_powers)
        if total_requested <= 0:
            return charging_powers

        scale = available / total_requested
        return [min(p * scale, p) for p in charging_powers]

    def load_balance(self, charging_powers: list,
                     hour: float) -> list:
        """
        Distribute available capacity evenly across active EVs.
        CN concept: load balancing
        """
        n_active = sum(1 for p in charging_powers if p > 0)
        if n_active == 0:
            return charging_powers

        available = self.available_ev_capacity(hour)
        per_ev = available / n_active
        return [min(p, per_ev) if p > 0 else 0
                for p in charging_powers]

    def utilization(self, charging_powers: list,
                    hour: float) -> float:
        """Return transformer utilization as 0-1."""
        base = self.get_base_load(hour)
        total = base + sum(charging_powers)
        return total / self.transformer_kw


if __name__ == "__main__":
    grid = GridModel(transformer_kva=63)
    powers = [7.2, 7.2, 7.2, 7.2]
    hour = 18.0

    feasible, headroom = grid.check_feasibility(powers, hour)
    print(f"4 EVs at 7.2 kW each at 6 PM:")
    print(f"  Feasible: {feasible}, Headroom: {headroom:.1f} kW")

    throttled = grid.throttle(powers, hour)
    print(f"  After throttling: {[round(p,2) for p in throttled]}")
    print(f"  Utilization: {grid.utilization(throttled, hour):.1%}")
