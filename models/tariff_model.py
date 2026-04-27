# models/tariff_model.py
# BESCOM Time-of-Day Tariff Model
# Rates from BESCOM Tariff Order 2024-25 (Rs/kWh)

class TariffModel:
    """
    BESCOM time-of-day tariff for EV charging stations.
    Used to compute charging cost and shape RL reward.
    """

    TARIFF_SCHEDULE = [
        # (start_hour, end_hour, price_rs_per_kwh, label)
        (0,  6,  5.50, "night"),        # cheap
        (6,  9,  6.50, "morning"),      # medium
        (9,  17, 5.90, "afternoon"),    # medium-low
        (17, 22, 8.25, "evening_peak"), # expensive
        (22, 24, 5.50, "late_night"),   # cheap
    ]

    def get_tariff(self, hour: float) -> float:
        """Return tariff in Rs/kWh for given hour (0-24)."""
        for start, end, price, _ in self.TARIFF_SCHEDULE:
            if start <= hour < end:
                return price
        return 6.0  # fallback

    def get_label(self, hour: float) -> str:
        """Return human-readable tariff label."""
        for start, end, _, label in self.TARIFF_SCHEDULE:
            if start <= hour < end:
                return label
        return "unknown"

    def get_normalized_tariff(self, hour: float) -> float:
        """Return tariff normalized 0-1 for use as RL state feature."""
        price = self.get_tariff(hour)
        min_price = 5.50
        max_price = 8.25
        return (price - min_price) / (max_price - min_price)

    def charging_cost(self, power_kw: float,
                      duration_hr: float, hour: float) -> float:
        """Compute cost in Rs for a charging action."""
        return power_kw * duration_hr * self.get_tariff(hour)

    def is_peak(self, hour: float) -> bool:
        """True if current hour is evening peak tariff."""
        return 17 <= hour < 22

    def is_cheap(self, hour: float) -> bool:
        """True if current hour is cheap off-peak tariff."""
        return hour < 6 or hour >= 22

    def timestep_to_hour(self, timestep: int,
                         timescale_min: int = 15) -> float:
        """Convert simulation timestep to hour of day."""
        return (timestep * timescale_min) / 60.0

    def get_all_rates(self) -> dict:
        """Return full tariff schedule as dict for display."""
        return {label: price for _, _, price, label
                in self.TARIFF_SCHEDULE}


if __name__ == "__main__":
    tariff = TariffModel()
    print("BESCOM Time-of-Day Tariff Schedule:")
    print("-" * 35)
    for hour in [0, 7, 12, 18, 23]:
        print(f"  {hour:02d}:00  Rs {tariff.get_tariff(hour):.2f}/kWh"
              f"  [{tariff.get_label(hour)}]")
    print()
    cost = tariff.charging_cost(7.2, 0.5, 18)
    print(f"Example: 7.2 kW for 30 min at 6 PM = Rs {cost:.2f}")
