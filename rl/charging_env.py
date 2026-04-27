# rl/charging_env.py
# OpenAI Gym-compatible EV Charging Environment
# Supports ALL stations in a single training run

import gymnasium as gym
import numpy as np
import yaml
from models.queue_model import EVPriorityQueue, EV
from models.soc_model import SOCModel
from models.grid_model import GridModel
from models.tariff_model import TariffModel
from data.arrival_generator import ArrivalGenerator


class EVChargingEnv(gym.Env):
    """
    EV Charging Scheduling Environment for DQN training.
    When station_idx = -1, randomly picks a station each episode
    so the agent learns to generalise across all stations.

    State:  [soc_1..N, urgency_1..N, deadline_1..N,
             tariff_normalized, grid_utilization,
             timestep_normalized, n_chargers_normalized,
             transformer_normalized]

    Action: 0 = Hold / Skip
            1-8 = Charge top-N urgent EVs (respects station charger count)
            9 = Charge cheapest (lowest remaining energy needed)

    """

    metadata = {"render_modes": []}

    # Fixed slot size so state dim is constant across all stations
    N_EV_SLOTS = 6

    def __init__(self, config_path: str = "config/nh44_config.yaml",
                 station_idx: int = -1):
        """
        Args:
            station_idx: -1 = random station per episode (recommended for training)
                         0-4 = fixed station
        """
        super().__init__()

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.all_stations = self.config["stations"]
        self.station_idx = station_idx
        self.timescale_min = self.config["simulation"]["timescale"]
        self.timescale_hr  = self.timescale_min / 60.0
        self.sim_length    = self.config["simulation"]["simulation_length"]

        # State: [soc, urgency, deadline] x N_EV_SLOTS
        #        + tariff + grid_util + timestep
        #        + n_chargers_norm + transformer_norm  (station identity)
        n_state = self.N_EV_SLOTS * 3 + 5
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(n_state,),
            dtype=np.float32
        )

        # Actions:
        # 0: Hold
        # 1-8: Charge top-1 to top-8
        # 9: Charge cheapest
        self.n_actions = 10
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # Placeholders — set properly in reset()
        self.station      = None
        self.n_chargers   = None
        self.charger_kw   = None
        self.soc_model    = None
        self.grid_model   = None
        self.tariff_model = TariffModel()
        self.arrival_gen  = None
        self.queue        = None
        self.timestep     = 0
        self.episode_stats = {}

    def _load_station(self, idx: int):
        """Initialise sub-models for a specific station."""
        s = self.all_stations[idx]
        self.station    = s
        self.n_chargers = s["n_chargers"]
        self.charger_kw = s["charger_kw"]
        self.soc_model  = SOCModel(timescale_min=self.timescale_min)
        self.grid_model = GridModel(transformer_kva=s["transformer_kva"])
        self.arrival_gen = ArrivalGenerator(
            station_id=s["id"],
            n_chargers=s["n_chargers"],
            timescale_min=self.timescale_min,
            station_charger_kw=s["charger_kw"],
            station_charger_type=s["charger_type"],
            config_arrival=self.config.get("arrival", {}),
            ev_specs=self.config.get("ev_specs", None)
        )

        self.queue = EVPriorityQueue()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick station: random during training, fixed during eval
        if self.station_idx == -1:
            idx = np.random.randint(0, len(self.all_stations))
        else:
            idx = self.station_idx

        self._load_station(idx)
        self.timestep = 0
        self.episode_stats = {
            "station":          self.station["name"],
            "evs_served":       0,
            "deadlines_missed": 0,
            "total_cost_rs":    0.0,
            "grid_violations":  0
        }
        return self._get_obs(), {}

    def step(self, action: int):
        hour = self.timestep * self.timescale_hr % 24

        # 1. New EV arrivals
        for ev in self.arrival_gen.get_arrivals(self.timestep):
            self.queue.add(ev, self.timestep)

        # 2. Decode action → charging plan
        charging_plan = self._decode_action(action, hour)

        # 3. Apply grid constraints
        if charging_plan:
            powers   = list(charging_plan.values())
            throttled = self.grid_model.throttle(powers, hour)
            ev_ids   = list(charging_plan.keys())
            charging_plan = dict(zip(ev_ids, throttled))

        feasible, _ = self.grid_model.check_feasibility(
            list(charging_plan.values()), hour
        )
        if not feasible:
            self.episode_stats["grid_violations"] += 1

        # 4. Update SOC
        cost = 0.0
        for ev_id, power in charging_plan.items():
            ev = self.queue.get_ev(ev_id)
            if ev is None:
                continue
            sm = SOCModel(
                capacity_kwh=ev.battery_kwh,
                efficiency=ev.efficiency,
                timescale_min=self.timescale_min
            )
            ev.soc = sm.update(ev.soc, power)
            cost += self.tariff_model.charging_cost(
                power, self.timescale_hr, hour
            )
        self.episode_stats["total_cost_rs"] += cost

        # 5. Handle departures
        departed = [ev for ev in self.queue.get_top_n(self.queue.size())
                    if ev.steps_remaining(self.timestep) <= 0]
        for ev in departed:
            if ev.soc >= 0.5:
                self.episode_stats["evs_served"] += 1
            else:
                self.episode_stats["deadlines_missed"] += 1
            self.queue.remove(ev.ev_id)

        # 6. Reward
        from rl.reward import compute_reward
        reward = compute_reward(
            charging_plan=charging_plan,
            departed_evs=departed,
            cost=cost,
            grid_violated=not feasible,
            hour=hour,
            queue=self.queue,
            timestep=self.timestep,
            tariff_model=self.tariff_model
        )

        self.queue.update_priorities(self.timestep)
        self.timestep += 1
        done = self.timestep >= self.sim_length
        state = self._get_obs()
        return state, reward, done, False, self.episode_stats

    def get_valid_actions(self):
        """
        Returns a list of valid action indices for this specific station.
        Action 0 (Hold) and 9 (Charge cheapest) are always valid.
        Actions 1 to N are valid where N is the station's max chargers.
        """
        valid_actions = [0, 9]
        n_chargers = self.station.get("n_chargers", 4)
        # Add actions 1 to n_chargers (up to max action space of 8)
        valid_actions.extend(list(range(1, min(n_chargers + 1, 9))))
        return valid_actions

    def _decode_action(self, action: int, hour: float) -> dict:
        available_kw = self.grid_model.available_ev_capacity(hour)

        if action == 0:
            return {}  # hold

        if action == 9:
            # Charge cheapest EV (least energy needed)
            top_evs = self.queue.get_top_n(self.queue.size())
            if not top_evs:
                return {}
            cheapest = min(top_evs,
                           key=lambda ev: (ev.target_soc - ev.soc)
                                          * ev.battery_kwh)
            power = min(cheapest.max_charge_kw, self.charger_kw, available_kw)
            return {cheapest.ev_id: power} if power > 0 else {}

        # Charge top-(action) most urgent EVs
        n_to_charge = action
        top_evs = self.queue.get_top_n(n_to_charge)
        plan, remaining = {}, available_kw
        for ev in top_evs:
            if remaining <= 0:
                break
            power = min(ev.max_charge_kw, self.charger_kw, remaining)
            if power > 0:
                plan[ev.ev_id] = power
                remaining -= power
        return plan

    def _get_obs(self) -> np.ndarray:
        hour = self.timestep * self.timescale_hr % 24

        queue_state = self.queue.get_state_vector(
            self.N_EV_SLOTS, self.timestep
        )

        max_chargers    = max(s["n_chargers"]    for s in self.all_stations)
        max_transformer = max(s["transformer_kva"] for s in self.all_stations)

        extra = np.array([
            self.tariff_model.get_normalized_tariff(hour),
            self.grid_model.utilization([], hour),
            self.timestep / self.sim_length,
            self.n_chargers / max_chargers,          # station identity
            self.station["transformer_kva"] / max_transformer
        ], dtype=np.float32)

        return np.concatenate([queue_state, extra])
