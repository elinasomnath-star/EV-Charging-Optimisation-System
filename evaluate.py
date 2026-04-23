# evaluate.py
# Evaluation: DQN vs Greedy vs EDF vs FCFS
# Generates comparison metrics for EL report

import numpy as np
import yaml
import torch
from rl.charging_env import EVChargingEnv
from rl.train import DQNAgent
from baselines.greedy import GreedyScheduler, EDFScheduler, FCFSScheduler
from models.queue_model import EVPriorityQueue
from data.arrival_generator import ArrivalGenerator
from models.grid_model import GridModel
from models.tariff_model import TariffModel
from models.soc_model import SOCModel


def run_baseline_episode(scheduler, config_path, station_idx=0):
    """Run one episode with a baseline scheduler."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    station = config["stations"][station_idx]
    timescale_min = config["simulation"]["timescale"]
    timescale_hr = timescale_min / 60.0
    sim_length = config["simulation"]["simulation_length"]

    queue = EVPriorityQueue()
    arrival_gen = ArrivalGenerator(
        station_id=station["id"],
        n_chargers=station["n_chargers"],
        timescale_min=timescale_min
    )
    grid = GridModel(transformer_kva=station["transformer_kva"])
    tariff = TariffModel()

    stats = {
        "evs_served": 0,
        "deadlines_missed": 0,
        "total_cost_rs": 0.0,
        "grid_violations": 0,
        "waiting_steps": 0
    }

    for step in range(sim_length):
        hour = step * timescale_hr % 24

        # Arrivals
        for ev in arrival_gen.get_arrivals(step):
            queue.add(ev, step)

        # Schedule
        available_kw = grid.available_ev_capacity(hour)
        plan = scheduler.schedule(queue, available_kw, step)

        # Apply grid check
        if plan:
            feasible, _ = grid.check_feasibility(
                list(plan.values()), hour
            )
            if not feasible:
                stats["grid_violations"] += 1

        # Update SOC
        for ev_id, power in plan.items():
            ev = queue.get_ev(ev_id)
            if ev is None:
                continue
            soc_model = SOCModel(
                capacity_kwh=ev.battery_kwh,
                efficiency=ev.efficiency,
                timescale_min=timescale_min
            )
            ev.soc = soc_model.update(ev.soc, power)
            stats["total_cost_rs"] += tariff.charging_cost(
                power, timescale_hr, hour
            )

        # Count waiting EVs
        stats["waiting_steps"] += max(0,
            queue.size() - len(plan))

        # Departures
        for ev in queue.get_top_n(queue.size()):
            if ev.steps_remaining(step) <= 0:
                if ev.soc >= ev.target_soc:
                    stats["evs_served"] += 1
                else:
                    stats["deadlines_missed"] += 1
                queue.remove(ev.ev_id)

        queue.update_priorities(step)

    return stats


def run_dqn_episode(agent, env):
    """Run one episode with DQN agent."""
    state, _ = env.reset()
    while True:
        action = agent.act(state, training=False)
        state, _, done, _, info = env.step(action)
        if done:
            break
    return info


def evaluate(config_path: str = "config/nh44_config.yaml",
             model_path: str = "models/saved/evocs_best.pt",
             station_idx: int = 0,
             n_episodes: int = 100):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    station = config["stations"][station_idx]
    n_chargers = station["n_chargers"]
    charger_kw = station["charger_kw"]

    print(f"\nEVOCS Evaluation — {station['name']}")
    print(f"Running {n_episodes} episodes per algorithm")
    print("=" * 60)

    # Baseline schedulers
    schedulers = {
        "FCFS (Apps)": FCFSScheduler(n_chargers, charger_kw),
        "Greedy":      GreedyScheduler(n_chargers, charger_kw),
        "EDF":         EDFScheduler(n_chargers, charger_kw),
    }

    results = {}

    # Run baselines
    for name, scheduler in schedulers.items():
        ep_stats = [
            run_baseline_episode(scheduler, config_path, station_idx)
            for _ in range(n_episodes)
        ]
        results[name] = {
            "evs_served":       np.mean([s["evs_served"] for s in ep_stats]),
            "deadlines_missed": np.mean([s["deadlines_missed"] for s in ep_stats]),
            "total_cost_rs":    np.mean([s["total_cost_rs"] for s in ep_stats]),
            "grid_violations":  np.mean([s["grid_violations"] for s in ep_stats]),
        }

    # Run DQN
    env = EVChargingEnv(config_path=config_path,
                        station_idx=station_idx)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, config["dqn"])
    agent.load(model_path)
    agent.epsilon = 0.0  # no exploration during eval

    dqn_stats = [run_dqn_episode(agent, env)
                 for _ in range(n_episodes)]
    results["DQN (EVOCS)"] = {
        "evs_served":       np.mean([s["evs_served"] for s in dqn_stats]),
        "deadlines_missed": np.mean([s["deadlines_missed"] for s in dqn_stats]),
        "total_cost_rs":    np.mean([s["total_cost_rs"] for s in dqn_stats]),
        "grid_violations":  np.mean([s["grid_violations"] for s in dqn_stats]),
    }

    # Print results table
    print(f"\n{'Algorithm':<18} {'Served':>8} {'Missed':>8} "
          f"{'Cost(Rs)':>10} {'Grid Viol':>10}")
    print("-" * 60)
    for name, r in results.items():
        marker = " ◄" if name == "DQN (EVOCS)" else ""
        print(f"{name:<18} {r['evs_served']:>8.1f} "
              f"{r['deadlines_missed']:>8.1f} "
              f"{r['total_cost_rs']:>10.1f} "
              f"{r['grid_violations']:>10.1f}{marker}")

    # DQN improvement over FCFS
    fcfs = results["FCFS (Apps)"]
    dqn = results["DQN (EVOCS)"]
    cost_saved = fcfs["total_cost_rs"] - dqn["total_cost_rs"]
    missed_reduced = fcfs["deadlines_missed"] - dqn["deadlines_missed"]

    print(f"\nDQN vs FCFS (existing apps):")
    print(f"  Cost saved per day:       Rs {cost_saved:.1f}")
    print(f"  Missed deadlines reduced: {missed_reduced:.1f}")
    print(f"  Grid violations reduced:  "
          f"{fcfs['grid_violations'] - dqn['grid_violations']:.1f}")

    return results


if __name__ == "__main__":
    evaluate()
