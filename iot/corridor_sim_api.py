# iot/corridor_sim_api.py
# System-Level Simulation of Highway Corridor (API version)

import yaml
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.charging_env import EVChargingEnv
from rl.train import DQNAgent
from iot.guidance_system import GuidanceSystem
from data.arrival_generator import ArrivalGenerator, EV_TYPES
from models.queue_model import EV

def run_simulation_generator(config_path="config/nh44_config.yaml", model_path=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config_prefix = os.path.basename(config_path).split('.')[0].replace('_config', '').upper()
    if config_prefix == "NH44":
        corridor_name = "NH-44"
    elif config_prefix == "NH275":
        corridor_name = "NH-275"
    else:
        corridor_name = config_prefix

    if model_path is None:
        corridor_key = config_prefix.lower()
        model_path = os.path.join("models", corridor_key, "evocs_best.pt")

    # Initialize 5 Envs
    envs = []
    station_statuses = []
    for i in range(5):
        env = EVChargingEnv(config_path, station_idx=i)
        env.reset()
        # Disable internal random arrivals so we can inject routed EVs
        env.arrival_gen.get_arrivals = lambda t: []
        envs.append(env)
        
        station_statuses.append({
            "idx": i,
            "id": env.station["id"],
            "name": env.station["name"],
            "location_km": env.station["location_km"],
            "n_chargers": env.station["n_chargers"],
            "charger_kw": env.station["charger_kw"],
            "charger_type": env.station["charger_type"],
            "queue_size": 0
        })

    # Load DQN Agent
    action_names = ["Hold"] + [f"Charge top-{i}" for i in range(1, 9)] + ["Charge cheapest"]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    agent = DQNAgent(state_dim, action_dim, config["dqn"])
    agent.load(model_path)
    agent.epsilon = 0.0

    # Initialize Highway Components
    guidance = GuidanceSystem()
    highway_gen = ArrivalGenerator(
        station_id=0,
        n_chargers=20,
        timescale_min=15,
        config_arrival=config.get("arrival", {}),
        ev_specs=config.get("ev_specs", None)
    )

    # Boost highway arrival rate to simulate corridor traffic
    highway_gen.HOURLY_ARRIVAL_RATE = {k: v * 2 for k, v in highway_gen.HOURLY_ARRIVAL_RATE.items()}
    
    traveling_evs = []
    
    # Metrics
    total_routed = 0
    station_metrics = [{"served": 0, "missed": 0, "cost": 0.0, "reward": 0.0, "violations": 0} for _ in range(5)]
    station_logs = [[] for _ in range(5)]
    reward_accums = [0.0] * 5

    # Yield initial configuration
    yield {
        "type": "init",
        "corridor_name": corridor_name,
        "total_steps": config["simulation"]["simulation_length"],
        "stations": station_statuses
    }

    # Simulation Loop
    for step in range(config["simulation"]["simulation_length"]):
        hour = step * 15 / 60.0
        
        # Update station statuses for Guidance
        for i, env in enumerate(envs):
            station_statuses[i]["queue_size"] = env.queue.size()

        # Generate EVs entering the highway at km 0
        new_evs = highway_gen.get_arrivals(step)
        
        # Calculate the spawn range (80% of the highway length)
        max_station_km = max([st["location_km"] for st in station_statuses])
        spawn_range = max_station_km * 0.8
        
        routed_this_step = []
        for ev in new_evs:
            start_km = random.uniform(0, spawn_range)
            rec = guidance.recommend_station(ev.soc, ev.battery_kwh, start_km, station_statuses)
            
            if rec is None:
                continue
                
            total_routed += 1
            st_idx = rec["station"]["idx"]
            
            dist = rec["distance"]
            travel_steps = int(dist / 15.0)
            arrival_step = step + travel_steps
            
            safe_arrival_soc = max(0.05, rec["arrival_soc"])
            ev.soc = safe_arrival_soc
            ev.arrival_soc = safe_arrival_soc
            ev.arrival_step = arrival_step
            ev.deadline_step = arrival_step + (ev.deadline_step - step)
            
            ev_type_dict = next((et for et in highway_gen.ev_types if et["name"] == ev.ev_type), None)
            
            if "DC" in station_statuses[st_idx].get("charger_type", ""):
                max_dc = ev_type_dict.get("max_dc_kw", ev_type_dict["max_kw"]) if ev_type_dict else 50.0
                ev.max_charge_kw = min(station_statuses[st_idx]["charger_kw"], max_dc)
            else:
                ev.max_charge_kw = min(station_statuses[st_idx]["charger_kw"], ev.max_charge_kw)
                
            traveling_evs.append({"ev": ev, "station_idx": st_idx, "arrival_step": arrival_step})
            routed_this_step.append(station_statuses[st_idx]["name"])

        # Process Arrivals
        arrived_count = [0] * 5
        for travel in traveling_evs[:]:
            if travel["arrival_step"] <= step:
                st_idx = travel["station_idx"]
                envs[st_idx].queue.add(travel["ev"], step)
                arrived_count[st_idx] += 1
                traveling_evs.remove(travel)

        # Run DQN Step for all stations
        queues_str = []
        queues_data = []
        for i, env in enumerate(envs):
            obs = env._get_obs()
            valid_actions = env.get_valid_actions()
            action = agent.act(obs, valid_actions=valid_actions, training=False)
            _, reward, _, _, info = env.step(action)
            
            reward_accums[i] += float(reward)
            
            station_metrics[i]["served"] = int(info["evs_served"])
            station_metrics[i]["missed"] = int(info["deadlines_missed"])
            station_metrics[i]["cost"] = float(info["total_cost_rs"])
            station_metrics[i]["reward"] += float(reward)
            station_metrics[i]["violations"] = int(info["grid_violations"])
            
            if step % 4 == 0:
                tariff_label = env.tariff_model.get_label(hour)
                queue_before = env.queue.size()
                
                if queue_before == 0:
                    action_name = "---"
                elif 1 <= action <= 8:
                    actual_n = min(action, env.station["n_chargers"], queue_before)
                    action_name = f"Charge top-{actual_n}"
                else:
                    action_name = action_names[action]
                    
                station_logs[i].append({
                    "step": step,
                    "hour": round(hour, 1),
                    "tariff": tariff_label,
                    "queue": queue_before,
                    "action": action_name,
                    "reward": round(reward_accums[i], 2)
                })
                reward_accums[i] = 0.0
            
            queues_str.append(str(env.queue.size()))
            queues_data.append(env.queue.size())

        # Always yield step data (or maybe only every 4th step if UI doesn't need to be so fine-grained, but let's yield every step for smooth animation)
        routes_str = ", ".join(set(routed_this_step)) if routed_this_step else "None"
        yield {
            "type": "step",
            "step": step,
            "hour": round(hour, 1),
            "new_evs": len(new_evs),
            "routes": routes_str,
            "queues": queues_data,
            "metrics": station_metrics
        }

    # Final stats
    total_served = sum(m["served"] for m in station_metrics)
    total_missed = sum(m["missed"] for m in station_metrics)
    total_cost = sum(m["cost"] for m in station_metrics)
    total_reward = sum(m["reward"] for m in station_metrics)

    yield {
        "type": "finish",
        "total_routed": total_routed,
        "total_served": total_served,
        "total_missed": total_missed,
        "total_cost": total_cost,
        "total_reward": total_reward,
        "station_metrics": station_metrics,
        "station_logs": station_logs
    }
