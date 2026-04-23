# iot/corridor_sim.py
# System-Level Simulation of NH-44 Highway Corridor

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

def simulate_corridor(config_path="config/nh44_config.yaml", model_path="models/saved/evocs_best.pt"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Initializing NH-44 Corridor Simulation...")
    
    # 1. Initialize 5 Envs
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

    # 2. Load DQN Agent
    action_names = ["Hold"] + [f"Charge top-{i}" for i in range(1, 9)] + ["Charge cheapest"]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    agent = DQNAgent(state_dim, action_dim, config["dqn"])
    agent.load(model_path)
    agent.epsilon = 0.0

    # 3. Initialize Highway Components
    guidance = GuidanceSystem()
    highway_gen = ArrivalGenerator(station_id=0, n_chargers=20, timescale_min=15, config_arrival=config.get("arrival", {}))
    # Boost highway arrival rate to simulate corridor traffic
    highway_gen.HOURLY_ARRIVAL_RATE = {k: v * 2 for k, v in highway_gen.HOURLY_ARRIVAL_RATE.items()}
    
    traveling_evs = []
    
    # Metrics
    total_routed = 0
    station_metrics = [{"served": 0, "missed": 0, "cost": 0.0, "reward": 0.0, "violations": 0} for _ in range(5)]
    station_logs = [[] for _ in range(5)]
    reward_accums = [0.0] * 5

    print(f"\n{'Step':>4} | {'Traffic':>7} | {'Routing Decisions':>35} | {'Station Queues (0..4)':>25}")
    print("-" * 80)

    # 4. Simulation Loop
    for step in range(config["simulation"]["simulation_length"]):
        hour = step * 15 / 60.0
        
        # Update station statuses for Guidance
        for i, env in enumerate(envs):
            station_statuses[i]["queue_size"] = env.queue.size()

        # Generate EVs entering the highway at km 0
        new_evs = highway_gen.get_arrivals(step)
        
        routed_this_step = []
        for ev in new_evs:
            # Assume EVs enter with a random SOC, starting randomly along the highway
            start_km = random.uniform(0, 160.0)
            rec = guidance.recommend_station(ev.soc, ev.battery_kwh, start_km, station_statuses)
            
            if rec is None:
                # Fallback if no station is ahead (should not happen if start < 160 and Salem is at 200)
                continue
                
            total_routed += 1
            st_idx = rec["station"]["idx"]
            
            # Calculate arrival time (assume 60 km/h -> 1 km/min -> 15 km/step)
            dist = rec["distance"]
            travel_steps = int(dist / 15.0)
            arrival_step = step + travel_steps
            
            # Prepare EV for arrival (guarantee at least 5% SOC so it's not dead)
            safe_arrival_soc = max(0.05, rec["arrival_soc"])
            ev.soc = safe_arrival_soc
            ev.arrival_soc = safe_arrival_soc
            ev.arrival_step = arrival_step
            ev.deadline_step = arrival_step + (ev.deadline_step - step) # Keep same stay duration
            
            # Update max charge rate based on assigned station
            ev_type_dict = next((et for et in EV_TYPES if et["name"] == ev.ev_type), None)
            
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
        for i, env in enumerate(envs):
            obs = env._get_obs()
            valid_actions = env.get_valid_actions()
            action = agent.act(obs, valid_actions=valid_actions, training=False)
            _, reward, _, _, info = env.step(action)
            
            # Accumulate reward for logging
            reward_accums[i] += reward
            
            # Sync metrics
            station_metrics[i]["served"] = info["evs_served"]
            station_metrics[i]["missed"] = info["deadlines_missed"]
            station_metrics[i]["cost"] = info["total_cost_rs"]
            station_metrics[i]["reward"] += reward
            station_metrics[i]["violations"] = info["grid_violations"]
            
            # Store log
            if step % 4 == 0:
                tariff_label = env.tariff_model.get_label(hour)
                queue_before = env.queue.size()  # captured before step ran
                
                if queue_before == 0:
                    action_name = "---"
                elif 1 <= action <= 8:
                    actual_n = min(action, env.station["n_chargers"], queue_before)
                    action_name = f"Charge top-{actual_n}"
                else:
                    action_name = action_names[action]
                    
                station_logs[i].append(
                    f"{step:>4} {hour:>5.1f}h {tariff_label:>13} {queue_before:>6} {action_name:>15} {reward_accums[i]:>8.2f}"
                )
                reward_accums[i] = 0.0  # Reset after logging
            
            queues_str.append(str(env.queue.size()))

        if step % 4 == 0:
            routes_str = ", ".join(set(routed_this_step))[:33] + ".." if routed_this_step else "-"
            print(f"{step:>4} | {len(new_evs):>4} EVs | {routes_str:>35} | {' - '.join(queues_str):>25}")

    print("\n" + "="*50)
    print("NH-44 CORRIDOR SIMULATION COMPLETE")
    print("="*50)
    print(f"Total EVs Generated on Highway: {total_routed}")
    print(f"EVs successfully routed:        {total_routed}")
    print("\nPer-Station Performance:")
    print(f"{'Station':>25} | {'Served':>6} | {'Missed':>6} | {'Cost':>8} | {'Reward':>8}")
    print("-" * 66)
    
    total_served = 0
    total_missed = 0
    total_cost = 0.0
    total_reward = 0.0
    for i, st in enumerate(station_statuses):
        served = station_metrics[i]["served"]
        missed = station_metrics[i]["missed"]
        cost = station_metrics[i]["cost"]
        reward = station_metrics[i]["reward"]
        total_served += served
        total_missed += missed
        total_cost += cost
        total_reward += reward
        print(f"{st['name']:>25} | {served:>6} | {missed:>6} | Rs{cost:>6.0f} | {reward:>8.2f}")
        
    print("-" * 66)
    print(f"{'TOTAL SYSTEM':>25} | {total_served:>6} | {total_missed:>6} | Rs{total_cost:>6.0f} | {total_reward:>8.2f}")
    print("="*50)
    
    # Print detailed logs for each station
    for i, st in enumerate(station_statuses):
        print(f"\n\nDemo - {st['name']}")
        print(f"{'Step':>4} {'Hour':>6} {'Tariff':>13} {'Queue':>6} {'Action':>15} {'Reward':>8}")
        print("-" * 60)
        for log in station_logs[i]:
            print(log)
            
        print(f"\nEpisode complete:")
        print(f"  Total reward:     {station_metrics[i]['reward']:.2f}")
        print(f"  EVs served:       {station_metrics[i]['served']}")
        print(f"  Deadlines missed: {station_metrics[i]['missed']}")
        print(f"  Total cost:       Rs {station_metrics[i]['cost']:.2f}")
        print(f"  Grid violations:  {station_metrics[i]['violations']}")

if __name__ == "__main__":
    simulate_corridor()
