# main.py
# EVOCS — EV Optimal Charging Scheduler for NH-44 Highway Corridor

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_train(args):
    from rl.train import train
    train(config_path=args.config, save_dir=args.save_dir)


def cmd_evaluate(args):
    from evaluate import evaluate
    evaluate(
        config_path=args.config,
        model_path=args.model,
        station_idx=args.station,
        n_episodes=args.episodes
    )


def cmd_simulate_corridor(args):
    from iot.corridor_sim import simulate_corridor

    simulate_corridor(config_path=args.config, model_path=args.model)


def cmd_demo(args):
    import yaml
    import numpy as np
    from rl.charging_env import EVChargingEnv
    from rl.train import DQNAgent

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = EVChargingEnv(config_path=args.config,
                        station_idx=args.station)
    agent = DQNAgent(env.observation_space.shape[0],
                     env.action_space.n, config["dqn"])
    agent.load(args.model)
    agent.epsilon = 0.0

    action_names = ["Hold"] + [f"Charge top-{i}" for i in range(1, 9)] + ["Charge cheapest"]

    state, _ = env.reset()
    print(f"\nDemo - {env.station['name']}")
    print(f"{'Step':>4} {'Hour':>6} {'Tariff':>13} "
          f"{'Queue':>6} {'Action':>15} {'Reward':>8}")
    print("-" * 60)
    
    valid_actions = env.get_valid_actions()

    total_reward = 0.0
    reward_accum = 0.0
    step = 0
    while True:
        hour = step * env.timescale_hr % 24
        tariff_label = env.tariff_model.get_label(hour)
        queue_size = env.queue.size()
        action = agent.act(state, valid_actions=valid_actions, training=False)
        state, reward, done, _, info = env.step(action)
        total_reward += reward
        reward_accum += reward
        
        if step % 4 == 0:
            if queue_size == 0:
                action_str = "---"
            elif 1 <= action <= 8:
                actual_n = min(action, env.station["n_chargers"], queue_size)
                action_str = f"Charge top-{actual_n}"
            else:
                action_str = action_names[action]
                
            print(f"{step:>4} {hour:>5.1f}h {tariff_label:>13} "
                  f"{queue_size:>6} {action_str:>15} "
                  f"{reward_accum:>8.2f}")
            reward_accum = 0.0
        step += 1
        if done:
            break

    print(f"\nEpisode complete:")
    print(f"  Total reward:     {total_reward:.2f}")
    print(f"  EVs served:       {info['evs_served']}")
    print(f"  Deadlines missed: {info['deadlines_missed']}")
    print(f"  Total cost:       Rs {info['total_cost_rs']:.2f}")
    print(f"  Grid violations:  {info['grid_violations']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EVOCS - EV Charging Scheduler for NH-44"
    )
    parser.add_argument("--config", default="config/nh44_config.yaml")

    sub = parser.add_subparsers(dest="command")

    # train — no --station needed (trains all stations together)
    t = sub.add_parser("train", help="Train DQN on all stations")
    t.add_argument("--save-dir", default=None, help="Directory to save models. Auto-determined if not set.")
    t.set_defaults(func=cmd_train)

    # evaluate — pick a specific station to evaluate
    e = sub.add_parser("evaluate", help="Compare DQN vs baselines")
    e.add_argument("--station", type=int, default=-1,
                   help="Station index 0-4")
    e.add_argument("--model", default=None, help="Path to model. Auto-determined if not set.")
    e.add_argument("--episodes", type=int, default=100)
    e.set_defaults(func=cmd_evaluate)

    # demo
    d = sub.add_parser("demo", help="Run single episode live")
    d.add_argument("--station", type=int, default=0,
                   help="Station index 0-4")
    d.add_argument("--model", default=None, help="Path to model. Auto-determined if not set.")
    d.set_defaults(func=cmd_demo)

    # simulate_corridor
    sc = sub.add_parser("simulate_corridor", help="Run synchronized multi-station corridor simulation")
    sc.add_argument("--model", default=None, help="Path to model. Auto-determined if not set.")
    sc.set_defaults(func=cmd_simulate_corridor)

    args = parser.parse_args()
    
    if args.command is not None:
        # Automatically determine save_dir and model paths based on config
        config_name = os.path.basename(args.config).split('.')[0]
        config_prefix = config_name.replace("_config", "") # e.g. 'nh44' or 'nh275'
        
        auto_save_dir = f"models/{config_prefix}"
            
        auto_model = os.path.join(auto_save_dir, "evocs_best.pt")
        
        if hasattr(args, 'save_dir') and args.save_dir is None:
            args.save_dir = auto_save_dir
            
        if hasattr(args, 'model') and args.model is None:
            args.model = auto_model

        args.func(args)
    else:
        parser.print_help()
