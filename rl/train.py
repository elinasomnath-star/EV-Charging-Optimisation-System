# rl/train.py
# DQN Training Loop for EVOCS
# Trains ONE model across ALL NH-44 stations simultaneously

import numpy as np
import os
import yaml
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim

from rl.charging_env import EVChargingEnv


# ── DQN Neural Network ────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden: list = [128, 64]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states,      dtype=np.float32),
                np.array(actions),
                np.array(rewards,     dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones,       dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: dict):
        self.action_dim       = action_dim
        self.gamma            = config["gamma"]
        self.epsilon          = config["epsilon_start"]
        self.epsilon_end      = config["epsilon_end"]
        self.epsilon_decay    = config["epsilon_decay"]
        self.batch_size       = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = DQN(
            state_dim, action_dim, config["hidden_layers"]
        ).to(self.device)
        self.target_net = DQN(
            state_dim, action_dim, config["hidden_layers"]
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config["learning_rate"]
        )
        self.memory = ReplayBuffer(config["memory_size"])
        self.steps  = 0

    def act(self, state, valid_actions=None, training=True):
        if training and random.random() < self.epsilon:
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            if valid_actions is not None:
                mask = torch.full_like(q_values, float('-inf'))
                for a in valid_actions:
                    mask[0, a] = q_values[0, a]
                return mask.argmax().item()
            return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        q_current = self.policy_net(s).gather(
            1, a.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            q_next   = self.target_net(ns).max(1)[0]
            q_target = r + self.gamma * q_next * (1 - d)

        loss = nn.MSELoss()(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return loss.item()

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "steps":      self.steps
        }, path)
        print(f"  Model saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["policy_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        print(f"Model loaded <- {path}")


# ── Training Loop ─────────────────────────────────────────────────────
def train(config_path: str = "config/nh44_config.yaml",
          save_dir:    str = "models/saved"):

    os.makedirs(save_dir, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dqn_cfg    = config["dqn"]
    n_episodes = config["simulation"]["n_episodes_train"]
    stations   = config["stations"]

    # station_idx = -1  →  random station each episode
    env = EVChargingEnv(config_path=config_path, station_idx=-1)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("\nEVOCS DQN Training - ALL NH-44 Stations")
    print(f"Stations : {[s['name'] for s in stations]}")
    print(f"State dim: {state_dim},  Action dim: {action_dim}")
    print(f"Episodes : {n_episodes}")
    print(f"Device   : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("-" * 60)

    agent = DQNAgent(state_dim, action_dim, dqn_cfg)

    # Per-episode tracking
    ep_rewards  = []
    ep_served   = []
    ep_missed   = []
    ep_costs    = []
    best_reward = -np.inf

    # Per-station tracking
    station_rewards = {s["name"]: [] for s in stations}

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        valid_actions = env.get_valid_actions()
        total_reward = 0.0
        losses = []

        while True:
            action     = agent.act(state, valid_actions=valid_actions, training=True)
            next_state, reward, done, _, info = env.step(action)
            valid_actions = env.get_valid_actions()
            agent.memory.push(state, action, reward,
                              next_state, float(done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            state = next_state
            if done:
                break

        # Decay epsilon every episode
        agent.epsilon = max(
            agent.epsilon_end,
            agent.epsilon * agent.epsilon_decay
        )

        ep_rewards.append(total_reward)
        ep_served.append(info.get("evs_served",       0))
        ep_missed.append(info.get("deadlines_missed",  0))
        ep_costs.append(info.get("total_cost_rs",     0.0))
        stn = info.get("station", "unknown")
        if stn in station_rewards:
            station_rewards[stn].append(total_reward)

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(f"{save_dir}/evocs_best.pt")

        # Log every 100 episodes
        if ep % 100 == 0:
            w = min(100, len(ep_rewards))
            print(f"Ep {ep:4d}/{n_episodes} | "
                  f"Reward: {np.mean(ep_rewards[-w:]):7.2f} | "
                  f"Served: {np.mean(ep_served[-w:]):.1f} | "
                  f"Missed: {np.mean(ep_missed[-w:]):.1f} | "
                  f"Cost: Rs{np.mean(ep_costs[-w:]):.0f} | "
                  f"eps: {agent.epsilon:.4f} | "
                  f"Loss: {np.mean(losses):.4f}")

    # Save final model
    agent.save(f"{save_dir}/evocs_final.pt")

    # Save training history
    np.save(f"{save_dir}/rewards.npy", np.array(ep_rewards))
    np.save(f"{save_dir}/served.npy",  np.array(ep_served))
    np.save(f"{save_dir}/missed.npy",  np.array(ep_missed))
    np.save(f"{save_dir}/costs.npy",   np.array(ep_costs))

    # Per-station summary
    print("\nPer-station average reward (last 200 episodes):")
    for stn_name, rewards in station_rewards.items():
        if rewards:
            avg = np.mean(rewards[-200:])
            print(f"  {stn_name:<25} {avg:7.2f}")

    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    return agent, ep_rewards


if __name__ == "__main__":
    train()
