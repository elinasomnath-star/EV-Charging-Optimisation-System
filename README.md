# EVOCS — EV Optimal Charging Scheduler
## Central Guidance System for NH-44 Highway Corridor

> Reinforcement Learning meets IoT to intelligently route and charge Electric Vehicles across a 200 km highway corridor in real time.

**Institution:** RV College of Engineering, Bangalore  
**Program:** Experiential Learning Project, Semester IV, 2025–26   
**UN SDGs Addressed:** SDG 7 (Affordable and Clean Energy) | SDG 11 (Sustainable Cities and Communities)

---

## Problem Statement

As EV adoption grows along Indian highways, charging infrastructure faces two critical challenges:
1. **Uneven load distribution** — Drivers stop at the nearest station regardless of queue length, causing congestion at some stations while others remain idle.
2. **Unoptimized scheduling** — Once at a station, EVs are charged in FCFS order, ignoring urgency, electricity tariffs, and grid constraints.

EVOCS solves both problems simultaneously.

---

## Solution Architecture
### Component 1 — IoT Guidance System (`iot/guidance_system.py`)
- Acts as a centralized server broadcasting live station availability
- Routes each incoming EV to the optimal station based on:
  - Reachability (battery SOC vs. distance)
  - Real-time queue lengths (penalizes congested stations)
  - Charger speed (prioritizes DC fast chargers)
- EVs travel to their assigned station, simulating real highway transit time

### Component 2 — Deep Q-Network Scheduler (`rl/`)
- A **single shared DQN agent** is deployed at all 5 stations simultaneously
- Uses **Action Masking** to respect each station's hardware limits (n_chargers)
- At each 15-minute step, the AI decides how many EVs to charge and which ones
- The AI learns to balance competing objectives:
  - Serve all EVs before their departure deadline
  - Minimize electricity cost by scheduling during cheap tariff windows
  - Never exceed the transformer's rated capacity

### Component 3 — Physical Constraint Models (`models/`)
| Model | Purpose |
|-------|---------|
| `soc_model.py` | Tracks battery charge level using `SOC(t+1) = SOC(t) + (P × Δt × η) / C` |
| `grid_model.py` | Enforces transformer KVA limits per station |
| `tariff_model.py` | BESCOM Time-of-Day tariff (night: ₹4.5/kWh → evening peak: ₹8.0/kWh) |
| `queue_model.py` | Priority queue sorted by EV urgency (deadline proximity) |

---

## NH-44 Charging Stations

| # | Station | Location (km) | Operator | Chargers | Power | Type |
|---|---------|--------------|----------|----------|-------|------|
| 1 | Anekal Junction | 28 | Bolt.Earth | 4 | 50 kW | DC CCS2 |
| 2 | Hosur TataPower | 48 | TataPower | 6 | 50 kW | DC CCS2 |
| 3 | Krishnagiri Statiq | 95 | Statiq | 3 | 50 kW | DC CCS2 |
| 4 | Dharmapuri TataPower | 140 | TataPower | 4 | 50 kW | DC CCS2 |
| 5 | Salem Bolt.Earth | 200 | Bolt.Earth | 8 | 50 kW | DC CCS2 |

---

## EV Types Simulated

| EV Model | Battery | Max AC | Max DC |
|----------|---------|--------|--------|
| Tata Nexon EV | 40.5 kWh | 7.2 kW | 50 kW |
| MG ZS EV | 50.3 kWh | 7.4 kW | 76 kW |
| Hyundai Kona | 39.2 kWh | 7.2 kW | 50 kW |
| Tata Tigor EV | 26.0 kWh | 7.2 kW | 25 kW |
| Ola S1 Pro | 4.0 kWh | 1.5 kW | — |

---

## DQN Agent — Action Space

The agent has 10 possible actions at each 15-minute timestep:

| Action | Meaning |
|--------|---------|
| 0 — Hold | Skip this timestep (wait for cheaper tariff or less urgency) |
| 1–N — Charge top-N | Plug in the N most urgent EVs simultaneously |
| 9 — Charge cheapest | Charge the EV that requires the least remaining energy |

**Action Masking** ensures the agent never selects an action that exceeds a station's physical charger count. If Anekal has 4 chargers, actions 5–8 are mathematically blocked (masked to −∞ in the Q-value network).

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/EVOCS-NH44.git
cd EVOCS-NH44
pip install -r requirements.txt
```

---

## Usage

**Train the DQN agent (all stations):**
```bash
python main.py train
```

**Run full NH-44 corridor simulation (all 5 stations simultaneously):**
```bash
python main.py simulate_corridor
```

**Run live demo for a single station:**
```bash
python main.py demo --station 0   # Anekal Junction
python main.py demo --station 2   # Krishnagiri Statiq
```

**Evaluate DQN vs baselines (Greedy, FCFS, EDF):**
```bash
python main.py evaluate --station 0 --episodes 100
```

---

## Sample Output (Corridor Simulation)

```
NH-44 CORRIDOR SIMULATION COMPLETE
==================================================
Total EVs Generated on Highway: 95

Per-Station Performance:
                  Station | Served | Missed |     Cost |   Reward
------------------------------------------------------------------
          Anekal_Junction |     15 |      0 | Rs  3951 |    93.72
          Hosur_TataPower |     13 |      0 | Rs  3277 |    88.14
       Krishnagiri_Statiq |     21 |      1 | Rs  4447 |   165.20
     Dharmapuri_TataPower |     23 |      1 | Rs  4766 |   142.35
               Salem_Bolt |     15 |      0 | Rs  3534 |   104.38
------------------------------------------------------------------
             TOTAL SYSTEM |     87 |      2 | Rs 19975 |   593.79
==================================================
* Note: 6 EVs were still "In Progress" (charging or in transit) 
  at the simulation end time (midnight).
```

---

## Project Structure

```
evocs/
├── config/
│   └── nh44_config.yaml          # Station hardware & simulation config
├── data/
│   └── arrival_generator.py      # Stochastic EV arrival model (ToD-weighted)
├── models/
│   ├── soc_model.py              # Battery SOC dynamics
│   ├── grid_model.py             # Transformer KVA constraint
│   ├── tariff_model.py           # BESCOM Time-of-Day pricing
│   └── queue_model.py            # Priority queue (urgency-sorted)
├── baselines/
│   └── greedy.py                 # Greedy, FCFS, EDF baseline policies
├── rl/
│   ├── charging_env.py           # OpenAI Gym-compatible environment
│   ├── reward.py                 # Reward function (SOC gain − cost penalty)
│   └── train.py                  # DQN agent (experience replay + target net)
├── iot/
│   ├── guidance_system.py        # IoT routing & station selection logic
│   └── corridor_sim.py           # Synchronized 5-station simulation engine
├── evaluate.py                   # Metrics comparison across policies
├── main.py                       # CLI entry point
└── requirements.txt
```

---

## Key Results

- **~95% EVs successfully served** across the 200 km corridor per 24-hour simulation
- **0 transformer violations** across all stations
- **Tariff-aware scheduling** — agent naturally prefers night/morning tariff windows
- **Action Masking** ensures the shared brain respects each station's unique hardware specs

---
