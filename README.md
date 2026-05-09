# EVOCS — EV Optimal Charging Scheduler
## Central Guidance System for Highway Corridors

> Reinforcement Learning meets IoT to intelligently route and charge Electric Vehicles across highway corridors (e.g., NH-44 and NH-275) in real time.

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
| `tariff_model.py` | BESCOM Time-of-Day tariff (night: ₹5.5/kWh → evening peak: ₹8.25/kWh) |
| `queue_model.py` | Priority queue sorted by EV urgency (deadline proximity) |

---

## NH-44 Charging Stations (Bangalore to Salem)

| # | Station | Location (km) | Operator | Chargers | Power | Type |
|---|---------|--------------|----------|----------|-------|------|
| 1 | Anekal Junction | 28 | Bolt.Earth | 2 | 30 kW | DC CCS2 |
| 2 | Hosur TataPower | 48 | TataPower | 4 | 25 kW | DC CCS2 |
| 3 | Krishnagiri Statiq | 95 | Statiq | 3 | 30 kW | DC CCS2 |
| 4 | Dharmapuri TataPower | 140 | TataPower | 4 | 25 kW | DC CCS2 |
| 5 | Salem Bolt.Earth | 200 | Bolt.Earth | 4 | 30 kW | DC CCS2 |

---

## NH-275 Charging Stations (Bangalore to Mysore Expressway)

| # | Station | Location (km) | Operator | Chargers | Power | Type |
|---|---------|--------------|----------|----------|-------|------|
| 1 | Bidadi Zeon | 30 | Zeon | 2 | 50 kW | DC CCS2 |
| 2 | Channapatna TataPower | 65 | TataPower | 4 | 50 kW | DC CCS2 |
| 3 | Maddur Statiq | 85 | Statiq | 3 | 60 kW | DC CCS2 |
| 4 | Mandya JioBP | 105 | JioBP | 4 | 60 kW | DC CCS2 |
| 5 | Srirangapatna Zeon | 130 | Zeon | 2 | 50 kW | DC CCS2 |

---

## EV Types Simulated

| EV Model | Battery | Max Charge | Weight |
|----------|---------|------------|--------|
| Tata Nexon EV | 40.5 kWh | 50 kW | 30% |
| MG ZS EV | 50.3 kWh | 50 kW | 20% |
| Hyundai Kona | 39.2 kWh | 50 kW | 15% |
| Tata Tiago EV | 24.0 kWh | 25 kW | 20% |
| Citroen_eC3 | 29.2 kWh | 30 kW | 15% |




---

## DQN Agent — Action Space

The agent has 10 possible actions at each 15-minute timestep:

| Action | Meaning |
|--------|---------|
| 0 — Hold | Skip this timestep (wait for cheaper tariff or less urgency) |
| 1–N — Charge top-N | Plug in the N most urgent EVs simultaneously |
| 9 — Charge cheapest | Charge the EV that requires the least remaining energy |

**Action Masking** ensures the agent never selects an action that exceeds a station's physical charger count. If Anekal has 2 chargers, actions 3–8 are mathematically blocked (masked to −∞ in the Q-value network).

---

## Setup

```bash
git clone https://github.com/elinasomnath-star/EV-Charging-Optimisation-System.git
cd EV-Charging-Optimisation-System
pip install -r requirements.txt
```

---

## Usage

EVOCS supports multiple highway corridors via configuration files. 

**Train the DQN agent:**
```bash
# Trains for NH-44 and automatically saves to models/nh44/
python main.py --config config/nh44_config.yaml train

# Trains for NH-275 and automatically saves to models/nh275/
python main.py --config config/nh275_config.yaml train
```

**Run full corridor simulation (all stations simultaneously):**
```bash
# Simulates the full NH-44 Corridor
python main.py --config config/nh44_config.yaml simulate_corridor

# Simulates the full NH-275 Expressway
python main.py --config config/nh275_config.yaml simulate_corridor
```

**Run live demo for a single station:**
```bash
python main.py --config config/nh44_config.yaml demo --station 0   # Anekal Junction (NH-44)
python main.py --config config/nh275_config.yaml demo --station 2  # Maddur Statiq (NH-275)
```

**Evaluate DQN vs baselines (Full Corridor Report):**
```bash
# Evaluates all stations on NH-44
python main.py --config config/nh44_config.yaml evaluate

# Evaluates all stations on NH-275
python main.py --config config/nh275_config.yaml evaluate
```

---

## Sample Output (Corridor Simulation)

```
NH-44 CORRIDOR SIMULATION COMPLETE
==================================================
Total EVs Generated on Highway: 95
EVs successfully routed:        95

Per-Station Performance:
                  Station | Served | Missed |     Cost |   Reward
------------------------------------------------------------------
          Anekal_Junction |     10 |      0 | Rs  2213 |    68.98
          Hosur_TataPower |     12 |      0 | Rs  2488 |    74.90
       Krishnagiri_Statiq |     26 |      0 | Rs  4711 |   196.99
     Dharmapuri_TataPower |     24 |      0 | Rs  4627 |   203.14
               Salem_Bolt |     19 |      0 | Rs  4236 |   150.70
------------------------------------------------------------------
             TOTAL SYSTEM |     91 |      0 | Rs 18276 |   694.72
==================================================
* Note: 4 EVs were still "In Progress" (charging or in transit) 
  at the simulation end time (midnight).
```

---

## Project Structure

```
.
├── config/
│   ├── nh44_config.yaml          # NH-44 Station hardware & simulation config
│   └── nh275_config.yaml         # NH-275 Expressway configuration
├── data/
│   └── arrival_generator.py      # Stochastic EV arrival model (ToD-weighted)
├── models/
│   ├── nh44/                     # Trained models for NH-44
│   ├── nh275/                    # Trained models for NH-275
│   ├── soc_model.py              # Battery SOC dynamics
│   ├── grid_model.py             # Transformer KVA constraint
│   ├── tariff_model.py           # BESCOM Time-of-Day pricing
│   └── queue_model.py            # Priority queue (urgency-sorted)
├── baselines/
│   └── schedulers.py             # Greedy, FCFS baseline policies
├── rl/
│   ├── charging_env.py           # OpenAI Gym-compatible environment
│   ├── reward.py                 # Reward function (SOC gain − cost penalty)
│   └── train.py                  # DQN agent (experience replay + target net)
├── iot/
│   ├── guidance_system.py        # Intelligent routing & station selection logic
│   └── corridor_sim.py           # Synchronized 5-station simulation engine
├── evaluate.py                   # Metrics comparison across policies
├── main.py                       # CLI entry point
└── requirements.txt
```


---

## Key Results: NH-44 (Bangalore - Salem)

Based on a robust 500-episode evaluation across all 5 stations:

| Algorithm | Avg Served | Avg Missed | Avg Cost (Rs) |
|-----------|------------|------------|---------------|
| FCFS (Current Apps) | 32.7 | 12.2 | ₹ 10,013 |
| Greedy | 33.5 | 11.4 | ₹ 12,071 |
| **DQN (EVOCS)** | **39.0** | **0.0** | **₹ 6,957** |

### Net Impact vs. Existing Systems (NH-44):
- **Financial Savings**: **₹ 15,281 saved per day** across the corridor.
- **Reliability Improvement**: **61.1 fewer missed deadlines** per day.
- **Grid Safety**: **100% compliance** with transformer KVA limits.

---

## Key Results: NH-275 (Bangalore - Mysore Expressway)

Based on a robust 500-episode evaluation across all 5 stations:

| Algorithm | Avg Served | Avg Missed | Avg Cost (Rs) |
|-----------|------------|------------|---------------|
| FCFS (Current Apps) | 33.3 | 11.6 | ₹ 13,367 |
| Greedy | 33.4 | 11.5 | ₹ 17,330 |
| **DQN (EVOCS)** | **39.0** | **0.0** | **₹ 6,995** |

### Net Impact vs. Existing Systems (NH-275):
- **Financial Savings**: **₹ 31,858 saved per day** across the corridor.
- **Reliability Improvement**: **58.2 fewer missed deadlines** per day.
- **Grid Safety**: **100% compliance** with transformer KVA limits.

---
