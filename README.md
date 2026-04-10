# Lane Following with DDPG

This project trains a continuous-control actor-critic policy to keep a vehicle centered in a piecewise-curvature lane with left and right turns.

## What is included

- Source code for actor, critic, transition dynamics, lane definition, replay buffer, and training.
- Scripts to reproduce training curves and policy evaluation figures.
- Validation checks for core assumptions and deterministic dynamics.
- Saved example artifacts (`policy.pt`, `training_logs.pt`, plots).

## Repository layout

- `train.py`: DDPG training loop and checkpoint selection.
- `rollout.py`: environment rollout logic and reset behavior.
- `lane.py`: lane profile and centerline sampling utilities.
- `transition.py`: vehicle-state transition model.
- `plot_results.py`: training reward/loss figures.
- `generate_results.py`: final policy showcase + spread evaluation plots.

## Dependencies

Install dependencies from `requirements.txt`:

- `numpy`
- `torch`
- `matplotlib`

## Setup

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Ubuntu/Linux (bash)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce key results

Run from the repository root.

### 1. Train policy

```bash
python train.py
```

Outputs:

- `policy.pt`
- `training_logs.pt`

### 2. Plot training curves

```bash
python plot_results.py
```

Outputs:

- `reward_plot.png`
- `loss_plot.png`

### 3. Generate final policy figures

```bash
python generate_results.py --num-eval 32
```

Outputs:

- `policy_showcase.png`
- `policy_eval_spread.png`

Interpretation of `policy_eval_spread.png`:

- Each trajectory uses a random longitudinal start on the lane.
- Blue trajectories stay on-lane.
- Red trajectories go off-lane.

## Notes on reproducibility

- Training includes stochasticity from initialization and exploration noise.
- The lane profile, dynamics defaults, and plotting scripts are all in-repo.
- For consistent comparisons, use the same Python and package versions across runs.
