# Project Report

## Problem framing

Goal: learn a policy that follows a curved lane with multiple turns, while minimizing lateral and heading error.

State used by the policy:

- lateral displacement `d`
- heading error `theta`
- lane curvature `k`

Action:

- normalized steering command in `[-1, 1]`

Reward:

- negative quadratic penalty on tracking errors (`-(d^2 + theta^2)`)

## Model and method

- Actor-Critic with DDPG-style updates.
- Replay buffer for off-policy learning.
- Soft target updates for stable bootstrapping.
- Piecewise-curvature lane profile with straight and turning segments.

## Trial-and-error summary

Early versions looked acceptable on short stretches but failed near later turns.

Main issues identified:

1. Rollout horizon did not always expose later lane segments.
2. Starts were concentrated near the beginning of the lane.
3. Single-rollout eval was noisy for checkpoint selection.

Changes that fixed these issues:

1. Rollout horizon now scales with lane length.
2. Training and evaluation use random longitudinal starts.
3. Evaluation during training averages multiple deterministic rollouts.
4. Training budget was increased to improve robustness.

## Final outputs

- `reward_plot.png`: learning curve over episodes.
- `loss_plot.png`: critic optimization behavior.
- `policy_showcase.png`: single representative trajectory from lane start.
- `policy_eval_spread.png`: many random-start trajectories to visualize robustness.

## Practical conclusion

The final setup substantially improved lane-wide coverage and reduced endpoint failure modes. The evaluation spread figure gives a more honest view of policy quality than a single rollout.

## AI usage disclosure

AI tools were used as part of this project in the following ways:

- Brainstorming project structure and experimental directions.
- Coding assistance for implementation details and debugging support.
- Heavy assistance in `generate_results.py` to speed up writing utility/reporting code.
- Drafting and formatting support for `README.md` and this report using project details I provided.

I reviewed all generated content, verified behavior in my project context, and can explain the submitted code and results.
