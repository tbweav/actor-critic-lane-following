import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from actor import Actor
from lane import Lane
from transition import Transition


def load_actor(policy_path):
    actor = Actor(3, 1, 2, 64)
    actor.load_state_dict(torch.load(policy_path, weights_only=True))
    actor.eval()
    return actor


def interpolate_centerline_at_s(s_values, road_s, road_x, road_y, road_psi):
    s_clipped = np.clip(s_values, road_s[0], road_s[-1])
    x = np.interp(s_clipped, road_s, road_x)
    y = np.interp(s_clipped, road_s, road_y)
    psi = np.interp(s_clipped, road_s, road_psi)
    return x, y, psi


def run_policy_rollout(actor, lane, transition, max_steps, start_s=0.0, random_lateral_init=True):
    if random_lateral_init:
        d = lane.d_init_std * np.random.randn()
        theta = lane.theta_init_std * np.random.randn()
    else:
        d = 0.0
        theta = 0.0

    s = float(start_s)
    k = lane.lane_curvature_at_s(lane.active_lane_id, s)

    ds = transition.velocity * transition.dt
    lane_len = lane.lane_length()

    s_hist = [s]
    d_hist = [float(d)]
    theta_hist = [float(theta)]

    offroad = False
    reached_end = False

    for _ in range(max_steps):
        state = torch.tensor([d, theta, k], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(state).squeeze(0).item()
        action = float(np.clip(action, -1.0, 1.0))

        s_next = s + ds
        next_state = transition.get_next_state(
            d,
            theta,
            k,
            action,
            lane_generator=lane,
            progress_s=s_next,
            use_uncertainty=False,
        )

        d, theta, k = next_state
        s = min(s_next, lane_len)

        s_hist.append(float(s))
        d_hist.append(float(d))
        theta_hist.append(float(theta))

        if abs(d) > (lane.lane_width / 2.0):
            offroad = True
            break
        if s >= lane_len:
            reached_end = True
            break

    return {
        "s": np.array(s_hist),
        "d": np.array(d_hist),
        "theta": np.array(theta_hist),
        "start_s": float(start_s),
        "offroad": offroad,
        "reached_end": reached_end,
    }


def lane_xy_from_sd(s_vals, d_vals, road_s, road_x, road_y, road_psi):
    x_center, y_center, psi_center = interpolate_centerline_at_s(s_vals, road_s, road_x, road_y, road_psi)
    x_car = x_center - d_vals * np.sin(psi_center)
    y_car = y_center + d_vals * np.cos(psi_center)
    return x_car, y_car


def summarize_rollouts(rollouts, lane_len):
    completions = []
    rms_errors = []
    offroads = 0
    reached_end = 0

    for r in rollouts:
        remaining = max(1e-6, lane_len - r["start_s"])
        completion = (r["s"][-1] - r["start_s"]) / remaining
        completion = float(np.clip(completion, 0.0, 1.0))
        completions.append(completion)
        rms_errors.append(float(np.sqrt(np.mean(r["d"] ** 2))))
        if r["offroad"]:
            offroads += 1
        if r["reached_end"]:
            reached_end += 1

    return {
        "mean_completion": float(np.mean(completions)),
        "median_completion": float(np.median(completions)),
        "mean_rms_d": float(np.mean(rms_errors)),
        "offroad_rate": float(offroads / len(rollouts)),
        "end_reach_rate": float(reached_end / len(rollouts)),
    }


def save_showcase_plot(path, lane, road_x, road_y, road_psi, rollout, road_s):
    half_w = lane.lane_width / 2.0
    left_x = road_x - half_w * np.sin(road_psi)
    left_y = road_y + half_w * np.cos(road_psi)
    right_x = road_x + half_w * np.sin(road_psi)
    right_y = road_y - half_w * np.cos(road_psi)

    x_car, y_car = lane_xy_from_sd(rollout["s"], rollout["d"], road_s, road_x, road_y, road_psi)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Policy Showcase (start s=0)")

    ax.plot(left_x, left_y, "k--", linewidth=1)
    ax.plot(right_x, right_y, "k--", linewidth=1, label="Lane boundaries")
    ax.plot(road_x, road_y, color="gray", linewidth=1, alpha=0.7, label="Centerline")
    ax.plot(x_car, y_car, color="tab:blue", linewidth=2, label="Vehicle trajectory")
    ax.scatter([x_car[0]], [y_car[0]], color="green", s=40, label="Start")
    ax.scatter([x_car[-1]], [y_car[-1]], color="red", s=40, label="End")

    pad = 2.0
    ax.set_xlim(min(left_x.min(), right_x.min()) - pad, max(left_x.max(), right_x.max()) + pad)
    ax.set_ylim(min(left_y.min(), right_y.min()) - pad, max(left_y.max(), right_y.max()) + pad)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_eval_spread_plot(path, lane, road_x, road_y, road_psi, road_s, rollouts):
    half_w = lane.lane_width / 2.0
    left_x = road_x - half_w * np.sin(road_psi)
    left_y = road_y + half_w * np.cos(road_psi)
    right_x = road_x + half_w * np.sin(road_psi)
    right_y = road_y - half_w * np.cos(road_psi)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Evaluation Spread (random starts)")

    ax.plot(left_x, left_y, "k--", linewidth=1)
    ax.plot(right_x, right_y, "k--", linewidth=1)
    ax.plot(road_x, road_y, color="gray", linewidth=1, alpha=0.6)

    for r in rollouts:
        x_car, y_car = lane_xy_from_sd(r["s"], r["d"], road_s, road_x, road_y, road_psi)
        if r["offroad"]:
            ax.plot(x_car, y_car, color="tab:red", alpha=0.35, linewidth=1)
        else:
            ax.plot(x_car, y_car, color="tab:blue", alpha=0.25, linewidth=1)

    pad = 2.0
    ax.set_xlim(min(left_x.min(), right_x.min()) - pad, max(left_x.max(), right_x.max()) + pad)
    ax.set_ylim(min(left_y.min(), right_y.min()) - pad, max(left_y.max(), right_y.max()) + pad)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(policy_path="policy.pt", num_eval=24):
    actor = load_actor(policy_path)
    lane = Lane()
    transition = Transition()

    ds = transition.velocity * transition.dt
    max_steps = int(np.ceil(lane.lane_length() / ds)) + 20

    road_x, road_y, road_psi, road_s = lane.sample_centerline(ds=0.1)

    showcase = run_policy_rollout(
        actor,
        lane,
        transition,
        max_steps=max_steps,
        start_s=0.0,
        random_lateral_init=True,
    )

    eval_rollouts = []
    for _ in range(num_eval):
        start_s = np.random.rand() * lane.lane_length()
        rollout = run_policy_rollout(
            actor,
            lane,
            transition,
            max_steps=max_steps,
            start_s=start_s,
            random_lateral_init=True,
        )
        eval_rollouts.append(rollout)

    summary = summarize_rollouts(eval_rollouts, lane.lane_length())

    save_showcase_plot("policy_showcase.png", lane, road_x, road_y, road_psi, showcase, road_s)
    save_eval_spread_plot("policy_eval_spread.png", lane, road_x, road_y, road_psi, road_s, eval_rollouts)

    print("Saved policy_showcase.png")
    print("Saved policy_eval_spread.png")
    print(
        "Eval stats | "
        f"mean completion={summary['mean_completion']:.3f}, "
        f"offroad rate={summary['offroad_rate']:.3f}, "
        f"end reach rate={summary['end_reach_rate']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="policy.pt")
    parser.add_argument("--num-eval", type=int, default=24)
    args = parser.parse_args()
    main(policy_path=args.policy, num_eval=args.num_eval)
