import os
import numpy as np
import matplotlib.pyplot as plt

from lane import Lane


def _apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.facecolor": "#f7f6f3",
            "figure.facecolor": "white",
            "axes.edgecolor": "#4b4b4b",
            "grid.color": "#b9b6ae",
            "grid.alpha": 0.35,
        }
    )


def make_d_visual(lane_width, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    half_w = lane_width / 2.0

    road_len = 18.0
    ax.fill_between([0, road_len], -half_w, half_w, color="#e7ecf3", zorder=0)
    ax.plot([0, road_len], [half_w, half_w], "k--", linewidth=1.3)
    ax.plot([0, road_len], [-half_w, -half_w], "k--", linewidth=1.3)
    ax.plot([0, road_len], [0, 0], color="#566573", linewidth=2.0, label="Lane centerline")

    x_car = 8.5
    d_val = 0.9
    y_car = d_val

    ax.scatter([x_car], [y_car], s=180, color="#c23b22", zorder=5, label="Vehicle")
    ax.annotate(
        "",
        xy=(x_car, y_car),
        xytext=(x_car, 0.0),
        arrowprops=dict(arrowstyle="<->", color="black", linewidth=2.0),
        zorder=4,
    )
    ax.text(x_car + 0.25, y_car / 2.0, r"$d$", color="black", fontsize=16, va="center")

    ax.set_title(r"State Variable: $d$")
    ax.set_xlabel("Longitudinal direction")
    ax.set_ylabel("Lateral position [m]")
    ax.set_xlim(0, road_len)
    ax.set_ylim(-half_w - 0.8, half_w + 0.8)
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_theta_visual(out_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    origin = np.array([5.0, 2.0])
    heading_len = 3.2
    theta_val = np.deg2rad(22.0)

    tangent_dir = np.array([1.0, 0.0])
    vehicle_dir = np.array([np.cos(theta_val), np.sin(theta_val)])

    ax.plot([0, 11], [2, 2], color="#566573", linewidth=2.0, label="Lane tangent direction")

    ax.annotate(
        "",
        xy=origin + heading_len * tangent_dir,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", color="#2471a3", linewidth=3),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=origin + heading_len * vehicle_dir,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", color="#c23b22", linewidth=3),
        zorder=4,
    )

    arc_r = 1.3
    arc_t = np.linspace(0.0, theta_val, 80)
    ax.plot(origin[0] + arc_r * np.cos(arc_t), origin[1] + arc_r * np.sin(arc_t), color="black", linewidth=2.2)
    ax.text(origin[0] + 1.64, origin[1] + 0.17, r"$\theta$", color="black", fontsize=16)

    ax.set_title(r"State Variable: $\theta$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 11)
    ax.set_ylim(0.4, 5.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_k_visual(out_path):
    fig, ax = plt.subplots(figsize=(10, 5.2))

    x = np.linspace(0.0, 20.0, 220)
    y_left = 0.018 * (x - 10.0) ** 2 + 1.6
    y_mid = np.full_like(x, 0.0)
    y_right = -0.018 * (x - 10.0) ** 2 - 1.6

    ax.plot(x, y_left, color="#2471a3", linewidth=3.0)
    ax.plot(x, y_mid, color="#566573", linewidth=3.0)
    ax.plot(x, y_right, color="#c23b22", linewidth=3.0)

    ax.text(20.35, y_left[-1], r"$k > 0$", fontsize=15, color="#2471a3", va="center")
    ax.text(20.35, y_mid[-1], r"$k = 0$", fontsize=15, color="#566573", va="center")
    ax.text(20.35, y_right[-1], r"$k < 0$", fontsize=15, color="#c23b22", va="center")

    ax.set_title(r"State Variable: $\kappa$")
    ax.set_xlabel("Lane Progress")
    ax.set_ylabel("Lane Shape")
    ax.set_xlim(0.0, 23.5)
    ax.set_ylim(-5.0, 5.0)
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_lane_geometry_visual(out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    lane = Lane()
    road_x, road_y, road_psi, road_s = lane.sample_centerline(ds=0.1)
    half_w = lane.lane_width / 2.0

    left_x = road_x - half_w * np.sin(road_psi)
    left_y = road_y + half_w * np.cos(road_psi)
    right_x = road_x + half_w * np.sin(road_psi)
    right_y = road_y - half_w * np.cos(road_psi)

    ax.plot(left_x, left_y, color="black", linewidth=2.2, linestyle="--", label="Lane boundary")
    ax.plot(right_x, right_y, color="black", linewidth=2.2, linestyle="--")
    ax.plot(road_x, road_y, color="#566573", linewidth=3.0, label="Centerline")

    ax.set_title("Lane Geometry")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="upper left")

    pad = 2.0
    ax.set_xlim(min(left_x.min(), right_x.min()) - pad, max(left_x.max(), right_x.max()) + pad)
    ax.set_ylim(min(left_y.min(), right_y.min()) - pad, max(left_y.max(), right_y.max()) + pad)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(output_dir="images"):
    _apply_style()
    os.makedirs(output_dir, exist_ok=True)

    lane = Lane()
    d_path = os.path.join(output_dir, "state_d.png")
    theta_path = os.path.join(output_dir, "state_theta.png")
    k_path = os.path.join(output_dir, "state_k.png")
    lane_geometry_path = os.path.join(output_dir, "lane_geometry.png")

    make_d_visual(lane.lane_width, d_path)
    make_theta_visual(theta_path)
    make_k_visual(k_path)
    make_lane_geometry_visual(lane_geometry_path)

    print(f"Saved {d_path}")
    print(f"Saved {theta_path}")
    print(f"Saved {k_path}")
    print(f"Saved {lane_geometry_path}")


if __name__ == "__main__":
    main()