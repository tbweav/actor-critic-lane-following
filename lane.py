import torch
import numpy as np

class Lane:
    def __init__(self, d_init_std=0.2, theta_init_std=0.1, seed=1):
        self.d_init_std = d_init_std
        self.theta_init_std = theta_init_std
        self.lane_width = 3.6
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        self.complex_segments = [
            (20.0, 0.00),
            (12.0, 0.015),
            (15.0, 0.00),
            (12.0, -0.030),
            (15.0, 0.00),
            (10.0, 0.050),
            (12.0, 0.00),
            (10.0, -0.070),
            (18.0, 0.00),
        ]
        self.profile_segments_map = {
            "complex_road": list(self.complex_segments)
        }
        self.active_lane_id = "complex_road"

    def lane_curvature_at_s(self, lane_id, s_value):
        segments = self.profile_segments_map[lane_id]
        s = max(0.0, float(s_value))

        total_len = sum(seg_len for seg_len, _ in segments)
        if s > total_len:
            s = total_len

        accum = 0.0
        for seg_len, seg_k in segments:
            if s <= accum + seg_len:
                return float(seg_k)
            accum += seg_len

        return float(segments[-1][1])

    def lane_length(self, lane_id=None):
        lane_id = self.active_lane_id if lane_id is None else lane_id
        return sum(seg_len for seg_len, _ in self.profile_segments_map[lane_id])

    def sample_centerline(self, lane_id=None, ds=0.1):
        lane_id = self.active_lane_id if lane_id is None else lane_id
        total_len = self.lane_length(lane_id)

        s_values = np.arange(0.0, total_len + ds, ds, dtype=np.float32)
        x = np.zeros_like(s_values)
        y = np.zeros_like(s_values)
        psi = np.zeros_like(s_values)

        for idx in range(1, len(s_values)):
            s_mid = min(float(s_values[idx - 1] + 0.5 * ds), total_len)
            k = self.lane_curvature_at_s(lane_id, s_mid)
            psi[idx] = psi[idx - 1] + k * ds
            x[idx] = x[idx - 1] + ds * np.cos(psi[idx - 1])
            y[idx] = y[idx - 1] + ds * np.sin(psi[idx - 1])

        return x, y, psi, s_values

    def get_state_curvature(self, state, progress_s=None):
        if progress_s is None:
            s_val = 0.0
        else:
            s_val = float(progress_s)
        k = self.lane_curvature_at_s(self.active_lane_id, s_val)
        return np.array(k, dtype=np.float32)