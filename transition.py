import numpy as np


class Transition:
    def __init__(self, velocity=4.0, velocity_uncertainty_std=0.1, state_uncertainty_std=0.1, action_uncertainty_std=0.1, dt=0.1, max_turn_rate=np.pi/6, ):
        self.velocity = velocity
        self.velocity_uncertainty_std = velocity_uncertainty_std
        self.state_uncertainty_std = state_uncertainty_std
        self.action_uncertainty_std = action_uncertainty_std
        self.dt = dt
        self.max_turn_rate = max_turn_rate

    def get_next_state(self, d, theta, k, action, lane_generator=None, progress_s=None, use_uncertainty=True):
        if use_uncertainty:
            v = self.velocity * (1 + self.velocity_uncertainty_std*np.random.randn())
            action_norm = action + self.action_uncertainty_std*np.random.randn()
        else:
            v = self.velocity
            action_norm = action
        action_norm = np.clip(action_norm, -1.0, 1.0)
        turn_rate = action_norm * self.max_turn_rate
        
        dd = v * np.sin(theta)
        dtheta = turn_rate - k * v
        
        if use_uncertainty:
            d_new = (d + dd * self.dt) + self.state_uncertainty_std * np.random.randn()
            theta_new = (theta + dtheta * self.dt) + self.state_uncertainty_std * np.random.randn()
        else:
            d_new = d + dd * self.dt
            theta_new = theta + dtheta * self.dt
        theta_new = np.atan2(np.sin(theta_new), np.cos(theta_new))             

        if lane_generator is not None:
            provisional_state = np.array([d_new, theta_new, k])
            k_new = lane_generator.get_state_curvature(provisional_state, progress_s=progress_s)
        else:
            k_new = k

        next_state = np.array([d_new, theta_new, k_new])
        return next_state