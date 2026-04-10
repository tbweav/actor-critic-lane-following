import numpy as np
import torch

from lane import Lane
from transition import Transition


class Rollout:
    def __init__(self, buffer, randomize_progress=True):
        self.buffer = buffer
        self.transition = Transition()
        self.lane = Lane()
        self.randomize_progress = bool(randomize_progress)

        self.max_steps = int(np.ceil(self.lane.lane_length() / (self.transition.velocity * self.transition.dt))) + 20
        self.lane_width = self.lane.lane_width
        self.explore_std = 0.2

        self.state = None
        self.progress_s = 0.0
        self.step_count = 0

    def _initialize_state(self, progress_s):
        d_init = self.lane.d_init_std * np.random.randn()
        theta_init = self.lane.theta_init_std * np.random.randn()
        k_init = self.lane.lane_curvature_at_s(self.lane.active_lane_id, progress_s)
        return d_init, theta_init, k_init

    def reset(self, randomize_progress=None):
        if randomize_progress is None:
            randomize_progress = self.randomize_progress

        if randomize_progress:
            self.progress_s = np.random.rand() * self.lane.lane_length()
        else:
            self.progress_s = 0.0

        d, theta, k = self._initialize_state(self.progress_s)
        self.state = np.array([d, theta, k], dtype=np.float32)
        self.step_count = 0
        return self.state

    def _get_action(self, actor, state, explore=True):
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = actor(s_tensor).squeeze(0).item()
        if explore:
            a = a + self.explore_std * np.random.randn()
        a = np.clip(a, -1.0, 1.0)
        return float(a)

    def step(self, actor, explore=True, use_uncertainty=True, store_transition=True):
        if self.state is None:
            self.reset()

        d, theta, k = self.state
        action = self._get_action(actor, self.state, explore=explore)

        ds = self.transition.velocity * self.transition.dt
        next_progress = self.progress_s + float(ds)

        next_state = self.transition.get_next_state(
            d,
            theta,
            k,
            action,
            lane_generator=self.lane,
            progress_s=next_progress,
            use_uncertainty=use_uncertainty,
        )
        reward = - (next_state[0] ** 2 + next_state[1] ** 2)

        self.step_count += 1
        done = bool(abs(next_state[0]) > (self.lane_width / 2.0) or self.step_count >= self.max_steps)

        if store_transition:
            self.buffer.add(self.state, action, reward, next_state, done)

        self.state = next_state
        self.progress_s = next_progress
        if done:
            self.state = None

        return reward, done

    def perform_rollout(self, actor, explore=True, use_uncertainty=True, store_transition=True, randomize_progress=None):
        self.reset(randomize_progress=randomize_progress)
        total_reward = 0.0
        done = False
        while not done:
            reward, done = self.step(actor, explore=explore, use_uncertainty=use_uncertainty, store_transition=store_transition)
            total_reward += reward
        return total_reward