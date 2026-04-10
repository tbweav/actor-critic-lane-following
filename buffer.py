import torch
import numpy as np

class Buffer:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32).copy()
        action = float(action)
        reward = float(reward)
        next_state = np.array(next_state, dtype=np.float32).copy()
        done = float(done)

        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i.item()] for i in indices])

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        batch = states, actions, rewards, next_states, dones
        return batch

    def __len__(self):
        return len(self.buffer)