import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hlayers, hwidth):
        super(Actor, self).__init__()

        layers = []

        layers.append(nn.Linear(state_dim, hwidth))
        layers.append(nn.ReLU())

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(hwidth, hwidth))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hwidth, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        action = self.network(state)
        return action
