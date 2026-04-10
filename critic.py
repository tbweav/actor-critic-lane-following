import torch
from torch import nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, action_value_dim, hlayers, hwidth):
        super(Critic, self).__init__()

        layers = []

        layers.append(nn.Linear(state_dim+action_dim, hwidth))
        layers.append(nn.ReLU())

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(hwidth, hwidth))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hwidth, action_value_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        action_value_function = self.network(x)
        return action_value_function
