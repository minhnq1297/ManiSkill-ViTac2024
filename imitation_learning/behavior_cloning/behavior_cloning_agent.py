import torch
import torch.nn as nn

class BCAgent(nn.Module):
    def __init__(self, feature_dim, hidden_dim, actions_dim):
        super(BCAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, actions_dim),
            nn.Tanh()
        )

    def forward(self, encoding):
        return self.net(encoding)
