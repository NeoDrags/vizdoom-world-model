import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.controller = nn.Linear(
            latent_dim + hidden_dim,
            action_dim
        )

    def forward(self, z, h):
        x = torch.cat([z, h], dim=1)
        return self.controller(x)