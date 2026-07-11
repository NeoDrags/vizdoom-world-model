import torch
import torch.nn as nn
from torch.distributions import Independent, Categorical, Normal, MixtureSameFamily

class MDNRNN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_dim, game_state_dim, mixtures):
        super().__init__()
        self.m = mixtures
        self.l = latent_dim
        self.rnn = nn.LSTMCell(latent_dim + action_dim + game_state_dim, hidden_dim)
        self.mdn = nn.Linear(hidden_dim, mixtures * (2 * latent_dim + 1))

    def makeGMM(self, mu, sigma, pi):
        categories = Categorical(logits=pi)
        norm = Independent(Normal(mu, sigma), 1)
        return MixtureSameFamily(categories, norm)

    def forward(self, z, a, g, hidden, tau):
        inp = torch.cat([z, a, g], dim=1)
        h, c = self.rnn(inp, hidden)
        v = self.mdn(h)
        tau = torch.tensor(tau)
        pi = v[:, : self.m]
        mu = v[:, self.m : self.m + self.m * self.l]
        sigma = v[:, self.m + self.m * self.l :]
        mu = mu.view(-1, self.m, self.l)
        sigma = sigma.view(-1, self.m, self.l)
        sigma = torch.exp(sigma)
        sigma = sigma * torch.sqrt(tau)
        pi = pi/tau
        gmm = self.makeGMM(mu, sigma, pi)
        return gmm, (h, c)
