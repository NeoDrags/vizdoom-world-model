import torch
import torch.nn as nn
from torch.distributions import Independent, Categorical, Normal, MixtureSameFamily

class MDNRNN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_dim, game_state_dim, mixtures):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.m = mixtures
        self.l = latent_dim
        self.rnn = nn.LSTM(latent_dim + action_dim + game_state_dim, hidden_dim)
        self.mdn = nn.Linear(hidden_dim, mixtures * (2 * latent_dim + 1))
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def makeGMM(self, mu, sigma, pi):
        categories = Categorical(logits=pi)
        norm = Independent(Normal(mu, sigma), 1)
        return MixtureSameFamily(categories, norm)

    def forward(self, z, a, g, h, c, tau = 1):
        inp = torch.cat([z, a, g], dim=-1)
        output, (h, c) = self.rnn(inp, (h, c))
        v = self.mdn(output).reshape(-1, self.m * (2 * self.l + 1))
        pi = v[:, : self.m]
        mu = v[:, self.m : self.m + self.m * self.l]
        sigma = v[:, self.m + self.m * self.l :]
        mu = mu.view(-1, self.m, self.l)
        sigma = sigma.view(-1, self.m, self.l)
        sigma = torch.exp(sigma)
        sigma = sigma * tau ** 1/2
        pi = pi/tau
        gmm = self.makeGMM(mu, sigma, pi)
        reward = self.reward_head(output)
        done = self.done_head(output) 
        return gmm, h, c, reward, done
