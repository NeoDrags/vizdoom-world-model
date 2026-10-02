import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim = 1024):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 16 -> 8
            nn.ReLU(),
            nn.Flatten()
        )

        self.mu = nn.Linear(8192, latent_dim)
        self.logvar = nn.Linear(8192, latent_dim)
        self.restructure = nn.Linear(latent_dim, 8192)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), # 32 -> 64
        )

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma 

    def forward(self, x):
        x_hat = self.encoder(x)
        mu = self.mu(x_hat)
        logvar = self.logvar(x_hat)
        z = self.reparametrize(mu, logvar)
        x_hat = self.restructure(z)
        x_hat = x_hat.view(-1, 128, 8, 8)
        x_hat = self.decoder(x_hat)
        return x_hat, z, mu, logvar
