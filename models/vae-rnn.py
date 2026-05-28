import torch
import torch.nn as nn
import math

class VAE(nn.Module):
        def __init__(self, d=512):
            super(VAE, self).__init__()
            
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),    # 224 -> 112
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # 112 -> 56
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56 -> 28
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 28 -> 14
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1), # 14 -> 7
                nn.ReLU(),
                nn.Flatten()
            )
            
            self.fc_mu     = nn.Linear(12544, d)
            self.fc_logvar = nn.Linear(12544, d)
            
            self.decoder_input = nn.Linear(d, 12544)
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1), # 7  -> 14
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 14 -> 28
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28 -> 56
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 56 -> 112
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 112 -> 224
                nn.Sigmoid()  # Normalize for output
            )

        def reparameterise(self, mu, logvar):
            sigma = torch.exp(0.5 * logvar)
            eps   = torch.randn_like(sigma)
            return mu + sigma * eps

        def encode(self, x):
            h      = self.encoder(x)
            mu     = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def decode(self, z):
            h = self.decoder_input(z)
            h = h.view(-1, 256, 7, 7)
            return self.decoder(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z          = self.reparameterise(mu, logvar)
            x_hat      = self.decode(z)
            return x_hat, mu, logvar
        