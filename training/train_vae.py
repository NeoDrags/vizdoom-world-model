import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def train_vae(vae, dataloader, lr, epochs, device, beta):
    optimizer = optim.Adam(vae.parameters(), lr)
    lowest_loss = float(np.inf)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Training Epoch: {epoch + 1}/{epochs}")
        total_loss = 0
        e_id = 1
        for episode in pbar:
            episode = episode.to(device)
            x_hat, _, mu, logvar = vae(episode)
            recon_loss = F.binary_cross_entropy_with_logits(x_hat, episode, reduction="sum") / len(episode)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(episode)
            loss = recon_loss + beta * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            curr_loss = total_loss/e_id
            pbar.set_postfix({"loss": curr_loss, "kl_loss": kl.item(), "recon_loss": recon_loss.item()})
            e_id += 1
        if curr_loss < lowest_loss:
            lowest_loss = curr_loss
            torch.save(vae.state_dict(), f"./trained_models/vae_{timestamp}.pt")
            torch.save(vae.state_dict(), './trained_models/latest_vae.pt')
            print(f"Saving model... loss = {curr_loss}")
        scheduler.step(total_loss)