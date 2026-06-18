import torch
import yaml
import gymnasium
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import vizdoom
from vizdoom import gymnasium_wrapper
import numpy as np
from datetime import datetime

# Custom Module Imports
import sys
sys.path.append(".")
from models.vae import VAE
from scripts.env_setup import collect_rollouts, batch_frames

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

VIZDOOM_CONFIG = CONFIG["vizdoom-env"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_TRAINING_PARAMS = CONFIG["vae"]["training-params"]
if CONFIG["vae"]["train"]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CONFIG["vae"]["model-path"] = f"./trained_models/vae_{timestamp}.pt"
    with open("config.yaml", "r+") as f:
        yaml.dump(CONFIG, f)
VAE_MODEL_LOC = CONFIG["vae"]["model-path"]

def train_vae(vae, dataloader, lr, epochs, device):
    optimizer = optim.Adam(vae.parameters(), lr)
    lowest_loss = float(np.inf)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Training Epoch: {epoch + 1}/{epochs}")
        total_loss = 0
        for episode in pbar:
            episode = episode.to(device)
            x_hat, mu, logvar = vae(episode)
            recon_loss = F.binary_cross_entropy_with_logits(x_hat, episode, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl = torch.clamp(kl, min=0.3)
            loss = recon_loss + kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / len(pbar)
            pbar.set_postfix({"loss": total_loss, "kl_loss": kl.item(), "recon_loss": recon_loss.item()})
        if total_loss < lowest_loss:
            lowest_loss = total_loss
            torch.save(vae.state_dict(), VAE_MODEL_LOC)
            print(f"Saving model... loss = {lowest_loss}")
        scheduler.step(total_loss)


def main():
    env = gymnasium.make(VIZDOOM_CONFIG, screen_resolution=vizdoom.ScreenResolution.RES_200X125)
    frames, _, _, _ = collect_rollouts(env=env, episodes=CONFIG["episodes"])
    vae = VAE().to(DEVICE)
    # Training VAE
    if CONFIG["vae"]["train"]:
        frames_loader = batch_frames(frames, VAE_TRAINING_PARAMS["batch-size"])
        train_vae(
            vae=vae, 
            dataloader=frames_loader,
            lr=float(VAE_TRAINING_PARAMS["lr"]),
            epochs=int(VAE_TRAINING_PARAMS["epochs"]),
            device=DEVICE
        )
    else:
        vae_dict = torch.load(VAE_MODEL_LOC)
        vae = vae.load_state_dict(vae_dict)

if __name__ == "__main__":
    main()
