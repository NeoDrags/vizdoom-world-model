import torch
from torch.utils.data import dataloader
import yaml
import gymnasium
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import vizdoom
from vizdoom import gymnasium_wrapper
import numpy as np
from datetime import datetime
import cma
import torchvision.transforms as tf

# Custom Module Imports
import sys
sys.path.append(".")
from models.vae import VAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from scripts.env_setup import collect_rollouts, batch_frames

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Vizdoom env variables
VIZDOOM_CONFIG = CONFIG["vizdoom-env"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

VAE_TRAINING_PARAMS = CONFIG["vae"]["training-params"]
VAE_MODEL_LOC = CONFIG["vae"]["model-path"]

MDNRNN_TRAINING_PARAMS = CONFIG["mdnrnn"]["training-params"]
MDNRNN_MODEL_LOC = CONFIG["mdnrnn"]["model-path"]

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
            x_hat, mu, logvar = vae(episode)
            recon_loss = F.binary_cross_entropy_with_logits(x_hat, episode, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl = torch.clamp(kl, min=0.2)
            loss = recon_loss + beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / e_id
            pbar.set_postfix({"loss": total_loss, "kl_loss": kl.item(), "recon_loss": recon_loss.item()})
            e_id += 1
        if total_loss < lowest_loss:
            lowest_loss = total_loss
            torch.save(vae.state_dict(), f"./trained_models/vae_{timestamp}.pt")
            torch.save(vae.state_dict(), './trained_models/latest_vae.pt')
            print(f"Saving model... loss = {lowest_loss}")
        scheduler.step(total_loss)

def train_mdnrnn(vae, mdnrnn, lr, epochs, frames, actions, game_states, hidden_size):
    pass

def main():
    env = gymnasium.make(VIZDOOM_CONFIG, screen_resolution=vizdoom.ScreenResolution.RES_200X125)
    frames, actions, game_states = collect_rollouts(env=env, episodes=CONFIG["episodes"])

    # Training VAE
    vae = VAE(CONFIG["latent-dim"]).to(DEVICE)
    if CONFIG["vae"]["train"]:
        print("Training vae....")
        frames_loader = batch_frames(frames, VAE_TRAINING_PARAMS["batch-size"])
        train_vae(
            vae=vae, 
            dataloader=frames_loader,
            lr=float(VAE_TRAINING_PARAMS["lr"]),
            epochs=int(VAE_TRAINING_PARAMS["epochs"]),
            device=DEVICE,
            beta = float(VAE_TRAINING_PARAMS["beta"]),
        )
    else:
        print(f"Loading vae from: {VAE_MODEL_LOC} ...")
        vae_dict = torch.load(VAE_MODEL_LOC)
        vae.load_state_dict(vae_dict)
    
    # Training mdnrnn
    mdnrnn = MDNRNN(
            latent_dim=CONFIG["latent-dim"],
            hidden_dim=MDNRNN_TRAINING_PARAMS["hidden-dim"],
            action_dim=CONFIG["action-dim"],
            game_state_dim=CONFIG["game-state-dim"],
            mixtures=MDNRNN_TRAINING_PARAMS["mixtures"]
            ).to(DEVICE)

    if CONFIG["mdnrnn"]["train"]:
        print("Training mdn-rnn....")
        train_mdnrnn(
                vae,
                mdnrnn,
                lr = float(MDNRNN_TRAINING_PARAMS["lr"]),
                epochs=int(MDNRNN_TRAINING_PARAMS["epochs"]),
                frames=frames,
                actions=actions,
                game_states=game_states,
                hidden_size=int(MDNRNN_TRAINING_PARAMS["hidden-dim"]),
                )
    else:
        print(f"Loading mdn-rnn from: {MDNRNN_MODEL_LOC} ...")
        mdnrnn_dict = torch.load(MDNRNN_MODEL_LOC)
        mdnrnn.load_state_dict(state_dict=mdnrnn_dict)

    
    # controller = Controller(latent_dim=1024, hidden_dim=2048, action_dim=4)
    # train_controller(
    #     controller=controller,
    #     vae=vae,
    #     mdnrnn=mdnrnn,
    #     num_eval_episodes=4,
    #     env=env,
    #     hidden_dim=2048
    # ) 


if __name__ == "__main__":
    main()
