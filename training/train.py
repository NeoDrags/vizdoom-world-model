import torch
import yaml
import gymnasium
import torch.optim as optim
import torch.nn.functional as F
import vizdoom
from vizdoom import gymnasium_wrapper
import numpy as np

# Custom Module Imports
import sys
sys.path.append(".")
from models.vae import VAE
from models.mdnrnn import MDNRNN
from training.train_vae import train_vae
from training.train_mdnrnn import train_mdnrnn
from training.train_controller import train_controller
from scripts.env_setup import collect_rollouts, batch_frames

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Vizdoom env variables
VIZDOOM_CONFIG = CONFIG["vizdoom-env"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


VAE_TRAINING_PARAMS = CONFIG["vae"]["training-params"]
VAE_MODEL_LOC = CONFIG["vae"]["model-path"]

MDNRNN_TRAINING_PARAMS = CONFIG["mdnrnn"]["training-params"]
MDNRNN_MODEL_LOC = CONFIG["mdnrnn"]["model-path"]

def main():
    env = gymnasium.make(VIZDOOM_CONFIG, screen_resolution=vizdoom.ScreenResolution.RES_200X125)
    frames, actions, game_states, rewards, done = collect_rollouts(env=env, episodes=CONFIG["episodes"])

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
                vae=vae,
                mdnrnn=mdnrnn,
                lr=float(MDNRNN_TRAINING_PARAMS["lr"]),
                epochs=int(MDNRNN_TRAINING_PARAMS["epochs"]),
                frames=frames,
                actions=actions,
                game_states=game_states,
                rewards=rewards,
                done_state=done,
                device=DEVICE,
                lambda_reward=float(MDNRNN_TRAINING_PARAMS["lambda-reward"]),
                lambda_done=float(MDNRNN_TRAINING_PARAMS["lambda-done"])
                )
    else:
        print(f"Loading mdn-rnn from: {MDNRNN_MODEL_LOC} ...")
        mdnrnn_dict = torch.load(MDNRNN_MODEL_LOC)
        mdnrnn.load_state_dict(state_dict=mdnrnn_dict)

if __name__ == "__main__":
    main()
