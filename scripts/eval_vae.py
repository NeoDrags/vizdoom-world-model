import sys
sys.path.append(".")
import yaml
import torch
import gymnasium
import matplotlib.pyplot as plt
from vizdoom import gymnasium_wrapper
import vizdoom
from models.vae import VAE
from scripts.env_setup import collect_rollouts

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gymnasium.make(CONFIG["vizdoom-env"],screen_resolution=vizdoom.ScreenResolution.RES_200X125)

frames, _, _ = collect_rollouts(
    env,
    episodes=1
)

vae = VAE()
vae.load_state_dict(torch.load(CONFIG["vae"]["model-path"], map_location=DEVICE))

vae = vae.to(DEVICE)
vae.eval()

idx = torch.randint(0,len(frames),(1,)).item()

frame = frames[idx].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    recon, mu, logvar = vae(frame)

frame = frame.squeeze(0).cpu().permute(1, 2, 0)
print(recon.min().item())
print(recon.max().item())
print(recon.mean().item())
recon = torch.sigmoid(
    recon.squeeze(0)
).cpu().permute(1, 2, 0)

print(f"Sigma: Mean -> {torch.mean(torch.exp(logvar/2)):.4f} | Min -> {torch.min(torch.exp(logvar/2)):.4f} | Max -> {torch.max(torch.exp(logvar/2))}")
print(f"Mu: Mean -> {torch.mean(mu):.4f} | Min -> {torch.min(mu):.4f} | Max -> {torch.max(mu)}")

print(recon.min().item())
print(recon.max().item())
print(recon.mean().item())

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(frame)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(recon)
ax[1].set_title("Reconstruction")
ax[1].axis("off")

plt.tight_layout()
plt.savefig("./imgs/vae_reconstruction_3.png", dpi=300)
plt.close()