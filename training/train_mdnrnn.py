import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import random

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def encode_frames(vae, frames):
    vae.eval()
    with torch.no_grad():
        z = vae(frames)
    return z.to(device="cpu")

def create_sequence_batches(frames, actions, game_states, rewards, done_state, seq_length, batch_size):
    sequences = []
    for ep_frames, ep_actions, ep_gs, ep_rewards, ep_done in zip(frames, actions, game_states, rewards, done_state):
        if len(ep_frames) < 2:
            continue
        if isinstance(ep_frames, list):
            ep_z = torch.stack(ep_frames)
        else:
            ep_z = ep_frames
        if isinstance(ep_actions, torch.Tensor):
            ep_a = ep_actions
        else:
            ep_a = torch.as_tensor(ep_actions)
        if isinstance(ep_gs, list):
            try:
                ep_g = torch.as_tensor(np.array(ep_gs))
            except Exception:
                ep_g = torch.as_tensor(ep_gs)
        else:
            ep_g = ep_gs if isinstance(ep_gs, torch.Tensor) else torch.as_tensor(ep_gs)
        if isinstance(ep_rewards, list):
            ep_r = torch.as_tensor(ep_rewards, dtype=torch.float32)
        else:
            ep_r = ep_rewards.float() if isinstance(ep_rewards, torch.Tensor) else torch.as_tensor(ep_rewards, dtype=torch.float32)
        if isinstance(ep_done, list):
            ep_d = torch.as_tensor(ep_done, dtype=torch.float32)
        else:
            ep_d = ep_done.float() if isinstance(ep_done, torch.Tensor) else torch.as_tensor(ep_done, dtype=torch.float32)
        for i in range(0, len(ep_z) - seq_length):
            sequences.append((ep_z[i:i + seq_length], ep_a[i:i + seq_length], ep_g[i:i + seq_length], ep_r[i:i + seq_length], ep_d[i:i + seq_length]))
    random.shuffle(sequences)
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        if len(batch) == 0:
            continue
        z_batch = torch.stack([b[0] for b in batch], dim=1)
        a_batch = torch.stack([b[1] for b in batch], dim=1)
        g_batch = torch.stack([b[2] for b in batch], dim=1)
        r_batch = torch.stack([b[3] for b in batch], dim=1)
        d_batch = torch.stack([b[4] for b in batch], dim=1)
        yield z_batch, a_batch, g_batch, r_batch, d_batch

def train_mdnrnn(vae, mdnrnn, lr, epochs, frames, actions, game_states, rewards, done_state, device, 
                 batch_size=16, seq_length=32, lambda_reward=1.0, lambda_done=1.0):
    vae.eval()
    encoded_frames = []
    for ep in frames:
        if isinstance(ep, list):
            ep_tensor = torch.stack(ep).to(device)
        else:
            ep_tensor = ep.to(device)
        with torch.no_grad():
            _, z, _, _ = vae(ep_tensor)
        encoded_frames.append(z.cpu())
    frames = encoded_frames
    actions_proc = []
    for ep_a in actions:
        if isinstance(ep_a, torch.Tensor):
            actions_proc.append(ep_a.float() if ep_a.dtype != torch.float32 else ep_a)
        else:
            actions_proc.append(torch.as_tensor(ep_a, dtype=torch.float32))
    game_states_proc = []
    for ep_g in game_states:
        if isinstance(ep_g, list):
            try:
                game_states_proc.append(torch.as_tensor(np.array(ep_g), dtype=torch.float32))
            except Exception:
                game_states_proc.append(torch.as_tensor(ep_g, dtype=torch.float32))
        else:
            game_states_proc.append(ep_g.float() if isinstance(ep_g, torch.Tensor) and ep_g.dtype != torch.float32 else (torch.as_tensor(ep_g, dtype=torch.float32) if not isinstance(ep_g, torch.Tensor) else ep_g))
    rewards_proc = []
    for ep_r in rewards:
        if isinstance(ep_r, torch.Tensor):
            rewards_proc.append(ep_r.float())
        else:
            rewards_proc.append(torch.as_tensor(ep_r, dtype=torch.float32))
    done_proc = []
    for ep_d in done_state:
        if isinstance(ep_d, torch.Tensor):
            done_proc.append(ep_d.float())
        else:
            done_proc.append(torch.as_tensor(ep_d, dtype=torch.float32))
    optimizer = optim.Adam(mdnrnn.parameters(), lr)
    lowest_loss = float(np.inf)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    for epoch in range(epochs):
        pbar = tqdm(
            create_sequence_batches(frames, actions_proc, game_states_proc, rewards_proc, done_proc, seq_length, batch_size),
            total=len(frames),
            desc=f"Training Epoch: {epoch + 1}/{epochs}"
        )
        total_loss = 0
        e_id = 1
        for batch in pbar:
            z_batch, a_batch, g_batch, r_batch, d_batch = batch
            z_batch = z_batch.to(device)
            a_batch = a_batch.to(device)
            g_batch = g_batch.to(device)
            r_batch = r_batch.to(device)
            d_batch = d_batch.to(device)
            h = torch.zeros(1, z_batch.size(1), mdnrnn.hidden_dim, device=device)
            c = torch.zeros(1, z_batch.size(1), mdnrnn.hidden_dim, device=device)
            z_in = z_batch[:-1]
            z_out = z_batch[1:]
            a_in = a_batch[:-1]
            g_in = g_batch[:-1]
            r_target = r_batch[1:].unsqueeze(-1)
            d_target = d_batch[1:].unsqueeze(-1)
            gmm, h, c, r, d = mdnrnn(z=z_in, a=a_in, g=g_in, h=h, c=c)
            z_pred_loss = -gmm.log_prob(z_out.reshape(-1, z_out.size(-1))).reshape(z_out.shape[0], z_out.shape[1]).mean()
            r_pred_loss = F.mse_loss(r, r_target)
            done_pred_loss = F.binary_cross_entropy_with_logits(d, d_target)
            loss = z_pred_loss + lambda_reward * r_pred_loss + lambda_done * done_pred_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdnrnn.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            curr_loss = total_loss / e_id
            pbar.set_postfix({"loss": curr_loss,
                              "reward_loss": r_pred_loss.item(),
                              "done_loss": done_pred_loss.item()
                              })
            e_id += 1
        if curr_loss < lowest_loss:
            lowest_loss = curr_loss
            torch.save(mdnrnn.state_dict(), f"./trained_models/mdnrnn_{timestamp}.pt")
            torch.save(mdnrnn.state_dict(), './trained_models/latest_mdnrnn.pt')
            print(f"Saving model... loss = {curr_loss}")
        scheduler.step(total_loss)
