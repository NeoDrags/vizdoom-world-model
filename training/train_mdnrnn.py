import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def train_mdnrnn(vae, mdnrnn, lr, epochs, frames, actions, game_states, rewards, done_state, device, 
                 lambda_reward, lambda_done):
    vae.eval()
    optimizer = optim.Adam(mdnrnn.parameters(), lr)
    lowest_loss = float(np.inf)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    for epoch in range(epochs):
        pbar = tqdm(
            zip(frames, actions, game_states, rewards, done_state), 
            desc=f"Training Epoch: {epoch + 1}/{epochs}"
            )
        total_loss = 0
        curr_loss = 0
        e_id = 1
        h = torch.zeros(1, 1, mdnrnn.hidden_dim, device=device)
        c = torch.zeros(1, 1, mdnrnn.hidden_dim, device=device)
        for episode in pbar:
            frame, action, game_state, reward, done = episode
            frame = torch.stack(frame).to(device)
            action = torch.as_tensor(action, device=device).unsqueeze(1)
            game_state = torch.as_tensor(np.array(game_state), device=device)
            reward = torch.as_tensor(reward, dtype=torch.float16, device=device)
            done = torch.as_tensor(done, dtype=torch.float16, device=device)
            with torch.no_grad():
                _, z, _, _ = vae(frame)
            z = z.unsqueeze(1)
            game_state = game_state.unsqueeze(1)
            z_in = z[:-1]
            z_out = z[1:]
            a_in = action[:-1]
            g_in = game_state[:-1] 
            r_target = reward[1:].float().reshape(-1, 1, 1)
            d_target = done[1:].float().reshape(-1, 1, 1)
            gmm, h, c, r, d = mdnrnn(z = z_in, a = a_in, g = g_in, h = h, c = c)
            z_pred_loss = -gmm.log_prob(z_out).mean()
            r_pred_loss = F.mse_loss(r, r_target)
            done_pred_loss = F.binary_cross_entropy_with_logits(d, d_target)
            loss = z_pred_loss + lambda_reward * r_pred_loss + lambda_done * done_pred_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdnrnn.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            curr_loss = total_loss/e_id
            pbar.set_postfix({"loss": curr_loss, 
                              "reward_loss": r_pred_loss.item(), 
                              "done_loss": done_pred_loss.item()
                              })
            e_id += 1
        if curr_loss < lowest_loss:
            lowest_loss = curr_loss
            torch.save(vae.state_dict(), f"./trained_models/mdnrnn_{timestamp}.pt")
            torch.save(vae.state_dict(), './trained_models/latest_mdnrnn.pt')
            print(f"Saving model... loss = {curr_loss}")
        scheduler.step(total_loss)

