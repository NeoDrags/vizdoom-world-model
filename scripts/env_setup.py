from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as tf
import torch
import torch.nn.functional as F

def collect_rollouts(env, episodes):
    frames_list, action_list, game_state_list, reward_list, done_list = [], [], [], [], []
    pbar = tqdm(range(episodes))
    episode_id = 1
    for episode in pbar:
        episode_frames, episode_actions, episode_game, episode_rewards, episode_done = [], [], [], [], []
        pbar.set_description(f"Rolling out Episode: {episode + 1}/{episodes}")
        obs, _ = env.reset()
        done = False
        transform = tf.Compose([
            tf.ToTensor(),
            tf.Resize((64, 64))
        ])
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = obs["screen"]
            frame = transform(frame)
            episode_frames.append(frame)
            episode_game.append(obs["gamevariables"])
            episode_done.append(done)
            episode_actions.append(action)
            episode_rewards.append(reward)
            obs = next_obs
        episode_id += 1
        frames_list.append(episode_frames)
        game_state_list.append(episode_game)
        action_list.append(F.one_hot(torch.tensor(episode_actions)))
        reward_list.append(episode_rewards)
        done_list.append(episode_done)
    return frames_list, action_list, game_state_list, reward_list, done_list

def batch_frames(frames, batch_size):
    frames = [frame for episode in frames for frame in episode]
    frames = DataLoader(dataset=frames, batch_size=batch_size, shuffle=True)
    return frames