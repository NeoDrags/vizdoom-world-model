from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as tf

def collect_rollouts(env, episodes):
    frames_list, action_list, game_state_list = [], [], []
    pbar = tqdm(range(episodes))
    episode_id = 1
    for episode in pbar:
        pbar.set_description(f"Rolling out Episode: {episode + 1}/{episodes}")
        obs, _ = env.reset()
        done = False
        transform = tf.Compose([
            tf.ToTensor(),
            tf.Resize((64, 64))
        ])
        while not done:
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            action_list.append(action)
            frame = obs["screen"]
            frame = transform(frame)
            frames_list.append(frame)
            game_state_list.append(obs["gamevariables"])
            obs = next_obs
        episode_id += 1
    return frames_list, action_list, game_state_list 

def batch_frames(frames, batch_size):
    frames = DataLoader(dataset=frames,batch_size=batch_size, shuffle=True)
    return frames
