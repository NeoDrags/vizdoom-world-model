import time
from argparse import ArgumentParser
import gymnasium
import numpy as np
import vizdoom as vzd
from vizdoom import gymnasium_wrapper  # noqa

DEFAULT_ENV = "VizdoomBasic-v1"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]  # type: ignore


def _print_obs_space(obs, obs_space, indent=0):
    """
    Help function to print the observation space and some stats about the observation
    """
    prefix = " " * indent
    if isinstance(obs_space, gymnasium.spaces.Dict):
        print(f"{prefix}Dict:")
        for key, space in obs_space.spaces.items():
            print(f"{prefix} Key: {key}")
            _print_obs_space(obs[key], space, indent + 4)
    else:
        print(f"{prefix}Space: {obs_space}")
        if isinstance(obs, np.ndarray):
            print(
                f"{prefix} Shape: {obs.shape}, Dtype: {obs.dtype}, Min value: {obs.min()}, Max value: {obs.max()}, Mean value: {obs.mean()}"
            )


if __name__ == "__main__":
    parser = ArgumentParser("ViZDoom example showing how to use Gymnasium wrapper.")
    parser.add_argument(
        dest="env",
        default=DEFAULT_ENV,
        choices=AVAILABLE_ENVS,
        help="Name of the environment to play",
    )

    # Create the Gymnasium environment
    env = gymnasium.make(
        parser.parse_args().env,
        render_mode="human",
        frame_skip=4,
        screen_resolution=vzd.ScreenResolution.RES_320X240,
    )
    
    for _ in range(10):
        done = False
        obs, info = env.reset(seed=42)
        while not done:
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            print("Observation:")
            print(obs["screen"])
            _print_obs_space(obs, env.observation_space)
            print(f"Reward: {rew}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            print(f"Info: {info}")
            print("=====================")

            time.sleep(
                1.0 / vzd.DEFAULT_TICRATE * env.unwrapped.frame_skip
            )  # Make it run at real-time speed

    env.close()