import time
from argparse import ArgumentParser
import gymnasium
import numpy as np
import vizdoom as vzd
from vizdoom import gymnasium_wrapper  # noqa
DEFAULT_ENV = "VizdoomBasic-v1"

AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]  # type: ignore

class Loader():
    def __init__(self, env):
        self.env = env
        pass

    def _print_obs_space(self, obs, obs_space, indent=0):
        """
        Help function to print the observation space and some stats about the observation
        """
        prefix = " " * indent
        if isinstance(obs_space, gymnasium.spaces.Dict):
            print(f"{prefix}Dict:")
            for key, space in obs_space.spaces.items():
                print(f"{prefix} Key: {key}")
                self._print_obs_space(obs[key], space, indent + 4)
        else:
            print(f"{prefix}Space: {obs_space}")
            if isinstance(obs, np.ndarray):
                print(
                    f"{prefix} Shape: {obs.shape}, Dtype: {obs.dtype}, Min value: {obs.min()}, Max value: {obs.max()}, Mean value: {obs.mean()}"
                )

    def runner(self):
        # Create the Gymnasium environment
        env = gymnasium.make(
            # Env ID, render_mode and frame_skip can be changed as needed
            self.env,
            render_mode="human",
            frame_skip=4,
            # Additional parameters can be passed to override the default environment config, however they should not be used for evaluation
            # Any kwargs that are supported by vizdoom.DoomGame.set_config can be passed here
            screen_resolution=vzd.ScreenResolution.RES_640X480,
        )

        # Rendering random rollouts for ten episodes
        for _ in range(10):
            done = False
            obs, info = env.reset(seed=42)
            while not done:
                obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
                done = terminated or truncated
                print("Observation:")
                self._print_obs_space(obs, env.observation_space)
                print(f"Reward: {rew}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"Info: {info}")
                print("=====================")

                time.sleep(
                    1.0 / vzd.DEFAULT_TICRATE * env.unwrapped.frame_skip
                )  # Make it run at real-time speed

        env.close()
